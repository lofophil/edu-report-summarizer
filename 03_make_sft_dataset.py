# -*- coding: utf-8 -*-
"""
03_make_sft_dataset.py (doc_id aligned)

你的目录：
BASE_DIR = D:\\语义分析模型训练\\PythonProject
DATA_DIR = BASE_DIR\\data
SEED_DIR = BASE_DIR\\seed
OUT_DIR  = BASE_DIR\\data

输入：
- data/reports_sections.jsonl   (第二步结构化原文，包含 doc_id/title/sections/full_text)
- seed/*.txt                    (你手写的 50 个 seed，要求与 doc_id 对齐)

输出：
- data/sft_train.jsonl
- data/sft_valid.jsonl
- data/03_unmatched_seeds.json          (seed 未匹配到 doc_id)
- data/03_unmatched_docs.json           (doc_id 没有对应 seed)
- data/03_docid_to_safe_filename.json   (doc_id -> 安全文件名建议，用于重命名 seed)

关键点：
- 对齐以 doc_id 为主，支持 seed 文件名为：
  1) 完全等于 doc_id
  2) 等于 doc_id 的“安全文件名版本”（替换 Windows 非法字符）
  3) 宽松匹配：清洗后相等（去空格、符号）
- 训练输入用 full_text（长原文），不再用 sections（避免训练成“只会改写结构化文本”）
- 每条样本写入固定 system+user，并把 seed 风格约束写进 user instruction（训练期固化）
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================
# 0. 路径（按你给的）
# =========================
BASE_DIR = Path(r"D:\语义分析模型训练\PythonProject")
DATA_DIR = BASE_DIR / "data"
SEED_DIR = BASE_DIR / "seed"
OUT_DIR  = BASE_DIR / "data"

IN_SECTIONS_JSONL = DATA_DIR / "reports_sections.jsonl"

OUT_TRAIN_JSONL = OUT_DIR / "sft_train.jsonl"
OUT_VALID_JSONL = OUT_DIR / "sft_valid.jsonl"

OUT_UNMATCHED_SEEDS = OUT_DIR / "03_unmatched_seeds.json"
OUT_UNMATCHED_DOCS  = OUT_DIR / "03_unmatched_docs.json"
OUT_DOCID_SAFE_MAP  = OUT_DIR / "03_docid_to_safe_filename.json"

# 长文处理（按字符近似；显存够就尽量大）
CHUNK_CHARS   = 60000
OVERLAP_CHARS = 600

VALID_RATIO = 0.02
RANDOM_SEED = 42

# =========================
# 1. 固化 seed 风格约束（每条样本都携带）
# =========================
STYLE_INSTRUCTION_CN = (
    "你是一名教育科研报告“主旨提取”助手。"
    "请仅依据原文内容，按固定结构输出五段："
    "研究问题、核心概念、研究目标和内容、研究成果、研究效果。"
    "硬性约束："
    "一、每段不超过三百字；二、总字数不超过一千五百字；"
    "三、不得出现英文、字母、阿拉伯数字或罗马数字；"
    "四、不得编造、不得扩展、不得加入原文没有的信息；"
    "五、不得引用原文句子（不要出现引号或逐句摘抄），用概括表达；"
    "六、语言风格正式、收束、与 seed 样式一致；"
    "七、若原文信息不足以支撑某段，仍需用审慎表达给出该段的可证据化概括，不得虚构细节。"
)

SYSTEM_PROMPT = (
    "你是一个严格遵循指令的中文写作模型。"
    "输出必须由给定原文直接支撑，且必须完全满足格式与长度约束。"
)

# =========================
# 2. 基础 I/O
# =========================
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u3000", " ").replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# =========================
# 3. doc_id 对齐相关：安全文件名与宽松匹配
# =========================
_WINDOWS_FORBIDDEN = r'\/:*?"<>|'

def to_safe_filename(name: str, repl: str = "_") -> str:
    """把 doc_id 转成 Windows 可用的文件名（不含扩展名）"""
    name = normalize_text(name)
    # 替换非法字符
    name = re.sub(r'[\\/:*?"<>|]', repl, name)
    # 压缩空白
    name = re.sub(r"\s+", " ", name).strip()
    # 避免结尾点或空格（Windows 不友好）
    name = name.rstrip(" .")
    return name

def loose_key(s: str) -> str:
    """宽松匹配 key：去空白和符号，只保留中英文数字下划线和中文"""
    s = normalize_text(s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s.lower()

# =========================
# 4. 长文切块（段落级 + overlap），但最终合并成单条输入避免重复学习
# =========================
def split_by_paragraph(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if "\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    elif "\n" in text:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    else:
        parts = [text]
    return parts

def chunk_paragraphs(paras: List[str], chunk_chars: int, overlap_chars: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
        buf, buf_len = [], 0

    for p in paras:
        p = p.strip()
        if not p:
            continue

        if len(p) > chunk_chars:
            flush()
            start = 0
            while start < len(p):
                end = min(len(p), start + chunk_chars)
                chunks.append(p[start:end].strip())
                if end >= len(p):
                    break
                start = max(end - overlap_chars, end)
            continue

        if buf_len + len(p) + 2 <= chunk_chars:
            buf.append(p)
            buf_len += len(p) + 2
        else:
            flush()
            if overlap_chars > 0 and chunks:
                tail = chunks[-1][-overlap_chars:].strip()
                if tail:
                    buf = [tail, p]
                    buf_len = len(tail) + len(p) + 2
                else:
                    buf = [p]
                    buf_len = len(p)
            else:
                buf = [p]
                buf_len = len(p)

    flush()

    # 弱去重，防止 overlap 导致完全重复块
    dedup: List[str] = []
    seen = set()
    for c in chunks:
        key = c[:200]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    return dedup

# =========================
# 5. 构造训练样本（ChatML）
# =========================
def make_chatml_sample(source_text: str, target: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = (
        f"{STYLE_INSTRUCTION_CN}\n\n"
        f"【原文】\n{source_text}\n\n"
        f"请开始输出。"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": normalize_text(target)},
        ],
        "meta": meta
    }

# =========================
# 6. 读取 seed 并按 doc_id 对齐
# =========================
def read_seed_txts(seed_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted(seed_dir.glob("*.txt")):
        txt = normalize_text(p.read_text(encoding="utf-8", errors="ignore"))
        if not txt:
            continue
        items.append({
            "seed_stem": p.stem,
            "seed_path": str(p),
            "target": txt
        })
    return items

def build_docid_index(docs: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    返回：
    - doc_map: doc_id -> doc
    - safe_map: doc_id -> safe_filename(doc_id)  （用于你批量重命名 seed）
    """
    doc_map: Dict[str, Dict[str, Any]] = {}
    safe_map: Dict[str, str] = {}
    for d in docs:
        doc_id = normalize_text(str(d.get("doc_id", "")))
        if not doc_id:
            continue
        doc_map[doc_id] = d
        safe_map[doc_id] = to_safe_filename(doc_id)
    return doc_map, safe_map

def align_seed_to_docid(seed_stem: str, doc_map: Dict[str, Dict[str, Any]], safe_map: Dict[str, str]) -> Optional[str]:
    """
    返回匹配到的 doc_id（严格以 doc_id 为核心）
    匹配优先级：
    1) seed_stem == doc_id
    2) seed_stem == safe_filename(doc_id)
    3) loose_key(seed_stem) == loose_key(doc_id)
    4) loose_key(seed_stem) == loose_key(safe_filename(doc_id))
    """
    seed_stem = normalize_text(seed_stem)
    if not seed_stem:
        return None

    # 1) 直接命中 doc_id
    if seed_stem in doc_map:
        return seed_stem

    # 2) 命中 safe_filename
    for doc_id, safe_name in safe_map.items():
        if seed_stem == safe_name:
            return doc_id

    # 3) 宽松匹配 doc_id
    sk = loose_key(seed_stem)
    if sk:
        for doc_id in doc_map.keys():
            if loose_key(doc_id) == sk:
                return doc_id

    # 4) 宽松匹配 safe_filename
    if sk:
        for doc_id, safe_name in safe_map.items():
            if loose_key(safe_name) == sk:
                return doc_id

    return None

# =========================
# 7. 主流程
# =========================
def main():
    random.seed(RANDOM_SEED)

    if not IN_SECTIONS_JSONL.exists():
        raise FileNotFoundError(f"未找到：{IN_SECTIONS_JSONL}")
    if not SEED_DIR.exists():
        raise FileNotFoundError(f"未找到：{SEED_DIR}")

    docs = read_jsonl(IN_SECTIONS_JSONL)
    if not docs:
        raise RuntimeError(f"{IN_SECTIONS_JSONL} 为空或解析失败。")

    # 你第二步的 records 结构：doc_id/title/sections/full_text
    doc_map, safe_map = build_docid_index(docs)
    OUT_DOCID_SAFE_MAP.write_text(json.dumps(safe_map, ensure_ascii=False, indent=2), encoding="utf-8")

    seed_items = read_seed_txts(SEED_DIR)
    if not seed_items:
        raise RuntimeError(f"{SEED_DIR} 下未读取到任何 seed txt。")

    samples: List[Dict[str, Any]] = []
    unmatched_seeds: List[Dict[str, Any]] = []
    matched_docids = set()

    for s in seed_items:
        seed_stem = s["seed_stem"]
        doc_id = align_seed_to_docid(seed_stem, doc_map, safe_map)
        if doc_id is None:
            unmatched_seeds.append({
                "seed_stem": seed_stem,
                "seed_path": s["seed_path"]
            })
            continue

        doc = doc_map[doc_id]
        matched_docids.add(doc_id)

        # 训练输入：full_text（长原文）
        full_text = normalize_text(str(doc.get("full_text", "")))
        if not full_text:
            # 若 full_text 缺失，则退回 title+sections 拼装（但你的文件里 full_text 是存在的）
            title = normalize_text(str(doc.get("title", "")))
            sections = doc.get("sections", {})
            parts = []
            if title:
                parts.append(f"标题：{title}")
            if isinstance(sections, dict):
                for k, v in sections.items():
                    v2 = normalize_text(str(v))
                    if v2:
                        parts.append(f"{k}：\n{v2}")
            full_text = "\n\n".join(parts).strip()

        if not full_text:
            unmatched_seeds.append({
                "seed_stem": seed_stem,
                "seed_path": s["seed_path"],
                "reason": "empty_full_text"
            })
            continue

        # 长文切块 + overlap，但合并为单条（避免同一 target 学 N 次造成重复）
        paras = split_by_paragraph(full_text)
        chunks = chunk_paragraphs(paras, CHUNK_CHARS, OVERLAP_CHARS)
        merged_source = "\n\n".join(chunks).strip()

        meta = {
            "doc_id": doc_id,
            "safe_doc_id": safe_map.get(doc_id, ""),
            "seed_stem": seed_stem,
            "seed_path": s["seed_path"],
            "title": normalize_text(str(doc.get("title", ""))),
            "n_chunks": len(chunks),
            "source_chars": len(merged_source),
            "target_chars": len(s["target"]),
        }
        samples.append(make_chatml_sample(merged_source, s["target"], meta))

    # doc_id 没有 seed 的清单（反向检查）
    unmatched_docs = []
    for doc_id in doc_map.keys():
        if doc_id not in matched_docids:
            unmatched_docs.append({
                "doc_id": doc_id,
                "safe_doc_id": safe_map.get(doc_id, "")
            })

    if unmatched_seeds:
        OUT_UNMATCHED_SEEDS.write_text(json.dumps(unmatched_seeds, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WARN] seed 未匹配到 doc_id：{len(unmatched_seeds)} -> {OUT_UNMATCHED_SEEDS}")

    if unmatched_docs:
        OUT_UNMATCHED_DOCS.write_text(json.dumps(unmatched_docs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WARN] doc_id 没有对应 seed：{len(unmatched_docs)} -> {OUT_UNMATCHED_DOCS}")

    if not samples:
        raise RuntimeError(
            "没有生成任何 SFT 样本：请确保 seed 文件名与 doc_id 对齐。\n"
            f"已输出 doc_id->安全文件名映射：{OUT_DOCID_SAFE_MAP}"
        )

    random.shuffle(samples)
    n_valid = max(1, int(len(samples) * VALID_RATIO))
    valid = samples[:n_valid]
    train = samples[n_valid:]

    write_jsonl(OUT_TRAIN_JSONL, train)
    write_jsonl(OUT_VALID_JSONL, valid)

    print(f"[OK] docs: {len(docs)}")
    print(f"[OK] seeds: {len(seed_items)}")
    print(f"[OK] matched samples: {len(samples)}")
    print(f"[OK] train: {len(train)} -> {OUT_TRAIN_JSONL}")
    print(f"[OK] valid: {len(valid)} -> {OUT_VALID_JSONL}")
    print(f"[OK] doc_id->safe filename map -> {OUT_DOCID_SAFE_MAP}")


if __name__ == "__main__":
    main()
