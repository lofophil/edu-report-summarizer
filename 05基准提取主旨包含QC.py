# -*- coding: utf-8 -*-
"""
infer_batch_docx.py

输入：BASE_DIR/data/*.docx
输出：BASE_DIR/result/<原docx同名>.txt
额外输出：BASE_DIR/result/run_report.jsonl（每篇一行汇总）
"""

from __future__ import annotations
from pathlib import Path
import re, json, hashlib
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from docx import Document  # pip install python-docx


# =========================
# 0) 路径（按你要求改）
# =========================
BASE_DIR = Path(r"D:\语义分析模型训练\PythonProject")

DATA_DIR = BASE_DIR / "data"         # 这里放 docx
RESULT_DIR = BASE_DIR / "result"     # 这里保存结果
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = "qwen2.5-mainidea-final-stable-fast-infer5-seedguide"
CKPT_DIR = BASE_DIR / "checkpoints" / RUN_NAME
LORA_DIR = CKPT_DIR / "final_best"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 召回缓存（建议放 result 下，避免污染训练缓存）
CACHE_DIR = RESULT_DIR / "_cache_embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# （可选）seed 引导召回：{"doc_id":"seed五段文本"}，不存在则自动退化
SEED_MAP_JSON = CKPT_DIR / "seed_map.json"


# =========================
# 1) 推理参数（与你当前版一致）
# =========================
CHUNK_TOKENS  = 1100
CHUNK_OVERLAP = 160

TOPN_PER_SECTION = 16
MMR_SELECT_K = 6
MMR_LAMBDA = 0.62

EVIDENCE_MAX_TOKENS_TOTAL = 3600
EVIDENCE_MAX_CHARS_PER_CHUNK = 2200

SECTION_TITLES = ["研究问题", "核心概念", "研究目标和内容", "研究成果", "研究效果"]

PER_SECTION_MAX_NEW_TOKENS = 300
PER_SECTION_RETRIES = 2

# embedding
EMB_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
EMB_DEVICE = "cpu"
EMB_BATCH_SIZE = 32


# =========================
# 2) 固定风格约束（保持一致）
# =========================
STYLE_INSTRUCTION_CN = (
    "你是一名教育科研报告“主旨提取”助手。"
    "请仅依据原文证据内容，按固定结构输出五段："
    "研究问题、核心概念、研究目标和内容、研究成果、研究效果。"
    "硬性约束："
    "一、每段不超过三百字；二、总字数不超过一千五百字；"
    "三、不得出现英文、字母、阿拉伯数字或罗马数字；"
    "四、不得编造、不得扩展、不得加入原文没有的信息；"
    "五、不得引用原文句子（不要出现引号或逐句摘抄），用概括表达；"
    "六、语言风格正式、收束、与 seed 样式一致；"
    "七、若原文信息不足以支撑某段，仍需用审慎表达给出可证据化概括，不得虚构细节。"
)

SYSTEM_PROMPT = (
    "你是一个严格遵循指令的中文写作模型。"
    "输出必须由给定原文证据直接支撑，且必须完全满足格式与长度约束。"
)


# =========================
# 3) 工具
# =========================
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u3000", " ").replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def to_safe_filename(name: str, repl: str = "_") -> str:
    name = normalize_text(name)
    name = re.sub(r'[\\/:*?"<>|]', repl, name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(" .")
    return name[:160]

def read_docx_text(docx_path: Path) -> str:
    doc = Document(str(docx_path))
    paras = []
    for p in doc.paragraphs:
        t = normalize_text(p.text)
        if t:
            paras.append(t)
    return "\n".join(paras)

def chunk_by_tokens(tokenizer, text: str, chunk_tokens: int, overlap: int) -> List[List[int]]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return []
    chunks, start = [], 0
    while start < len(ids):
        end = min(len(ids), start + chunk_tokens)
        chunks.append(ids[start:end])
        if end >= len(ids):
            break
        start = max(0, end - overlap)
    return chunks

def ids_to_text(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)

def budget_clip_with_tokenizer(tokenizer, blocks: List[str]) -> List[str]:
    kept, used = [], 0
    for b in blocks:
        b = normalize_text(b)
        if not b:
            continue
        if len(b) > EVIDENCE_MAX_CHARS_PER_CHUNK + 50:
            b = b[:EVIDENCE_MAX_CHARS_PER_CHUNK] + "…"
        tlen = len(tokenizer(b, add_special_tokens=False).input_ids)
        if used + tlen > EVIDENCE_MAX_TOKENS_TOTAL:
            break
        used += tlen
        kept.append(b)
    return kept


# =========================
# 4) 召回：bge + cache + MMR
# =========================
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL_NAME, device=EMB_DEVICE)

def encode_texts(embedder, texts: List[str]) -> np.ndarray:
    emb = embedder.encode(
        texts,
        batch_size=EMB_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    return emb

def cache_key(doc_id: str, text_hash: str) -> str:
    raw = f"{doc_id}||{text_hash}||{EMB_MODEL_NAME}||{CHUNK_TOKENS}||{CHUNK_OVERLAP}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def get_chunk_embeddings_cached(embedder, doc_id: str, chunk_texts: List[str]) -> np.ndarray:
    text_hash = hashlib.md5("".join(chunk_texts).encode("utf-8")).hexdigest()
    key = cache_key(doc_id, text_hash)
    path = CACHE_DIR / f"{key}.npz"
    if path.exists():
        data = np.load(path)
        emb = data["emb"]
        if emb.shape[0] == len(chunk_texts):
            return emb
    emb = encode_texts(embedder, chunk_texts)
    np.savez_compressed(path, emb=emb)
    return emb

def mmr_select(query_emb: np.ndarray, cand_embs: np.ndarray, cand_indices: List[int], k: int, lam: float) -> List[int]:
    q = query_emb
    E = cand_embs
    sim_q = E @ q
    selected = []
    selected_mask = np.zeros(E.shape[0], dtype=bool)

    for _ in range(min(k, E.shape[0])):
        if not selected:
            j = int(np.argmax(sim_q))
            selected.append(j)
            selected_mask[j] = True
            continue
        sel_E = E[selected]
        sim_sel = E @ sel_E.T
        max_sim_sel = sim_sel.max(axis=1)
        mmr = lam * sim_q - (1.0 - lam) * max_sim_sel
        mmr[selected_mask] = -1e9
        j = int(np.argmax(mmr))
        selected.append(j)
        selected_mask[j] = True

    return [cand_indices[j] for j in selected]

def recall_topn_then_mmr(embedder, doc_id: str, chunk_texts: List[str], query_text: str, topn: int, k: int, lam: float):
    query_text = normalize_text(query_text)[:2000]
    if not query_text:
        return []
    chunk_emb = get_chunk_embeddings_cached(embedder, doc_id, chunk_texts)
    q_emb = encode_texts(embedder, [query_text])[0]
    sims = chunk_emb @ q_emb

    topn = min(topn, len(chunk_texts))
    cand_idx = np.argpartition(-sims, topn - 1)[:topn].tolist()
    cand_idx.sort(key=lambda i: float(sims[i]), reverse=True)

    cand_embs = chunk_emb[cand_idx]
    chosen_idx = mmr_select(q_emb, cand_embs, cand_idx, k=k, lam=lam)
    return [(i, chunk_texts[i]) for i in chosen_idx]


# =========================
# 5) seed 引导召回
# =========================
def load_seed_map() -> Dict[str, str]:
    if SEED_MAP_JSON.exists():
        return json.loads(SEED_MAP_JSON.read_text(encoding="utf-8"))
    return {}

def guided_query(title: str, seed_hint: Optional[str]) -> str:
    return f"{title}\n{seed_hint}" if seed_hint else title


# =========================
# 6) QC
# =========================
def has_forbidden_chars(text: str) -> bool:
    if re.search(r"[A-Za-z0-9]", text):
        return True
    if re.search(r"\b[IVXLCDM]+\b", text):
        return True
    return False

def _strip_title_prefix(text: str, title: str) -> str:
    t = normalize_text(text)
    t = re.sub(rf"^\s*{re.escape(title)}\s*[:：]?\s*", "", t)
    return t

def _count_zh_chars(s: str) -> int:
    return len(re.sub(r"\s+", "", normalize_text(s)))

def qc_section(title: str, content: str) -> Tuple[bool, List[str]]:
    reasons = []
    c = normalize_text(content)
    if not c:
        return False, ["空段落"]
    if has_forbidden_chars(c):
        reasons.append("包含英文或数字（禁用）")
    if _count_zh_chars(c) > 300:
        reasons.append("本段超过三百字")
    if any(x in c for x in ["“", "”", "\"", "《", "》"]):
        reasons.append("包含引号或书名号，疑似引用")
    return (len(reasons) == 0), reasons

def split_five_sections(text: str) -> Optional[Dict[str, str]]:
    t = normalize_text(text)
    patt = r"(研究问题|核心概念|研究目标和内容|研究成果|研究效果)\s*[:：]?"
    parts = re.split(patt, t)
    if len(parts) < 11:
        return None
    d = {}
    for i in range(1, len(parts)-1, 2):
        title = parts[i]
        content = normalize_text(parts[i+1])
        d[title] = content
    if any(k not in d or not d[k] for k in SECTION_TITLES):
        return None
    return d

def qc_summary(text: str) -> Tuple[bool, List[str]]:
    reasons = []
    t = normalize_text(text)
    if not t:
        return False, ["空输出"]
    if has_forbidden_chars(t):
        reasons.append("包含英文或数字（禁用）")
    sec = split_five_sections(t)
    if sec is None:
        reasons.append("未按五段结构输出或标题缺失")
        return False, reasons
    total_chars = _count_zh_chars(t)
    if total_chars > 1500:
        reasons.append("总字数超限")
    for k in SECTION_TITLES:
        if _count_zh_chars(sec[k]) > 300:
            reasons.append(f"{k} 超过三百字")
    if any(x in t for x in ["“", "”", "\"", "《", "》"]):
        reasons.append("包含引号或书名号，疑似引用")
    return (len(reasons) == 0), reasons


# =========================
# 7) 五段独立生成
# =========================
@torch.no_grad()
def generate_section_once(model, tokenizer, title: str, evidence_blocks: List[str], extra_tip: str = "") -> str:
    user_content = (
        f"{STYLE_INSTRUCTION_CN}\n\n"
        f"【任务】只输出《{title}》这一段内容。\n"
        f"【严格要求】\n"
        f"1）只输出段落内容本身，不要输出其它标题，不要列条目编号；\n"
        f"2）不得出现英文、字母、阿拉伯数字或罗马数字；\n"
        f"3）本段不超过三百字；\n"
        f"4）不得引用原文句子，不要出现引号或书名号；\n"
        f"5）只依据下方证据块概括，不得补充未出现信息。\n\n"
        f"【证据块】\n" + "\n\n".join(evidence_blocks) +
        ("\n\n" + extra_tip if extra_tip else "") +
        "\n\n请输出该段。"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=PER_SECTION_MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = normalize_text(gen)
    gen = _strip_title_prefix(gen, title)
    return gen


@torch.no_grad()
def infer_one(model, tok, embedder, doc_id: str, doc_text: str, seed_hint: Optional[str], out_txt: Path):
    full_text = normalize_text(doc_text)
    chunk_ids_list = chunk_by_tokens(tok, full_text, CHUNK_TOKENS, CHUNK_OVERLAP)
    chunk_texts = [ids_to_text(tok, ids) for ids in chunk_ids_list]

    section_outputs: Dict[str, str] = {}
    section_qc: Dict[str, List[str]] = {}

    for title in SECTION_TITLES:
        q = guided_query(title, seed_hint)
        hits = recall_topn_then_mmr(embedder, doc_id, chunk_texts, q, TOPN_PER_SECTION, MMR_SELECT_K, MMR_LAMBDA)
        evidence = []
        for _, txt in hits:
            txt = normalize_text(txt)
            if txt:
                evidence.append(f"【{title}证据】\n{txt}")
        evidence = budget_clip_with_tokenizer(tok, evidence)

        ans = generate_section_once(model, tok, title, evidence)
        ok, reasons = qc_section(title, ans)

        tries = 0
        while (not ok) and (tries < PER_SECTION_RETRIES):
            tip = "【质检提醒】你上一版未通过：" + "；".join(reasons) + "。请严格改正：只输出该段正文，三百字内，不含英文数字与引号书名号。"
            ans2 = generate_section_once(model, tok, title, evidence, extra_tip=tip)
            ok2, reasons2 = qc_section(title, ans2)
            ans, ok, reasons = ans2, ok2, reasons2
            tries += 1

        section_outputs[title] = ans
        section_qc[title] = ([] if ok else reasons)

    merged = "\n".join([f"{t}：{normalize_text(section_outputs.get(t,''))}" for t in SECTION_TITLES])
    merged = normalize_text(merged)

    ok_all, reasons_all = qc_summary(merged)
    tag = "[QC_OK]" if ok_all else "[QC_FAIL] " + "；".join(reasons_all)

    detail = []
    for t in SECTION_TITLES:
        detail.append(f"{t}=>OK" if not section_qc[t] else f"{t}=>FAIL：{'；'.join(section_qc[t])}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(merged + "\n\n" + tag + "\n" + "\n".join(detail), encoding="utf-8")

    return merged, ok_all, reasons_all, section_qc


# =========================
# 8) 批处理：data/*.docx -> result/同名.txt
# =========================
def main():
    assert torch.cuda.is_available(), "CUDA 不可用"
    assert LORA_DIR.exists(), f"Missing LoRA: {LORA_DIR}"
    assert DATA_DIR.exists(), f"Missing data dir: {DATA_DIR}"

    seed_map = load_seed_map()

    # 只扫描 data 下的 docx
    docx_list = sorted(DATA_DIR.glob("*.docx"))
    print(f"[INFO] docx_files={len(docx_list)}  data_dir={DATA_DIR}")
    print(f"[INFO] lora_dir={LORA_DIR}")
    print(f"[INFO] result_dir={RESULT_DIR}")

    # 一次加载（避免每篇重复加载模型）
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")

    model = PeftModel.from_pretrained(base, str(LORA_DIR)).to("cuda")
    model.eval()

    embedder = load_embedder()

    report_path = RESULT_DIR / "run_report.jsonl"
    with report_path.open("w", encoding="utf-8") as rep:
        for i, p in enumerate(docx_list, 1):
            doc_id = p.stem
            text = read_docx_text(p)

            if not normalize_text(text):
                rec = {"doc_id": doc_id, "src_docx": str(p), "qc_ok": False, "qc_reasons": ["空文档或无法读取"], "out_path": ""}
                rep.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(docx_list)}] {doc_id} -> SKIP(empty)")
                continue

            out_txt = RESULT_DIR / f"{to_safe_filename(doc_id)}.txt"
            seed_hint = seed_map.get(doc_id)

            merged, ok_all, reasons_all, section_qc = infer_one(
                model=model,
                tok=tok,
                embedder=embedder,
                doc_id=doc_id,
                doc_text=text,
                seed_hint=seed_hint,
                out_txt=out_txt,
            )

            rec = {
                "doc_id": doc_id,
                "src_docx": str(p),
                "out_path": str(out_txt),
                "qc_ok": bool(ok_all),
                "qc_reasons": reasons_all,
                "section_fail": {k: v for k, v in section_qc.items() if v},
            }
            rep.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(docx_list)}] {doc_id} -> {out_txt.name}  QC={'OK' if ok_all else 'FAIL'}")

    print(f"[DONE] report: {report_path}")


if __name__ == "__main__":
    main()
