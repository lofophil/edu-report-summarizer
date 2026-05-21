# -*- coding: utf-8 -*-
"""
04_train_sft_qwen_lora_FINAL_STABLE_FAST_INFER5_SEEDGUIDE.py

最终稳定 + 提速 + 推理稳态版：
- SafeEvalLossTrainer（修法A，避免 eval_loss 缺失导致中断）
- base fp16 + LoRA trainable fp32（关闭 AMP，稳）
- 证据召回：bge + TopN + MMR + 证据预算 + 缓存（缓存键含文本hash，防脏缓存）
- 训练样本：每 doc 多 variants，但通过 jitter 避免重复样本浪费
- 推理：五段独立生成 + 分段QC + 失败段重试 + 合并总QC
- Seed 引导召回：若能为 doc 找到 seed，则 query=标题 + seed；否则退化为标题
"""

from __future__ import annotations
from pathlib import Path
import re, json, random, hashlib, csv, time, inspect
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel


# =========================
# 0) 路径配置
# =========================
BASE_DIR = Path(r"D:\语义分析模型训练\PythonProject")
DATA_DIR = BASE_DIR / "data"
SEED_DIR = BASE_DIR / "seed"
IN_REPORTS = DATA_DIR / "reports_sections.jsonl"

RUN_NAME = "qwen2.5-mainidea-final-stable-fast-infer5-seedguide"
OUT_DIR = BASE_DIR / "checkpoints" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

QA_DIR = OUT_DIR / "qa_outputs"
QA_DIR.mkdir(parents=True, exist_ok=True)

EVAL_LOG_CSV = OUT_DIR / "eval_log.csv"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = DATA_DIR / "cache_embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1) 参数（进度 + 稳定优先）
# =========================
MAX_SEQ_LEN = 4096

CHUNK_TOKENS  = 1100
CHUNK_OVERLAP = 160

TOPN_PER_SECTION = 16
MMR_SELECT_K = 6
MMR_LAMBDA = 0.62

VARIANTS_PER_DOC = 12          # 仍保留 12，但通过 jitter 让它“真不同”
VALID_RATIO = 0.10
RANDOM_SEED = 42

EVIDENCE_MAX_TOKENS_TOTAL = 3600
EVIDENCE_MAX_CHARS_PER_CHUNK = 2200
MIN_SUP_TOKENS = 220

EMB_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
EMB_DEVICE = "cpu"
EMB_BATCH_SIZE = 32

# 推理：五段独立生成
PER_SECTION_MAX_NEW_TOKENS = 300
PER_SECTION_RETRIES = 2       # 单段失败重试次数（1 一般足够）


# =========================
# 2) 固定风格约束
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

SECTION_TITLES = ["研究问题", "核心概念", "研究目标和内容", "研究成果", "研究效果"]


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
    return name

def loose_key(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s.lower()

def read_reports_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def read_seed_txts(seed_dir: Path) -> List[Dict[str, Any]]:
    items = []
    for p in sorted(seed_dir.glob("*.txt")):
        txt = normalize_text(p.read_text(encoding="utf-8", errors="ignore"))
        if txt:
            items.append({"seed_stem": p.stem, "seed_path": str(p), "target": txt})
    return items

def build_docid_maps(docs: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    doc_map, safe_map = {}, {}
    for d in docs:
        doc_id = normalize_text(str(d.get("doc_id", "")))
        if doc_id:
            doc_map[doc_id] = d
            safe_map[doc_id] = to_safe_filename(doc_id)
    return doc_map, safe_map

def align_seed_to_docid(seed_stem: str, doc_map: Dict[str, Dict[str, Any]], safe_map: Dict[str, str]) -> Optional[str]:
    seed_stem = normalize_text(seed_stem)
    if not seed_stem:
        return None
    if seed_stem in doc_map:
        return seed_stem
    for doc_id, safe_name in safe_map.items():
        if seed_stem == safe_name:
            return doc_id
    sk = loose_key(seed_stem)
    if sk:
        for doc_id in doc_map.keys():
            if loose_key(doc_id) == sk:
                return doc_id
        for doc_id, safe_name in safe_map.items():
            if loose_key(safe_name) == sk:
                return doc_id
    return None


# =========================
# 4) token 切块
# =========================
def chunk_by_tokens(tokenizer, text: str, chunk_tokens: int, overlap: int) -> List[List[int]]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return []
    chunks = []
    start = 0
    while start < len(ids):
        end = min(len(ids), start + chunk_tokens)
        chunks.append(ids[start:end])
        if end >= len(ids):
            break
        start = max(0, end - overlap)
    return chunks

def ids_to_text(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


# =========================
# 5) 语义召回（SentenceTransformer + Cache + MMR）
# =========================
def cache_key(doc_id: str, text_hash: str) -> str:
    raw = f"{doc_id}||{text_hash}||{EMB_MODEL_NAME}||{CHUNK_TOKENS}||{CHUNK_OVERLAP}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

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
# 6) 证据预算裁剪 + 轻扰动（防重复）
# =========================
def budget_clip_with_tokenizer(tokenizer, blocks: List[str]) -> List[str]:
    kept = []
    used_tokens = 0
    for b in blocks:
        b = normalize_text(b)
        if not b:
            continue
        if len(b) > EVIDENCE_MAX_CHARS_PER_CHUNK + 50:
            b = b[:EVIDENCE_MAX_CHARS_PER_CHUNK] + "…"
        tlen = len(tokenizer(b, add_special_tokens=False).input_ids)
        if used_tokens + tlen > EVIDENCE_MAX_TOKENS_TOTAL:
            break
        used_tokens += tlen
        kept.append(b)
    return kept

def jitter_evidence_blocks(ev_blocks: List[str], drop_p=0.12, max_blocks=24) -> List[str]:
    ev = ev_blocks[:]
    random.shuffle(ev)
    ev = [b for b in ev if random.random() > drop_p]
    return ev[:max_blocks]


# =========================
# 7) ChatML 编码：只监督 assistant
# =========================
def encode_assistant_only(tokenizer, messages: List[Dict[str, str]], max_len: int) -> Dict[str, Any]:
    prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer_text = messages[-1]["content"]

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids[:]

    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    pad_id = tokenizer.pad_token_id
    attn = [1] * len(input_ids)
    if len(input_ids) < max_len:
        pad_n = max_len - len(input_ids)
        input_ids += [pad_id] * pad_n
        labels += [-100] * pad_n
        attn += [0] * pad_n

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def count_supervised(labels: List[int]) -> int:
    return sum(1 for x in labels if x != -100)


# =========================
# 8) Seed 引导召回（训练/QA/推理通用）
# =========================
def make_seed_map(docs: List[Dict[str, Any]], seeds: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    返回 doc_id -> seed_target 映射（尽可能对齐）
    """
    doc_map, safe_map = build_docid_maps(docs)
    out = {}
    for s in seeds:
        doc_id = align_seed_to_docid(s["seed_stem"], doc_map, safe_map)
        if doc_id:
            out[doc_id] = s["target"]
    return out

def guided_query(title: str, seed_hint: Optional[str]) -> str:
    """
    seed 引导召回：query = 标题 + seed（找不到 seed 就只用标题）
    """
    if seed_hint:
        return f"{title}\n{seed_hint}"
    return title


# =========================
# 9) 分段证据召回（训练样本构造）
# =========================
def build_evidence_by_sections(embedder, tokenizer, doc_id: str, chunk_texts: List[str], seed_hint: Optional[str]) -> List[str]:
    evidence_blocks = []
    for title in SECTION_TITLES:
        q = guided_query(title, seed_hint)
        hits = recall_topn_then_mmr(
            embedder=embedder,
            doc_id=doc_id,
            chunk_texts=chunk_texts,
            query_text=q,
            topn=TOPN_PER_SECTION,
            k=MMR_SELECT_K,
            lam=MMR_LAMBDA,
        )
        for _, txt in hits:
            txt = normalize_text(txt)
            if txt:
                evidence_blocks.append(f"【{title}证据】\n{txt}")

    # 去重
    uniq, seen = [], set()
    for b in evidence_blocks:
        h = hashlib.md5(b.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append(b)

    # 轻扰动（避免 VARIANTS 重复浪费）
    uniq = jitter_evidence_blocks(uniq, drop_p=0.12, max_blocks=24)
    uniq = budget_clip_with_tokenizer(tokenizer, uniq)
    return uniq


# =========================
# 10) 构造训练样本
# =========================
def build_samples(tokenizer, docs: List[Dict[str, Any]], seed_map: Dict[str, str]) -> List[Dict[str, Any]]:
    doc_map, _ = build_docid_maps(docs)
    embedder = load_embedder()
    print(f"[INFO] embedder={EMB_MODEL_NAME} device={EMB_DEVICE}")

    samples = []
    missed, dropped = 0, 0

    # 只对有 seed 的 doc 做训练（与你现有策略一致）
    for doc_id, seed_target in seed_map.items():
        doc = doc_map.get(doc_id)
        if not doc:
            missed += 1
            continue

        full_text = normalize_text(str(doc.get("full_text", "")))
        if not full_text:
            missed += 1
            continue

        chunk_ids_list = chunk_by_tokens(tokenizer, full_text, CHUNK_TOKENS, CHUNK_OVERLAP)
        if not chunk_ids_list:
            missed += 1
            continue
        chunk_texts = [ids_to_text(tokenizer, ids) for ids in chunk_ids_list]

        for _ in range(VARIANTS_PER_DOC):
            ev_blocks = build_evidence_by_sections(
                embedder=embedder,
                tokenizer=tokenizer,
                doc_id=doc_id,
                chunk_texts=chunk_texts,
                seed_hint=seed_target,  # ✅ seed 引导召回（训练）
            )
            if not ev_blocks:
                dropped += 1
                continue

            # 训练提示：可少量注入“质检提醒”增强二次纠错能力
            qc_tip = ""
            if random.random() < 0.20:
                qc_tip = "\n\n【质检提醒】请再次检查：五段齐全；每段三百字内；总字数一千五百字内；不含英文数字与引号书名号。"

            user_content = (
                f"{STYLE_INSTRUCTION_CN}\n\n"
                f"以下为从原文中抽取的证据块集合（并非全文）。"
                f"你必须仅依据这些证据块进行主旨提取，不得补充未出现的信息。\n\n"
                + "\n\n".join(ev_blocks)
                + qc_tip
                + "\n\n请开始输出。"
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": seed_target},
            ]
            enc = encode_assistant_only(tokenizer, messages, MAX_SEQ_LEN)
            sup = count_supervised(enc["labels"])
            if sup < MIN_SUP_TOKENS:
                dropped += 1
                continue
            samples.append(enc)

    if missed:
        print(f"[WARN] missed docs: {missed}")
    if dropped:
        print(f"[WARN] dropped samples: {dropped}")

    if samples:
        sup_list = [count_supervised(x["labels"]) for x in samples]
        print(f"[INFO] supervised tokens: min={min(sup_list)} avg={sum(sup_list)/len(sup_list):.1f} max={max(sup_list)}")
    return samples


# =========================
# 11) 单卡 FP16 基座 + LoRA(fp32 trainable)
# =========================
def load_model_tokenizer_fp16_single_gpu():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading base model (fp16, single GPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer

def apply_lora(model):
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # LoRA 参数 fp32
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    model.print_trainable_parameters()
    return model

def simple_collator(features):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }


# =========================
# 12) eval_loss 记录
# =========================
class EvalLoggerCallback(TrainerCallback):
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["time", "global_step", "epoch", "eval_loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if "eval_loss" in metrics:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), state.global_step, state.epoch, metrics["eval_loss"]])


# =========================
# 13) 修法A：保证 evaluate() 一定返回 eval_loss
# =========================
class SafeEvalLossTrainer(Trainer):
    @torch.no_grad()
    def _manual_eval_loss(self, eval_dataset=None):
        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        losses = []
        for batch in dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            if loss is not None:
                losses.append(loss.detach().float().cpu().item())
        if not losses:
            return None
        return float(np.mean(losses))

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        k = f"{metric_key_prefix}_loss"
        if k not in metrics:
            manual = self._manual_eval_loss(eval_dataset)
            if manual is not None:
                metrics[k] = manual
        return metrics


# =========================
# 14) 推理质检（总 QC + 分段 QC）
# =========================
def has_forbidden_chars(text: str) -> bool:
    if re.search(r"[A-Za-z0-9]", text):
        return True
    if re.search(r"\b[IVXLCDM]+\b", text):
        return True
    return False

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

    total_chars = len(re.sub(r"\s+", "", t))
    if total_chars > 1500:
        reasons.append("总字数超限")

    for k in SECTION_TITLES:
        c = len(re.sub(r"\s+", "", sec[k]))
        if c > 300:
            reasons.append(f"{k} 超过三百字")

    if "“" in t or "”" in t or "\"" in t or "《" in t or "》" in t:
        reasons.append("包含引号或书名号，疑似引用")

    ok = (len(reasons) == 0)
    return ok, reasons

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
    if "“" in c or "”" in c or "\"" in c or "《" in c or "》" in c:
        reasons.append("包含引号或书名号，疑似引用")
    ok = (len(reasons) == 0)
    return ok, reasons


# =========================
# 15) 五段独立生成（终极稳态）+ seed 引导召回（推理/QA用）
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
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
def infer_five_sections_stable(
    base_model_name: str,
    lora_dir: Path,
    doc_id: str,
    doc_text: str,
    out_txt: Path,
    *,
    seed_hint: Optional[str] = None,
    per_section_retries: int = PER_SECTION_RETRIES,
):
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")

    model = PeftModel.from_pretrained(base, str(lora_dir)).to("cuda")
    model.eval()

    embedder = load_embedder()

    full_text = normalize_text(doc_text)
    chunk_ids_list = chunk_by_tokens(tok, full_text, CHUNK_TOKENS, CHUNK_OVERLAP)
    chunk_texts = [ids_to_text(tok, ids) for ids in chunk_ids_list]

    section_outputs: Dict[str, str] = {}
    section_qc: Dict[str, List[str]] = {}

    for title in SECTION_TITLES:
        q = guided_query(title, seed_hint)  # ✅ seed 引导召回（推理/QA）
        hits = recall_topn_then_mmr(
            embedder=embedder,
            doc_id=doc_id,
            chunk_texts=chunk_texts,
            query_text=q,
            topn=TOPN_PER_SECTION,
            k=MMR_SELECT_K,
            lam=MMR_LAMBDA,
        )
        evidence = []
        for _, txt in hits:
            txt = normalize_text(txt)
            if txt:
                evidence.append(f"【{title}证据】\n{txt}")
        evidence = budget_clip_with_tokenizer(tok, evidence)

        ans = generate_section_once(model, tok, title, evidence)
        ok, reasons = qc_section(title, ans)

        tries = 0
        while (not ok) and (tries < per_section_retries):
            tip = (
                "\n【质检提醒】你上一版未通过："
                + "；".join(reasons)
                + "。请严格改正：只输出该段正文，三百字内，不含英文数字与引号书名号。"
            )
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
        if section_qc[t]:
            detail.append(f"{t}=>FAIL：{'；'.join(section_qc[t])}")
        else:
            detail.append(f"{t}=>OK")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(merged + "\n\n" + tag + "\n" + "\n".join(detail), encoding="utf-8")

    return merged, ok_all, reasons_all, section_qc


# =========================
# 16) QA 抽检：用五段独立生成（更稳）+ seed 引导召回
# =========================
def qa_probe_infer5(docs: List[Dict[str, Any]], seed_map: Dict[str, str], lora_dir: Path, n=2, tag="pre_train"):
    sample_docs = [d for d in docs if normalize_text(str(d.get("full_text", "")))]
    random.shuffle(sample_docs)
    sample_docs = sample_docs[:n]

    for idx, d in enumerate(sample_docs, 1):
        doc_id = normalize_text(str(d.get("doc_id", f"doc{idx}")))
        text = normalize_text(str(d.get("full_text", "")))
        out_path = QA_DIR / f"{tag}_{idx}_{to_safe_filename(doc_id)[:80]}.txt"

        seed_hint = seed_map.get(doc_id)  # 找不到就 None，自动退化为标题召回
        infer_five_sections_stable(
            base_model_name=MODEL_NAME,
            lora_dir=lora_dir,
            doc_id=doc_id,
            doc_text=text,
            out_txt=out_path,
            seed_hint=seed_hint,
            per_section_retries=1
        )

    print(f"[QA] saved: {QA_DIR} ({tag})")


# =========================
# 17) TrainingArguments 兼容（eval_strategy / evaluation_strategy）
# =========================
def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    if "eval_strategy" in params and "evaluation_strategy" not in params:
        if "evaluation_strategy" in kwargs:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in params and "eval_strategy" not in params:
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    else:
        if "evaluation_strategy" in kwargs and "eval_strategy" not in kwargs:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


# =========================
# 18) main
# =========================
def main():
    random.seed(RANDOM_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    assert torch.cuda.is_available(), "CUDA 不可用"
    assert IN_REPORTS.exists(), f"Missing: {IN_REPORTS}"
    assert SEED_DIR.exists(), f"Missing: {SEED_DIR}"

    docs = read_reports_jsonl(IN_REPORTS)
    seeds = read_seed_txts(SEED_DIR)
    print(f"[INFO] docs={len(docs)} seeds={len(seeds)}")

    # seed_map：doc_id -> seed_target（✅ 推理/QA/训练都可用）
    seed_map = make_seed_map(docs, seeds)
    print(f"[INFO] aligned seed docs: {len(seed_map)}")

    base_model, tokenizer = load_model_tokenizer_fp16_single_gpu()
    model = apply_lora(base_model)

    # 训练样本：只用 seed_map 对齐的 doc
    samples = build_samples(tokenizer, docs, seed_map)
    if not samples:
        raise RuntimeError("No training samples built. Check doc_id alignment / seed filenames.")

    random.shuffle(samples)
    n_valid = max(1, int(len(samples) * VALID_RATIO))
    valid = samples[:n_valid]
    train = samples[n_valid:]
    print(f"[INFO] train={len(train)} valid={len(valid)}")

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    # ✅ 提速：eval/save 降频（你可按训练时长调到 200~400）
    args = make_training_args(
        output_dir=str(OUT_DIR),

        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=300,

        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,

        logging_strategy="steps",
        logging_steps=20,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=2e-5,
        num_train_epochs=4,
        warmup_ratio=0.08,
        weight_decay=0.01,
        max_grad_norm=1.0,

        prediction_loss_only=True,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=False,
        bf16=False,

        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
    )

    trainer = SafeEvalLossTrainer(
        model=model,
        args=args,
        train_dataset=ListDataset(train),
        eval_dataset=ListDataset(valid),
        data_collator=simple_collator,
    )
    trainer.add_callback(EvalLoggerCallback(EVAL_LOG_CSV))

    # === 训练前：用“五段独立生成 + seed 引导召回”做 QA 抽检（用当前 LoRA 权重目录先保存一次）
    tmp_pre = OUT_DIR / "tmp_pretrain_lora"
    tmp_pre.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(tmp_pre))
    tokenizer.save_pretrained(str(tmp_pre))
    qa_probe_infer5(docs, seed_map, lora_dir=tmp_pre, n=2, tag="pre_train")

    print("[INFO] Start training...")
    trainer.train()

    # === 保存 best
    final_dir = OUT_DIR / "final_best"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"[OK] Saved BEST LoRA to: {final_dir}")
    print(f"[OK] Eval log: {EVAL_LOG_CSV}")

    # === 训练后：再做一次 QA 抽检（五段独立生成）
    qa_probe_infer5(docs, seed_map, lora_dir=final_dir, n=2, tag="post_train")

    # ===== 推理示例（五段独立生成 + seed 引导召回，如果该 doc 有 seed）=====
    # d0 = docs[0]
    # doc_id = normalize_text(str(d0.get("doc_id","infer_doc")))
    # doc_text = normalize_text(str(d0.get("full_text","")))
    # out_txt = OUT_DIR / "infer_demo_infer5_seedguide.txt"
    # infer_five_sections_stable(MODEL_NAME, final_dir, doc_id, doc_text, out_txt, seed_hint=seed_map.get(doc_id), per_section_retries=1)

if __name__ == "__main__":
    main()
