# -*- coding: utf-8 -*-
"""
config/config.py

Centralised path and hyperparameter configuration.
Edit BASE_DIR to point to your project root before running any script.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\语义分析模型训练\PythonProject")   # ← change this

DATA_DIR       = BASE_DIR / "data"
SEED_DIR       = BASE_DIR / "seed"
RESULT_DIR     = BASE_DIR / "result"

IN_REPORTS     = DATA_DIR / "reports_sections.jsonl"
OUT_TRAIN      = DATA_DIR / "sft_train.jsonl"
OUT_VALID      = DATA_DIR / "sft_valid.jsonl"

RUN_NAME       = "qwen2.5-mainidea-final-stable-fast-infer5-seedguide"
CKPT_ROOT      = BASE_DIR / "checkpoints"
OUT_DIR        = CKPT_ROOT / RUN_NAME
LORA_BEST_DIR  = OUT_DIR / "final_best"

CACHE_DIR      = DATA_DIR / "cache_embeddings"

# ── Models ─────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-7B-Instruct"
EMB_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
EMB_DEVICE     = "cpu"
EMB_BATCH_SIZE = 32

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_TOKENS   = 1100
CHUNK_OVERLAP  = 160

# ── Retrieval ───────────────────────────────────────────────────────────────
TOPN_PER_SECTION           = 16
MMR_SELECT_K               = 6
MMR_LAMBDA                 = 0.62
EVIDENCE_MAX_TOKENS_TOTAL  = 3600
EVIDENCE_MAX_CHARS_PER_CHUNK = 2200

# ── Generation ──────────────────────────────────────────────────────────────
PER_SECTION_MAX_NEW_TOKENS = 300
PER_SECTION_RETRIES        = 2

# ── Training ────────────────────────────────────────────────────────────────
MAX_SEQ_LEN            = 4096
VARIANTS_PER_DOC       = 12
VALID_RATIO            = 0.10
RANDOM_SEED            = 42
LEARNING_RATE          = 2e-5
NUM_TRAIN_EPOCHS       = 4
WARMUP_RATIO           = 0.08
WEIGHT_DECAY           = 0.01
GRAD_ACCUM_STEPS       = 4
EVAL_STEPS             = 300
SAVE_STEPS             = 300

# ── Section titles (fixed) ──────────────────────────────────────────────────
SECTION_TITLES = [
    "研究问题",
    "核心概念",
    "研究目标和内容",
    "研究成果",
    "研究效果",
]
