# EduReport-Summarizer

**A QLoRA fine-tuned model for structured abstract extraction from Chinese educational research reports.**

> Base model: `Qwen2.5-7B-Instruct` | Adapter: LoRA via PEFT  
> Checkpoint: `qwen2.5-mainidea-final-stable-fast-infer5-seedguide`

---

## Project Description

EduReport-Summarizer is an end-to-end pipeline for automatically extracting structured abstracts from Chinese educational research reports. Given a raw `.doc`, `.docx`, or `.pdf` report as input, the system produces a formal five-section abstract covering Research Question, Core Concepts, Research Objectives and Content, Research Outcomes, and Research Impact, with each section capped at 300 Chinese characters and the full output not exceeding 1,500 characters.

The pipeline consists of five stages. First, source documents are batch-converted to `.docx` using Microsoft Word COM automation with LibreOffice and `pdf2docx` as fallbacks. Second, each document is parsed into the five canonical sections by matching paragraph headings against an alias table, and the full text is retained for downstream use. Third, human-written gold-standard seed summaries are aligned to their corresponding documents by `doc_id` and assembled into a ChatML-format supervised fine-tuning dataset. Fourth, `Qwen2.5-7B-Instruct` is fine-tuned with QLoRA (rank 16, alpha 32) over four epochs using a retrieval-augmented evidence construction strategy: document chunks are embedded with `BAAI/bge-base-zh-v1.5`, Top-16 candidates are recalled by cosine similarity, and six diverse evidence blocks are selected per section via Maximal Marginal Relevance re-ranking within a 3,600-token budget. Fifth, the trained model runs batch inference section-by-section, with a rule-based quality-control layer that checks character limits, forbidden characters, and quotation marks, retrying generation up to twice on failure before logging the QC result.

The model strictly forbids English text, Arabic numerals, direct quotation from the source, and fabrication of information not present in the original document, making it suitable for formal educational research evaluation contexts.

---

## Overview

Given a research report (`.docx`), the model outputs a structured abstract with five sections:

| Section | Description |
|---|---|
| 研究问题 | Research Question |
| 核心概念 | Core Concepts |
| 研究目标和内容 | Research Objectives and Content |
| 研究成果 | Research Outcomes |
| 研究效果 | Research Impact |

Each section: max **300 Chinese characters**. Total: max **1,500 characters**.

---

## Pipeline

```
Raw .doc / .pdf reports
        │
        ▼
[01] 01_convert_to_docx.py       Batch convert → .docx
        │
        ▼
[02] 02_docx_to_sections.py      Parse → 5-section JSONL
        │
        ▼
[03] 03_make_sft_dataset.py      Align seeds → SFT train/valid JSONL
        │
        ▼
[04] 04_train_sft_qwen_lora.py   QLoRA fine-tuning
        │
        ▼
[05] 05_batch_infer_mainidea.py  Batch inference → .txt abstracts
```

---

## Project Structure

```
PythonProject/
├── 01_convert_to_docx.py
├── 02_docx_to_sections.py
├── 03_make_sft_dataset.py
├── 04_train_sft_qwen_lora.py
├── 05_batch_infer_mainidea.py
├── pipeline_walkthrough.ipynb   ← interactive walkthrough
├── config.py                    ← centralised path & hyperparameter config
├── requirements.txt
├── README.md
├── example/                     ← anonymised sample outputs
├── data/                        ← converted .docx + JSONL (not uploaded)
├── seed/                        ← human-written seeds (not uploaded)
└── checkpoints/                 ← LoRA weights (HuggingFace)
```

---

## Quick Start

```bash
git clone https://github.com/lofophil/edu-report-summarizer.git
cd edu-report-summarizer
pip install -r requirements.txt
```

Download the base model:
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./checkpoints/base
```

Download the LoRA weights:
```bash
hf download lofo361/edu-report-summarizer-lora --local-dir ./checkpoints/final_best
```

Run inference:
```bash
python 05_batch_infer_mainidea.py
```

Or follow the notebook: `pipeline_walkthrough.ipynb`

---

## Output Constraints

The model is trained to enforce:

- Exactly 5 sections with canonical headings
- No English letters, Arabic numerals, or Roman numerals
- No direct quotation from source text
- No fabricated information
- Max 300 characters per section; max 1,500 total
- Formal, concise style consistent with human-written seeds

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 2e-5 |
| Epochs | 4 |
| Batch size | 1 (grad accum = 4) |
| Max sequence length | 4096 tokens |
| Warmup ratio | 0.08 |
| Retrieval model | BAAI/bge-base-zh-v1.5 |
| Chunk size | 1100 tokens, 160 overlap |
| MMR lambda | 0.62 |
| Evidence budget | 3600 tokens |

---

## RAG Strategy

For each section, the system:
1. Chunks the document by token count (1100 tok, 160 overlap)
2. Embeds chunks with BGE (`bge-base-zh-v1.5`), cached to disk
3. Recalls Top-16 chunks by cosine similarity
4. Re-ranks with MMR (lambda=0.62) → 6 diverse evidence blocks
5. Clips to 3600-token evidence budget
6. Generates each section independently (up to 2 retries on QC failure)

If a human seed exists for the document, it enriches the retrieval query.

---

## Example Outputs

See [`example/`](example/) for anonymised sample outputs.

---

## Requirements

```
torch>=2.0 (CUDA build)
transformers>=4.40
peft>=0.10
accelerate>=0.27
sentence-transformers>=2.6
python-docx
pdf2docx
numpy
tqdm
```

---

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@misc{lofophil2025edureport,
  author       = {lofophil},
  title        = {EduReport-Summarizer: Structured Abstract Extraction
                  from Chinese Educational Research Reports via QLoRA},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/lofophil/edu-report-summarizer}},
  note         = {Contact: lofocdut@gmail.com}
}
```

Or in plain text:

> lofophil. (2025). *EduReport-Summarizer: Structured Abstract Extraction from Chinese Educational Research Reports via QLoRA*. GitHub. https://github.com/lofophil/edu-report-summarizer. Contact: lofocdut@gmail.com

---

## License

MIT License

---

## Acknowledgements

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by Alibaba Cloud
- [BGE](https://github.com/FlagAI-Open/FlagEmbedding) by BAAI
- [PEFT](https://github.com/huggingface/peft) by Hugging Face
- LoRA weights: [lofo361/edu-report-summarizer-lora](https://huggingface.co/lofo361/edu-report-summarizer-lora)
