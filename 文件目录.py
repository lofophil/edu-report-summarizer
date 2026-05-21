D:\语义分析模型训练\PythonProject\
│
├─ data\
│   ├─ reports_sections.jsonl      # 02 从 docx 抽模块的结果
│   ├─ seed_summaries.jsonl        # 你人工标注的“金标准摘要”
│   ├─ sft_train.jsonl             # 03 生成的训练集
│   ├─ sft_eval.jsonl              # 03 生成的验证集
│   └─ infer_inputs\               # 以后要批量摘要的报告/sections
│
├─ 报告_docx\                       # 01 转码后的 docx（你已经有）
│
├─ scripts\
│   ├─ 01_convert_to_docx.py       # ✅ 你已有
│   ├─ 02_docx_to_sections.py      # docx → 11 个模块的文本
│   ├─ 03_make_sft_dataset.py      # sections + 人工摘要 → SFT 数据
│   ├─ 04_train_sft_qwen_lora.py   # 用 Qwen2.5-7B 做 QLoRA 训练
│   ├─ 05_eval_and_inference.py    # 加载 LoRA，生成摘要 & 简单评估
│   └─ utils_text.py               # 共用的小工具（清洗、截断等，可选）
│
└─ checkpoints\
    ├─ qwen2.5-7b-base\            # 从 HF 下载的基础模型（缓存）
    ├─ qwen2.5-7b-sft-v1\          # 第一次 SFT LoRA 权重
    └─ qwen2.5-7b-sft-v2\          # 第二次迭代后 LoRA 权重
