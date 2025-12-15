# Distill–Prune–Quantize Pipeline (SST-2) | 双语 README

## Overview | 概述
- Lightweight repo with runnable scripts, latest checkpoints, and concise docs for SST-2 distillation → pruning → quantization.  
- 轻量化仓库，仅保留可运行脚本与最新模型，用于 SST-2 的蒸馏 → 剪枝 → 量化全流程。

## What’s Included | 包含内容
- Scripts / 脚本: `download_model.py`, `8_pruning_with_finetuning.py`, `9_quantize_pruned_model.py`, `10_qat_kd_4bit.py`, `7_evaluate_and_generate_report.py`, `evaluate_plot.py`.
- Docs / 文档: `OPTIMIZATION_GUIDE.md`, `ALGORITHM_ENHANCEMENTS.md`, `compression_pipeline_overview.md`, `CHANGES_SUMMARY.md`, `paper_innovations_summary.txt`.
- Models & data / 模型与数据:  
  - Teacher: `teacher_model/`；Student: `student_model/`；Tokenizer/cache: `distilbert-local/`  
  - Latest artifacts 最新产物:  
    - Pruned FP32: `models/pruning_with_finetuning_20251213_131504`  
    - PTQ 4-bit: `models/pruned_quantized_ptq_fast_20251212_110837`  
    - QKD-4bit (QAT): `models/pruned_qkd4bit_full_20251213_205841`  
    - Quantized distilled baseline: `models/distilled_quantized_student/`
  - Dataset cache 数据缓存: `sst2_data/` (GLUE SST-2)

## Quickstart (EN)
```bash
pip install -r requirements.txt
python download_model.py                               # fetch distilbert-base-uncased if missing
python 8_pruning_with_finetuning.py                    # gradual pruning + recovery finetuning
python 9_quantize_pruned_model.py                      # 4-bit PTQ on pruned model
python 10_qat_kd_4bit.py                               # 4-bit QAT + KD (use --fast for quick pass)
python 7_evaluate_and_generate_report.py               # eval + final_report.json/png
```

## 快速开始 (ZH)
```bash
pip install -r requirements.txt
python download_model.py                               # 如无则下载 distilbert-base-uncased
python 8_pruning_with_finetuning.py                    # 渐进剪枝 + 恢复微调
python 9_quantize_pruned_model.py                      # 剪枝模型 4-bit PTQ 量化
python 10_qat_kd_4bit.py                               # 4-bit 量化感知训练 + 蒸馏 (加 --fast 走快速模式)
python 7_evaluate_and_generate_report.py               # 统一评估，生成 final_report.json/png
```

## Fast Path | 快速路径
- For quick iteration / 快速迭代：  
  ```bash
  python 8_pruning_with_finetuning.py --fast
  python 10_qat_kd_4bit.py --fast
  python 7_evaluate_and_generate_report.py
  ```

## Key Results | 核心结果 (from `final_report.json`)
| Model                               | Acc   | GFLOPs | Size (MB) | Latency (ms, bs=32) |
|-------------------------------------|-------|--------|-----------|---------------------|
| Distilled Student (Baseline)        | 0.900 | 0.6813 | 255.4     | 33.24               |
| Quantized Distilled Student         | 0.904 | 0.6813 | 66.8      | 34.56               |
| Pruned Student (Sparse FP32)        | 0.886 | 0.5450 | 255.4     | 33.21               |
| Pruned + Quantized Student (PTQ)    | 0.839 | 0.5450 | 68.7      | 13.37               |
| Pruned + QKD-4bit (QAT)             | 0.849 | 0.5450 | 68.7      | 13.30               |

`final_report.png` shows the accuracy–GFLOPs trade-off (最佳点在左上角)。

## Repo Layout | 仓库结构
```
.
├─ distilbert-local/                 # student init/tokenizer cache
├─ teacher_model/                    # BERT teacher
├─ student_model/                    # distilled baseline
├─ models/                           # latest pruned/PTQ/QKD/quantized artifacts
├─ sst2_data/                        # GLUE SST-2 cache (auto-downloads if absent)
├─ 7_evaluate_and_generate_report.py # unified evaluation + plot
├─ 8_pruning_with_finetuning.py      # pruning + recovery finetuning
├─ 9_quantize_pruned_model.py        # PTQ 4-bit
├─ 10_qat_kd_4bit.py                 # QAT + KD (4-bit)
├─ evaluate_plot.py                  # auxiliary comparison plotter
├─ OPTIMIZATION_GUIDE.md             # fast/full knobs for 4080 mobile
├─ ALGORITHM_ENHANCEMENTS.md         # KGSP / MSP / QKD-4bit usage notes
├─ compression_pipeline_overview.md  # high-level walkthrough
├─ CHANGES_SUMMARY.md                # summary of code changes
├─ paper_innovations_summary.txt     # literature-aligned observations
├─ final_report.{json,png}
├─ requirements.txt
└─ download_model.py
```

## Notes | 使用提示
- If you delete `sst2_data/`, scripts will re-download via `datasets`; 已加 `.gitignore` 避免误提交数据集。  
- Checkpoints under `models/` stay ignored by git; 清理无用产物可节省空间。  
- `evaluate_plot.py` aligns with the cleaned artifacts and skips missing paths gracefully; 如需自定义对照只需补充路径。
