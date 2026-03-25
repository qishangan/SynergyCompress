# DGSP-WKD Compression Pipeline

This repository contains the cleaned experiment version of a task-oriented Transformer compression pipeline on SST-2.

Current main line:
- Teacher: BERT
- Student: DistilBERT
- Task: SST-2
- Pipeline: Distillation -> Structured Pruning -> 4-bit PTQ / 4-bit QKD recovery
- Final method version: `DGSP-WKD`

## What Is Implemented

`DGSP-WKD` stands for `Distillation-Guided Sensitivity Pruning with Weighted Knowledge Distillation Recovery`.

The current codebase implements:
- Distillation-guided pruning scores that fuse weight magnitude and KD-only first-order sensitivity.
- Layer-adaptive sparsity allocation under a fixed global sparsity target.
- Sensitivity-weighted hidden-state distillation for 4-bit QKD recovery.
- Unified evaluation that reports:
  - `accuracy`
  - `loss`
  - `real_sparsity`
  - `theoretical_gflops`
  - `measured_latency_ms`
  - `model_size_mb`

Important:
- The pruning in this repo is zero-masked pruning.
- `Theoretical GFLOPs` do not imply structural acceleration.
- Measured latency is reported separately from theoretical complexity.

## Cleaned Repository Layout

Core scripts:
- `8_pruning_with_finetuning.py`: pruning + recovery finetuning, including DGSP scoring.
- `adaptive_pruner.py`: global and layer-adaptive structured pruning helpers.
- `dgsp_utils.py`: DGSP scoring and sensitivity utility functions.
- `9_quantize_pruned_model.py`: 4-bit PTQ export for pruned checkpoints.
- `10_qat_kd_4bit.py`: 4-bit QKD recovery with optional weighted hidden KD.
- `7_evaluate_and_generate_report.py`: manifest-based evaluation and report generation.
- `run_dgsp_wkd_suite.py`: quick/formal experiment runner.

Main result files:
- `baseline_results.json`
- `dgsp_results.json`
- `dgsp_wkd_results.json`
- `ablation_results.json`
- `final_report.json`
- `final_report.png`
- `layer_sensitivity.json`
- `pruning_allocation.json`
- `weighted_hidden_config.json`

Main result docs:
- `EXPERIMENT_RESULTS_ARCHIVE.md`
- `FINAL_EXPERIMENT_SUMMARY.md`
- `PAPER_READY_NOTES.md`

Retained output directories:
- `outputs/exp_formal_dgsp_wkd_20260325_074046`
- `outputs/exp_baseline_refresh_20260325_120004`
- `outputs/exp_dgsp_wkd_quickrefresh_20260325_153003`

Notes:
- `models/`, `teacher_model/`, `student_model/`, `distilbert-local/`, and `sst2_data/` are ignored by git.
- The repository keeps result summaries and logs, but not large model weights.

## Latest Experiment Version

Current synced experiment version:
- Name: `DGSP-WKD cleaned experiment release`
- Date: `2026-03-25`
- Scope:
  - refreshed baselines after fixing global pruning target accounting
  - DGSP formal experiments at sparsity `0.15` and `0.20`
  - quick ablation refresh on fixed code
  - hidden-KD weighting ablation on `sp20`

Environment used for the retained results:
- Python: `3.12.3`
- CUDA: `12.1`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- torch: `2.5.1+cu121`
- transformers: `4.53.0`
- bitsandbytes: `0.46.1`
- datasets: `3.6.0`
- accelerate: `1.8.1`

## Main Results

### Formal `sp15`

| Model | Acc | Loss | Real Sparsity | Theoretical GFLOPs | Latency ms | Size MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Distilled Student | 0.9002 | 0.4175 | 0.0000 | 0.6813 | 38.82 | 255.43 |
| Baseline Pruned FP32 | 0.8761 | 0.3690 | 0.1500 | 0.5791 | 38.22 | 255.43 |
| Baseline PTQ | 0.8681 | 0.3698 | 0.1500 | 0.5791 | 14.47 | 68.71 |
| Baseline QKD | 0.8704 | 0.3592 | 0.1500 | 0.5791 | 14.52 | 68.71 |
| DGSP FP32 | 0.9048 | 0.2913 | 0.1503 | 0.5789 | 38.81 | 255.43 |
| DGSP PTQ | 0.9060 | 0.2855 | 0.1503 | 0.5789 | 14.95 | 68.71 |
| DGSP-WKD | 0.9048 | 0.2751 | 0.1503 | 0.5789 | 14.24 | 68.71 |

### Formal `sp20`

| Model | Acc | Loss | Real Sparsity | Theoretical GFLOPs | Latency ms | Size MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Distilled Student | 0.9002 | 0.4175 | 0.0000 | 0.6813 | 40.25 | 255.43 |
| Baseline Pruned FP32 | 0.8796 | 0.3646 | 0.2000 | 0.5450 | 37.46 | 255.43 |
| Baseline PTQ | 0.8784 | 0.3568 | 0.2000 | 0.5450 | 14.00 | 68.71 |
| Baseline QKD | 0.8842 | 0.3460 | 0.2000 | 0.5450 | 15.21 | 68.71 |
| DGSP FP32 | 0.9094 | 0.2871 | 0.2003 | 0.5448 | 47.47 | 255.43 |
| DGSP PTQ | 0.9060 | 0.2831 | 0.2003 | 0.5448 | 21.13 | 68.71 |
| DGSP-WKD | 0.9060 | 0.2676 | 0.2003 | 0.5448 | 17.94 | 68.71 |

## Ablation Takeaways

From the refreshed quick ablation and formal hidden-weight ablation:
- `DGSP sensitivity score` is the strongest and most reliable gain source.
- `Layer-adaptive sparsity` is implemented and stable, but its isolated gain is mixed on SST-2.
- `Weighted hidden KD` completes the unified method story, but it did not beat uniform hidden KD in the formal `sp20` hidden ablation.

So the most conservative paper narrative is:
- strong claim: `DGSP score works`
- moderate claim: `adaptive allocation and weighted hidden KD form a unified sensitivity-guided pipeline`
- avoid strong claim: `weighted hidden KD always improves results`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a pruning experiment:

```bash
python 8_pruning_with_finetuning.py --sensitivity_method dgsp --target_sparsity 0.20
```

Run PTQ:

```bash
python 9_quantize_pruned_model.py --pruned_model_path <pruned_model_dir> --output_dir <ptq_dir>
```

Run 4-bit QKD:

```bash
python 10_qat_kd_4bit.py --pruned_model_path <pruned_model_dir> --output_dir <qkd_dir> --enable_hidden_kd --weighted_hidden
```

Run evaluation:

```bash
python 7_evaluate_and_generate_report.py --manifest <manifest.json> --output_json <report.json> --output_png <report.png>
```

Run the quick suite:

```bash
python run_dgsp_wkd_suite.py --skip_formal
```

## Canonical Experiment References

If you want the retained logs and manifests, use these directories:
- `outputs/exp_formal_dgsp_wkd_20260325_074046`
- `outputs/exp_baseline_refresh_20260325_120004`
- `outputs/exp_dgsp_wkd_quickrefresh_20260325_153003`

If you want the final summarized view first, open:
- `EXPERIMENT_RESULTS_ARCHIVE.md`
- `FINAL_EXPERIMENT_SUMMARY.md`
- `PAPER_READY_NOTES.md`

## GitHub Sync Note

This README corresponds to the cleaned GitHub experiment version that keeps:
- final code
- final result JSONs
- retained experiment logs
- paper-ready summaries

while removing:
- obsolete temporary experiments
- smoke outputs
- stale baseline artifacts
- intermediate debug files
