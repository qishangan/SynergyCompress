# Experiment Results Archive

This file is the single retained entry point for the final DGSP-WKD experiment results after repository cleanup.

## Final Result Files
- `baseline_results.json`
- `dgsp_results.json`
- `dgsp_wkd_results.json`
- `ablation_results.json`
- `final_report.json`
- `layer_sensitivity.json`
- `pruning_allocation.json`
- `weighted_hidden_config.json`

## Formal Results

### sp15
| Model | Acc | Loss | Real Sparsity | Theoretical GFLOPs | Latency ms | Size MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Distilled Student | 0.9002 | 0.4175 | 0.0000 | 0.6813 | 38.82 | 255.43 |
| Baseline Pruned FP32 | 0.8761 | 0.3690 | 0.1500 | 0.5791 | 38.22 | 255.43 |
| Baseline PTQ | 0.8681 | 0.3698 | 0.1500 | 0.5791 | 14.47 | 68.71 |
| Baseline QKD | 0.8704 | 0.3592 | 0.1500 | 0.5791 | 14.52 | 68.71 |
| DGSP FP32 | 0.9048 | 0.2913 | 0.1503 | 0.5789 | 38.81 | 255.43 |
| DGSP PTQ | 0.9060 | 0.2855 | 0.1503 | 0.5789 | 14.95 | 68.71 |
| DGSP-WKD | 0.9048 | 0.2751 | 0.1503 | 0.5789 | 14.24 | 68.71 |

### sp20
| Model | Acc | Loss | Real Sparsity | Theoretical GFLOPs | Latency ms | Size MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Distilled Student | 0.9002 | 0.4175 | 0.0000 | 0.6813 | 40.25 | 255.43 |
| Baseline Pruned FP32 | 0.8796 | 0.3646 | 0.2000 | 0.5450 | 37.46 | 255.43 |
| Baseline PTQ | 0.8784 | 0.3568 | 0.2000 | 0.5450 | 14.00 | 68.71 |
| Baseline QKD | 0.8842 | 0.3460 | 0.2000 | 0.5450 | 15.21 | 68.71 |
| DGSP FP32 | 0.9094 | 0.2871 | 0.2003 | 0.5448 | 47.47 | 255.43 |
| DGSP PTQ | 0.9060 | 0.2831 | 0.2003 | 0.5448 | 21.13 | 68.71 |
| DGSP-WKD | 0.9060 | 0.2676 | 0.2003 | 0.5448 | 17.94 | 68.71 |

## Key Ablation Notes
- DGSP score is the strongest contributor.
- Layer-adaptive sparsity is implemented and stable, but its isolated gain is mixed on SST-2.
- Weighted hidden KD completes the unified method story, but did not beat uniform hidden KD in the formal sp20 hidden ablation.

## Retained Experiment Roots
- `outputs/exp_formal_dgsp_wkd_20260325_074046`
- `outputs/exp_baseline_refresh_20260325_120004`
- `outputs/exp_dgsp_wkd_quickrefresh_20260325_153003`

## Retained Canonical Model Directories
- `models/baseline_exact_sp15_prune`
- `models/baseline_exact_sp15_ptq`
- `models/baseline_exact_sp15_qkd`
- `models/baseline_exact_sp20_prune_rerun`
- `models/baseline_exact_sp20_ptq_rerun`
- `models/baseline_exact_sp20_qkd_rerun`
- `models/formal_dgsp_wkd_20260325_074046_formal_dgsp_sp15_prune_sp15`
- `models/formal_dgsp_wkd_20260325_074046_formal_dgsp_sp15_ptq`
- `models/formal_dgsp_wkd_20260325_074046_formal_dgsp_wkd_sp15_qkd`
- `models/formal_dgsp_wkd_20260325_074046_formal_dgsp_sp20_prune_sp20`
- `models/formal_dgsp_wkd_20260325_074046_formal_dgsp_sp20_ptq`
- `models/formal_dgsp_wkd_20260325_074046_formal_dgsp_wkd_sp20_qkd`
- `models/formal_dgsp_wkd_20260325_074046_formal_dgsp_sp20_uniform_qkd`
- `models/formal_dgsp_wkd_20260325_074046_formal_dgsp_sp20_weighted_tau2_qkd`

## Reporting Note
- `Theoretical GFLOPs` are retained as theory-side complexity indicators.
- The current pruning is zero-masked pruning, so the reported latency should not be interpreted as guaranteed structural acceleration.
