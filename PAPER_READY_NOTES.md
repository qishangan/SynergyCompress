# PAPER_READY_NOTES

## 1. ????
We propose DGSP-WKD, a lightweight compression enhancement for task-oriented Transformers that reuses a single distillation-guided sensitivity signal across both pruning and low-bit recovery. During pruning, each structured group is scored by a convex combination of normalized weight magnitude and KD-only first-order sensitivity estimated from `abs(grad(W_g) * W_g)` with EMA smoothing. The resulting layer-level sensitivity is then used both for adaptive sparsity allocation and for sensitivity-weighted hidden-state distillation during 4-bit QKD recovery.

## 2. ?????
- A distillation-guided structured pruning score that augments magnitude pruning with KD-only first-order sensitivity while keeping the original pruning loop nearly unchanged.
- A layer-adaptive sparsity allocator driven by the same sensitivity signal, so sensitive layers are pruned less and insensitive layers are pruned more under a fixed global sparsity target.
- A unified recovery design in which pruning-derived sensitivity is reused to weight hidden-state distillation in 4-bit QKD, closing the loop between pruning and quantization recovery.

## 3. ??????
- Sensitivity score is the dominant contributor: in quick sp20, DGSP score-only QKD reached 0.9060 vs baseline QKD 0.8222.
- Adaptive sparsity is implemented and stable, but its isolated gain is mixed on SST-2 quick ablations.
- Weighted hidden KD also shows mixed gains: it slightly improved over adaptive QKD in quick sp20, but uniform hidden KD remained better than weighted tau=1/tau=2 in the formal sp20 hidden ablation.

## 4. ??????
### Formal sp15
| Model | Acc | Loss | Real Sparsity | Theoretical GFLOPs | Latency ms | Size MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Distilled Student | 0.9002 | 0.4175 | 0.0000 | 0.6813 | 38.82 | 255.43 |
| Baseline Pruned FP32 | 0.8761 | 0.3690 | 0.1500 | 0.5791 | 38.22 | 255.43 |
| Baseline PTQ | 0.8681 | 0.3698 | 0.1500 | 0.5791 | 14.47 | 68.71 |
| Baseline QKD | 0.8704 | 0.3592 | 0.1500 | 0.5791 | 14.52 | 68.71 |
| DGSP FP32 | 0.9048 | 0.2913 | 0.1503 | 0.5789 | 38.81 | 255.43 |
| DGSP PTQ | 0.9060 | 0.2855 | 0.1503 | 0.5789 | 14.95 | 68.71 |
| DGSP-WKD | 0.9048 | 0.2751 | 0.1503 | 0.5789 | 14.24 | 68.71 |

### Formal sp20
| Model | Acc | Loss | Real Sparsity | Theoretical GFLOPs | Latency ms | Size MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Distilled Student | 0.9002 | 0.4175 | 0.0000 | 0.6813 | 40.25 | 255.43 |
| Baseline Pruned FP32 | 0.8796 | 0.3646 | 0.2000 | 0.5450 | 37.46 | 255.43 |
| Baseline PTQ | 0.8784 | 0.3568 | 0.2000 | 0.5450 | 14.00 | 68.71 |
| Baseline QKD | 0.8842 | 0.3460 | 0.2000 | 0.5450 | 15.21 | 68.71 |
| DGSP FP32 | 0.9094 | 0.2871 | 0.2003 | 0.5448 | 47.47 | 255.43 |
| DGSP PTQ | 0.9060 | 0.2831 | 0.2003 | 0.5448 | 21.13 | 68.71 |
| DGSP-WKD | 0.9060 | 0.2676 | 0.2003 | 0.5448 | 17.94 | 68.71 |

## 5. ?????????
- ?DGSP-WKD consistently improves over the original pruning plus QKD baseline under matched sparsity and 4-bit settings on SST-2.?
- ?The largest gains come from the distillation-guided sensitivity score, while layer-adaptive allocation and sensitivity-weighted hidden KD provide a unified compression narrative with mixed but stable task-level behavior.?
- ?The reported GFLOPs are theoretical because the current implementation applies zero-masking rather than structural reparameterization.?

## 6. ????????
- ????real acceleration???deployment speedup????????????????
- ????weighted hidden KD consistently outperforms uniform KD?????? formal sp20 ??????
- ????state-of-the-art????????????????????? SST-2 ????
