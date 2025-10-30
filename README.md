# Distill-Prune-Quantize Compression Pipeline

[English](#english) | [中文](#中文)

## English

### Project Overview

This repository accompanies a study of a four-stage compression workflow that combines knowledge distillation, gradual structured pruning with recovery fine-tuning, and 4-bit weight quantization. The pipeline is evaluated on the SST-2 sentiment classification task using a BERT teacher and a DistilBERT student, building on prior work in model compression [1-6]. The primary aim is to document a reproducible recipe and quantify the trade-offs between accuracy and compute cost in a modest, deployment-oriented setting.

### Methodology

1. **Knowledge Distillation:** A DistilBERT student is initialized from a BERT teacher following the approach popularized in DistilBERT [1].
2. **Gradual Structured Pruning:** We apply a polynomial sparsity schedule inspired by Movement Pruning and recent large-model pruning studies [2][3], targeting 20% sparsity.
3. **Recovery Fine-tuning:** After reaching the sparsity target, the sparse structure is frozen and the model is fine-tuned with a reduced learning rate to stabilize accuracy.
4. **4-bit Quantization:** The sparse model is quantized to 4-bit weights, echoing low-bit deployment practices such as SmoothQuant and QLoRA [4][5].

### Results

| Model                                | Accuracy | GFLOPs | Size (MB) |
|--------------------------------------|----------|--------|-----------|
| Distilled Student (Baseline)         | 0.9002   | 0.6813 | 255.43    |
| Quantized Distilled Student          | 0.9037   | 0.6813 | 66.84     |
| Pruned + Quantized Student (Final)   | 0.8888   | 0.5450 | 68.71     |

- The final model reduces theoretical compute by ~20% versus the distilled baseline with a 1.1% absolute accuracy drop.
- Quantization alone mainly affects storage; pruning is responsible for the GFLOPs reduction.
- The workflow highlights where recovery fine-tuning is necessary to maintain accuracy after pruning, echoing observations from [2][3].

### Reproducing the Pipeline

```bash
pip install -r requirements.txt
python download_model.py
python 8_pruning_with_finetuning.py
python 9_quantize_pruned_model.py
python 7_evaluate_and_generate_report.py
```

Artifacts such as `final_report.json` and `final_report.png` summarize the accuracy/compute trade-off.

### Repository Layout

```
.
├── models/                         # Saved checkpoints (distilled, pruned, quantized)
├── teacher_model/                  # BERT teacher weights
├── 7_evaluate_and_generate_report.py
├── 8_pruning_with_finetuning.py
├── 9_quantize_pruned_model.py
├── download_model.py
├── final_report.json / final_report.png
├── paper_innovations_summary.txt   # Literature-aligned summary of observations
└── requirements.txt
```

### Limitations and Next Steps

- Experiments are limited to one downstream task and a DistilBERT-sized student; broader validation is required.
- Only weight sparsity and weight quantization are explored; activation quantization or hardware-specific sparsity patterns (e.g., 2:4) remain open.
- Extending the pipeline to larger LLM checkpoints would require revisiting the pruning schedule and optimizer settings.

### References

[1] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT: a distilled version of BERT. arXiv:1910.01108.  
[2] Sanh, V., Wolf, T., & Rush, A. M. (2020). Movement Pruning: Adaptive Sparsity by Fine-Tuning. arXiv:2005.07683.  
[3] Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. arXiv:2301.00774.  
[4] Xiao, S., Zhang, S., Chen, S., et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arXiv:2211.10438.  
[5] Dettmers, T., Lewis, M., Shleifer, S., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.  
[6] Gale, T., Elsen, E., & Hooker, S. (2019). The State of Sparsity in Deep Neural Networks. arXiv:1902.09574.

---

## 中文

### 项目概述

本仓库面向“四阶段模型压缩”流程的复现研究：依次执行知识蒸馏、渐进式结构化剪枝、低学习率恢复微调以及 4-bit 权重量化。实验以 BERT 教师模型和 DistilBERT 学生模型为基础，在 SST-2 情感分类任务上验证流程效果，参考并对比近期压缩文献的实践经验 [1-6]。目标是在部署友好的小规模场景下，提供客观的准确率与计算成本权衡数据。

### 实验流程

1. **知识蒸馏**：按照 DistilBERT 的做法 [1]，利用 BERT 教师得到高质量学生模型。
2. **渐进式剪枝**：参考 Movement Pruning 及 SparseGPT 的调度思想 [2][3]，通过多项式稀疏度曲线逐步提升到 20% 稀疏率。
3. **恢复期微调**：在达成目标稀疏度后冻结掩码，以更低学习率继续训练，稳定剪枝后的性能。
4. **4-bit 量化**：沿用 SmoothQuant 与 QLoRA 所倡导的低比特部署范式 [4][5]，对稀疏模型进行 4-bit 权重量化。

### 实验结果

| 模型                               | 准确率  | GFLOPs | 大小 (MB) |
|------------------------------------|---------|--------|-----------|
| 蒸馏学生 (基线)                    | 0.9002  | 0.6813 | 255.43    |
| 蒸馏学生 + 量化                    | 0.9037  | 0.6813 | 66.84     |
| 剪枝 + 恢复微调 + 4-bit 量化 (最终) | 0.8888  | 0.5450 | 68.71     |

- 在约 1.1% 的准确率下降下，GFLOPs 降低约 20%，验证了“剪枝先于量化”对计算效率的益处。
- 单独量化主要影响模型大小；GFLOPs 改善主要来自结构化剪枝。
- 结果印证了文献中关于恢复期微调重要性的观点 [2][3]。

### 复现实验

```bash
pip install -r requirements.txt
python download_model.py
python 8_pruning_with_finetuning.py
python 9_quantize_pruned_model.py
python 7_evaluate_and_generate_report.py
```

运行后可获得 `final_report.json` 与 `final_report.png`，用于查看准确率-计算量的对比。

### 仓库结构

```
.
├── models/                         # 各阶段模型检查点
├── teacher_model/                  # 教师模型权重
├── 7_evaluate_and_generate_report.py
├── 8_pruning_with_finetuning.py
├── 9_quantize_pruned_model.py
├── download_model.py
├── final_report.json / final_report.png
├── paper_innovations_summary.txt   # 对应的文献综述与实验总结
└── requirements.txt
```

### 局限与后续方向

- 目前仅在单一任务与 DistilBERT 规模上评估，仍需扩展到更多任务和更大模型。
- 尚未尝试硬件友好的 2:4 稀疏或与 LoRA 的协同策略。
- 激活量化和高吞吐部署相关的优化尚未涉及，后续可结合 SmoothQuant 的激活平滑策略 [4] 继续探索。

### 参考文献

[1] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT: a distilled version of BERT. arXiv:1910.01108.  
[2] Sanh, V., Wolf, T., & Rush, A. M. (2020). Movement Pruning: Adaptive Sparsity by Fine-Tuning. arXiv:2005.07683.  
[3] Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. arXiv:2301.00774.  
[4] Xiao, S., Zhang, S., Chen, S., et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arXiv:2211.10438.  
[5] Dettmers, T., Lewis, M., Shleifer, S., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.  
[6] Gale, T., Elsen, E., & Hooker, S. (2019). The State of Sparsity in Deep Neural Networks. arXiv:1902.09574.
