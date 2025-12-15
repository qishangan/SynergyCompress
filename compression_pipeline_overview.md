# Distill-Prune-Quantize 压缩流水线说明

本说明文档梳理了仓库 `llm_compression_project` 中的关键脚本、数据流、实验产出与可扩展点，帮助你或合作者快速理解并复现实验。

## 1. 项目概览

- **目标任务**：以 SST-2 情感分类为例，串联“知识蒸馏 → 渐进式结构化剪枝 → 恢复微调 → 4-bit 权重量化”的四阶段压缩流程，量化准确率、GFLOPs 与模型体积之间的权衡。
- **教师/学生**：教师为 BERT 序列分类模型，学生为 DistilBERT。仓库已经包含教师/学生及部分阶段性检查点。
- **核心成果**：`final_report.(json|png)` 给出三个模型（蒸馏基线、蒸馏+量化、剪枝+量化）的准确率与 GFLOPs，对比显示最终模型在仅下降 1.1% 准确率的情况下将理论算力需求降低 ~20%。

## 2. 仓库结构速览

| 路径 | 说明 |
| --- | --- |
| `teacher_model/` | BERT 教师模型的权重与 tokenizer。 |
| `distilbert-local/` | `download_model.py` 拉取的 DistilBERT 检查点，用作学生初始权重。 |
| `student_model/`、`student_*` | 不同蒸馏或量化阶段的学生模型产物，可作为额外对照。 |
| `models/` | 剪枝+恢复微调及其量化结果默认保存的位置。 |
| `7_evaluate_and_generate_report.py` | 统一评估所有目标模型并生成报告。 |
| `8_pruning_with_finetuning.py` | 负责渐进式结构化剪枝与恢复微调的主脚本。 |
| `9_quantize_pruned_model.py` | 基于 bitsandbytes 将剪枝后的模型加载为 4-bit 并保存。 |
| `evaluate_plot.py` | 对多个模型进行加载/推理耗时分析并输出图表，可作为额外对照实验。 |
| `paper_innovations_summary.txt` | 文献对比与实验观察的总结，可补充写作素材。 |
| `requirements.txt` | 完整依赖列表（Torch 2.5.1 + CUDA 12.1、bitsandbytes、datasets、thop 等）。 |

> **提示**：所有脚本默认在仓库根目录执行，并通过相对路径访问模型与缓存。

## 3. 环境与依赖

1. **Python 与硬件**  
   - 建议 Python ≥ 3.10，脚本在 3.12 环境下验证。  
   - 如果可用 GPU，`8_pruning_with_finetuning.py` 将自动使用 CUDA；否则会退回 CPU（训练耗时显著增加）。  
   - `bitsandbytes` 与 4-bit 量化对 GPU 架构与 CUDA 版本有要求，仓库默认 `torch==2.5.1+cu121`、`bitsandbytes==0.46.1`。

2. **依赖安装**  
   ```bash
   python -m venv venv
   venv\\Scripts\\activate   # PowerShell
   pip install -r requirements.txt
   ```

3. **额外工具**  
   - `thop` 被用于 FLOPs 统计（需要 PyTorch 能加载模型本体）。
   - Matplotlib 用于生成性能-算力散点图与对照图。

## 4. 数据准备

- SST-2 数据集优先从 `./sst2_data/glue/sst2/...` 读取（避免重复下载），不存在时会自动调用 `datasets` 从 Hugging Face 拉取。
- 如果处于离线环境，可先运行：
  ```bash
  python -c "from datasets import load_dataset; load_dataset('glue','sst2')"
  ```
  然后将缓存复制到 `./sst2_data/...`，以便后续脚本直接本地加载。

## 5. 工作流分解

### 5.1 模型准备：`download_model.py`

- 下载 `distilbert-base-uncased`（包含 tokenizer 与二分类头），保存到 `./distilbert-local`。  
- 该目录既是 `8_pruning_with_finetuning.py` 的学生起点，也是 `7_evaluate...` 里的 tokenizer 来源。若已有自定义蒸馏学生，可直接覆盖该目录。

### 5.2 渐进剪枝 + 恢复微调：`8_pruning_with_finetuning.py`

1. **配置类** (`GradualPruningWithFinetuningConfig`, 行 16 起)  
   - 训练：`batch_size=32`、`learning_rate=5e-5`、`total_epochs=15`。  
   - 蒸馏：温度 10→2，`alpha` 0.9→0.2，用于平衡 KL 与 CE。  
   - 剪枝：目标稀疏度 20%，从第 2 到 8 轮执行，每 20 个 step 触发一次；目标稀疏度随训练进度按三次多项式平滑提升。  
   - 微调：第 9 轮起进入 “Finetuning” 阶段，学习率降到 `1e-5`，并启用早停（耐心值 5、`min_delta=0.001`）。  
   - 日志：`experiment_name` 带时间戳，便于区分不同实验。

2. **结构化剪枝器** (`StructuralPruner`, 行 50 起)  
   - 仅作用于除分类头以外的 `Linear` 层，对每个输出通道计算 L2 范数，依据全局阈值决定是否保留。  
   - 保留/剪枝通过逐行 mask 完成，同时记录 mask 以便训练过程中反复应用。  
   - `compute_importance_and_prune` 返回当前全局稀疏度，供日志记录。

3. **训练主循环** (`main`, 行 118 起)  
   - 加载教师/学生与 tokenizer，使用 `datasets` 进行 `max_length=128` 的静态 padding。  
   - `phase` 根据 epoch 自动切换为 Warmup、Pruning 或 Finetuning，并动态调节学习率与蒸馏温度。  
   - 在剪枝或微调阶段，每个优化步后都会重新应用 mask，防止 pruned 权重被更新。  
   - 每轮评估验证集准确率与稀疏度；在微调阶段监控最佳准确率并进行早停。  
   - 输出：  
     - 最佳模型与 tokenizer 保存到 `./models/{experiment_name}`。  
     - 同目录下写出 `training_history.json` 记录每轮准确率/稀疏度/阶段。

> **常见修改点**  
> - 提高稀疏度：调整 `target_sparsity` 或延长 `pruning_end_epoch`。  
> - 更换数据集：修改 `load_dataset` 与 `tokenize_function`。  
> - 控制显存：调小 `batch_size`，或在 `DataLoader` 中传入 `num_workers`。

### 5.3 4-bit 量化：`9_quantize_pruned_model.py`

- 读取剪枝产物路径（默认硬编码为 `./models/pruning_with_finetuning_20251029_092719`，可按实际运行结果替换）。  
- 使用 `BitsAndBytesConfig(load_in_4bit=True, quant_type='nf4', compute_dtype=bfloat16)` 以 4-bit 方式加载，并启用 `device_map='auto'`。  
- 量化后的模型与 tokenizer 默认保存到 `./models/pruned_quantized_final`，供评估与部署使用。  
- 若需要以 QLoRA 形式继续微调，可在加载后追加 LoRA 适配器训练。

### 5.4 统一评估与报告：`7_evaluate_and_generate_report.py`

1. **模型列表** (`MODELS_TO_EVALUATE`, 行 12 起)  
   - 预定义三种模型：蒸馏基线、蒸馏+量化、剪枝+量化。  
   - `eval_path` 指向加载推理所需的目录，`gflops_path` 指定用于 FLOPs 计算的结构（剪枝模型使用稀疏度估算），`sparsity` 用于线性缩放 GFLOPs。

2. **评估流程**  
   - **准确率**：`evaluate_accuracy` (行 55) 逐批次前向计算；量化模型通过 4-bit 配置放到 GPU。  
   - **模型体积**：`get_model_size` (行 37) 累计 `bin/safetensors` 文件大小。  
   - **GFLOPs**：基线通过 `calculate_gflops` (行 46) 使用 `thop.profile` 精确计算，剪枝模型按 `baseline_gflops * (1 - sparsity)` 近似。  
   - **可视化**：`generate_report_plot` (行 69) 输出散点图 `final_report.png`，并写出 `final_report.json`。

3. **扩展**  
   - 若需要添加更多模型，只需在 `MODELS_TO_EVALUATE` 字典附加条目并确保路径存在。  
   - 在无 GPU 环境下运行量化模型评估会较慢，可将 `BitsAndBytesConfig` 切换为 `load_in_8bit`。

### 5.5 对照评估脚本：`evaluate_plot.py`

- 通过 Hugging Face `Trainer`（针对全精度模型）与自定义 `DataLoader`（针对量化模型）对 `teacher_model`、`student_model`、`student_softlabel_model`、`student_quantized_model` 进行推理耗时/加载耗时/准确率统计。  
- 生成的四联图 (`results/model_comparison.(png|pdf)`) 展示不同模型的准确率、推理时间与加载时间，并将数值写入 `results/model_comparison.csv`。  
- 可根据需要将剪枝+量化模型加入 `models` 列表，用于与蒸馏阶段的更多实验对比。

## 6. 实验产出与辅助文件

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `models/<experiment>/training_history.json` | 每轮准确率、稀疏度、阶段标签 | 可视化学习曲线或排查训练阶段表现。 |
| `final_report.json` | 三个对比模型的准确率 / GFLOPs / 体积 | 供论文或 PPT 直接引用。 |
| `final_report.png` | Accuracy vs. GFLOPs 散点图 | 帮助展示算力收益。 |
| `paper_innovations_summary.txt` | 文献对比与结论总结 | 撰写报告或引言时引用。 |
| `results/` | `evaluate_plot.py` 的多维度对比结果 | 分析不同蒸馏/量化策略的综合表现。 |

## 7. 自定义与扩展建议

1. **更高稀疏度或不同结构约束**  
   - 可在 `StructuralPruner` 中按层、按 heads 或 attention blocks 区分阈值，或引入 2:4 稀疏等硬件友好模式。

2. **替换学生模型**  
   - 将 `download_model.py` 指向新的 checkpoint，并确保 `AutoModelForSequenceClassification` 头的 `num_labels` 与任务一致。  
   - 注意同时更新 `MODELS_TO_EVALUATE` 中的参考路径。

3. **更丰富的评估指标**  
   - 在 `evaluate_accuracy` 基础上增加 F1、AUC 或混淆矩阵，或通过 `datasets` 的 `metric` API 计算官方指标。  
   - 结合 `evaluate_plot.py` 中的推理耗时，形成 “准确率-速度-体积” 三维对比。

4. **部署与集成**  
   - 量化后模型可直接用作 `transformers` 推理；若目标是 TensorRT/ONNX，可在量化前导出稀疏结构，再交由对应后端做进一步压缩。

## 8. 快速复现命令

```bash
# 1. 安装依赖（首次）
pip install -r requirements.txt

# 2. 准备学生模型
python download_model.py

# 3. 剪枝 + 恢复微调
python 8_pruning_with_finetuning.py

# 4. 量化剪枝模型（根据输出路径调整参数）
python 9_quantize_pruned_model.py

# 5. 统一评估并生成报告
python 7_evaluate_and_generate_report.py
```

脚本运行完成后，可在 `models/`、`final_report.*`、`results/` 下查看各阶段产物。

## 9. 常见问题 & 排查

| 症状 | 可能原因 | 建议 |
| --- | --- | --- |
| `bitsandbytes` 初始化失败 | CUDA / GPU 版本不匹配 | 确认 `torch`、`bitsandbytes` 与驱动版本一致，或改用 8-bit/全精度推理。 |
| SST-2 下载超时 | 离线环境或网络受限 | 先在联网环境缓存 `datasets`, 再复制到 `./sst2_data/...`。 |
| 训练早停过早 | 验证集波动或学习率过低 | 放宽 `patience`、`min_delta` 或延后 `finetuning_start_epoch`。 |
| GFLOPs 为 0 | `thop` 未能 profile | 确保已安装 CUDA 版本的 PyTorch，或手动提供 GFLOPs 数值。 |

---

如需进一步撰写论文/报告，可将本说明与 `paper_innovations_summary.txt` 搭配使用：前者负责复现细节，后者提供研究背景与创新点总结。
