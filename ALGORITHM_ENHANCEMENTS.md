# 算法增强功能使用说明

本文档说明如何使用新增的三类算法升级功能：KGSP、MSP 和 QKD-4bit。

## 一、KGSP（蒸馏感知的渐进结构化剪枝）

### 功能说明
在剪枝阶段结合知识蒸馏损失的梯度信息来评估参数重要性，相比纯L2范数更准确。

### 配置参数（在 `8_pruning_with_finetuning.py` 中）

```python
class GradualPruningWithFinetuningConfig:
    # KGSP 参数
    self.importance_mode = "l2+kd_grad"  # 选项: "l2", "kd_grad", "l2+kd_grad"
    self.lambda_mag = 0.5  # L2范数重要性权重
    self.lambda_grad = 0.5  # 梯度重要性权重
    self.pruning_window_steps = 50  # 累计重要性的步数窗口
```

### 重要性模式说明
- **"l2"**: 仅使用L2范数（原始方法，向后兼容）
- **"kd_grad"**: 仅使用梯度×权重的重要性
- **"l2+kd_grad"**: 组合两种方法（推荐，默认）

### 使用方式
直接运行脚本，KGSP会自动启用：
```bash
python 8_pruning_with_finetuning.py
```

---

## 二、MSP（多粒度Transformer结构化剪枝）

### 功能说明
区分attention heads和FFN neurons两类结构，分别应用不同的稀疏度目标，实现更精细的剪枝控制。

### 配置参数

```python
class GradualPruningWithFinetuningConfig:
    # MSP 参数
    self.enable_multi_granularity = True  # 是否启用多粒度剪枝
    self.head_final_sparsity = 0.30  # attention heads目标稀疏度
    self.ffn_final_sparsity = 0.20  # FFN neurons目标稀疏度
```

### 结构识别
- **Attention Heads**: 自动识别DistilBERT中每层的Q/K/V/O投影，按head分组
- **FFN Neurons**: 自动识别FFN中间层(lin1)和输出层(lin2)，按neuron分组

### 使用方式
1. 启用多粒度剪枝（默认已启用）：
```python
config.enable_multi_granularity = True
```

2. 调整不同结构的稀疏度目标：
```python
config.head_final_sparsity = 0.30  # 30%的heads被剪枝
config.ffn_final_sparsity = 0.20   # 20%的FFN neurons被剪枝
```

3. 如需禁用多粒度剪枝，恢复为全局统一剪枝：
```python
config.enable_multi_granularity = False
```

---

## 三、QKD-4bit（4-bit量化感知蒸馏）

### 功能说明
在剪枝后的模型基础上，进行短期QAT+KD训练，获得比纯PTQ更高精度的4-bit量化模型。

### 使用方式

#### 1. 基本使用（自动检测最新剪枝模型）
```bash
python 10_qat_kd_4bit.py
```

#### 2. 指定剪枝模型路径
```bash
python 10_qat_kd_4bit.py --pruned_model_path ./models/pruning_with_finetuning_20251029_092719
```

#### 3. 自定义训练参数
```bash
python 10_qat_kd_4bit.py \
    --pruned_model_path ./models/your_pruned_model \
    --num_epochs 3 \
    --learning_rate 5e-6 \
    --batch_size 16
```

### 配置参数（在 `10_qat_kd_4bit.py` 中）

```python
class QATKDConfig:
    # 训练参数
    self.batch_size = 16
    self.learning_rate = 1e-5
    self.num_epochs = 2
    self.max_grad_norm = 1.0

    # KD损失权重
    self.beta_ce = 0.3  # 任务损失权重
    self.beta_logits = 0.5  # logits蒸馏权重
    self.beta_hidden = 0.2  # 隐藏层蒸馏权重
    self.temperature = 4.0  # 蒸馏温度
```

### 损失组成
1. **CE Loss**: 对真实标签的交叉熵损失
2. **Logits KD**: teacher和student输出logits的KL散度
3. **Hidden States KD**: teacher和student中间层表示的MSE损失（可选）

### 输出
- 模型保存路径: `./models/pruned_qkd4bit_YYYYMMDD_HHMMSS/`
- 训练历史: `training_history.json`
- 配置信息: `config.json`

---

## 四、完整训练流程

### 标准流程
```bash
# Step 1: KGSP + MSP 剪枝与微调
python 8_pruning_with_finetuning.py

# Step 2: 传统4-bit PTQ（可选，用于对比）
python 9_quantize_pruned_model.py

# Step 3: 4-bit QAT + KD（推荐）
python 10_qat_kd_4bit.py

# Step 4: 统一评估所有模型
python 7_evaluate_and_generate_report.py
```

### 评估报告
运行评估脚本后，会生成：
- `final_report.json`: 详细的数值结果
- `final_report.png`: Accuracy vs GFLOPs 可视化图表

报告包含以下模型：
1. Distilled Student (Baseline)
2. Quantized Distilled Student
3. Pruned + Quantized Student (PTQ)
4. **Pruned + QKD-4bit (QAT)** ← 新增

---

## 五、性能调优建议

### KGSP调优
- 如果重要性累计窗口太小，可能导致重要性估计不稳定：
  ```python
  config.pruning_window_steps = 100  # 增加窗口大小
  ```
- 调整L2和梯度的权重比例：
  ```python
  config.lambda_mag = 0.7  # 更重视L2范数
  config.lambda_grad = 0.3  # 减少梯度权重
  ```

### MSP调优
- 如果attention性能下降明显，减少head剪枝：
  ```python
  config.head_final_sparsity = 0.20  # 从0.30降到0.20
  ```
- 如果FFN参数量仍然过大，增加FFN剪枝：
  ```python
  config.ffn_final_sparsity = 0.30  # 从0.20提高到0.30
  ```

### QKD-4bit调优
- 增加QAT训练轮数以获得更好收敛：
  ```python
  config.num_epochs = 3  # 从2增加到3
  ```
- 调整KD损失权重：
  ```python
  config.beta_ce = 0.2      # 减少任务损失
  config.beta_logits = 0.6  # 增加logits蒸馏
  config.beta_hidden = 0.2  # 保持hidden states蒸馏
  ```

---

## 六、故障排查

### 问题1: 多粒度剪枝时出现维度错误
**原因**: 模型结构与DistilBERT不匹配
**解决**: 确保使用DistilBERT或修改 `_build_structure_groups()` 以适配您的模型

### 问题2: QAT训练时OOM
**原因**: 4-bit模型仍然占用较多显存
**解决**:
- 减小batch size: `--batch_size 8`
- 禁用hidden states蒸馏: `config.beta_hidden = 0.0`

### 问题3: 评估脚本找不到QKD-4bit模型
**原因**: 模型保存路径名称不匹配
**解决**:
1. 检查 `./models/` 下的实际目录名
2. 更新 `7_evaluate_and_generate_report.py` 中的路径配置

---

## 七、兼容性说明

### 向后兼容性
所有改动保持向后兼容：
- 设置 `importance_mode="l2"` 和 `enable_multi_granularity=False` 可恢复原始行为
- 原有脚本 `9_quantize_pruned_model.py` 保持不变

### 文件格式
- `training_history.json`: 格式不变，可与旧版本互操作
- `final_report.json`: 增加了新模型条目，但结构保持一致

---

## 八、参考配置

### 保守配置（更高精度）
```python
# 8_pruning_with_finetuning.py
config.importance_mode = "l2+kd_grad"
config.lambda_mag = 0.6
config.lambda_grad = 0.4
config.head_final_sparsity = 0.20
config.ffn_final_sparsity = 0.15
config.pruning_window_steps = 100

# 10_qat_kd_4bit.py
config.num_epochs = 3
config.beta_ce = 0.2
config.beta_logits = 0.6
config.beta_hidden = 0.2
```

### 激进配置（更高压缩率）
```python
# 8_pruning_with_finetuning.py
config.importance_mode = "l2+kd_grad"
config.lambda_mag = 0.5
config.lambda_grad = 0.5
config.head_final_sparsity = 0.40
config.ffn_final_sparsity = 0.35
config.pruning_window_steps = 50

# 10_qat_kd_4bit.py
config.num_epochs = 2
config.beta_ce = 0.3
config.beta_logits = 0.5
config.beta_hidden = 0.2
```
