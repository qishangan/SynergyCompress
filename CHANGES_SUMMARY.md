# 代码改动总结

本文档总结了为实现KGSP、MSP和QKD-4bit三类算法升级所做的所有代码改动。

## 改动文件清单

### 1. `8_pruning_with_finetuning.py` - 增强剪枝脚本

#### 改动内容：
- **扩展配置类** (行 16-59)
  - 新增 `importance_mode`、`lambda_mag`、`lambda_grad`、`pruning_window_steps` (KGSP参数)
  - 新增 `enable_multi_granularity`、`head_final_sparsity`、`ffn_final_sparsity` (MSP参数)

- **完全重写 StructuralPruner 类** (行 61-342)
  - 新增 `_build_structure_groups()`: 构建attention heads和FFN neurons的结构分组
  - 新增 `_get_module_by_name()`: 根据名称获取模块
  - 新增 `_compute_l2_importance()`: L2范数重要性计算
  - 新增 `_compute_grad_importance()`: 梯度重要性计算
  - 新增 `accumulate_importance_scores()`: 窗口累计重要性分数
  - 新增 `_get_combined_importance()`: 组合L2和梯度重要性
  - 修改 `compute_importance_and_prune()`: 支持多粒度剪枝
  - 新增 `_prune_uniform()`: 原始的统一剪枝逻辑
  - 新增 `_prune_multi_granularity()`: 多粒度剪枝逻辑

- **更新训练循环** (行 390, 430-459, 478-485, 491-498)
  - 初始化pruner时传入config参数
  - 添加多项式调度分别计算head和FFN稀疏度
  - 剪枝时根据配置选择uniform或multi-granularity模式
  - backward后调用accumulate_importance_scores()
  - 评估时使用正确的剪枝方法

#### 向后兼容性：
- 设置 `importance_mode="l2"` 和 `enable_multi_granularity=False` 恢复原始行为
- 保持原有接口和输出格式不变

---

### 2. `10_qat_kd_4bit.py` - 新增QAT+KD脚本

#### 文件说明：
全新创建的脚本，实现4-bit量化感知训练结合知识蒸馏。

#### 主要功能：
- **QATKDConfig类**: 配置QAT训练参数和KD损失权重
- **load_data_and_models()**: 加载4-bit学生模型和FP32教师模型
- **compute_hidden_state_loss()**: 计算隐藏层表示的MSE损失
- **train_one_epoch()**: 单epoch训练，支持三种KD损失：
  - CE Loss (任务损失)
  - Logits KD (输出蒸馏)
  - Hidden States KD (中间层蒸馏)
- **evaluate()**: 评估模型准确率
- **main()**: 主函数，支持命令行参数

#### 命令行参数：
- `--pruned_model_path`: 剪枝模型路径（默认自动检测最新）
- `--num_epochs`: QAT训练轮数（默认2）
- `--learning_rate`: 学习率（默认1e-5）
- `--batch_size`: 批大小（默认16）

---

### 3. `7_evaluate_and_generate_report.py` - 更新评估脚本

#### 改动内容：
- **扩展模型列表** (行 12-33)
  - 更新 `MODELS_TO_EVALUATE` 字典
  - 重命名 "Pruned + Quantized Student (Final)" → "Pruned + Quantized Student (PTQ)"
  - 新增 "Pruned + QKD-4bit (QAT)" 条目

- **更新量化检测逻辑** (行 132)
  - 修改 `is_quantized` 判断条件，增加对 "qkd" 关键字的检测

- **增加错误处理** (行 124-163)
  - 检查模型路径是否存在
  - 使用try-except捕获评估错误
  - 跳过不存在或错误的模型而不中断整体评估

- **更新可视化配置** (行 82-83)
  - 增加第4种颜色 `#d62728` 用于QKD-4bit模型
  - 修改为动态选择颜色数量: `colors[:len(labels)]`

#### 输出变化：
- `final_report.json`: 可能包含4个模型（增加了QKD-4bit）
- `final_report.png`: 支持显示4个模型点

---

### 4. `9_quantize_pruned_model.py` - 保持不变

#### 说明：
该脚本完全保持原样，继续提供传统的4-bit PTQ量化功能，用于与QKD-4bit对比。

---

### 5. `ALGORITHM_ENHANCEMENTS.md` - 新增使用文档

#### 文件说明：
全新创建的详细使用说明文档。

#### 内容包括：
1. KGSP功能说明和配置
2. MSP功能说明和配置
3. QKD-4bit使用方法
4. 完整训练流程
5. 性能调优建议
6. 故障排查
7. 兼容性说明
8. 参考配置

---

## 技术实现细节

### KGSP实现
```
重要性计算 = λ_mag × normalize(L2_norm) + λ_grad × normalize(|grad·weight|)
```
- 在每个训练步的backward后收集梯度信息
- 使用滑动窗口累计最近N步的梯度重要性
- 剪枝时使用累计平均值而非单步梯度

### MSP实现
```
对于DistilBERT:
- 每层6个attention heads → 6×6=36个head groups (6层)
- 每层3072个FFN neurons → 3072×6=18432个neuron groups

分别排序后应用不同稀疏度阈值
```

### QKD-4bit实现
```
Total Loss = β_ce × CE + β_logits × KL(T||S) × τ² + β_hidden × MSE(H_T, H_S)
```
- Teacher: FP32精度
- Student: 4-bit量化（BitsAndBytes NF4）
- 训练1-2个epoch，学习率1e-5

---

## 代码统计

### 新增代码量
- `8_pruning_with_finetuning.py`: +280行 (重写StructuralPruner)
- `10_qat_kd_4bit.py`: +300行 (全新文件)
- `7_evaluate_and_generate_report.py`: +40行 (错误处理和新模型)
- `ALGORITHM_ENHANCEMENTS.md`: +300行 (文档)

**总计**: ~920行新代码

### 修改代码量
- `8_pruning_with_finetuning.py`: ~50行 (配置和训练循环)
- `7_evaluate_and_generate_report.py`: ~20行 (模型列表和可视化)

**总计**: ~70行修改

---

## 测试建议

### 单元测试
1. **KGSP测试**
   ```python
   # 测试不同importance_mode
   config.importance_mode = "l2"
   config.importance_mode = "kd_grad"
   config.importance_mode = "l2+kd_grad"
   ```

2. **MSP测试**
   ```python
   # 测试structure groups构建
   pruner = StructuralPruner(model, config)
   assert len(pruner.head_groups) == 6 * 6  # DistilBERT
   assert len(pruner.ffn_groups) == 6 * 3072
   ```

3. **QKD-4bit测试**
   ```bash
   # 测试短训练
   python 10_qat_kd_4bit.py --num_epochs 1 --batch_size 8
   ```

### 集成测试
```bash
# 完整流程测试
python 8_pruning_with_finetuning.py
python 10_qat_kd_4bit.py
python 7_evaluate_and_generate_report.py
```

---

## 性能预期

### KGSP vs 原始L2剪枝
- 预期准确率提升: +0.5% ~ +1.5%
- 额外计算开销: ~5% (梯度累计)

### MSP vs 全局统一剪枝
- 预期准确率提升: +0.3% ~ +1.0%
- 内存开销: 忽略不计 (只在初始化时构建分组)

### QKD-4bit vs PTQ
- 预期准确率提升: +1.0% ~ +2.0%
- 额外训练时间: 2 epochs × 训练集大小

---

## 未来改进方向

1. **支持更多模型架构**
   - BERT、RoBERTa、GPT等
   - 需修改 `_build_structure_groups()` 以适配不同模型结构

2. **动态剪枝调度**
   - 根据验证集表现动态调整head/FFN稀疏度

3. **混合精度QAT**
   - 支持不同层使用不同量化位宽

4. **蒸馏策略优化**
   - 添加attention map蒸馏
   - 尝试其他蒸馏损失（CRD、DKD等）

---

## 参考文献

- **KGSP**: Knowledge Distillation Guided Structural Pruning
- **MSP**: Multi-granularity Structural Pruning for Transformers
- **QAT**: Quantization-Aware Training
- **NF4**: 4-bit NormalFloat Quantization (BitsAndBytes)
