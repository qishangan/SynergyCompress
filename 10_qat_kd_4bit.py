import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from bitsandbytes.optim import AdamW8bit
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse
import json
from datetime import datetime


class QATKDConfig:
    """Configuration for Quantization-Aware Training with Knowledge Distillation."""
    def __init__(self, fast_mode=False):
        """
        Initialize configuration.

        Args:
            fast_mode (bool): If True, use faster settings for RTX 4080 mobile GPU.
                             Reduces training time significantly.
        """
        # Model paths
        self.pruned_model_path = None  # Will be set via argparse
        self.teacher_model_path = "./teacher_model"

        if fast_mode:
            print("=" * 60)
            print("FAST MODE ENABLED - QAT optimized for quick training")
            print("=" * 60)

            # Training parameters - optimized for speed
            self.batch_size = 32  # Larger batch for 4-bit model
            self.learning_rate = 2e-5  # Higher for faster convergence
            self.num_epochs = 1  # Reduced from 2
            self.max_grad_norm = 1.0
            self.max_seq_length = 64  # Reduced sequence length

            # Dataset subsampling
            self.use_subset = True
            self.train_subset_ratio = 0.3  # 30% of training data
            self.val_subset_ratio = 0.5  # 50% of validation data

            # KD loss weights - simplified
            self.beta_ce = 0.4  # Task loss weight
            self.beta_logits = 0.6  # Logits distillation weight
            self.beta_hidden = 0.0  # Disable hidden state distillation for speed
            self.temperature = 3.0  # Lower temperature

        else:
            # Original full training mode
            # Training parameters
            self.batch_size = 16
            self.learning_rate = 1e-5
            self.num_epochs = 2
            self.max_grad_norm = 1.0
            self.max_seq_length = 128

            # No subsampling
            self.use_subset = False
            self.train_subset_ratio = 1.0
            self.val_subset_ratio = 1.0

            # KD loss weights
            self.beta_ce = 0.3  # Task loss weight
            self.beta_logits = 0.5  # Logits distillation weight
            self.beta_hidden = 0.2  # Hidden states distillation weight
            self.temperature = 4.0

        # Experiment tracking
        mode_suffix = "fast" if fast_mode else "full"
        self.experiment_name = f"qat_kd_4bit_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = f"./models/pruned_qkd4bit_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def load_data_and_models(config):
    """Load dataset, teacher model (FP32), and student model (4-bit)."""
    print("--- Loading Dataset ---")
    try:
        dataset = load_dataset("./sst2_data/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c")
    except Exception:
        print("Failed to load local dataset, downloading...")
        dataset = load_dataset("glue", "sst2")

    print(f"--- Loading Teacher Model from: {config.teacher_model_path} ---")
    teacher = AutoModelForSequenceClassification.from_pretrained(config.teacher_model_path)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"--- Loading Pruned Student Model (4-bit) from: {config.pruned_model_path} ---")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Use bfloat16 compute for better numeric stability while training
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )

    student = AutoModelForSequenceClassification.from_pretrained(
        config.pruned_model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.pruned_model_path)

    return dataset, teacher, student, tokenizer


def compute_hidden_state_loss(student_hidden, teacher_hidden):
    """
    Compute MSE loss between student and teacher hidden states.
    Only compute for matching layers (take every other layer if student has fewer).
    """
    if len(student_hidden) != len(teacher_hidden):
        # Sample teacher layers to match student layers
        teacher_indices = torch.linspace(0, len(teacher_hidden) - 1, len(student_hidden)).long()
        teacher_hidden = [teacher_hidden[i] for i in teacher_indices]

    total_loss = 0.0
    for s_hidden, t_hidden in zip(student_hidden, teacher_hidden):
        # Both are [batch, seq_len, hidden_dim]
        total_loss += F.mse_loss(s_hidden, t_hidden)

    return total_loss / len(student_hidden)


def train_one_epoch(model, teacher, dataloader, optimizer, config, device, epoch, masks=None):
    """Train for one epoch with QAT and KD."""
    model.train()
    teacher.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_hidden_loss = 0.0
    valid_steps = 0

    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden = teacher_outputs.hidden_states

        # Student forward
        student_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        student_logits = student_outputs.logits
        student_hidden = student_outputs.hidden_states

        # Work in float32 for the losses to avoid instability in half precision
        student_logits_fp32 = student_logits.float()
        teacher_logits_fp32 = teacher_logits.float()

        # Compute losses
        # 1. Task loss (CE)
        ce_loss = ce_loss_fn(student_logits_fp32, labels)

        # 2. Logits distillation (KL divergence)
        soft_targets = F.softmax(teacher_logits_fp32 / config.temperature, dim=-1)
        soft_probs = F.log_softmax(student_logits_fp32 / config.temperature, dim=-1)
        kd_loss = kl_loss_fn(soft_probs, soft_targets) * (config.temperature ** 2)

        # 3. Hidden states distillation (MSE) - optional
        if config.beta_hidden > 0:
            student_hidden_fp32 = [h.float() for h in student_hidden]
            teacher_hidden_fp32 = [h.float() for h in teacher_hidden]
            hidden_loss = compute_hidden_state_loss(student_hidden_fp32, teacher_hidden_fp32)
        else:
            hidden_loss = torch.tensor(0.0, device=device)

        # Combined loss
        loss = (config.beta_ce * ce_loss +
                config.beta_logits * kd_loss +
                config.beta_hidden * hidden_loss)

        # Skip unstable steps
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  -> Skipping batch {batch_idx + 1} due to invalid loss (nan/inf)")
            continue

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()

        # 训练中强制保持剪枝 mask
        if masks:
            apply_masks(model, masks)

        # Accumulate losses
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_kd_loss += kd_loss.item()
        total_hidden_loss += hidden_loss.item()
        valid_steps += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | "
                  f"KD: {kd_loss.item():.4f} | Hidden: {hidden_loss.item():.4f}")

    denom = max(1, valid_steps)
    avg_loss = total_loss / denom
    avg_ce = total_ce_loss / denom
    avg_kd = total_kd_loss / denom
    avg_hidden = total_hidden_loss / denom

    return avg_loss, avg_ce, avg_kd, avg_hidden


def evaluate(model, dataloader, device, masks=None):
    """Evaluate model accuracy."""
    model.eval()
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # 评估时也保持剪枝 mask
            if masks:
                apply_masks(model, masks)

    return total_correct / total_samples if total_samples > 0 else 0


def build_masks(model):
    """基于当前零权重位置构建剪枝 mask（仅作用于 Linear 层）。"""
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            mask = (module.weight != 0).float().to(module.weight.device)
            masks[name] = mask
    return masks


def apply_masks(model, masks):
    """将 mask 重新施加到模型权重，防止训练/优化器破坏稀疏结构。"""
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in masks:
                mask_w = masks[name].to(module.weight.device, dtype=module.weight.dtype)
                module.weight.data.mul_(mask_w)


def main():
    parser = argparse.ArgumentParser(description="4-bit QAT with Knowledge Distillation")
    parser.add_argument(
        "--pruned_model_path",
        type=str,
        default=None,
        help="Path to the pruned model checkpoint (default: latest in ./models/)"
    )
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of QAT epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode for quick iteration")

    args = parser.parse_args()

    config = QATKDConfig(fast_mode=args.fast)

    # Override config with command-line arguments if provided
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Auto-detect latest pruned model if not specified
    if args.pruned_model_path:
        config.pruned_model_path = args.pruned_model_path
    else:
        models_dir = "./models"
        if os.path.exists(models_dir):
            pruned_dirs = [
                d for d in os.listdir(models_dir)
                if d.startswith("pruning_with_finetuning_")
            ]
            if pruned_dirs:
                pruned_dirs.sort(reverse=True)
                config.pruned_model_path = os.path.join(models_dir, pruned_dirs[0])
                print(f"Auto-detected pruned model: {config.pruned_model_path}")
            else:
                print("Error: No pruned model found in ./models/")
                return
        else:
            print("Error: ./models/ directory not found")
            return

    if not os.path.exists(config.pruned_model_path):
        print(f"Error: Pruned model not found at '{config.pruned_model_path}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and models
    dataset, teacher, student, tokenizer = load_data_and_models(config)

    teacher = teacher.to(device)

    # 构建剪枝 mask，并在后续训练/评估中保持稀疏结构
    masks = build_masks(student)

    # Prepare data loaders
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=config.max_seq_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Apply dataset subsampling if enabled
    if config.use_subset:
        train_size = int(len(tokenized_datasets['train']) * config.train_subset_ratio)
        val_size = int(len(tokenized_datasets['validation']) * config.val_subset_ratio)

        tokenized_datasets['train'] = tokenized_datasets['train'].select(range(train_size))
        tokenized_datasets['validation'] = tokenized_datasets['validation'].select(range(val_size))

        print(f"Using SUBSET: Train={train_size} samples, Val={val_size} samples")
    else:
        print(f"Using FULL dataset: Train={len(tokenized_datasets['train'])} samples, Val={len(tokenized_datasets['validation'])} samples")

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_datasets['validation'], batch_size=config.batch_size)

    # Setup optimizer (only optimize trainable parameters)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW8bit(trainable_params, lr=config.learning_rate)

    # Training loop
    print(f"\n--- Starting QAT with KD for {config.num_epochs} Epochs ---")
    print(f"Loss weights: CE={config.beta_ce}, Logits={config.beta_logits}, Hidden={config.beta_hidden}")

    history = []
    best_val_acc = 0

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{config.num_epochs} ---")

        # Train
        avg_loss, avg_ce, avg_kd, avg_hidden = train_one_epoch(
            student, teacher, train_loader, optimizer, config, device, epoch, masks=masks
        )

        # Evaluate
        val_acc = evaluate(student, val_loader, device, masks=masks)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | KD: {avg_kd:.4f} | Hidden: {avg_hidden:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")

        history.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'avg_ce': avg_ce,
            'avg_kd': avg_kd,
            'avg_hidden': avg_hidden,
            'val_acc': val_acc
        })

        # Save best model checkpoint immediately to avoid state_dict issues with 4-bit params
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.output_dir, exist_ok=True)
            student.save_pretrained(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)
            print(f"  -> New best validation accuracy! Checkpoint saved to {config.output_dir}")

    # Save final model (last epoch)
    print("\n--- Training Finished ---")
    os.makedirs(config.output_dir, exist_ok=True)
    student.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Save training history and config
    with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(config.output_dir, 'qat_config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
