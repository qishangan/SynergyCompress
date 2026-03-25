import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from bitsandbytes.optim import AdamW8bit
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from dgsp_utils import aggregate_hidden_layer_sensitivity, load_layer_sensitivity_file


class QATKDConfig:
    """Configuration for 4-bit KD recovery."""

    def __init__(self, fast_mode: bool = False) -> None:
        self.pruned_model_path = None
        self.teacher_model_path = "./teacher_model"
        self.sensitivity_path = None
        self.output_dir = None
        self.weighted_hidden = False
        self.enable_hidden_kd = False
        self.hidden_tau = 1.0
        self.hidden_weight_epoch1 = 0.3
        self.hidden_weight_late = 0.1

        if fast_mode:
            print("=" * 60)
            print("FAST MODE ENABLED - QKD optimized for quick validation")
            print("=" * 60)
            self.batch_size = 32
            self.learning_rate = 2e-5
            self.num_epochs = 1
            self.max_grad_norm = 1.0
            self.max_seq_length = 64
            self.use_subset = True
            self.train_subset_ratio = 0.3
            self.val_subset_ratio = 0.5
            self.beta_ce = 0.4
            self.beta_logits = 0.6
            self.beta_hidden = 0.0
            self.temperature = 3.0
        else:
            self.batch_size = 16
            self.learning_rate = 1e-5
            self.num_epochs = 2
            self.max_grad_norm = 1.0
            self.max_seq_length = 128
            self.use_subset = False
            self.train_subset_ratio = 1.0
            self.val_subset_ratio = 1.0
            self.beta_ce = 0.3
            self.beta_logits = 0.5
            self.beta_hidden = 0.2
            self.temperature = 4.0

        mode_suffix = "fast" if fast_mode else "full"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"qat_kd_4bit_{mode_suffix}_{timestamp}"
        self.output_dir = f"./models/pruned_qkd4bit_{mode_suffix}_{timestamp}"

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def find_latest_model(prefixes: List[str], root: str = "./models") -> Optional[str]:
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if any(name.startswith(prefix) for prefix in prefixes):
            candidates.append(full)
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def resolve_sensitivity_path(config: QATKDConfig, explicit_path: Optional[str]) -> Optional[str]:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    if config.pruned_model_path:
        candidates.extend([
            os.path.join(config.pruned_model_path, "layer_sensitivity.json"),
            os.path.join(config.pruned_model_path, "sensitivity_analysis", "layer_sensitivity.json"),
        ])
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def load_data_and_models(config: QATKDConfig):
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
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    student = AutoModelForSequenceClassification.from_pretrained(
        config.pruned_model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.pruned_model_path)
    return dataset, teacher, student, tokenizer


def align_teacher_hidden(student_hidden: List[torch.Tensor], teacher_hidden: List[torch.Tensor]) -> List[torch.Tensor]:
    if len(student_hidden) == len(teacher_hidden):
        return teacher_hidden
    teacher_indices = torch.linspace(0, len(teacher_hidden) - 1, len(student_hidden)).long()
    return [teacher_hidden[i] for i in teacher_indices]


def compute_hidden_state_loss(
    student_hidden: List[torch.Tensor],
    teacher_hidden: List[torch.Tensor],
    layer_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    aligned_teacher = align_teacher_hidden(student_hidden, teacher_hidden)

    if layer_weights is None:
        layer_weights = torch.ones(len(student_hidden), device=student_hidden[0].device, dtype=torch.float32)
        layer_weights = layer_weights / layer_weights.sum()
    else:
        layer_weights = layer_weights.to(student_hidden[0].device, dtype=torch.float32)
        layer_weights = layer_weights / layer_weights.sum().clamp_min(1e-8)

    losses = []
    for s_hidden, t_hidden in zip(student_hidden, aligned_teacher):
        losses.append(F.mse_loss(s_hidden, t_hidden))
    stacked = torch.stack(losses)
    return torch.sum(stacked * layer_weights)


def build_masks(model):
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            masks[name] = (module.weight != 0).float().to(module.weight.device)
    return masks


def apply_masks(model, masks):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in masks:
                mask_w = masks[name].to(module.weight.device, dtype=module.weight.dtype)
                module.weight.data.mul_(mask_w)


def build_hidden_layer_weights(
    sensitivity_path: Optional[str],
    num_student_hidden_states: int,
    tau: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    info: Dict[str, object] = {
        "weighted_hidden_enabled": sensitivity_path is not None,
        "sensitivity_path": sensitivity_path,
        "fallback_to_uniform": False,
        "tau": float(tau),
    }

    if sensitivity_path is None:
        info["fallback_to_uniform"] = True
        weights = torch.ones(num_student_hidden_states, dtype=torch.float32)
        weights = weights / weights.sum()
        info["hidden_weights"] = weights.tolist()
        return weights, info

    layer_sensitivity = load_layer_sensitivity_file(sensitivity_path)
    block_sensitivity = aggregate_hidden_layer_sensitivity(layer_sensitivity)
    info["layer_sensitivity"] = layer_sensitivity
    info["block_sensitivity"] = {str(k): float(v) for k, v in block_sensitivity.items()}

    if not block_sensitivity:
        info["fallback_to_uniform"] = True
        weights = torch.ones(num_student_hidden_states, dtype=torch.float32)
        weights = weights / weights.sum()
        info["hidden_weights"] = weights.tolist()
        return weights, info

    default_block_score = float(sum(block_sensitivity.values()) / len(block_sensitivity))
    hidden_scores = []
    for hidden_idx in range(num_student_hidden_states):
        if hidden_idx == 0:
            hidden_scores.append(default_block_score)
        else:
            hidden_scores.append(float(block_sensitivity.get(hidden_idx - 1, default_block_score)))

    hidden_score_tensor = torch.tensor(hidden_scores, dtype=torch.float32)
    weights = torch.softmax(hidden_score_tensor / max(float(tau), 1e-6), dim=0)
    info["hidden_scores"] = hidden_scores
    info["hidden_weights"] = weights.tolist()
    return weights, info


def get_hidden_loss_weight(config: QATKDConfig, epoch: int) -> float:
    if not config.enable_hidden_kd and not config.weighted_hidden:
        return 0.0
    if config.weighted_hidden:
        return config.hidden_weight_epoch1 if epoch == 1 else config.hidden_weight_late
    return config.beta_hidden


def train_one_epoch(
    model,
    teacher,
    dataloader,
    optimizer,
    config: QATKDConfig,
    device,
    epoch: int,
    masks=None,
    hidden_layer_weights: Optional[torch.Tensor] = None,
):
    model.train()
    teacher.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_hidden_loss = 0.0
    valid_steps = 0

    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()
    hidden_weight = get_hidden_loss_weight(config, epoch)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden = teacher_outputs.hidden_states

        student_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=hidden_weight > 0.0,
        )
        student_logits = student_outputs.logits
        student_hidden = student_outputs.hidden_states if hidden_weight > 0.0 else None

        student_logits_fp32 = student_logits.float()
        teacher_logits_fp32 = teacher_logits.float()

        ce_loss = ce_loss_fn(student_logits_fp32, labels)
        soft_targets = F.softmax(teacher_logits_fp32 / config.temperature, dim=-1)
        soft_probs = F.log_softmax(student_logits_fp32 / config.temperature, dim=-1)
        kd_loss = kl_loss_fn(soft_probs, soft_targets) * (config.temperature ** 2)

        if hidden_weight > 0.0 and student_hidden is not None:
            student_hidden_fp32 = [h.float() for h in student_hidden]
            teacher_hidden_fp32 = [h.float() for h in teacher_hidden]
            hidden_loss = compute_hidden_state_loss(
                student_hidden_fp32,
                teacher_hidden_fp32,
                layer_weights=hidden_layer_weights,
            )
        else:
            hidden_loss = torch.tensor(0.0, device=device)

        loss = (
            config.beta_ce * ce_loss
            + config.beta_logits * kd_loss
            + hidden_weight * hidden_loss
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  -> Skipping batch {batch_idx + 1} due to invalid loss")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        if masks:
            apply_masks(model, masks)

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_kd_loss += kd_loss.item()
        total_hidden_loss += hidden_loss.item()
        valid_steps += 1

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | "
                f"KD: {kd_loss.item():.4f} | Hidden: {hidden_loss.item():.4f} | "
                f"HiddenWeight: {hidden_weight:.3f}"
            )

    denom = max(1, valid_steps)
    return (
        total_loss / denom,
        total_ce_loss / denom,
        total_kd_loss / denom,
        total_hidden_loss / denom,
        hidden_weight,
    )


def evaluate(model, dataloader, device, masks=None):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            total_loss += ce_loss_fn(logits, labels).item() * labels.size(0)
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            if masks:
                apply_masks(model, masks)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    loss = total_loss / total_samples if total_samples > 0 else 0.0
    return accuracy, loss


def main():
    parser = argparse.ArgumentParser(description="4-bit QKD recovery")
    parser.add_argument("--pruned_model_path", type=str, default=None, help="Path to the pruned model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the recovered 4-bit model")
    parser.add_argument("--sensitivity_path", type=str, default=None, help="Path to layer_sensitivity.json")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of QKD epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--train_subset_ratio", type=float, default=None, help="Optional training subset ratio")
    parser.add_argument("--val_subset_ratio", type=float, default=None, help="Optional validation subset ratio")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode")
    parser.add_argument("--enable_hidden_kd", action="store_true", help="Enable uniform hidden-state distillation")
    parser.add_argument("--weighted_hidden", action="store_true", help="Use sensitivity-weighted hidden-state distillation")
    parser.add_argument("--hidden_tau", type=float, default=1.0, help="Temperature for weighted hidden softmax")

    args = parser.parse_args()
    config = QATKDConfig(fast_mode=args.fast)

    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.train_subset_ratio is not None:
        config.use_subset = args.train_subset_ratio < 1.0 or config.val_subset_ratio < 1.0
        config.train_subset_ratio = args.train_subset_ratio
    if args.val_subset_ratio is not None:
        config.use_subset = args.val_subset_ratio < 1.0 or config.train_subset_ratio < 1.0
        config.val_subset_ratio = args.val_subset_ratio
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length

    config.enable_hidden_kd = args.enable_hidden_kd or args.weighted_hidden
    config.weighted_hidden = args.weighted_hidden
    config.hidden_tau = args.hidden_tau
    if config.enable_hidden_kd and config.beta_hidden == 0.0 and not config.weighted_hidden:
        config.beta_hidden = 0.2

    if args.pruned_model_path:
        config.pruned_model_path = args.pruned_model_path
    else:
        config.pruned_model_path = find_latest_model(
            ["pruning_", "pruning_with_finetuning_"],
            root="./models",
        )
        if config.pruned_model_path:
            print(f"Auto-detected pruned model: {config.pruned_model_path}")

    if config.pruned_model_path is None or not os.path.exists(config.pruned_model_path):
        raise FileNotFoundError("Could not resolve pruned model path for QKD recovery")

    if args.output_dir:
        config.output_dir = args.output_dir

    config.sensitivity_path = resolve_sensitivity_path(config, args.sensitivity_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, teacher, student, tokenizer = load_data_and_models(config)
    teacher = teacher.to(device)
    masks = build_masks(student)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    if config.use_subset:
        train_size = max(1, int(len(tokenized_datasets["train"]) * config.train_subset_ratio))
        val_size = max(1, int(len(tokenized_datasets["validation"]) * config.val_subset_ratio))
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(train_size))
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(val_size))
        print(f"Using subset: train={train_size}, validation={val_size}")
    else:
        print(
            f"Using full dataset: train={len(tokenized_datasets['train'])}, "
            f"validation={len(tokenized_datasets['validation'])}"
        )

    train_loader = DataLoader(tokenized_datasets["train"], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=config.batch_size)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = AdamW8bit(trainable_params, lr=config.learning_rate)

    hidden_layer_weights = None
    weighted_hidden_info: Dict[str, object] = {
        "weighted_hidden_enabled": False,
        "fallback_to_uniform": True,
        "hidden_weights": [],
    }
    if config.enable_hidden_kd:
        student_hidden_count = student.config.n_layers + 1 if hasattr(student.config, "n_layers") else student.config.num_hidden_layers + 1
        hidden_layer_weights, weighted_hidden_info = build_hidden_layer_weights(
            sensitivity_path=config.sensitivity_path if config.weighted_hidden else None,
            num_student_hidden_states=student_hidden_count,
            tau=config.hidden_tau,
        )
        weighted_hidden_info["weighted_hidden_enabled"] = bool(config.weighted_hidden)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weighted_hidden_config.json").write_text(
        json.dumps(weighted_hidden_info, indent=2),
        encoding="utf-8",
    )
    (output_dir / "qat_config.json").write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )

    print(f"\n--- Starting QKD recovery for {config.num_epochs} epochs ---")
    print(
        f"Loss weights: CE={config.beta_ce}, Logits={config.beta_logits}, "
        f"Hidden={'weighted' if config.weighted_hidden else config.beta_hidden}"
    )

    history = []
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{config.num_epochs} ---")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        avg_loss, avg_ce, avg_kd, avg_hidden, hidden_weight = train_one_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            config,
            device,
            epoch,
            masks=masks,
            hidden_layer_weights=hidden_layer_weights,
        )
        val_acc, val_loss = evaluate(student, val_loader, device, masks=masks)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        print(f"\nEpoch {epoch} summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  CE: {avg_ce:.4f} | KD: {avg_kd:.4f} | Hidden: {avg_hidden:.4f}")
        print(f"  Hidden Weight: {hidden_weight:.3f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Peak Memory: {peak_memory_mb:.1f} MB")

        history.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_ce": avg_ce,
            "avg_kd": avg_kd,
            "avg_hidden": avg_hidden,
            "hidden_loss_weight": hidden_weight,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "peak_memory_mb": peak_memory_mb,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            student.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  -> New best validation accuracy, checkpoint saved to {output_dir}")

    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    qkd_stats = {
        "best_val_acc": best_val_acc,
        "weighted_hidden": config.weighted_hidden,
        "enable_hidden_kd": config.enable_hidden_kd,
        "sensitivity_path": config.sensitivity_path,
        "total_time_sec": time.time() - start_time,
        "weighted_hidden_config_file": "weighted_hidden_config.json",
    }
    (output_dir / "qkd_stats.json").write_text(
        json.dumps(qkd_stats, indent=2),
        encoding="utf-8",
    )

    print("\n--- QKD recovery finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
