import argparse
import copy
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from adaptive_pruner import AdaptivePruner
from dgsp_utils import DistillationGuidedScorer, is_prunable_linear
from sensitivity_analyzer import KnowledgeSensitivityAnalyzer


class GradualPruningWithFinetuningConfig:
    """Configuration for pruning with recovery finetuning."""

    def __init__(
        self,
        sensitivity_method: str = "gradient",
        target_sparsity: float = 0.20,
        alpha: float = 1.0,
    ) -> None:
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.max_seq_length = 128
        self.use_subset = False
        self.train_subset_ratio = 1.0
        self.val_subset_ratio = 1.0
        self.seed = 42

        if target_sparsity <= 0.20:
            self.total_epochs = 15
        elif target_sparsity <= 0.40:
            self.total_epochs = 18
        else:
            self.total_epochs = 20

        self.initial_temperature = 10.0
        self.final_temperature = 2.0
        self.alpha_initial = 0.9
        self.alpha_final = 0.2

        self.target_sparsity = target_sparsity
        self.pruning_start_epoch = 2
        self.pruning_end_epoch = min(8, self.total_epochs - 4)
        self.pruning_frequency_steps = 20

        self.finetuning_start_epoch = self.pruning_end_epoch + 1
        self.finetuning_lr = 1e-5

        self.sensitivity_method = sensitivity_method
        self.alpha = alpha
        self.lambda_score = 0.5
        self.ema_beta = 0.9
        self.use_layer_adaptive = True
        self.rho_min_factor = 0.5
        self.rho_max_factor = 1.5

        self.patience = 5
        self.min_delta = 0.001

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adaptive_tag = "adaptive" if self.use_layer_adaptive else "global"
        self.experiment_name = (
            f"pruning_{sensitivity_method}_"
            f"sp{int(target_sparsity * 100):02d}_"
            f"{adaptive_tag}_"
            f"a{alpha:.1f}_"
            f"{timestamp}"
        )

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_and_models():
    try:
        dataset = load_dataset("./sst2_data/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c")
    except Exception:
        print("Failed to load local dataset, downloading...")
        dataset = load_dataset("glue", "sst2")

    teacher = AutoModelForSequenceClassification.from_pretrained("./teacher_model")
    student = AutoModelForSequenceClassification.from_pretrained("./distilbert-local")
    tokenizer = AutoTokenizer.from_pretrained("./distilbert-local")
    return dataset, teacher, student, tokenizer


def evaluate(model, dataloader, device):
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
            logits = outputs.logits
            total_loss += ce_loss_fn(logits, labels).item() * labels.size(0)
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    loss = total_loss / total_samples if total_samples > 0 else 0.0
    return accuracy, loss


def format_layer_sensitivity_payload(
    snapshot: Dict[str, object],
    config: GradualPruningWithFinetuningConfig,
    global_step: int,
) -> Dict[str, object]:
    layer_sensitivity = snapshot.get("layer_sensitivity", {})
    layer_details = snapshot.get("layer_details", {})
    layers = []
    for layer_name in sorted(layer_sensitivity.keys()):
        detail = layer_details.get(layer_name, {})
        layers.append({
            "layer_name": layer_name,
            "mean_sensitivity": float(detail.get("mean_sensitivity", layer_sensitivity[layer_name])),
            "normalized_sensitivity": float(layer_sensitivity[layer_name]),
            "mean_l2_score": float(detail.get("mean_l2_score", 0.0)),
            "mean_kd_score": float(detail.get("mean_kd_score", 0.0)),
            "num_groups": int(detail.get("num_groups", 0)),
            "fused_min": float(detail.get("fused_min", 0.0)),
            "fused_max": float(detail.get("fused_max", 0.0)),
        })

    return {
        "method": config.sensitivity_method,
        "global_step": int(global_step),
        "lambda_score": float(config.lambda_score),
        "ema_beta": float(config.ema_beta),
        "layer_sensitivity": {
            k: float(v) for k, v in sorted(layer_sensitivity.items())
        },
        "layers": layers,
    }


def format_pruning_allocation_payload(
    pruner: AdaptivePruner,
    snapshot: Dict[str, object],
    actual_sparsities: Dict[str, float],
    config: GradualPruningWithFinetuningConfig,
    global_step: int,
) -> Dict[str, object]:
    layer_sensitivity = snapshot.get("layer_sensitivity", {})
    allocation_layers = []
    for layer_name in sorted(pruner.layer_sparsities.keys()):
        allocation_layers.append({
            "layer_name": layer_name,
            "mean_sensitivity": float(layer_sensitivity.get(layer_name, 0.0)),
            "target_sparsity": float(pruner.layer_sparsities[layer_name]),
            "actual_sparsity": float(actual_sparsities.get(layer_name, 0.0)),
        })

    return {
        "method": config.sensitivity_method,
        "global_step": int(global_step),
        "global_target_sparsity": float(pruner.global_sparsity),
        "adaptive_enabled": bool(config.use_layer_adaptive),
        "rho_min_factor": float(config.rho_min_factor),
        "rho_max_factor": float(config.rho_max_factor),
        "layers": allocation_layers,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_static_sensitivity(
    sensitivity_method: str,
    analyzer: KnowledgeSensitivityAnalyzer,
    val_loader: DataLoader,
    args: argparse.Namespace,
) -> Dict[str, object]:
    if sensitivity_method == "gradient":
        print("  -> Using gradient-based sensitivity")
        raw_sensitivities = analyzer.compute_gradient_based_sensitivity(
            val_loader,
            max_batches=args.sensitivity_batches,
        )
        normalized_sensitivities = analyzer.normalize_sensitivities(raw_sensitivities)
    elif sensitivity_method == "kd":
        print("  -> Using KD-based sensitivity")
        raw_sensitivities = analyzer.compute_layer_sensitivity(
            val_loader,
            max_batches=args.sensitivity_batches,
        )
        normalized_sensitivities = analyzer.normalize_sensitivities(raw_sensitivities)
    elif sensitivity_method == "random":
        print("  -> Using random sensitivity")
        raw_sensitivities = analyzer.compute_random_sensitivity(seed=args.random_seed)
        normalized_sensitivities = analyzer.normalize_sensitivities(raw_sensitivities)
    elif sensitivity_method == "uniform":
        print("  -> Using uniform sensitivity")
        layer_names = analyzer.get_prunable_layer_names()
        raw_sensitivities = {name: 1.0 for name in layer_names}
        normalized_sensitivities = raw_sensitivities.copy()
    else:
        raise ValueError(f"Unsupported static sensitivity method: {sensitivity_method}")

    layer_details = {}
    for layer_name, normalized_value in normalized_sensitivities.items():
        layer_details[layer_name] = {
            "mean_sensitivity": float(normalized_value),
            "normalized_sensitivity": float(normalized_value),
            "mean_l2_score": 0.0,
            "mean_kd_score": 0.0,
            "num_groups": 0,
            "fused_min": 0.0,
            "fused_max": 0.0,
        }

    return {
        "layer_sensitivity": {k: float(v) for k, v in normalized_sensitivities.items()},
        "layer_details": layer_details,
        "raw_layer_sensitivity": {k: float(v) for k, v in raw_sensitivities.items()},
        "group_scores": None,
    }


def main(config: GradualPruningWithFinetuningConfig, args: argparse.Namespace):
    assert args.sensitivity_batches > 0, "sensitivity_batches must be > 0"
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("--- Loading Data and Models ---")
    dataset, teacher, student, tokenizer = load_data_and_models()
    teacher.to(device)
    student.to(device)
    teacher.eval()

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

    model_save_path = Path("./models") / config.experiment_name
    sensitivity_dir = model_save_path / "sensitivity_analysis"
    model_save_path.mkdir(parents=True, exist_ok=True)
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_save_path / "pruning_config.json", config.to_dict())

    runtime_scorer: Optional[DistillationGuidedScorer] = None
    initial_snapshot: Optional[Dict[str, object]] = None

    print("\n" + "=" * 60)
    print("PHASE 0: Preparing Sensitivity Signal")
    print("=" * 60)

    if config.sensitivity_method in {"weight", "dgsp"}:
        lambda_score = 0.0 if config.sensitivity_method == "weight" else config.lambda_score
        runtime_scorer = DistillationGuidedScorer(
            student,
            lambda_score=lambda_score,
            ema_beta=config.ema_beta,
        )
        initial_snapshot = runtime_scorer.collect_scores()
    else:
        analyzer = KnowledgeSensitivityAnalyzer(student, teacher, device)
        initial_snapshot = compute_static_sensitivity(config.sensitivity_method, analyzer, val_loader, args)

    assert initial_snapshot is not None, "Initial sensitivity snapshot must not be None"
    initial_payload = format_layer_sensitivity_payload(initial_snapshot, config, global_step=0)
    save_json(model_save_path / "layer_sensitivity.json", initial_payload)
    save_json(sensitivity_dir / "layer_sensitivity.json", initial_payload)

    sorted_layers = sorted(
        initial_snapshot["layer_sensitivity"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    print(f"  -> Top 3 sensitive layers: {sorted_layers[:3]}")
    print(f"  -> Bottom 3 sensitive layers: {list(reversed(sorted_layers[-3:]))}")

    pruner = AdaptivePruner(
        model=student,
        sensitivity_scores=initial_snapshot["layer_sensitivity"],
        global_sparsity=config.target_sparsity,
        alpha=config.alpha,
    )

    if config.use_layer_adaptive:
        pruner.compute_adaptive_sparsities(
            sensitivity_scores=initial_snapshot["layer_sensitivity"],
            adaptive_enabled=True,
            rho_min_factor=config.rho_min_factor,
            rho_max_factor=config.rho_max_factor,
        )
    else:
        pruner.compute_adaptive_sparsities(adaptive_enabled=False)

    initial_allocation = format_pruning_allocation_payload(
        pruner=pruner,
        snapshot=initial_snapshot,
        actual_sparsities={},
        config=config,
        global_step=0,
    )
    save_json(model_save_path / "pruning_allocation.json", initial_allocation)
    save_json(sensitivity_dir / "pruning_allocation.json", initial_allocation)

    optimizer = AdamW(student.parameters(), lr=config.learning_rate)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()

    best_finetuned_acc = 0.0
    patience_counter = 0
    best_finetuned_model_state = None
    history = []
    start_time = time.time()
    latest_snapshot = initial_snapshot

    print(f"\n--- Starting Training for {config.total_epochs} epochs ---")
    print(f"Pruning phase: epoch {config.pruning_start_epoch} to {config.pruning_end_epoch}")
    print(f"Finetuning phase: epoch {config.finetuning_start_epoch} to {config.total_epochs}")

    pruning_start_step = len(train_loader) * (config.pruning_start_epoch - 1)
    pruning_end_step = len(train_loader) * config.pruning_end_epoch

    global_step = 0
    for epoch in range(1, config.total_epochs + 1):
        epoch_start = time.time()
        student.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if epoch >= config.finetuning_start_epoch:
            phase = "Finetuning"
            if epoch == config.finetuning_start_epoch:
                print(f"  -> Switching to finetuning learning rate: {config.finetuning_lr}")
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.finetuning_lr
        elif epoch >= config.pruning_start_epoch:
            phase = "Pruning"
        else:
            phase = "Warmup"

        print(f"\n--- Epoch {epoch}/{config.total_epochs} | Phase: {phase} ---")

        progress_bar = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{config.total_epochs} [{phase}]",
            leave=False,
            dynamic_ncols=True,
        )

        last_allocation_range = "0.000-0.000"
        latest_allocation_payload = initial_allocation

        for _, batch in progress_bar:
            global_step += 1

            progress_denom = max(1, pruning_end_step - pruning_start_step)
            progress = min(1.0, max(0.0, (global_step - pruning_start_step) / progress_denom))
            if phase == "Pruning":
                target_sparsity = config.target_sparsity * (3 * progress**2 - 2 * progress**3)
            elif phase == "Finetuning":
                target_sparsity = config.target_sparsity
            else:
                target_sparsity = 0.0

            temperature = config.initial_temperature - (config.initial_temperature - config.final_temperature) * progress
            distill_alpha = config.alpha_initial - (config.alpha_initial - config.alpha_final) * progress

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids, attention_mask=attention_mask).logits

            if phase == "Pruning" and global_step % config.pruning_frequency_steps == 0:
                if runtime_scorer is not None and config.sensitivity_method == "dgsp":
                    optimizer.zero_grad(set_to_none=True)
                    kd_student_logits = student(input_ids, attention_mask=attention_mask).logits
                    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
                    soft_probs = F.log_softmax(kd_student_logits / temperature, dim=-1)
                    kd_only_loss = kl_loss_fn(soft_probs, soft_targets) * (temperature ** 2)
                    kd_only_loss.backward()
                    runtime_scorer.update_kd_sensitivity_from_grads()
                    optimizer.zero_grad(set_to_none=True)

                if runtime_scorer is not None:
                    latest_snapshot = runtime_scorer.collect_scores()
                    pruner.set_sensitivity_scores(latest_snapshot["layer_sensitivity"])

                pruner.global_sparsity = target_sparsity
                if config.use_layer_adaptive:
                    pruner.compute_adaptive_sparsities(
                        sensitivity_scores=pruner.sensitivity_scores,
                        adaptive_enabled=True,
                        rho_min_factor=config.rho_min_factor,
                        rho_max_factor=config.rho_max_factor,
                    )
                    actual_sparsities = pruner.prune_with_layer_targets(
                        group_scores=latest_snapshot.get("group_scores")
                    )
                else:
                    pruner.compute_adaptive_sparsities(adaptive_enabled=False)
                    actual_sparsities = pruner.prune_globally(
                        group_scores=latest_snapshot.get("group_scores"),
                        target_sparsity=target_sparsity,
                    )

                latest_layer_payload = format_layer_sensitivity_payload(latest_snapshot, config, global_step)
                latest_allocation_payload = format_pruning_allocation_payload(
                    pruner=pruner,
                    snapshot=latest_snapshot,
                    actual_sparsities=actual_sparsities,
                    config=config,
                    global_step=global_step,
                )
                save_json(model_save_path / "layer_sensitivity.json", latest_layer_payload)
                save_json(model_save_path / "pruning_allocation.json", latest_allocation_payload)
                save_json(sensitivity_dir / "layer_sensitivity.json", latest_layer_payload)
                save_json(sensitivity_dir / "pruning_allocation.json", latest_allocation_payload)

                if actual_sparsities:
                    last_allocation_range = (
                        f"{min(actual_sparsities.values()):.3f}-"
                        f"{max(actual_sparsities.values()):.3f}"
                    )
            elif phase == "Finetuning":
                pruner.apply_masks()

            student_logits = student(input_ids, attention_mask=attention_mask).logits
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            soft_probs = F.log_softmax(student_logits / temperature, dim=-1)
            distill_loss = kl_loss_fn(soft_probs, soft_targets) * (temperature ** 2)
            label_loss = ce_loss_fn(student_logits, labels)
            loss = distill_alpha * distill_loss + (1.0 - distill_alpha) * label_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if phase in {"Pruning", "Finetuning"}:
                pruner.apply_masks()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "T": f"{temperature:.2f}",
                "alpha": f"{distill_alpha:.2f}",
                "global_sp": f"{pruner.get_global_sparsity():.3f}",
            })

        if phase == "Pruning" and epoch == config.pruning_end_epoch:
            if runtime_scorer is not None:
                latest_snapshot = runtime_scorer.collect_scores()
                pruner.set_sensitivity_scores(latest_snapshot["layer_sensitivity"])
            pruner.global_sparsity = config.target_sparsity
            if config.use_layer_adaptive:
                pruner.compute_adaptive_sparsities(
                    sensitivity_scores=pruner.sensitivity_scores,
                    adaptive_enabled=True,
                    rho_min_factor=config.rho_min_factor,
                    rho_max_factor=config.rho_max_factor,
                )
                actual_sparsities = pruner.prune_with_layer_targets(
                    group_scores=latest_snapshot.get("group_scores")
                )
            else:
                pruner.compute_adaptive_sparsities(adaptive_enabled=False)
                actual_sparsities = pruner.prune_globally(
                    group_scores=latest_snapshot.get("group_scores"),
                    target_sparsity=config.target_sparsity,
                )

            latest_layer_payload = format_layer_sensitivity_payload(latest_snapshot, config, global_step)
            latest_allocation_payload = format_pruning_allocation_payload(
                pruner=pruner,
                snapshot=latest_snapshot,
                actual_sparsities=actual_sparsities,
                config=config,
                global_step=global_step,
            )
            save_json(model_save_path / "layer_sensitivity.json", latest_layer_payload)
            save_json(model_save_path / "pruning_allocation.json", latest_allocation_payload)
            save_json(sensitivity_dir / "layer_sensitivity.json", latest_layer_payload)
            save_json(sensitivity_dir / "pruning_allocation.json", latest_allocation_payload)

        val_acc, val_loss = evaluate(student, val_loader, device)
        actual_sparsity = pruner.get_global_sparsity()
        compression_ratio = 1.0 / (1.0 - actual_sparsity) if actual_sparsity < 1.0 else float("inf")
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        current_layer_sparsities = pruner.layer_sparsities
        if current_layer_sparsities:
            layer_range = (
                f"{min(current_layer_sparsities.values()):.3f}-"
                f"{max(current_layer_sparsities.values()):.3f}"
            )
        else:
            layer_range = last_allocation_range

        epoch_record = {
            "epoch": epoch,
            "phase": phase,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "global_sparsity": actual_sparsity,
            "target_sparsity": config.target_sparsity,
            "layer_target_range": layer_range,
            "compression_ratio": compression_ratio,
            "peak_memory_mb": peak_memory_mb,
            "epoch_time_sec": time.time() - epoch_start,
        }
        history.append(epoch_record)
        save_json(model_save_path / "training_history.json", {"history": history})

        print(f"\nEpoch {epoch} results:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Global Sparsity: {actual_sparsity:.4f}")
        print(f"  Layer Target Range: {layer_range}")
        print(f"  Peak Memory: {peak_memory_mb:.1f} MB")

        if phase == "Finetuning":
            if val_acc > best_finetuned_acc + config.min_delta:
                best_finetuned_acc = val_acc
                patience_counter = 0
                best_finetuned_model_state = copy.deepcopy(student.state_dict())
                print("  -> New best finetuned model")
            else:
                patience_counter += 1
                print(f"  -> Patience: {patience_counter}/{config.patience}")
        else:
            patience_counter = 0

        if epoch == config.pruning_end_epoch:
            pruning_ckpt_path = model_save_path / "checkpoint_pruning"
            pruning_ckpt_path.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(pruning_ckpt_path)
            tokenizer.save_pretrained(pruning_ckpt_path)
            save_json(pruning_ckpt_path / "layer_sensitivity.json", format_layer_sensitivity_payload(latest_snapshot, config, global_step))
            save_json(pruning_ckpt_path / "pruning_allocation.json", latest_allocation_payload)
            print(f"  -> Saved pruning checkpoint to {pruning_ckpt_path}")

        if patience_counter >= config.patience:
            print("--- Early stopping triggered ---")
            break

    if best_finetuned_model_state is not None:
        student.load_state_dict(best_finetuned_model_state)

    student.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    final_layer_payload = format_layer_sensitivity_payload(latest_snapshot, config, global_step)
    final_allocation_payload = format_pruning_allocation_payload(
        pruner=pruner,
        snapshot=latest_snapshot,
        actual_sparsities={name: entry.get("actual_sparsity", 0.0) for name, entry in pruner.last_allocation.items()},
        config=config,
        global_step=global_step,
    )
    save_json(model_save_path / "layer_sensitivity.json", final_layer_payload)
    save_json(model_save_path / "pruning_allocation.json", final_allocation_payload)

    final_stats = {
        "method": config.sensitivity_method,
        "use_layer_adaptive": config.use_layer_adaptive,
        "global_sparsity": pruner.get_global_sparsity(),
        "target_sparsity": config.target_sparsity,
        "layer_sparsities": pruner.layer_sparsities,
        "best_accuracy": best_finetuned_acc,
        "lambda_score": config.lambda_score,
        "ema_beta": config.ema_beta,
        "alpha": config.alpha,
        "total_time_sec": time.time() - start_time,
        "training_history_file": "training_history.json",
        "layer_sensitivity_file": "layer_sensitivity.json",
        "pruning_allocation_file": "pruning_allocation.json",
    }
    save_json(model_save_path / "kgapq_stats.json", final_stats)

    print("\n--- Training finished ---")
    print(f"Model saved to: {model_save_path}")
    print(f"Best finetuned accuracy: {best_finetuned_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DGSP pruning with recovery finetuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sensitivity_method",
        type=str,
        default="gradient",
        choices=["gradient", "kd", "weight", "random", "uniform", "dgsp"],
        help="Sensitivity source for pruning decisions",
    )
    parser.add_argument("--target_sparsity", type=float, default=0.20, help="Target global sparsity")
    parser.add_argument("--alpha", type=float, default=1.0, help="Adaptive allocation strength")
    parser.add_argument("--lambda_score", type=float, default=0.5, help="Weight for KD-gradient sensitivity in DGSP")
    parser.add_argument("--ema_beta", type=float, default=0.9, help="EMA factor for KD-gradient sensitivity")
    parser.add_argument("--fast", action="store_true", help="Use a reduced quick-validation setup")
    parser.add_argument("--disable_layer_adaptive", action="store_true", help="Use global threshold pruning instead of layer-adaptive budgets")
    parser.add_argument("--sensitivity_batches", type=int, default=50, help="Validation batches for static sensitivity analysis")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--total_epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--pruning_frequency_steps", type=int, default=None, help="How often to update pruning masks")
    parser.add_argument("--train_subset_ratio", type=float, default=None, help="Optional training subset ratio")
    parser.add_argument("--val_subset_ratio", type=float, default=None, help="Optional validation subset ratio")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum tokenized sequence length")
    parser.add_argument("--rho_min_factor", type=float, default=0.5, help="Lower clipping factor for adaptive layer sparsity")
    parser.add_argument("--rho_max_factor", type=float, default=1.5, help="Upper clipping factor for adaptive layer sparsity")
    parser.add_argument("--experiment_name", type=str, default=None, help="Optional explicit output directory name under ./models")

    args = parser.parse_args()

    config = GradualPruningWithFinetuningConfig(
        sensitivity_method=args.sensitivity_method,
        target_sparsity=args.target_sparsity,
        alpha=args.alpha,
    )
    config.seed = args.random_seed
    config.lambda_score = args.lambda_score
    config.ema_beta = args.ema_beta
    config.use_layer_adaptive = not args.disable_layer_adaptive
    config.rho_min_factor = args.rho_min_factor
    config.rho_max_factor = args.rho_max_factor

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.total_epochs is not None:
        config.total_epochs = args.total_epochs
        config.pruning_end_epoch = min(config.pruning_end_epoch, max(2, config.total_epochs - 2))
        config.finetuning_start_epoch = config.pruning_end_epoch + 1
    if args.pruning_frequency_steps is not None:
        config.pruning_frequency_steps = args.pruning_frequency_steps
    if args.train_subset_ratio is not None:
        config.use_subset = args.train_subset_ratio < 1.0 or config.val_subset_ratio < 1.0
        config.train_subset_ratio = args.train_subset_ratio
    if args.val_subset_ratio is not None:
        config.use_subset = args.val_subset_ratio < 1.0 or config.train_subset_ratio < 1.0
        config.val_subset_ratio = args.val_subset_ratio
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length

    if args.fast:
        print("\nFAST MODE ENABLED - Using quick-validation setup")
        config.total_epochs = 5 if args.total_epochs is None else config.total_epochs
        config.pruning_end_epoch = min(3, max(2, config.total_epochs - 2))
        config.finetuning_start_epoch = config.pruning_end_epoch + 1
        config.use_subset = True
        config.train_subset_ratio = args.train_subset_ratio if args.train_subset_ratio is not None else 0.15
        config.val_subset_ratio = args.val_subset_ratio if args.val_subset_ratio is not None else 0.5
        config.max_seq_length = args.max_seq_length if args.max_seq_length is not None else 64
        config.batch_size = args.batch_size if args.batch_size is not None else 32

    adaptive_tag = "adaptive" if config.use_layer_adaptive else "global"
    config.experiment_name = (
        args.experiment_name
        if args.experiment_name
        else (
            f"pruning_{config.sensitivity_method}_"
            f"sp{int(config.target_sparsity * 100):02d}_"
            f"{adaptive_tag}_"
            f"a{config.alpha:.1f}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )

    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"  Sensitivity Method: {config.sensitivity_method}")
    print(f"  Target Sparsity: {config.target_sparsity * 100:.0f}%")
    print(f"  Layer Adaptive: {config.use_layer_adaptive}")
    print(f"  Lambda Score: {config.lambda_score}")
    print(f"  EMA Beta: {config.ema_beta}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Total Epochs: {config.total_epochs}")
    print(f"  Pruning Epochs: {config.pruning_start_epoch}-{config.pruning_end_epoch}")
    print(f"  Finetuning Epochs: {config.finetuning_start_epoch}-{config.total_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Max Seq Length: {config.max_seq_length}")
    print(f"  Subset: {config.use_subset} (train={config.train_subset_ratio}, val={config.val_subset_ratio})")
    print("=" * 60 + "\n")

    main(config, args)
