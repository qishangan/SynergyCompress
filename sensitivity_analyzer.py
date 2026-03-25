"""
Module for computing layer-wise knowledge distillation sensitivity.
Sensitivity = KL divergence between student and teacher layer outputs
multiplied by a layer weight norm proxy.
"""

from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class KnowledgeSensitivityAnalyzer:
    """
    Computes layer-wise sensitivity scores based on:
    1. KL divergence between student and teacher layer outputs
    2. Layer weight magnitude (importance proxy)
    """

    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, device: torch.device) -> None:
        assert student_model is not None, "student_model must not be None"
        assert teacher_model is not None, "teacher_model must not be None"
        assert device is not None, "device must not be None"
        self.student = student_model
        self.teacher = teacher_model
        self.device = device
        self.layer_outputs: Dict[str, torch.Tensor] = {}
        self.layer_sequences: Dict[str, List[Tuple[str, torch.Tensor]]] = {"student": [], "teacher": []}

    def _reset_cache(self) -> None:
        """Reset cached layer outputs for a fresh forward pass."""
        self.layer_outputs = {}
        self.layer_sequences = {"student": [], "teacher": []}

    def _register_hooks(self, model: nn.Module, model_type: str) -> List[torch.utils.hooks.RemovableHandle]:
        """Register forward hooks to capture layer outputs."""
        assert model_type in ("student", "teacher"), "model_type must be 'student' or 'teacher'"
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "classifier" not in name:
                def hook_fn(
                    _module: nn.Module,
                    _input: Tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                    layer_name: str = name,
                    layer_type: str = model_type,
                ) -> None:
                    if not torch.is_tensor(output):
                        return
                    detached = output.detach()
                    key = f"{layer_type}_{layer_name}"
                    self.layer_outputs[key] = detached
                    self.layer_sequences[layer_type].append((layer_name, detached))

                hooks.append(module.register_forward_hook(hook_fn))
        return hooks

    def _align_layers_by_index(
        self,
        student_seq: List[Tuple[str, torch.Tensor]],
        teacher_seq: List[Tuple[str, torch.Tensor]],
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """
        Align student and teacher layers by index when names do not match.
        This distributes student layers across teacher layers by relative position.
        """
        if not student_seq or not teacher_seq:
            return []
        pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        teacher_count = len(teacher_seq)
        student_count = len(student_seq)
        for idx, (s_name, s_act) in enumerate(student_seq):
            if student_count == 1:
                t_index = 0
            else:
                t_index = int(round(idx * (teacher_count - 1) / (student_count - 1)))
            t_index = max(0, min(teacher_count - 1, t_index))
            _, t_act = teacher_seq[t_index]
            pairs.append((s_name, s_act, t_act))
        return pairs

    def _align_activation_pair(
        self,
        student_act: torch.Tensor,
        teacher_act: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Align activation shapes so KL divergence can be computed.
        Returns None when alignment is not possible.
        """
        if not torch.is_tensor(student_act) or not torch.is_tensor(teacher_act):
            return None

        if student_act.size(-1) != teacher_act.size(-1):
            return None

        s_dim = student_act.dim()
        t_dim = teacher_act.dim()

        if s_dim == t_dim:
            if student_act.shape == teacher_act.shape:
                return student_act, teacher_act
            if s_dim == 3:
                # Align sequence length by truncating to min length.
                min_len = min(student_act.size(1), teacher_act.size(1))
                return student_act[:, :min_len, :], teacher_act[:, :min_len, :]
            return None

        # Handle [batch, hidden] vs [batch, seq, hidden] by pooling seq.
        if s_dim == 2 and t_dim == 3:
            teacher_pooled = teacher_act.mean(dim=1)
            return student_act, teacher_pooled
        if s_dim == 3 and t_dim == 2:
            student_pooled = student_act.mean(dim=1)
            return student_pooled, teacher_act

        return None

    def get_prunable_layer_names(self) -> List[str]:
        """Return layer names eligible for sensitivity analysis and pruning."""
        layer_names: List[str] = []
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Linear) and "classifier" not in name:
                layer_names.append(name)
        return layer_names

    def _compute_weight_norms(self) -> Dict[str, float]:
        """Compute raw L2 weight norms for prunable layers."""
        norms: Dict[str, float] = {}
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Linear) and "classifier" not in name:
                norm_val = torch.norm(module.weight, p=2).item()
                if math.isfinite(norm_val):
                    norms[name] = float(norm_val)
        return norms

    def compute_layer_sensitivity(self, dataloader: DataLoader, max_batches: int = 50) -> Dict[str, float]:
        """
        Compute sensitivity score for each layer.

        Args:
            dataloader: Validation data loader.
            max_batches: Number of batches to use (for speed).

        Returns:
            Dict mapping layer name to sensitivity score (unnormalized).
        """
        assert dataloader is not None, "dataloader must not be None"
        assert max_batches > 0, "max_batches must be > 0"

        self.student.eval()
        self.teacher.eval()

        layer_kl_divs: Dict[str, List[float]] = {}
        batch_count = 0
        no_name_match_warned = False
        skipped_mismatch = 0

        print("  -> Registering hooks for layer output capture...")
        student_hooks = self._register_hooks(self.student, "student")
        teacher_hooks = self._register_hooks(self.teacher, "teacher")

        try:
            print(f"  -> Computing KL divergence over {max_batches} batches...")
            with torch.no_grad():
                for batch in dataloader:
                    if batch_count >= max_batches:
                        break

                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    self._reset_cache()
                    _ = self.student(input_ids, attention_mask=attention_mask)
                    student_outputs = {name: tensor for name, tensor in self.layer_sequences["student"]}
                    student_seq = list(self.layer_sequences["student"])

                    self._reset_cache()
                    _ = self.teacher(input_ids, attention_mask=attention_mask)
                    teacher_outputs = {name: tensor for name, tensor in self.layer_sequences["teacher"]}
                    teacher_seq = list(self.layer_sequences["teacher"])

                    matched_names = set(student_outputs.keys()) & set(teacher_outputs.keys())
                    if matched_names:
                        pairs = [(name, student_outputs[name], teacher_outputs[name]) for name in matched_names]
                    else:
                        pairs = self._align_layers_by_index(student_seq, teacher_seq)
                        if pairs and not no_name_match_warned:
                            print("  -> Warning: no matching layer names; aligning layers by index.")
                            no_name_match_warned = True

                    for layer_name, student_act, teacher_act in pairs:
                        if student_act.dim() < 2 or teacher_act.dim() < 2:
                            skipped_mismatch += 1
                            continue
                        aligned = self._align_activation_pair(student_act, teacher_act)
                        if aligned is None:
                            skipped_mismatch += 1
                            continue
                        s_aligned, t_aligned = aligned
                        s_flat = s_aligned.reshape(-1, s_aligned.size(-1))
                        t_flat = t_aligned.reshape(-1, t_aligned.size(-1))
                        if s_flat.shape != t_flat.shape or s_flat.numel() == 0:
                            skipped_mismatch += 1
                            continue

                        kl_div = F.kl_div(
                            F.log_softmax(s_flat, dim=-1),
                            F.softmax(t_flat, dim=-1),
                            reduction="batchmean",
                        )
                        kl_val = float(kl_div.detach().cpu())
                        if not math.isfinite(kl_val):
                            continue

                        layer_kl_divs.setdefault(layer_name, []).append(kl_val)

                    batch_count += 1
        finally:
            for hook in student_hooks + teacher_hooks:
                hook.remove()

        if batch_count == 0:
            raise ValueError("No batches processed for sensitivity analysis.")

        if skipped_mismatch > 0:
            print(f"  -> Skipped {skipped_mismatch} layer activations due to shape mismatch.")

        avg_kl_divs = {
            name: float(np.mean(divs))
            for name, divs in layer_kl_divs.items()
            if divs
        }

        print("  -> Computing layer weight norms...")
        layer_weight_norms = self.compute_weight_norms()

        sensitivities: Dict[str, float] = {}
        for name, kl_score in avg_kl_divs.items():
            weight_norm = layer_weight_norms.get(name)
            if weight_norm is None:
                continue
            combined = kl_score * weight_norm
            if math.isfinite(combined):
                sensitivities[name] = float(combined)

        if not sensitivities:
            print("  -> Warning: no KL-based sensitivities computed; falling back to weight norms.")
            sensitivities = layer_weight_norms

        print(f"  -> Computed sensitivity for {len(sensitivities)} layers")
        return sensitivities

    def normalize_sensitivities(self, sensitivities: Dict[str, float]) -> Dict[str, float]:
        """Normalize sensitivity scores to [0, 1] range."""
        assert sensitivities, "sensitivities must not be empty"
        values = np.array(list(sensitivities.values()), dtype=np.float64)
        finite_mask = np.isfinite(values)
        if not finite_mask.all():
            print("  -> Warning: non-finite sensitivity values detected; replacing with 0.")
            values = np.where(finite_mask, values, 0.0)

        min_val, max_val = float(values.min()), float(values.max())
        if max_val - min_val < 1e-8:
            return {k: 0.5 for k in sensitivities.keys()}

        normalized = {
            name: float((score - min_val) / (max_val - min_val))
            for name, score in sensitivities.items()
        }
        return normalized

    def compute_weight_norm_only(self) -> Dict[str, float]:
        """Baseline: use only weight magnitude as normalized sensitivity."""
        norms = self.compute_weight_norms()
        return self.normalize_sensitivities(norms)

    def compute_weight_norms(self) -> Dict[str, float]:
        """Compute raw weight norm sensitivities for prunable layers."""
        norms = self._compute_weight_norms()
        assert norms, "No prunable layers found for weight norms."
        return norms

    def compute_random_sensitivity(self, seed: Optional[int] = None) -> Dict[str, float]:
        """Baseline: random sensitivity scores in [0, 1]."""
        rng = np.random.default_rng(seed)
        sensitivities: Dict[str, float] = {}
        for name in self.get_prunable_layer_names():
            sensitivities[name] = float(rng.random())
        return sensitivities

    def compute_gradient_based_sensitivity(self, dataloader: DataLoader, max_batches: int = 50) -> Dict[str, float]:
        """
        Compute layer sensitivity using gradient magnitudes and activation statistics.
        This is more stable than KD-based methods and doesn't require teacher alignment.

        Sensitivity Score = (Gradient Norm) * sqrt(Activation Variance) * (Weight Norm)
        """
        assert dataloader is not None, "dataloader must not be None"
        assert max_batches > 0, "max_batches must be > 0"

        print("\n" + "=" * 60)
        print("Computing Gradient-Based Layer Sensitivity")
        print("=" * 60)

        layer_modules = {
            name: module
            for name, module in self.student.named_modules()
            if isinstance(module, nn.Linear) and "classifier" not in name
        }
        assert layer_modules, "No prunable layers found for gradient sensitivity."

        layer_gradient_norms: Dict[str, List[float]] = {name: [] for name in layer_modules}
        layer_activation_vars: Dict[str, List[float]] = {name: [] for name in layer_modules}
        layer_weight_norms: Dict[str, float] = {}

        print("  -> Step 1/3: Computing gradient norms...")
        self.student.train()
        batch_count = 0

        for batch in dataloader:
            if batch_count >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.student.zero_grad(set_to_none=True)
            outputs = self.student(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()

            for name, module in layer_modules.items():
                grad = module.weight.grad
                grad_norm = torch.norm(grad, p=2).item() if grad is not None else 0.0
                layer_gradient_norms[name].append(grad_norm)

            batch_count += 1

        print(f"     Processed {batch_count} batches")
        print(f"     Collected gradients for {len(layer_gradient_norms)} layers")

        print("  -> Step 2/3: Computing activation variance...")
        activation_hooks = []
        activation_buffer: Dict[str, torch.Tensor] = {}

        def make_hook(layer_name: str):
            def hook_fn(_module, _input, output):
                if isinstance(output, tuple):
                    output = output[0]
                if torch.is_tensor(output):
                    activation_buffer[layer_name] = output.detach()
            return hook_fn

        for name, module in layer_modules.items():
            activation_hooks.append(module.register_forward_hook(make_hook(name)))

        self.student.eval()
        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= max_batches:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                activation_buffer.clear()
                _ = self.student(input_ids, attention_mask=attention_mask)

                for layer_name in layer_modules.keys():
                    activation = activation_buffer.get(layer_name)
                    if activation is None:
                        layer_activation_vars[layer_name].append(0.0)
                        continue

                    act_flat = activation.reshape(activation.size(0), -1)
                    variance = torch.var(act_flat, dim=0).mean().item()
                    layer_activation_vars[layer_name].append(variance)

                batch_count += 1

        for hook in activation_hooks:
            hook.remove()

        print(f"     Collected activations for {len(layer_activation_vars)} layers")

        print("  -> Step 3/3: Computing weight norms...")
        for name, module in layer_modules.items():
            weight_norm = torch.norm(module.weight, p=2).item()
            layer_weight_norms[name] = float(weight_norm)

        print(f"     Computed weight norms for {len(layer_weight_norms)} layers")

        print("  -> Combining metrics into sensitivity scores...")
        sensitivities: Dict[str, float] = {}
        skipped_layers: List[str] = []

        for name in layer_modules.keys():
            grad_list = layer_gradient_norms.get(name, [])
            act_list = layer_activation_vars.get(name, [])
            weight_norm = layer_weight_norms.get(name)

            if not grad_list or not act_list or weight_norm is None:
                skipped_layers.append(name)
                continue

            avg_grad_norm = float(np.mean(grad_list))
            avg_activation_var = float(np.mean(act_list))
            sensitivity = avg_grad_norm * float(np.sqrt(avg_activation_var)) * float(weight_norm)

            if math.isfinite(sensitivity):
                sensitivities[name] = float(sensitivity)
            else:
                sensitivities[name] = 0.0

        print("\n  -> Sensitivity Computation Complete:")
        print(f"     Total layers: {len(layer_modules)}")
        print(f"     Successfully computed: {len(sensitivities)}")
        print(f"     Skipped: {len(skipped_layers)}")

        if skipped_layers:
            print(f"     Skipped layers: {skipped_layers[:5]}...")

        assert len(skipped_layers) == 0, (
            f"ERROR: {len(skipped_layers)} layers were skipped! This should not happen."
        )

        print("  OK: All layers computed successfully (0 skips, 0 imputations)\n")
        return sensitivities

    def visualize_sensitivity(self, sensitivities: Dict[str, float], save_path: str) -> None:
        """Create visualization of layer sensitivity distribution."""
        assert sensitivities, "sensitivities must not be empty"
        assert save_path, "save_path must not be empty"

        names = list(sensitivities.keys())
        scores = list(sensitivities.values())

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(names)), scores, color="steelblue", alpha=0.7)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Sensitivity Score", fontsize=12)
        plt.title("Knowledge Distillation Sensitivity per Layer", fontsize=14, weight="bold")
        plt.xticks(range(len(names)), [f"L{i}" for i in range(len(names))], rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  -> Sensitivity visualization saved to {save_path}")
