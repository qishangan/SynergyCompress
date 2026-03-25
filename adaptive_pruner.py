"""
Adaptive pruning helpers for global and layer-adaptive structured sparsity.
"""

from typing import Dict, List, Optional
import math

import numpy as np
import torch
import torch.nn as nn

from dgsp_utils import is_prunable_linear


class AdaptivePruner:
    """
    Maintains pruning masks and supports:
    1. Global threshold pruning with group scores.
    2. Layer-adaptive pruning with per-layer sparsity targets.
    """

    def __init__(
        self,
        model: nn.Module,
        sensitivity_scores: Dict[str, float],
        global_sparsity: float = 0.20,
        alpha: float = 1.0,
    ) -> None:
        assert model is not None, "model must not be None"
        assert sensitivity_scores is not None, "sensitivity_scores must not be None"
        assert 0.0 <= global_sparsity < 1.0, "global_sparsity must be in [0, 1)"
        assert alpha >= 0.0, "alpha must be >= 0"

        self.model = model
        self.sensitivity_scores = {k: float(v) for k, v in sensitivity_scores.items()}
        self.global_sparsity = float(global_sparsity)
        self.alpha = float(alpha)
        self.layer_sparsities: Dict[str, float] = {}
        self.pruned_masks: Dict[str, torch.Tensor] = {}
        self.last_allocation: Dict[str, Dict[str, float]] = {}

    def _get_prunable_layer_names(self) -> List[str]:
        names: List[str] = []
        for name, module in self.model.named_modules():
            if is_prunable_linear(name, module):
                names.append(name)
        return names

    def _get_layer_param_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for name, module in self.model.named_modules():
            if is_prunable_linear(name, module):
                counts[name] = int(module.weight.numel())
        return counts

    def set_sensitivity_scores(self, sensitivity_scores: Dict[str, float]) -> None:
        self.sensitivity_scores = {k: float(v) for k, v in sensitivity_scores.items()}

    def compute_adaptive_sparsities(
        self,
        sensitivity_scores: Optional[Dict[str, float]] = None,
        adaptive_enabled: bool = True,
        rho_min_factor: float = 0.5,
        rho_max_factor: float = 1.5,
        eps: float = 1e-8,
    ) -> Dict[str, float]:
        print("\n" + "=" * 60)
        print("Computing Adaptive Sparsity Allocation")
        print("=" * 60)

        layer_param_counts = self._get_layer_param_counts()
        assert layer_param_counts, "No prunable layers found in model"

        print(f"  -> Found {len(layer_param_counts)} layers to prune")
        print(f"  -> Total parameters: {sum(layer_param_counts.values()):,}")

        if self.global_sparsity <= 0.0:
            self.layer_sparsities = {name: 0.0 for name in layer_param_counts}
            return self.layer_sparsities

        if not adaptive_enabled:
            self.layer_sparsities = {
                name: float(self.global_sparsity)
                for name in layer_param_counts
            }
            return self.layer_sparsities

        scores_source = sensitivity_scores if sensitivity_scores is not None else self.sensitivity_scores
        scores: Dict[str, float] = {}
        missing = []
        for name in layer_param_counts:
            if name in scores_source:
                value = float(scores_source[name])
                if not math.isfinite(value):
                    value = 0.0
                scores[name] = max(0.0, value)
            else:
                missing.append(name)

        fallback = float(np.mean(list(scores.values()))) if scores else 0.5
        for name in missing:
            scores[name] = fallback
        if missing:
            print(f"  -> Warning: missing sensitivity for {len(missing)} layers; using fallback {fallback:.4f}")

        rho_min = max(0.0, self.global_sparsity * rho_min_factor)
        rho_max = min(0.95, self.global_sparsity * rho_max_factor)
        if rho_max < rho_min:
            rho_max = rho_min

        if self.alpha <= 0.0:
            inverse_scores = {name: 1.0 for name in scores}
        else:
            inverse_scores = {
                name: (1.0 / (float(score) + eps)) ** self.alpha
                for name, score in scores.items()
            }
        total_params = float(sum(layer_param_counts.values()))

        def weighted_avg(scale: float) -> float:
            total = 0.0
            for name, inv_score in inverse_scores.items():
                rho = float(np.clip(scale * inv_score, rho_min, rho_max))
                total += rho * layer_param_counts[name]
            return total / total_params

        low = 0.0
        high = 1.0
        while weighted_avg(high) < self.global_sparsity and high < 1e6:
            high *= 2.0

        for _ in range(60):
            mid = (low + high) / 2.0
            avg = weighted_avg(mid)
            if avg < self.global_sparsity:
                low = mid
            else:
                high = mid

        scale = high
        self.layer_sparsities = {}
        for name, inv_score in inverse_scores.items():
            rho = float(np.clip(scale * inv_score, rho_min, rho_max))
            self.layer_sparsities[name] = rho

        final_weighted_avg = sum(
            self.layer_sparsities[name] * layer_param_counts[name]
            for name in self.layer_sparsities
        ) / total_params

        sparsity_values = list(self.layer_sparsities.values())
        print("\n  -> Sparsity Allocation Summary:")
        print(f"     Target global: {self.global_sparsity:.4f}")
        print(f"     Actual global: {final_weighted_avg:.4f}")
        print(f"     Layer range: {min(sparsity_values):.4f} - {max(sparsity_values):.4f}")
        print(f"     Layer std dev: {float(np.std(sparsity_values)):.4f}")
        print(f"     rho_min / rho_max: {rho_min:.4f} / {rho_max:.4f}")

        return self.layer_sparsities

    def _default_group_scores(self) -> Dict[str, torch.Tensor]:
        scores: Dict[str, torch.Tensor] = {}
        for name, module in self.model.named_modules():
            if is_prunable_linear(name, module):
                scores[name] = torch.norm(module.weight.detach().float(), p=2, dim=1).cpu()
        return scores

    def _apply_mask(self, module: nn.Linear, mask: torch.Tensor) -> None:
        with torch.no_grad():
            mask = mask.to(module.weight.device, dtype=module.weight.dtype)
            module.weight.data *= mask
            if module.bias is not None:
                module.bias.data *= mask.squeeze()

    def prune_globally(
        self,
        group_scores: Optional[Dict[str, torch.Tensor]] = None,
        target_sparsity: Optional[float] = None,
    ) -> Dict[str, float]:
        scores = group_scores if group_scores is not None else self._default_group_scores()
        target = self.global_sparsity if target_sparsity is None else float(target_sparsity)
        if target <= 0.0:
            return {}

        scored_rows = []
        total_params = 0
        for name, module in self.model.named_modules():
            if not is_prunable_linear(name, module):
                continue
            score = scores.get(name)
            if score is None or score.numel() == 0:
                continue
            clean_score = torch.nan_to_num(score.detach().float().flatten(), nan=0.0, posinf=0.0, neginf=0.0)
            row_param_count = int(module.weight.shape[1])
            total_params += int(module.weight.numel())
            for idx, value in enumerate(clean_score.tolist()):
                scored_rows.append((float(value), name, idx, row_param_count))

        if not scored_rows or total_params <= 0:
            return {}

        scored_rows.sort(key=lambda item: item[0])
        target_zero_params = total_params * target
        rows_to_prune = set()
        pruned_params = 0
        for _, name, idx, row_param_count in scored_rows:
            if pruned_params >= target_zero_params:
                break
            rows_to_prune.add((name, idx))
            pruned_params += row_param_count

        actual_sparsities: Dict[str, float] = {}
        self.layer_sparsities = {}
        self.last_allocation = {}

        for name, module in self.model.named_modules():
            if not is_prunable_linear(name, module):
                continue

            score = scores.get(name)
            if score is None or score.numel() == 0:
                continue
            clean_score = torch.nan_to_num(score.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).to(module.weight.device)
            row_mask = torch.ones(clean_score.size(0), device=module.weight.device, dtype=torch.float32)
            for row_idx in range(clean_score.size(0)):
                if (name, row_idx) in rows_to_prune:
                    row_mask[row_idx] = 0.0
            mask = row_mask.unsqueeze(1)
            self.pruned_masks[name] = mask.detach().cpu()
            self._apply_mask(module, mask)

            actual_sparsity = 1.0 - float(mask.mean().item())
            actual_sparsities[name] = actual_sparsity
            self.layer_sparsities[name] = target
            self.last_allocation[name] = {
                "target_sparsity": float(target),
                "actual_sparsity": float(actual_sparsity),
            }

        return actual_sparsities

    def prune_with_layer_targets(
        self,
        group_scores: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        if not self.layer_sparsities:
            self.compute_adaptive_sparsities()

        scores = group_scores if group_scores is not None else self._default_group_scores()
        actual_sparsities: Dict[str, float] = {}
        self.last_allocation = {}

        for name, module in self.model.named_modules():
            if not is_prunable_linear(name, module):
                continue
            if name not in self.layer_sparsities:
                continue

            score = scores.get(name)
            if score is None or score.numel() == 0:
                continue

            target_sparsity = float(self.layer_sparsities[name])
            clean_score = torch.nan_to_num(score.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).to(module.weight.device)
            if target_sparsity > 0.0:
                threshold = torch.quantile(clean_score, target_sparsity)
            else:
                threshold = torch.tensor(-float("inf"), device=clean_score.device)

            mask = (clean_score > threshold).float().unsqueeze(1)
            self.pruned_masks[name] = mask.detach().cpu()
            self._apply_mask(module, mask)

            actual_sparsity = 1.0 - float(mask.mean().item())
            actual_sparsities[name] = actual_sparsity
            self.last_allocation[name] = {
                "target_sparsity": target_sparsity,
                "actual_sparsity": actual_sparsity,
            }

        return actual_sparsities

    def prune(
        self,
        group_scores: Optional[Dict[str, torch.Tensor]] = None,
        adaptive_enabled: bool = True,
    ) -> Dict[str, float]:
        if adaptive_enabled:
            return self.prune_with_layer_targets(group_scores=group_scores)
        return self.prune_globally(group_scores=group_scores, target_sparsity=self.global_sparsity)

    def apply_masks(self) -> None:
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name not in self.pruned_masks:
                    continue
                mask = self.pruned_masks[name].to(module.weight.device, dtype=module.weight.dtype)
                module.weight.data *= mask
                if module.bias is not None:
                    module.bias.data *= mask.squeeze()

    def get_global_sparsity(self) -> float:
        total_params = 0
        zero_params = 0
        for name, module in self.model.named_modules():
            if is_prunable_linear(name, module):
                weight = module.weight
                total_params += weight.numel()
                zero_params += (weight.abs() < 1e-8).sum().item()
        return zero_params / total_params if total_params > 0 else 0.0
