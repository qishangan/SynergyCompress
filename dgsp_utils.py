import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


PRUNABLE_LAYER_PATTERN = re.compile(r"transformer\.layer\.(\d+)")


def is_prunable_linear(name: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and "classifier" not in name


def normalize_score_dict(
    score_dict: Dict[str, torch.Tensor],
    fill_value: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    if not score_dict:
        return {}

    tensors: List[torch.Tensor] = []
    for tensor in score_dict.values():
        if tensor.numel() == 0:
            continue
        clean = torch.nan_to_num(tensor.detach().float().flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        tensors.append(clean)

    if not tensors:
        return {
            name: torch.full_like(tensor.detach().float(), fill_value)
            for name, tensor in score_dict.items()
        }

    merged = torch.cat(tensors)
    min_value = float(merged.min().item())
    max_value = float(merged.max().item())

    if max_value - min_value < eps:
        return {
            name: torch.full_like(tensor.detach().float(), fill_value)
            for name, tensor in score_dict.items()
        }

    normalized: Dict[str, torch.Tensor] = {}
    scale = max_value - min_value
    for name, tensor in score_dict.items():
        clean = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        normalized[name] = ((clean - min_value) / scale).clamp_(0.0, 1.0)
    return normalized


def extract_transformer_layer_index(layer_name: str) -> Optional[int]:
    match = PRUNABLE_LAYER_PATTERN.search(layer_name)
    if not match:
        return None
    return int(match.group(1))


def aggregate_hidden_layer_sensitivity(layer_sensitivity: Dict[str, float]) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {}
    for layer_name, score in layer_sensitivity.items():
        layer_idx = extract_transformer_layer_index(layer_name)
        if layer_idx is None:
            continue
        buckets.setdefault(layer_idx, []).append(float(score))

    aggregated: Dict[int, float] = {}
    for layer_idx, scores in buckets.items():
        aggregated[layer_idx] = float(np.mean(scores))
    return aggregated


def load_layer_sensitivity_file(path: str) -> Dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "layer_sensitivity" in data and isinstance(data["layer_sensitivity"], dict):
            return {k: float(v) for k, v in data["layer_sensitivity"].items()}
        if "normalized" in data and isinstance(data["normalized"], dict):
            return {k: float(v) for k, v in data["normalized"].items()}
        if "layers" in data and isinstance(data["layers"], list):
            parsed: Dict[str, float] = {}
            for item in data["layers"]:
                if not isinstance(item, dict):
                    continue
                layer_name = item.get("layer_name")
                score = item.get("normalized_sensitivity", item.get("mean_sensitivity"))
                if layer_name is None or score is None:
                    continue
                parsed[str(layer_name)] = float(score)
            if parsed:
                return parsed
    raise ValueError(f"Unsupported layer sensitivity format: {path}")


class DistillationGuidedScorer:
    def __init__(
        self,
        model: nn.Module,
        lambda_score: float = 0.5,
        ema_beta: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.lambda_score = float(lambda_score)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self.ema_kd_scores: Dict[str, torch.Tensor] = {}
        self.last_snapshot: Dict[str, object] = {}

    def iter_prunable_modules(self):
        for name, module in self.model.named_modules():
            if is_prunable_linear(name, module):
                yield name, module

    def update_kd_sensitivity_from_grads(self) -> None:
        for name, module in self.iter_prunable_modules():
            grad = module.weight.grad
            weight = module.weight.detach().float()
            if grad is None:
                if name not in self.ema_kd_scores:
                    self.ema_kd_scores[name] = torch.zeros(weight.size(0), dtype=torch.float32)
                continue

            raw_score = torch.mean(torch.abs(grad.detach().float() * weight), dim=1)
            raw_score = torch.nan_to_num(raw_score, nan=0.0, posinf=0.0, neginf=0.0).cpu()

            previous = self.ema_kd_scores.get(name)
            if previous is None:
                ema_score = raw_score
            else:
                ema_score = previous * self.ema_beta + raw_score * (1.0 - self.ema_beta)
            self.ema_kd_scores[name] = ema_score

    def _collect_l2_scores(self) -> Dict[str, torch.Tensor]:
        scores: Dict[str, torch.Tensor] = {}
        for name, module in self.iter_prunable_modules():
            l2_score = torch.norm(module.weight.detach().float(), p=2, dim=1).cpu()
            scores[name] = torch.nan_to_num(l2_score, nan=0.0, posinf=0.0, neginf=0.0)
        return scores

    def _collect_kd_scores(self, l2_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        kd_scores: Dict[str, torch.Tensor] = {}
        for name, l2_score in l2_scores.items():
            kd_score = self.ema_kd_scores.get(name)
            if kd_score is None:
                kd_scores[name] = torch.zeros_like(l2_score)
            else:
                kd_scores[name] = torch.nan_to_num(kd_score.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        return kd_scores

    def collect_scores(self) -> Dict[str, object]:
        l2_scores = self._collect_l2_scores()
        kd_scores = self._collect_kd_scores(l2_scores)

        normalized_l2 = normalize_score_dict(l2_scores, eps=self.eps)
        normalized_kd = normalize_score_dict(kd_scores, eps=self.eps)

        fused_scores: Dict[str, torch.Tensor] = {}
        layer_details: Dict[str, Dict[str, float]] = {}
        raw_layer_sensitivity: Dict[str, float] = {}

        for name in l2_scores:
            fused = (1.0 - self.lambda_score) * normalized_l2[name] + self.lambda_score * normalized_kd[name]
            fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0).cpu()
            fused_scores[name] = fused

            mean_l2 = float(normalized_l2[name].mean().item()) if normalized_l2[name].numel() else 0.0
            mean_kd = float(normalized_kd[name].mean().item()) if normalized_kd[name].numel() else 0.0
            mean_fused = float(fused.mean().item()) if fused.numel() else 0.0
            raw_layer_sensitivity[name] = mean_fused
            layer_details[name] = {
                "mean_l2_score": mean_l2,
                "mean_kd_score": mean_kd,
                "mean_sensitivity": mean_fused,
                "num_groups": int(fused.numel()),
                "fused_min": float(fused.min().item()) if fused.numel() else 0.0,
                "fused_max": float(fused.max().item()) if fused.numel() else 0.0,
            }

        normalized_layer_sensitivity = self.normalize_layer_scores(raw_layer_sensitivity)
        for name, normalized_value in normalized_layer_sensitivity.items():
            layer_details.setdefault(name, {})
            layer_details[name]["normalized_sensitivity"] = float(normalized_value)

        snapshot = {
            "group_scores": fused_scores,
            "layer_sensitivity": normalized_layer_sensitivity,
            "layer_details": layer_details,
            "raw_layer_sensitivity": raw_layer_sensitivity,
        }
        self.last_snapshot = snapshot
        return snapshot

    def update_and_collect(self) -> Dict[str, object]:
        self.update_kd_sensitivity_from_grads()
        return self.collect_scores()

    @staticmethod
    def normalize_layer_scores(layer_scores: Dict[str, float], fill_value: float = 0.5, eps: float = 1e-8) -> Dict[str, float]:
        if not layer_scores:
            return {}

        values = np.array([float(v) for v in layer_scores.values()], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        min_value = float(values.min())
        max_value = float(values.max())

        if max_value - min_value < eps:
            return {name: float(fill_value) for name in layer_scores}

        scale = max_value - min_value
        return {
            name: float(max(0.0, min(1.0, (float(score) - min_value) / scale)))
            for name, score in layer_scores.items()
        }
