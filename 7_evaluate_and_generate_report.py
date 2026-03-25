import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from thop import profile
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


EVAL_BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_size(model_path: str) -> float:
    total_size = 0
    for child in Path(model_path).iterdir():
        if child.is_file() and child.suffix in {".bin", ".safetensors"}:
            total_size += child.stat().st_size
    return total_size / (1024 * 1024)


def calculate_dense_gflops(model_path: str, tokenizer) -> float:
    print(f"  -> Calculating dense reference GFLOPs from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    dummy_input = tokenizer("This is a dummy sentence.", return_tensors="pt").to(DEVICE)
    macs, _ = profile(model, inputs=(dummy_input.input_ids,), verbose=False)
    return (macs * 2) / 1e9


def get_first_param_device(model):
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def measure_latency(model, tokenizer, seq_len=MAX_LENGTH, batch_size=32, warmup=5, iters=20):
    device = get_first_param_device(model)
    model.eval()

    dummy = tokenizer(
        ["This is a dummy sentence."] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    dummy = {k: v.to(device) for k, v in dummy.items()}

    use_cuda = device.type == "cuda"
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
    else:
        starter = ender = None

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**dummy)
    if use_cuda:
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iters):
            if use_cuda:
                starter.record()
                _ = model(**dummy)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
            else:
                t0 = time.perf_counter()
                _ = model(**dummy)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)

    return sum(timings) / len(timings) if timings else None


def load_eval_dataset(tokenizer, batch_size: int, max_length: int):
    try:
        dataset = load_dataset("./sst2_data/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c")
    except Exception:
        dataset = load_dataset("glue", "sst2")

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(tokenized["validation"], batch_size=batch_size)


def evaluate_metrics(model, dataloader) -> Dict[str, float]:
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            total_loss += ce_loss_fn(logits, labels).item() * labels.size(0)
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    return {
        "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        "loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "peak_memory_mb": peak_memory_mb,
    }


def get_real_sparsity_from_path(model_path: str) -> float:
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    total = 0
    nonzero = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "classifier" not in name:
                weight = module.weight
                total += weight.numel()
                nonzero += (weight != 0).sum().item()
    if total == 0:
        return 0.0
    return 1.0 - (nonzero / total)


def load_model_for_eval(model_path: str, quantized: bool):
    if quantized:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
        )
    return AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)


def default_manifest() -> List[Dict[str, object]]:
    entries = [
        {
            "name": "Distilled Student (Baseline)",
            "eval_path": "./student_model",
            "quantized": False,
            "sparsity_source_path": "./student_model",
        },
    ]

    models_root = Path("./models")
    if models_root.exists():
        latest_pruned = None
        latest_ptq = None
        latest_qkd = None
        for child in sorted(models_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not child.is_dir():
                continue
            if latest_pruned is None and child.name.startswith("pruning_"):
                latest_pruned = child
            if latest_ptq is None and child.name.startswith("pruned_quantized_ptq_"):
                latest_ptq = child
            if latest_qkd is None and child.name.startswith("pruned_qkd4bit_"):
                latest_qkd = child
        if latest_pruned is not None:
            entries.append({
                "name": "Latest Pruned Model",
                "eval_path": str(latest_pruned),
                "quantized": False,
                "sparsity_source_path": str(latest_pruned),
            })
        if latest_ptq is not None:
            entries.append({
                "name": "Latest PTQ 4-bit Model",
                "eval_path": str(latest_ptq),
                "quantized": True,
                "sparsity_source_path": str(latest_pruned or latest_ptq),
            })
        if latest_qkd is not None:
            entries.append({
                "name": "Latest QKD 4-bit Model",
                "eval_path": str(latest_qkd),
                "quantized": True,
                "sparsity_source_path": str(latest_pruned or latest_qkd),
            })
    return entries


def load_manifest(path: Optional[str]) -> List[Dict[str, object]]:
    if path is None:
        return default_manifest()
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "models" in payload:
        payload = payload["models"]
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a list of model entries or a dict with 'models'")
    return payload


def generate_report_plot(results: Dict[str, Dict[str, float]], output_path: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    labels = list(results.keys())
    accuracies = [res["accuracy"] for res in results.values()]
    gflops = [res["theoretical_gflops"] for res in results.values()]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#e377c2", "#7f7f7f"]
    if len(labels) <= len(palette):
        colors = palette[: len(labels)]
    else:
        cmap = plt.colormaps.get_cmap("tab20")
        denom = max(len(labels) - 1, 1)
        colors = [cmap(idx / denom) for idx in range(len(labels))]
    ax.scatter(gflops, accuracies, s=180, c=colors, alpha=0.8, edgecolors="white")

    for idx, label in enumerate(labels):
        ax.text(gflops[idx], accuracies[idx] + 0.001, label, fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("Theoretical GFLOPs", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Theoretical GFLOPs", fontsize=15, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)

    plt.figtext(
        0.5,
        0.01,
        "Measured latency is reported separately; zero-masked weights do not imply structural acceleration.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path)
    print(f"\n--- Report plot saved to {output_path} ---")


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense/pruned/PTQ/QKD checkpoints")
    parser.add_argument("--manifest", type=str, default=None, help="JSON manifest of models to evaluate")
    parser.add_argument("--output_json", type=str, default="final_report.json", help="Output JSON path")
    parser.add_argument("--output_png", type=str, default="final_report.png", help="Output figure path")
    parser.add_argument("--batch_size", type=int, default=EVAL_BATCH_SIZE, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Maximum sequence length")
    parser.add_argument("--dense_reference_path", type=str, default="./student_model", help="Dense reference for theoretical GFLOPs")
    args = parser.parse_args()

    print("--- Starting Final Model Evaluation ---")
    manifest = load_manifest(args.manifest)
    if not manifest:
        raise ValueError("No models found for evaluation")

    tokenizer = AutoTokenizer.from_pretrained("./distilbert-local")
    val_loader = load_eval_dataset(tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dense_reference_gflops = calculate_dense_gflops(args.dense_reference_path, tokenizer)

    results: Dict[str, Dict[str, float]] = {}
    for entry in manifest:
        name = entry["name"]
        eval_path = entry["eval_path"]
        quantized = bool(entry.get("quantized", False))
        sparsity_source_path = entry.get("sparsity_source_path", eval_path)

        print(f"\n--- Evaluating: {name} ---")
        if not os.path.exists(eval_path):
            print(f"  -> Skipped missing path: {eval_path}")
            continue
        if not os.path.exists(sparsity_source_path):
            raise FileNotFoundError(f"Sparsity source path not found: {sparsity_source_path}")

        model = load_model_for_eval(eval_path, quantized=quantized)
        metrics = evaluate_metrics(model, val_loader)
        size_mb = get_model_size(eval_path)
        real_sparsity = get_real_sparsity_from_path(sparsity_source_path)
        theoretical_gflops = dense_reference_gflops * (1.0 - real_sparsity)
        latency_ms = measure_latency(model, tokenizer, seq_len=args.max_length, batch_size=args.batch_size)

        result = {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "real_sparsity": real_sparsity,
            "theoretical_gflops": theoretical_gflops,
            "measured_latency_ms": latency_ms,
            "model_size_mb": size_mb,
            "peak_memory_mb": metrics["peak_memory_mb"],
            "structural_speedup_realized": False,
        }
        results[name] = result

        latency_str = f"{latency_ms:.2f}" if latency_ms is not None else "N/A"
        print(
            f"  -> Acc: {result['accuracy']:.4f} | Loss: {result['loss']:.4f} | "
            f"Real Sparsity: {result['real_sparsity']:.4f} | "
            f"Theoretical GFLOPs: {result['theoretical_gflops']:.4f} | "
            f"Latency(ms): {latency_str} | Size(MB): {result['model_size_mb']:.2f}"
        )

    header = (
        f"{'Model':<36} | {'Acc':<8} | {'Loss':<8} | {'Real Sparsity':<14} | "
        f"{'Theoretical GFLOPs':<19} | {'Latency(ms)':<12} | {'Size(MB)':<10}"
    )
    print("\n--- Evaluation Summary ---")
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        latency_str = f"{res['measured_latency_ms']:.2f}" if res["measured_latency_ms"] is not None else "N/A"
        print(
            f"{name:<36} | {res['accuracy']:<8.4f} | {res['loss']:<8.4f} | "
            f"{res['real_sparsity']:<14.4f} | {res['theoretical_gflops']:<19.4f} | "
            f"{latency_str:<12} | {res['model_size_mb']:<10.2f}"
        )

    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n--- Detailed results saved to {args.output_json} ---")
    generate_report_plot(results, args.output_png)


if __name__ == "__main__":
    main()
