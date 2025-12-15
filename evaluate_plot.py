from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
import csv

# Create results directory
os.makedirs("results", exist_ok=True)


def find_latest_model(prefix, root="./models"):
    """Return latest directory under root matching prefix, or None."""
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith(prefix):
            candidates.append(full)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def compute_metrics(p):
    """Compute accuracy metric."""
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}


def eval_quantized_model(model, dataset):
    """Manually evaluate 4-bit models (Trainer does not support NF4)."""
    device = next(model.parameters()).device
    eval_dataloader = DataLoader(dataset["validation"], batch_size=64, shuffle=False)

    model.eval()
    total_correct = 0
    total_samples = 0

    start_infer = time.time()
    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            labels = batch["label"].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    infer_time = time.time() - start_infer
    accuracy = total_correct / total_samples if total_samples else 0.0
    return accuracy, infer_time


def load_sst2(tokenizer):
    """Load SST-2 with local cache fallback."""
    try:
        dataset = load_dataset("./sst2_data/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c")
    except Exception:
        dataset = load_dataset("glue", "sst2")
    def tokenize(example):
        return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)
    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


def eval_model(model_path, model_name, quantized=False):
    """Evaluate one model and return metrics dict."""
    if not model_path or not os.path.exists(model_path):
        print(f"  -> Skipped (not found): {model_name} @ {model_path}")
        return None

    print(f"\nEvaluating model: {model_name}")
    start_load = time.time()
    if quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    load_time = time.time() - start_load

    tokenized = load_sst2(tokenizer)

    if quantized:
        accuracy, infer_time = eval_quantized_model(model, tokenized)
    else:
        args = TrainingArguments(
            output_dir="./eval_tmp",
            per_device_eval_batch_size=64,
            report_to="none"
        )
        trainer = Trainer(model=model, args=args, compute_metrics=compute_metrics)
        start_infer = time.time()
        metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
        infer_time = time.time() - start_infer
        accuracy = metrics["eval_accuracy"]

    del model
    torch.cuda.empty_cache()

    print(f"Load time: {load_time:.2f}s | Inference time: {infer_time:.2f}s | Accuracy: {accuracy:.4f}")
    return {
        "accuracy": accuracy,
        "load_time": load_time,
        "infer_time": infer_time
    }


def main():
    latest_pruned = find_latest_model("pruning_with_finetuning_")
    latest_ptq = find_latest_model("pruned_quantized")
    latest_qkd = find_latest_model("pruned_qkd4bit")

    models = [
        {"name": "BERT Teacher", "path": "teacher_model", "quantized": False},
        {"name": "Distilled Student (Baseline)", "path": "student_model", "quantized": False},
        {"name": "Quantized Distilled Student", "path": "models/distilled_quantized_student", "quantized": True},
    ]
    if latest_pruned:
        models.append({"name": "Pruned Student (Sparse FP32)", "path": latest_pruned, "quantized": False})
    if latest_ptq and latest_pruned:
        models.append({"name": "Pruned + Quantized Student (PTQ)", "path": latest_ptq, "quantized": True})
    if latest_qkd and latest_pruned:
        models.append({"name": "Pruned + QKD-4bit (QAT)", "path": latest_qkd, "quantized": True})

    results = {}
    for entry in models:
        metrics = eval_model(entry["path"], entry["name"], entry["quantized"])
        if metrics is not None:
            results[entry["name"]] = metrics

    if not results:
        print("No models were evaluated; nothing to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = palette[:len(results)]

    plt.figure(figsize=(14, 8))

    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]
    infer_times = [results[name]["infer_time"] for name in model_names]
    load_times = [results[name]["load_time"] for name in model_names]

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(model_names, accuracies, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy Comparison", fontsize=14)
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=15, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Inference time comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(model_names, infer_times, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    plt.ylabel("Inference Time (s)", fontsize=12)
    plt.title("Inference Time Comparison", fontsize=14)
    plt.xticks(rotation=15, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Load time comparison
    plt.subplot(2, 2, 3)
    bars = plt.bar(model_names, load_times, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    plt.ylabel("Load Time (s)", fontsize=12)
    plt.title("Load Time Comparison", fontsize=14)
    plt.xticks(rotation=15, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Accuracy vs Inference time scatter plot
    plt.subplot(2, 2, 4)
    for i, name in enumerate(model_names):
        plt.scatter(infer_times[i], accuracies[i], s=200, color=colors[i], label=name, alpha=0.8)
        plt.text(infer_times[i], accuracies[i], name, fontsize=10, ha='right', va='bottom')
    plt.xlabel("Inference Time (s)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy vs Inference Time", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle("Model Compression Performance Comparison", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig("results/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("results/model_comparison.pdf", bbox_inches='tight')
    plt.close()

    print("\nFinal Results:")
    for name, metrics in results.items():
        print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Inference Time={metrics['infer_time']:.2f}s, Load Time={metrics['load_time']:.2f}s")

    with open('results/model_comparison.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Accuracy', 'Inference Time (s)', 'Load Time (s)'])
        for name, metrics in results.items():
            writer.writerow([name, metrics['accuracy'], metrics['infer_time'], metrics['load_time']])


if __name__ == "__main__":
    main()
