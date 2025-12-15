import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from thop import profile
import time

# --- Configuration ---
EVAL_BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL_REPORT_IMAGE = "final_report.png"
FINAL_REPORT_JSON = "final_report.json"

# --- Helper Functions ---

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

def get_model_size(model_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(model_path):
        for f in filenames:
            if f.endswith((".bin", ".safetensors")):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def calculate_gflops(model_path, tokenizer):
    print(f"  -> Calculating GFLOPs from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    dummy_input = tokenizer("This is a dummy sentence.", return_tensors="pt").to(DEVICE)
    input_ids = dummy_input.input_ids
    macs, _ = profile(model, inputs=(input_ids,), verbose=False)
    return (macs * 2) / 1e9

def evaluate_accuracy(model, dataloader):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples if total_samples > 0 else 0

def get_first_param_device(model):
    for p in model.parameters():
        return p.device
    return torch.device("cpu")

def measure_latency(model, tokenizer, seq_len=MAX_LENGTH, batch_size=32, warmup=5, iters=20):
    """粗略测 GPU/CPU latency（平均单次 forward ms）。"""
    device = get_first_param_device(model)
    model.eval()

    # 构造固定长度的 dummy 输入
    dummy = tokenizer(["This is a dummy sentence."] * batch_size,
                      padding="max_length",
                      truncation=True,
                      max_length=seq_len,
                      return_tensors="pt")
    dummy = {k: v.to(device) for k, v in dummy.items()}

    # 选择计时方法
    use_cuda = device.type == "cuda"
    if use_cuda:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
    else:
        starter = ender = None

    # warmup
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
                timings.append(starter.elapsed_time(ender))  # ms
            else:
                t0 = time.perf_counter()
                _ = model(**dummy)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)  # ms
    return sum(timings) / len(timings) if timings else None

def get_real_sparsity(model):
    """统计 Linear 层（非 classifier）真实稀疏度。"""
    total = 0
    nonzero = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "classifier" not in name:
                w = module.weight
                total += w.numel()
                nonzero += (w != 0).sum().item()
    if total == 0:
        return 0.0
    density = nonzero / total
    return 1.0 - density

def generate_report_plot(results):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    labels = list(results.keys())
    accuracies = [res['accuracy'] for res in results.values()]
    gflops = [res['gflops'] for res in results.values()]
    
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = palette[:len(labels)]
    ax.scatter(gflops, accuracies, s=200, c=colors, alpha=0.7, edgecolors='w')

    for i, label in enumerate(labels):
        ax.text(gflops[i], accuracies[i] + 0.001, label, fontsize=10, ha='center', va='bottom')

    ax.set_xlabel("Computational Cost (GFLOPs)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy vs. Computational Cost", fontsize=16, weight='bold')
    ax.grid(True)
    ax.set_ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)
    
    plt.figtext(0.5, 0.01, "Ideal models are in the top-left (High Accuracy, Low Cost).", ha="center", fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(FINAL_REPORT_IMAGE)
    print(f"\n--- Report plot saved to {FINAL_REPORT_IMAGE} ---")

# --- Main Execution ---

def main():
    print("--- Starting Final Model Evaluation (Corrected) ---")
    
    # Build model list dynamically based on最新完整实验的产物
    models_to_eval = []
    # Baseline distilled
    models_to_eval.append({
        "name": "Distilled Student (Baseline)",
        "eval_path": "./student_model",
        "gflops_path": "./student_model",
        "sparsity": 0.0,
        "quantized": False,
    })
    # Quantized distilled
    models_to_eval.append({
        "name": "Quantized Distilled Student",
        "eval_path": "./models/distilled_quantized_student",
        "gflops_path": "./student_model",
        "sparsity": 0.0,
        "quantized": True,
    })
    # Latest pruned (full) model
    latest_pruned = find_latest_model("pruning_with_finetuning_")
    if latest_pruned:
        models_to_eval.append({
            "name": "Pruned Student (Sparse FP32)",
            "eval_path": latest_pruned,
            "gflops_path": latest_pruned,
            "sparsity": 0.20,
            "quantized": False,
        })
    # Latest PTQ 4bit
    latest_ptq = find_latest_model("pruned_quantized")
    if latest_ptq and latest_pruned:
        models_to_eval.append({
            "name": "Pruned + Quantized Student (PTQ)",
            "eval_path": latest_ptq,
            "gflops_path": latest_pruned,
            "sparsity": 0.20,
            "quantized": True,
        })
    # Latest QKD 4bit
    latest_qkd = find_latest_model("pruned_qkd4bit")
    if latest_qkd and latest_pruned:
        models_to_eval.append({
            "name": "Pruned + QKD-4bit (QAT)",
            "eval_path": latest_qkd,
            "gflops_path": latest_pruned,
            "sparsity": 0.20,
            "quantized": True,
        })
    if not models_to_eval:
        print("No models found for evaluation. Please check the models directory.")
        return
    
    try:
        dataset = load_dataset("./sst2_data/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c")
    except Exception:
        dataset = load_dataset("glue", "sst2")
    
    base_tokenizer = AutoTokenizer.from_pretrained("./distilbert-local")
    
    def tokenize_function(examples):
        return base_tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_loader = DataLoader(tokenized_datasets['validation'], batch_size=EVAL_BATCH_SIZE)

    results = {}
    baseline_gflops = None

    for entry in models_to_eval:
        name = entry["name"]
        eval_path = entry["eval_path"]
        gflops_path = entry["gflops_path"]
        sparsity = entry["sparsity"]
        is_quantized = entry["quantized"]

        print(f"\n--- Evaluating: {name} ---")

        if not os.path.exists(eval_path):
            print(f"  -> Skipped: path not found {eval_path}")
            continue
        
        # 1. Evaluate Accuracy
        if is_quantized:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            model_for_eval = AutoModelForSequenceClassification.from_pretrained(
                eval_path, quantization_config=quant_config, device_map="auto"
            )
        else:
            model_for_eval = AutoModelForSequenceClassification.from_pretrained(eval_path).to(DEVICE)
            
        accuracy = evaluate_accuracy(model_for_eval, val_loader)
        size_mb = get_model_size(eval_path)
        real_sparsity = get_real_sparsity(model_for_eval)
        latency_ms = measure_latency(model_for_eval, base_tokenizer, seq_len=MAX_LENGTH, batch_size=EVAL_BATCH_SIZE)
        
        # 2. Calculate GFLOPs
        if baseline_gflops is None: # First model must be the baseline
            baseline_gflops = calculate_gflops(gflops_path, base_tokenizer)
        
        gflops = baseline_gflops * (1 - sparsity)
        
        results[name] = {
            "accuracy": accuracy,
            "size_mb": size_mb,
            "gflops": gflops,
            "real_sparsity": real_sparsity,
            "latency_ms": latency_ms
        }
        lat_str = f"{latency_ms:.2f}" if latency_ms is not None else "N/A"
        print(f"  -> Accuracy: {accuracy:.4f} | Size: {size_mb:.2f} MB | GFLOPs: {gflops:.2f} | Sparsity(real): {real_sparsity:.3f} | Latency(ms): {lat_str}")

    print("\n--- Final Comparison Report (Corrected) ---")
    header = f"{'Model':<40} | {'Accuracy':<10} | {'Size (MB)':<12} | {'GFLOPs (Theoretical)':<25} | {'Sparsity(real)':<15} | {'Latency(ms)':<12}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        note = "(Calculated)" if res['gflops'] != baseline_gflops else ""
        lat = f"{res['latency_ms']:.2f}" if res.get("latency_ms") is not None else "N/A"
        print(f"{name:<40} | {res['accuracy']:<10.4f} | {res['size_mb']:<12.2f} | {res['gflops']:.2f} {note:<25} | {res.get('real_sparsity',0):<15.3f} | {lat:<12}")

    with open(FINAL_REPORT_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n--- Detailed results saved to {FINAL_REPORT_JSON} ---")

    generate_report_plot(results)

if __name__ == "__main__":
    main()
