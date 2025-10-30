import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from thop import profile

# --- Configuration ---
MODELS_TO_EVALUATE = {
    "Distilled Student (Baseline)": {
        "eval_path": "./student_model",
        "gflops_path": "./student_model",
        "sparsity": 0.0
    },
    "Quantized Distilled Student": {
        "eval_path": "./models/distilled_quantized_student",
        "gflops_path": "./student_model", # Structurally identical to baseline
        "sparsity": 0.0
    },
    "Pruned + Quantized Student (Final)": {
        "eval_path": "./models/pruned_quantized_final",
        "gflops_path": "./models/pruning_with_finetuning_20251029_092719", # This is just a placeholder
        "sparsity": 0.20 # We explicitly set the known sparsity
    },
}
EVAL_BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL_REPORT_IMAGE = "final_report.png"
FINAL_REPORT_JSON = "final_report.json"

# --- Helper Functions ---

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

def generate_report_plot(results):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    labels = list(results.keys())
    accuracies = [res['accuracy'] for res in results.values()]
    gflops = [res['gflops'] for res in results.values()]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
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

    for name, config in MODELS_TO_EVALUATE.items():
        print(f"\n--- Evaluating: {name} ---")
        
        # 1. Evaluate Accuracy
        eval_path = config['eval_path']
        is_quantized = "quantized" in name.lower()
        
        if is_quantized:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            model_for_eval = AutoModelForSequenceClassification.from_pretrained(
                eval_path, quantization_config=quant_config, device_map="auto"
            )
        else:
            model_for_eval = AutoModelForSequenceClassification.from_pretrained(eval_path).to(DEVICE)
            
        accuracy = evaluate_accuracy(model_for_eval, val_loader)
        size_mb = get_model_size(eval_path)
        
        # 2. Calculate GFLOPs
        if baseline_gflops is None: # First model must be the baseline
            baseline_gflops = calculate_gflops(config['gflops_path'], base_tokenizer)
        
        # For pruned model, calculate based on sparsity. Otherwise, it's the same as baseline.
        sparsity = config['sparsity']
        gflops = baseline_gflops * (1 - sparsity)
        
        results[name] = {
            "accuracy": accuracy,
            "size_mb": size_mb,
            "gflops": gflops
        }
        print(f"  -> Accuracy: {accuracy:.4f} | Size: {size_mb:.2f} MB | GFLOPs: {gflops:.2f}")

    print("\n--- Final Comparison Report (Corrected) ---")
    header = f"{'Model':<40} | {'Accuracy':<10} | {'Size (MB)':<12} | {'GFLOPs (Theoretical)':<25}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        note = "(Calculated)" if res['gflops'] != baseline_gflops else ""
        print(f"{name:<40} | {res['accuracy']:<10.4f} | {res['size_mb']:<12.2f} | {res['gflops']:.2f} {note:<25}")

    with open(FINAL_REPORT_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n--- Detailed results saved to {FINAL_REPORT_JSON} ---")

    generate_report_plot(results)

if __name__ == "__main__":
    main()
