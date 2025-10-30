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

def compute_metrics(p):
    """Compute accuracy metric"""
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

def eval_quantized_model(model, tokenizer, dataset):
    """Manually evaluate quantized model"""
    device = next(model.parameters()).device
    
    # Prepare dataloader
    eval_dataloader = DataLoader(
        dataset["validation"], 
        batch_size=64,
        shuffle=False
    )
    
    # Evaluate model
    model.eval()
    total_correct = 0
    total_samples = 0
    
    start_infer = time.time()
    for batch in eval_dataloader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        labels = batch["label"].to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    infer_time = time.time() - start_infer
    accuracy = total_correct / total_samples
    
    return accuracy, infer_time

def eval_model(model_path, model_name, quantized=False):
    """Evaluate model and return accuracy and inference time"""
    print(f"\nEvaluating model: {model_name}")
    
    # Load model
    start_load = time.time()
    if quantized:
        # Load model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
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
    
    # Prepare data
    dataset = load_dataset("glue", "sst2")
    def tokenize(example):
        return tokenizer(example['sentence'], truncation=True, padding='max_length')
    
    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Quantized models need manual evaluation
    if quantized:
        accuracy, infer_time = eval_quantized_model(model, tokenizer, tokenized)
    else:
        # Evaluation configuration
        args = TrainingArguments(
            output_dir="./eval_tmp",
            per_device_eval_batch_size=64,
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=args,
            compute_metrics=compute_metrics
        )
        
        # Measure inference time
        start_infer = time.time()
        metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
        infer_time = time.time() - start_infer
        accuracy = metrics["eval_accuracy"]
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    print(f"Load time: {load_time:.2f}s | Inference time: {infer_time:.2f}s | Accuracy: {accuracy:.4f}")
    return {
        "accuracy": accuracy,
        "load_time": load_time,
        "infer_time": infer_time
    }

# Evaluate all models
models = [
    {"name": "BERT Teacher", "path": "teacher_model", "quantized": False},
    {"name": "Distilled Student", "path": "student_model", "quantized": False},
    {"name": "Soft-label Distilled", "path": "student_softlabel_model", "quantized": False},
    {"name": "Quantized Student", "path": "student_quantized_model", "quantized": True}
]

results = {}
for model in models:
    metrics = eval_model(model["path"], model["name"], model["quantized"])
    results[model["name"]] = metrics

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

# Create plots
plt.figure(figsize=(14, 8))

# Accuracy comparison
plt.subplot(2, 2, 1)
model_names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in model_names]
bars = plt.bar(model_names, accuracies, color=colors)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', 
             ha='center', va='bottom', fontsize=10)

plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy Comparison", fontsize=14)
plt.ylim(0.4, 1.0)
plt.xticks(rotation=15, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Inference time comparison
plt.subplot(2, 2, 2)
infer_times = [results[name]["infer_time"] for name in model_names]
bars = plt.bar(model_names, infer_times, color=colors)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s', 
             ha='center', va='bottom', fontsize=10)

plt.ylabel("Inference Time (s)", fontsize=12)
plt.title("Inference Time Comparison", fontsize=14)
plt.xticks(rotation=15, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Load time comparison
plt.subplot(2, 2, 3)
load_times = [results[name]["load_time"] for name in model_names]
bars = plt.bar(model_names, load_times, color=colors)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s', 
             ha='center', va='bottom', fontsize=10)

plt.ylabel("Load Time (s)", fontsize=12)
plt.title("Load Time Comparison", fontsize=14)
plt.xticks(rotation=15, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Accuracy vs Inference time scatter plot
plt.subplot(2, 2, 4)
for i, name in enumerate(model_names):
    plt.scatter(results[name]["infer_time"], results[name]["accuracy"], 
               s=200, color=colors[i], label=name, alpha=0.8)
    
    # Add model name labels
    plt.text(results[name]["infer_time"], results[name]["accuracy"], 
             name, fontsize=10, ha='right', va='bottom')

plt.xlabel("Inference Time (s)", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy vs Inference Time", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Add main title
plt.suptitle("Model Compression Performance Comparison", fontsize=18, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save plots
plt.savefig("results/model_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("results/model_comparison.pdf", bbox_inches='tight')
plt.show()

# Output detailed results
print("\nFinal Results:")
for name, metrics in results.items():
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Inference Time={metrics['infer_time']:.2f}s, Load Time={metrics['load_time']:.2f}s")

# Save results to CSV
with open('results/model_comparison.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Accuracy', 'Inference Time (s)', 'Load Time (s)'])
    for name, metrics in results.items():
        writer.writerow([name, metrics['accuracy'], metrics['infer_time'], metrics['load_time']])