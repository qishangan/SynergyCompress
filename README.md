# Advanced Model Compression via Distillation, Pruning, and Quantization

This repository contains the code and results for a research project on advanced model compression techniques. The core innovation is a structured, multi-stage pipeline that synergistically combines knowledge distillation, gradual structural pruning, and 4-bit quantization to create highly efficient yet accurate language models.

Our final compressed model achieves a **20% reduction in computational cost (GFLOPs)** with only a **1.6% drop in accuracy** compared to a standard quantized model, demonstrating the superiority of our proposed `Distill-Prune-Quantize` pipeline for efficiency-critical applications.

## Core Methodology

Our approach is a four-stage process designed to maximize compression benefits while mitigating accuracy loss:

1.  **Knowledge Distillation:** We start by distilling knowledge from a large `BERT` teacher model into a smaller `DistilBERT` student. This provides a robust, high-accuracy baseline for further compression.
2.  **Gradual Structural Pruning:** To avoid the "accuracy collapse" common in pruning, we introduce a gradual pruning algorithm. The model's sparsity is smoothly increased to a target of 20% over several epochs.
3.  **Recovery Fine-tuning:** After pruning, the model undergoes a crucial fine-tuning phase at a lower learning rate. This allows the model to adapt to its new, sparser architecture and recover lost performance.
4.  **4-bit Quantization:** Finally, the pruned and fine-tuned model is quantized to 4-bit precision, further reducing its memory footprint and making it suitable for deployment.

## Repository Structure

```
.
├── models/                     # Contains final, deployable models
│   ├── distilled_quantized_student/
│   ├── pruned_quantized_final/
│   └── student_model/
├── teacher_model/              # Pre-trained teacher model
├── 7_evaluate_and_generate_report.py # Script to run final evaluation and generate report
├── 8_pruning_with_finetuning.py      # Script for the core pruning and fine-tuning experiment
├── 9_quantize_pruned_model.py        # Script to quantize the final pruned model
├── download_model.py           # Script to download initial models
├── final_report.json           # JSON file with final evaluation metrics
├── final_report.png            # Result visualization: Accuracy vs. GFLOPs
├── paper_innovations_summary.txt # Detailed summary of research innovations
└── requirements.txt            # Python dependencies
```

## How to Reproduce the Experiment

### 1. Setup

First, clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. Download Initial Models

Run the download script to fetch the pre-trained `BERT-base` (teacher) and `DistilBERT` (initial student) models. This will also download the SST-2 dataset.

```bash
python download_model.py
```

### 3. Run the Full Pipeline

The following scripts must be run in order to reproduce the final results.

**Step 1: Train the Teacher Model (Optional, pre-trained provided)**
*(This step is only needed if you want to retrain the teacher model.)*

**Step 2: Distill the Student Model (Optional, pre-trained provided)**
*(This step is only needed if you want to re-distill the student model.)*

**Step 3: Run the Pruning and Fine-tuning Experiment**
This is the core script of our research. It will perform gradual pruning and recovery fine-tuning.

```bash
python 8_pruning_with_finetuning.py
```

**Step 4: Quantize the Pruned Model**
This script takes the best model from the previous step and applies 4-bit quantization.

```bash
python 9_quantize_pruned_model.py
```

### 4. Generate the Final Report

After running the pipeline, execute the evaluation script. This will measure the accuracy, size, and GFLOPs of all relevant models and generate the final comparison report (`final_report.png` and `final_report.json`).

```bash
python 7_evaluate_and_generate_report.py
```

This will produce the final chart, scientifically demonstrating the trade-offs between accuracy and computational cost.
