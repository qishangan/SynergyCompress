import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertConfig
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from datetime import datetime
import time
from tqdm import tqdm

class GradualPruningWithFinetuningConfig:
    """Configuration for gradual pruning with a post-pruning finetuning phase."""
    def __init__(self):
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.total_epochs = 15  # Increased total epochs for longer finetuning
        
        # Distillation parameters
        self.initial_temperature = 10.0
        self.final_temperature = 2.0
        self.alpha_initial = 0.9
        self.alpha_final = 0.2
        
        # Pruning parameters
        self.target_sparsity = 0.20  # Reduced target sparsity to be more conservative
        self.pruning_start_epoch = 2
        self.pruning_end_epoch = 8  # Pruning stops at the end of this epoch
        self.pruning_frequency_steps = 20
        
        # Finetuning parameters
        self.finetuning_start_epoch = self.pruning_end_epoch + 1 # Finetuning starts after pruning ends
        self.finetuning_lr = 1e-5  # Use a smaller learning rate for finetuning
        
        # Early stopping
        self.patience = 5
        self.min_delta = 0.001
        
        # Experiment tracking
        self.experiment_name = f"pruning_with_finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class StructuralPruner:
    """A simple structural pruner based on L2 norm of weights."""
    def __init__(self, model):
        self.model = model
        self.pruned_masks = {}

    def compute_importance_and_prune(self, target_sparsity):
        all_scores = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'classifier' not in name:
                scores = torch.norm(module.weight, p=2, dim=1)
                all_scores.extend(scores.detach().cpu().numpy().tolist())

        threshold = np.percentile(all_scores, target_sparsity * 100)
        
        total_neurons, pruned_neurons = 0, 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'classifier' not in name:
                scores = torch.norm(module.weight, p=2, dim=1)
                mask = (scores > threshold).float().unsqueeze(1)
                self.pruned_masks[name] = mask
                
                with torch.no_grad():
                    module.weight.data *= mask
                    if module.bias is not None:
                        module.bias.data *= mask.squeeze()
                
                total_neurons += module.weight.shape[0]
                pruned_neurons += (1 - mask.squeeze()).sum().item()
        
        return pruned_neurons / total_neurons if total_neurons > 0 else 0

    def apply_masks(self):
        for name, module in self.model.named_modules():
            if name in self.pruned_masks:
                with torch.no_grad():
                    module.weight.data *= self.pruned_masks[name]
                    if module.bias is not None:
                        module.bias.data *= self.pruned_masks[name].squeeze()

def load_data_and_models():
    """Loads dataset and models from local paths."""
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
    """Evaluates the model."""
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples

def main():
    config = GradualPruningWithFinetuningConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- Loading Data and Models ---")
    dataset, teacher, student, tokenizer = load_data_and_models()
    teacher.to(device)
    student.to(device)

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_loader = DataLoader(tokenized_datasets['train'], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_datasets['validation'], batch_size=config.batch_size)

    optimizer = AdamW(student.parameters(), lr=config.learning_rate)
    pruner = StructuralPruner(student)
    
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()

    best_finetuned_acc = 0
    patience_counter = 0
    best_finetuned_model_state = None
    history = []

    print(f"\n--- Starting Training for {config.total_epochs} Epochs ---")
    print(f"Pruning phase: Epochs {config.pruning_start_epoch} to {config.pruning_end_epoch}")
    print(f"Finetuning phase: Epochs {config.finetuning_start_epoch} to {config.total_epochs}")

    total_steps = len(train_loader) * config.total_epochs
    pruning_start_step = len(train_loader) * (config.pruning_start_epoch - 1)
    pruning_end_step = len(train_loader) * config.pruning_end_epoch
    
    global_step = 0
    for epoch in range(1, config.total_epochs + 1):
        student.train()
        last_sparsity = 0.0
        
        # Determine current phase
        if epoch >= config.finetuning_start_epoch:
            phase = "Finetuning"
            # Set a lower learning rate for the finetuning phase
            if epoch == config.finetuning_start_epoch:
                print(f"  -> Switching to finetuning learning rate: {config.finetuning_lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.finetuning_lr
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

        for batch_idx, batch in progress_bar:
            global_step += 1
            
            # Dynamic parameter calculation
            progress = min(1.0, max(0.0, (global_step - pruning_start_step) / (pruning_end_step - pruning_start_step)))
            target_sparsity = config.target_sparsity * (3 * progress**2 - 2 * progress**3) if phase == "Pruning" else config.target_sparsity if phase == "Finetuning" else 0.0
            
            temperature = config.initial_temperature - (config.initial_temperature - config.final_temperature) * progress
            alpha = config.alpha_initial - (config.alpha_initial - config.alpha_final) * progress

            # Pruning step
            if phase == "Pruning" and global_step % config.pruning_frequency_steps == 0:
                actual_sparsity = pruner.compute_importance_and_prune(target_sparsity)
                last_sparsity = actual_sparsity
            elif phase == "Finetuning":
                pruner.apply_masks() # Ensure pruned weights stay zero

            # Training step
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids, attention_mask=attention_mask).logits
            
            student_logits = student(input_ids, attention_mask=attention_mask).logits

            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            soft_probs = F.log_softmax(student_logits / temperature, dim=-1)
            distill_loss = kl_loss_fn(soft_probs, soft_targets) * (temperature ** 2)
            
            label_loss = ce_loss_fn(student_logits, labels)
            loss = alpha * distill_loss + (1 - alpha) * label_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Ensure pruned weights stay zero after optimizer step
            if phase in ["Pruning", "Finetuning"]:
                pruner.apply_masks()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "T": f"{temperature:.2f}",
                "alpha": f"{alpha:.2f}",
                "sparsity": f"{last_sparsity:.3f}"
            })

        # Evaluation
        val_acc = evaluate(student, val_loader, device)
        actual_sparsity = pruner.compute_importance_and_prune(target_sparsity) # Get current sparsity
        
        print(f"Epoch {epoch} Results: Val Acc: {val_acc:.4f}, Sparsity: {actual_sparsity:.4f}")
        history.append({'epoch': epoch, 'val_acc': val_acc, 'sparsity': actual_sparsity, 'phase': phase})

        # Early stopping and model saving, focused on the finetuning phase
        if phase == "Finetuning":
            if val_acc > best_finetuned_acc + config.min_delta:
                best_finetuned_acc = val_acc
                patience_counter = 0
                best_finetuned_model_state = copy.deepcopy(student.state_dict())
                print(f"  -> New best FINETUNING accuracy! Saving model state.")
            else:
                patience_counter += 1
                print(f"  -> Patience: {patience_counter}/{config.patience}")
        
        elif phase == "Pruning":
            # Reset patience during pruning phase to avoid premature stopping
            patience_counter = 0
            print(f"  -> In pruning phase, patience is not checked.")

        if patience_counter >= config.patience:
            print("--- Early stopping triggered in finetuning phase ---")
            break

    # Save the best finetuned model
    print("\n--- Training Finished ---")
    if best_finetuned_model_state:
        student.load_state_dict(best_finetuned_model_state)
        model_save_path = f"./models/{config.experiment_name}"
        student.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Best FINETUNED model saved to {model_save_path} with accuracy: {best_finetuned_acc:.4f}")
        
        # Save history
        with open(os.path.join(model_save_path, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    else:
        print("No best finetuned model was saved. The model did not improve during the finetuning phase.")

if __name__ == "__main__":
    main()
