import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import os

def main():
    """
    Loads the pruned-and-finetuned model, applies 4-bit quantization, and saves it.
    """
    pruned_model_path = "./models/pruning_with_finetuning_20251029_092719"
    quantized_model_save_path = None
    
    print(f"--- Loading Pruned Model from: {pruned_model_path} ---")

    if not os.path.exists(pruned_model_path):
        # Auto-detect latest pruning_with_finetuning_* directory
        models_root = "./models"
        candidates = []
        if os.path.isdir(models_root):
            for name in os.listdir(models_root):
                full_path = os.path.join(models_root, name)
                if os.path.isdir(full_path) and name.startswith("pruning_with_finetuning_"):
                    candidates.append(full_path)
        if candidates:
            candidates.sort(reverse=True)
            pruned_model_path = candidates[0]
            print(f"  -> Auto-detected latest pruned model: {pruned_model_path}")
        else:
            print(f"Error: Pruned model path not found at '{pruned_model_path}'.")
            print("Please ensure the pruning and finetuning script (8) has been run successfully.")
            return

    # Default save path: align name with source pruning checkpoint
    if quantized_model_save_path is None:
        base_name = os.path.basename(pruned_model_path.rstrip(os.sep))
        quantized_model_save_path = os.path.join("./models", base_name.replace("pruning_with_finetuning", "pruned_quantized_ptq"))

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("--- Applying 4-bit Quantization ---")
    
    # Load the pruned model with quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        pruned_model_path,
        quantization_config=quantization_config,
        device_map="auto" # Automatically handle device placement
    )
    
    tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)

    print(f"--- Saving Quantized Model to: {quantized_model_save_path} ---")
    
    # Create directory if it doesn't exist
    os.makedirs(quantized_model_save_path, exist_ok=True)
    
    # Save the quantized model and tokenizer
    model.save_pretrained(quantized_model_save_path)
    tokenizer.save_pretrained(quantized_model_save_path)
    
    print("--- Quantization and Saving Complete ---")

if __name__ == "__main__":
    main()
