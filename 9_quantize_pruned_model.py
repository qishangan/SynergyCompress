import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import os

def main():
    """
    Loads the pruned-and-finetuned model, applies 4-bit quantization, and saves it.
    """
    pruned_model_path = "./models/pruning_with_finetuning_20251029_092719"
    quantized_model_save_path = "./models/pruned_quantized_final"
    
    print(f"--- Loading Pruned Model from: {pruned_model_path} ---")

    if not os.path.exists(pruned_model_path):
        print(f"Error: Pruned model path not found at '{pruned_model_path}'.")
        print("Please ensure the pruning and finetuning script (8) has been run successfully.")
        return

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
