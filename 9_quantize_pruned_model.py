import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


def find_latest_pruned_model(root: str = "./models") -> str | None:
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("pruning_") or name.startswith("pruning_with_finetuning_"):
            candidates.append(full)
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Apply 4-bit PTQ to a pruned checkpoint")
    parser.add_argument("--pruned_model_path", type=str, default=None, help="Path to the pruned model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the PTQ checkpoint")
    args = parser.parse_args()

    pruned_model_path = args.pruned_model_path or find_latest_pruned_model()
    if pruned_model_path is None or not os.path.exists(pruned_model_path):
        raise FileNotFoundError("Could not resolve a pruned checkpoint for PTQ")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(pruned_model_path.rstrip(os.sep))
        output_dir = Path("./models") / f"pruned_quantized_ptq_{base_name}_{timestamp}"

    print(f"--- Loading pruned model from: {pruned_model_path} ---")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pruned_model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Saving quantized model to: {output_dir} ---")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    for metadata_name in ["layer_sensitivity.json", "pruning_allocation.json", "kgapq_stats.json"]:
        source = Path(pruned_model_path) / metadata_name
        if source.exists():
            shutil.copy2(source, output_dir / metadata_name)

    ptq_config = {
        "pruned_model_path": pruned_model_path,
        "output_dir": str(output_dir),
        "quantization": "4bit-nf4",
    }
    (output_dir / "ptq_config.json").write_text(json.dumps(ptq_config, indent=2), encoding="utf-8")
    print("--- PTQ complete ---")


if __name__ == "__main__":
    main()
