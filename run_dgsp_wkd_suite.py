import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class ExperimentRunner:
    def __init__(self, suite_name: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.suite_tag = f"{suite_name}_{timestamp}"
        self.output_root = Path("outputs") / f"exp_{self.suite_tag}"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.python = sys.executable

    def run_command(self, name: str, command: List[str]) -> Path:
        run_dir = self.output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_path = run_dir / "stdout.log"
        start_time = time.time()

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                safe_line = line
                try:
                    safe_line.encode(sys.stdout.encoding or "utf-8")
                except UnicodeEncodeError:
                    safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                        sys.stdout.encoding or "utf-8",
                        errors="replace",
                    )
                print(safe_line, end="")
                log_file.write(line)
            return_code = process.wait()

        metadata = {
            "name": name,
            "return_code": return_code,
            "duration_sec": time.time() - start_time,
            "command": command,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if return_code != 0:
            raise RuntimeError(f"Command failed for {name}: {' '.join(command)}")
        return run_dir


def prune_command(
    runner: ExperimentRunner,
    model_name: str,
    sensitivity_method: str,
    sparsity: float,
    layer_adaptive: bool,
    quick: bool,
    total_epochs: int,
    train_subset_ratio: float | None = None,
    val_subset_ratio: float | None = None,
) -> Dict[str, str]:
    experiment_name = f"{runner.suite_tag}_{model_name}_prune_sp{int(sparsity * 100):02d}"
    command = [
        runner.python,
        "8_pruning_with_finetuning.py",
        "--sensitivity_method",
        sensitivity_method,
        "--target_sparsity",
        str(sparsity),
        "--alpha",
        "1.0",
        "--experiment_name",
        experiment_name,
        "--total_epochs",
        str(total_epochs),
    ]
    if sensitivity_method == "weight":
        command.extend(["--lambda_score", "0.0"])
    else:
        command.extend(["--lambda_score", "0.5", "--ema_beta", "0.9"])
    if not layer_adaptive:
        command.append("--disable_layer_adaptive")
    if quick:
        command.append("--fast")
    if train_subset_ratio is not None:
        command.extend(["--train_subset_ratio", str(train_subset_ratio)])
    if val_subset_ratio is not None:
        command.extend(["--val_subset_ratio", str(val_subset_ratio)])

    runner.run_command(f"{model_name}_prune_sp{int(sparsity * 100):02d}", command)
    return {
        "model_dir": str(Path("models") / experiment_name),
        "experiment_name": experiment_name,
    }


def ptq_command(runner: ExperimentRunner, model_name: str, pruned_model_path: str) -> str:
    output_dir = str(Path("models") / f"{runner.suite_tag}_{model_name}_ptq")
    command = [
        runner.python,
        "9_quantize_pruned_model.py",
        "--pruned_model_path",
        pruned_model_path,
        "--output_dir",
        output_dir,
    ]
    runner.run_command(f"{model_name}_ptq", command)
    return output_dir


def qkd_command(
    runner: ExperimentRunner,
    model_name: str,
    pruned_model_path: str,
    quick: bool,
    weighted_hidden: bool,
    train_subset_ratio: float | None = None,
    val_subset_ratio: float | None = None,
) -> str:
    output_dir = str(Path("models") / f"{runner.suite_tag}_{model_name}_qkd")
    command = [
        runner.python,
        "10_qat_kd_4bit.py",
        "--pruned_model_path",
        pruned_model_path,
        "--output_dir",
        output_dir,
        "--enable_hidden_kd",
    ]
    if quick:
        command.append("--fast")
    if weighted_hidden:
        command.extend(["--weighted_hidden", "--hidden_tau", "1.0"])
    if train_subset_ratio is not None:
        command.extend(["--train_subset_ratio", str(train_subset_ratio)])
    if val_subset_ratio is not None:
        command.extend(["--val_subset_ratio", str(val_subset_ratio)])
    runner.run_command(f"{model_name}_qkd", command)
    return output_dir


def evaluate_command(
    runner: ExperimentRunner,
    eval_name: str,
    manifest: List[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    manifest_path = runner.output_root / f"{eval_name}_manifest.json"
    output_json = runner.output_root / f"{eval_name}_results.json"
    output_png = runner.output_root / f"{eval_name}_results.png"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    command = [
        runner.python,
        "7_evaluate_and_generate_report.py",
        "--manifest",
        str(manifest_path),
        "--output_json",
        str(output_json),
        "--output_png",
        str(output_png),
    ]
    runner.run_command(f"{eval_name}_eval", command)
    return json.loads(output_json.read_text(encoding="utf-8"))


def run_quick_stage(runner: ExperimentRunner) -> Dict[str, Dict[str, float]]:
    sparsity = 0.20
    total_epochs = 4

    baseline_prune = prune_command(runner, "quick_a_baseline", "weight", sparsity, layer_adaptive=False, quick=True, total_epochs=total_epochs)
    baseline_qkd = qkd_command(runner, "quick_a_baseline", baseline_prune["model_dir"], quick=True, weighted_hidden=False)

    dgsp_global_prune = prune_command(runner, "quick_b_dgsp_global", "dgsp", sparsity, layer_adaptive=False, quick=True, total_epochs=total_epochs)
    dgsp_global_qkd = qkd_command(runner, "quick_b_dgsp_global", dgsp_global_prune["model_dir"], quick=True, weighted_hidden=False)

    dgsp_adaptive_prune = prune_command(runner, "quick_c_dgsp_adaptive", "dgsp", sparsity, layer_adaptive=True, quick=True, total_epochs=total_epochs)
    dgsp_adaptive_qkd = qkd_command(runner, "quick_c_dgsp_adaptive", dgsp_adaptive_prune["model_dir"], quick=True, weighted_hidden=False)
    dgsp_wkd_qkd = qkd_command(runner, "quick_d_dgsp_wkd", dgsp_adaptive_prune["model_dir"], quick=True, weighted_hidden=True)

    manifest = [
        {"name": "Distilled Student (Baseline)", "eval_path": "./student_model", "quantized": False, "sparsity_source_path": "./student_model"},
        {"name": "A. Baseline Pruned", "eval_path": baseline_prune["model_dir"], "quantized": False, "sparsity_source_path": baseline_prune["model_dir"]},
        {"name": "A. Baseline QKD", "eval_path": baseline_qkd, "quantized": True, "sparsity_source_path": baseline_prune["model_dir"]},
        {"name": "B. DGSP Score Only Pruned", "eval_path": dgsp_global_prune["model_dir"], "quantized": False, "sparsity_source_path": dgsp_global_prune["model_dir"]},
        {"name": "B. DGSP Score Only QKD", "eval_path": dgsp_global_qkd, "quantized": True, "sparsity_source_path": dgsp_global_prune["model_dir"]},
        {"name": "C. DGSP + Adaptive Pruned", "eval_path": dgsp_adaptive_prune["model_dir"], "quantized": False, "sparsity_source_path": dgsp_adaptive_prune["model_dir"]},
        {"name": "C. DGSP + Adaptive QKD", "eval_path": dgsp_adaptive_qkd, "quantized": True, "sparsity_source_path": dgsp_adaptive_prune["model_dir"]},
        {"name": "D. DGSP-WKD", "eval_path": dgsp_wkd_qkd, "quantized": True, "sparsity_source_path": dgsp_adaptive_prune["model_dir"]},
    ]
    return evaluate_command(runner, "quick_ablation", manifest)


def run_formal_stage(
    runner: ExperimentRunner,
    sparsities: List[float],
    total_epochs: int,
    train_subset_ratio: float | None = None,
    val_subset_ratio: float | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    formal_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sparsity in sparsities:
        tag = f"sp{int(sparsity * 100):02d}"
        baseline_prune = prune_command(
            runner,
            f"formal_baseline_{tag}",
            "weight",
            sparsity,
            layer_adaptive=False,
            quick=False,
            total_epochs=total_epochs,
            train_subset_ratio=train_subset_ratio,
            val_subset_ratio=val_subset_ratio,
        )
        baseline_ptq = ptq_command(runner, f"formal_baseline_{tag}", baseline_prune["model_dir"])
        baseline_qkd = qkd_command(
            runner,
            f"formal_baseline_{tag}",
            baseline_prune["model_dir"],
            quick=False,
            weighted_hidden=False,
            train_subset_ratio=train_subset_ratio,
            val_subset_ratio=val_subset_ratio,
        )

        dgsp_prune = prune_command(
            runner,
            f"formal_dgsp_{tag}",
            "dgsp",
            sparsity,
            layer_adaptive=True,
            quick=False,
            total_epochs=total_epochs,
            train_subset_ratio=train_subset_ratio,
            val_subset_ratio=val_subset_ratio,
        )
        dgsp_ptq = ptq_command(runner, f"formal_dgsp_{tag}", dgsp_prune["model_dir"])
        dgsp_wkd = qkd_command(
            runner,
            f"formal_dgsp_wkd_{tag}",
            dgsp_prune["model_dir"],
            quick=False,
            weighted_hidden=True,
            train_subset_ratio=train_subset_ratio,
            val_subset_ratio=val_subset_ratio,
        )

        manifest = [
            {"name": "Distilled Student (Baseline)", "eval_path": "./student_model", "quantized": False, "sparsity_source_path": "./student_model"},
            {"name": "Baseline Pruned Sparse FP32", "eval_path": baseline_prune["model_dir"], "quantized": False, "sparsity_source_path": baseline_prune["model_dir"]},
            {"name": "Baseline Pruned + PTQ", "eval_path": baseline_ptq, "quantized": True, "sparsity_source_path": baseline_prune["model_dir"]},
            {"name": "Baseline Pruned + QKD", "eval_path": baseline_qkd, "quantized": True, "sparsity_source_path": baseline_prune["model_dir"]},
            {"name": "DGSP Sparse FP32", "eval_path": dgsp_prune["model_dir"], "quantized": False, "sparsity_source_path": dgsp_prune["model_dir"]},
            {"name": "DGSP + PTQ", "eval_path": dgsp_ptq, "quantized": True, "sparsity_source_path": dgsp_prune["model_dir"]},
            {"name": "DGSP-WKD", "eval_path": dgsp_wkd, "quantized": True, "sparsity_source_path": dgsp_prune["model_dir"]},
        ]
        formal_results[tag] = evaluate_command(runner, f"formal_{tag}", manifest)

    return formal_results


def write_summary_files(
    quick_results: Dict[str, Dict[str, float]],
    formal_results: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    baseline_results = {}
    dgsp_results = {}
    dgsp_wkd_results = {}

    for tag, results in formal_results.items():
        baseline_results[tag] = {
            key: value
            for key, value in results.items()
            if key in {
                "Distilled Student (Baseline)",
                "Baseline Pruned Sparse FP32",
                "Baseline Pruned + PTQ",
                "Baseline Pruned + QKD",
            }
        }
        dgsp_results[tag] = {
            key: value
            for key, value in results.items()
            if key in {"DGSP Sparse FP32", "DGSP + PTQ"}
        }
        dgsp_wkd_results[tag] = {
            key: value
            for key, value in results.items()
            if key == "DGSP-WKD"
        }

    if quick_results:
        Path("ablation_results.json").write_text(json.dumps(quick_results, indent=2), encoding="utf-8")
    if formal_results:
        Path("baseline_results.json").write_text(json.dumps(baseline_results, indent=2), encoding="utf-8")
        Path("dgsp_results.json").write_text(json.dumps(dgsp_results, indent=2), encoding="utf-8")
        Path("dgsp_wkd_results.json").write_text(json.dumps(dgsp_wkd_results, indent=2), encoding="utf-8")

    if quick_results or formal_results:
        final_report = {}
        final_report_path = Path("final_report.json")
        if final_report_path.exists():
            try:
                final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                final_report = {}
        if quick_results:
            final_report["quick_ablation"] = quick_results
        if formal_results:
            final_report["formal_results"] = formal_results
        final_report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run DGSP-WKD quick and formal experiments")
    parser.add_argument("--suite_name", type=str, default="dgsp_wkd", help="Prefix for the output suite directory")
    parser.add_argument("--formal_total_epochs", type=int, default=8, help="Total epochs for formal pruning runs")
    parser.add_argument("--skip_quick", action="store_true", help="Skip quick ablation stage")
    parser.add_argument("--skip_formal", action="store_true", help="Skip formal stage")
    parser.add_argument("--sparsities", type=float, nargs="*", default=[0.15, 0.20], help="Formal sparsity levels")
    parser.add_argument("--formal_train_subset_ratio", type=float, default=None, help="Optional near-full training subset ratio for formal runs")
    parser.add_argument("--formal_val_subset_ratio", type=float, default=None, help="Optional validation subset ratio for formal runs")
    args = parser.parse_args()

    runner = ExperimentRunner(args.suite_name)
    quick_results: Dict[str, Dict[str, float]] = {}
    formal_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    if not args.skip_quick:
        quick_results = run_quick_stage(runner)
    if not args.skip_formal:
        formal_results = run_formal_stage(
            runner,
            args.sparsities,
            total_epochs=args.formal_total_epochs,
            train_subset_ratio=args.formal_train_subset_ratio,
            val_subset_ratio=args.formal_val_subset_ratio,
        )

    write_summary_files(quick_results, formal_results)
    print(f"\nAll requested stages completed. Logs and manifests are under: {runner.output_root}")


if __name__ == "__main__":
    main()
