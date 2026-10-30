"""
Benchmark evaluation script comparing all three mmWave radar representations:
- Representation 1: Kinematic / Doppler-Time Map (ResNet-18)
- Representation 2: Spatial / 2D Orthogonal Projections (ResNet-18)
- Representation 3: Geometric / Native 3D Point Tensor (PointNet++)

Evaluates out-of-fold classification metrics, false-alarm rates (FPR),
and computational latency per sequence (ms).
"""

import json
import time
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import pandas as pd

from src.models import ResNet18Adaptor, PointNetPlusPlusFallDetector

BENCHMARKS_DIR = Path("models/benchmarks")

def measure_inference_latency(model: torch.nn.Module, sample_input: torch.Tensor, num_runs: int = 100) -> float:
    """
    Measures mean inference latency per sequence in milliseconds (ms).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    sample_input = sample_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark run
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    mean_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    return float(mean_latency_ms)

def run_benchmark_evaluation(results_json_path: Path = BENCHMARKS_DIR / "loso_benchmark_results.json") -> pd.DataFrame:
    """
    Compiles side-by-side benchmark metrics and latency measurements for all 3 representations.
    """
    if not results_json_path.exists():
        print(f"Results JSON not found at {results_json_path}. Please run train_loso.py first.")
        return pd.DataFrame()

    with open(results_json_path, "r", encoding="utf-8") as f:
        loso_res = json.load(f)

    # Instantiate dummy models & inputs for latency benchmark
    models_dict = {
        "rep1": (ResNet18Adaptor(in_channels=3), torch.randn(1, 3, 32, 32)),
        "rep2": (ResNet18Adaptor(in_channels=3), torch.randn(1, 3, 32, 32)),
        "rep3": (PointNetPlusPlusFallDetector(in_channels=5), torch.randn(1, 5, 320))
    }

    rep_names = {
        "rep1": "Rep 1: Doppler-Time Map (ResNet-18)",
        "rep2": "Rep 2: 2D Orthogonal Projections (ResNet-18)",
        "rep3": "Rep 3: Native 3D Point Tensor (PointNet++)"
    }

    table_data = []
    for rep_key in ["rep1", "rep2", "rep3"]:
        r_info = loso_res.get(rep_key)
        if not r_info:
            for k in loso_res.keys():
                if rep_key in k.lower():
                    r_info = loso_res[k]
                    break
        r_info = r_info or {}
        model, dummy_in = models_dict[rep_key]
        latency_ms = measure_inference_latency(model, dummy_in)

        table_data.append({
            "Representation": rep_names[rep_key],
            "Accuracy (%)": round(r_info.get("accuracy", 0.0) * 100, 2),
            "Precision (%)": round(r_info.get("precision", 0.0) * 100, 2),
            "Recall (%)": round(r_info.get("recall", 0.0) * 100, 2),
            "Macro F1 (%)": round(r_info.get("macro_f1", 0.0) * 100, 2),
            "ROC-AUC": round(r_info.get("roc_auc", 0.0), 4),
            "False Alarm Rate / FPR (%)": round(r_info.get("fpr", 0.0) * 100, 2),
            "Latency (ms/seq)": round(latency_ms, 2)
        })


    df_summary = pd.DataFrame(table_data)

    print("\n" + "="*80)
    print("      TI mmWave Radar Fall Detection: Representation Benchmark Summary")
    print("="*80)
    print(df_summary.to_markdown(index=False))
    print("="*80)

    # Save summary table
    summary_md_path = BENCHMARKS_DIR / "loso_representation_comparison.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# TI mmWave Radar Fall Detection: Representation Comparison\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n")

    print(f"Saved benchmark summary table to {summary_md_path}")
    return df_summary

if __name__ == "__main__":
    run_benchmark_evaluation()
