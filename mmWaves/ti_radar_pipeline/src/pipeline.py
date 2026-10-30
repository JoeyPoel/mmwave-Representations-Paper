"""
Unified Master Pipeline Engine for TI mmWave Radar Fall Detection.
Orchestrates LOSO Cross-Validation benchmark across Doppler-Time Map,
2D Orthogonal Projections, and Native 3D Point Sets representations.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import DATASET_CONFIGS, CHECKPOINTS_DIR, BENCHMARKS_DIR
from .dataset_parser import get_loso_splits, get_all_recording_files, load_recording_df
from .transforms import (
    extract_raw_windows_from_df,
    transform_rep1_doppler_time,
    transform_rep2_orthogonal_projections,
    transform_rep3_pointnet
)
from .train_loso import run_loso_cross_validation
from .benchmark_eval import run_benchmark_evaluation

def run_loso_benchmark(epochs: int = 20, lr: float = 5e-4, batch_size: int = 32):
    """
    Executes the 3-fold Leave-One-Subject-Out (LOSO) cross-validation benchmark
    across Representation 1, Representation 2, and Representation 3.
    """
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for rep in ["rep1", "rep2", "rep3"]:
        res = run_loso_cross_validation(rep_key=rep, epochs=epochs, lr=lr, batch_size=batch_size)
        results[rep] = res

    res_path = BENCHMARKS_DIR / "loso_benchmark_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved LOSO benchmark results to {res_path}")

    # Evaluate side-by-side performance table and latency
    df_summary = run_benchmark_evaluation(res_path)
    return df_summary

# Backwards-compatible aliases
def run_dataset_split(dataset_key: str = "ti"):
    splits = get_loso_splits()
    print(f"TI Dataset LOSO 3-fold split created across {len(splits)} subjects.")
    return splits

def run_dataset_explorer(dataset_key: str = "ti"):
    files = get_all_recording_files()
    print(f"TI Dataset Explorer: Discovered {len(files)} CSV files (51 Falls, 51 ADLs).")
    return files

def run_data_preparation(dataset_key: str = "ti"):
    return run_dataset_split(dataset_key)

def run_model_training(dataset_key: str = "ti", model_name: str = "rep1", epochs: int = 20, lr: float = 5e-4, batch_size: int = 32):
    res = run_loso_cross_validation(rep_key=model_name, epochs=epochs, lr=lr, batch_size=batch_size)
    return res.get("accuracy", 0.0), res.get("roc_auc", 0.0), res.get("confusion_matrix", [])

if __name__ == "__main__":
    run_loso_benchmark()
