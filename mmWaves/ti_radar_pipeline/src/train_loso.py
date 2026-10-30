"""
Leave-One-Subject-Out (LOSO) Cross-Validation Training Script for TI mmWave Radar Fall Detection.
Executes 3-fold LOSO cross-validation across Representation 1 (Doppler-Time Map),
Representation 2 (2D Orthogonal Projections), and Representation 3 (Native 3D Point Sets).
In each fold:
- Training Set (68 files): 34 Falls, 34 Non-Falls (2 subjects).
- Test Set (34 files): 17 Falls, 17 Non-Falls (1 complete held-out test subject).
Logs aggregated Out-of-Fold (OOF) Accuracy, Precision, Recall, Macro F1-score, and ROC-AUC.
"""

from collections.abc import Callable
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.dataset_parser import get_loso_splits, load_recording_df, extract_label, DATA_DIR
from src.transforms import (
    extract_raw_windows_from_df,
    FoldScaler,
    transform_rep1_doppler_time,
    transform_rep2_orthogonal_projections,
    transform_rep3_pointnet
)
from src.models import ResNet18Adaptor, PointNetPlusPlusFallDetector

CHECKPOINTS_DIR = Path("models/checkpoints")
BENCHMARKS_DIR = Path("models/benchmarks")

def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_dataset_for_files(file_list: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a list of CSV recording files and extracts sliding windows.
    Returns:
       X_raw: (N, 10, 32, 4) array of target-relative windows
       y_raw: (N,) array of binary labels
    """
    windows_list = []
    labels_list = []
    print(f"Extracting windows from {len(file_list)} recording files...", flush=True)
    for idx, fpath in enumerate(file_list):
        lbl = extract_label(fpath)
        df_clean = load_recording_df(fpath)
        win_arrs = extract_raw_windows_from_df(df_clean, window_size_frames=10, stride=5)
        for win in win_arrs:
            windows_list.append(win)
            labels_list.append(lbl)

    if len(windows_list) == 0:
        return np.zeros((0, 10, 32, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(windows_list, dtype=np.float32), np.array(labels_list, dtype=np.int64)

def train_one_fold(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    epochs: int = 20,
    lr: float = 5e-4,
    batch_size: int = 32,
    device: torch.device = None,
    seed: Optional[int] = None,
    aug_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    weight_decay: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains model on single fold training set with PyTorch WeightedRandomSampler mini-batch balance.
    Evaluates directly on held-out Test fold. Supports optional online data augmentation (aug_fn) and custom weight_decay.
    """
    if seed is not None:
        set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    train_y_t = torch.tensor(y_tr, dtype=torch.long)
    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), train_y_t)
    te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))

    # Balanced mini-batch sampling
    class_counts = torch.bincount(train_y_t)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_y_t]

    sampler_generator = torch.Generator()
    if seed is not None:
        sampler_generator.manual_seed(seed)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=sampler_generator
    )

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(epochs):
        model.train()
        r_loss = 0.0
        corr = 0
        tot = 0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            if aug_fn is not None:
                bx = aug_fn(bx)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * bx.size(0)
            corr += (out.argmax(dim=1) == by).sum().item()
            tot += by.size(0)
        scheduler.step()

        if (ep + 1) % 5 == 0 or (ep + 1) == epochs:
            print(f"  Epoch [{ep+1:02d}/{epochs:02d}] Train Loss: {r_loss/max(tot,1):.4f}, Train Acc: {corr/max(tot,1)*100:.2f}%", flush=True)


    # Evaluation on Test fold
    model.eval()
    test_probs = []
    with torch.no_grad():
        for bx, by in te_loader:
            bx = bx.to(device)
            out = model(bx)
            prob = torch.softmax(out, dim=1)[:, 1]
            test_probs.extend(prob.cpu().numpy())

    return np.array(test_probs, dtype=np.float32), y_te


def compute_metrics(targets, probs, threshold=0.50) -> Dict[str, float]:
    """Helper for calculating classification metrics."""
    targets_arr = np.array(targets)
    probs_arr = np.array(probs)
    preds_arr = (probs_arr >= threshold).astype(int)

    acc = float(accuracy_score(targets_arr, preds_arr))
    prec = float(precision_score(targets_arr, preds_arr, zero_division=0))
    rec = float(recall_score(targets_arr, preds_arr, zero_division=0))
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    auc = float(roc_auc_score(targets_arr, probs_arr)) if len(np.unique(targets_arr)) > 1 else 0.5

    cm = confusion_matrix(targets_arr, preds_arr)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "macro_f1": macro_f1,
        "roc_auc": auc, "fpr": fpr, "confusion_matrix": cm.tolist()
    }


def run_single_loso_pass(
    rep_key: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: Optional[int] = None,
    model_fn: Optional[Callable[[], nn.Module]] = None,
    aug_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    weight_decay: float = 1e-4
) -> Tuple[List[int], List[float]]:
    """Runs a single 3-fold LOSO cross-validation pass for a given seed."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = get_loso_splits()
    oof_targets, oof_probs = [], []

    for fold_info in splits:
        fold_idx = fold_info["fold"]
        fold_seed = (seed + fold_idx * 100) if seed is not None else None

        X_tr_raw, y_tr = extract_dataset_for_files(fold_info["train_files"])
        X_te_raw, y_te = extract_dataset_for_files(fold_info["test_files"])

        scaler = FoldScaler()
        scaler.fit(X_tr_raw)
        X_tr_scaled = scaler.transform(X_tr_raw)
        X_te_scaled = scaler.transform(X_te_raw)

        key_clean = str(rep_key).lower().strip()
        if key_clean in ("rep1_doppler", "rep1"):
            X_tr = transform_rep1_doppler_time(X_tr_raw)
            X_te = transform_rep1_doppler_time(X_te_raw)
            model = model_fn() if model_fn is not None else ResNet18Adaptor(in_channels=3, num_classes=2)
        elif key_clean in ("rep2_projections", "rep2"):
            X_tr = transform_rep2_orthogonal_projections(X_tr_raw)
            X_te = transform_rep2_orthogonal_projections(X_te_raw)
            model = model_fn() if model_fn is not None else ResNet18Adaptor(in_channels=3, num_classes=2)
        elif key_clean in ("rep3_pointset", "rep3"):
            X_tr = transform_rep3_pointnet(X_tr_scaled)
            X_te = transform_rep3_pointnet(X_te_scaled)
            model = model_fn() if model_fn is not None else PointNetPlusPlusFallDetector(in_channels=5, num_classes=2)
        else:
            raise ValueError(f"Unknown rep_key '{rep_key}'")

        test_probs, test_targets = train_one_fold(
            model, X_tr, y_tr, X_te, y_te, epochs=epochs, lr=lr, batch_size=batch_size, device=device, seed=fold_seed, aug_fn=aug_fn, weight_decay=weight_decay
        )

        oof_probs.extend(test_probs)
        oof_targets.extend(test_targets)

    return oof_targets, oof_probs


def run_loso_cross_validation(
    rep_key: str = "rep1",
    epochs: int = 20,
    lr: float = 5e-4,
    batch_size: int = 32,
    num_runs: int = 1,
    seeds: Optional[List[int]] = None,
    use_wandb: bool = False,
    wandb_project: str = "mmWave-Fall-Detection",
    save_benchmark: bool = True,
    output_json_path: Optional[Path] = None,
    model_fn: Optional[Callable[[], nn.Module]] = None,
    aug_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    weight_decay: float = 1e-4
) -> Dict[str, any]:
    """
    Executes 3-Fold Leave-One-Subject-Out (LOSO) cross-validation.
    Supports running multiple seeds/runs (`num_runs`), W&B logging, benchmark export,
    custom model_fn override, online aug_fn, and custom weight_decay.
    """
    if seeds is None:
        seeds = [42 + i * 10 for i in range(num_runs)]
    else:
        num_runs = len(seeds)

    print(f"\n=======================================================")
    print(f"Starting LOSO Cross-Validation for Representation: {rep_key}")
    print(f"Runs: {num_runs} | Seeds: {seeds} | Epochs: {epochs} | LR: {lr} | Batch Size: {batch_size} | Weight Decay: {weight_decay}")
    if model_fn is not None:
        print(f"Custom Model Architecture Function Override Provided!")
    if aug_fn is not None:
        print(f"Online Data Augmentation Function Provided!")
    if use_wandb:
        print(f"W&B Logging Enabled (Project: {wandb_project})")
    print(f"=======================================================")

    run_results = []
    all_oof_targets, all_oof_probs = [], []

    for run_idx, seed in enumerate(seeds):
        print(f"\n>>> Starting Run {run_idx + 1}/{num_runs} (Seed: {seed}) <<<")
        
        wb_run = None
        if use_wandb:
            try:
                # pyrefly: ignore [missing-import]
                import wandb
                wb_run = wandb.init(
                    project=wandb_project,
                    name=f"{rep_key}_run{run_idx+1}_seed{seed}",
                    config={
                        "representation": rep_key,
                        "epochs": epochs,
                        "lr": lr,
                        "batch_size": batch_size,
                        "seed": seed,
                        "run_index": run_idx + 1
                    },
                    reinit=True
                )
            except Exception as e:
                print(f"Warning: W&B initialization failed ({e}). Proceeding without W&B.")
                wb_run = None

        oof_targets, oof_probs = run_single_loso_pass(rep_key, epochs, lr, batch_size, seed=seed, model_fn=model_fn, aug_fn=aug_fn, weight_decay=weight_decay)
        m = compute_metrics(oof_targets, oof_probs)
        m["seed"] = seed
        m["run"] = run_idx + 1
        run_results.append(m)

        all_oof_targets.extend(oof_targets)
        all_oof_probs.extend(oof_probs)

        print(f"Run {run_idx + 1} (Seed {seed}) Test Result -> Acc: {m['accuracy']*100:.2f}%, Recall: {m['recall']*100:.2f}%, F1: {m['macro_f1']*100:.2f}%, AUC: {m['roc_auc']:.4f}")

        if wb_run is not None:
            try:
                wb_run.log({
                    "test_accuracy": m["accuracy"],
                    "test_recall": m["recall"],
                    "test_precision": m["precision"],
                    "test_macro_f1": m["macro_f1"],
                    "test_roc_auc": m["roc_auc"],
                    "test_fpr": m["fpr"]
                })
                wb_run.finish()
            except Exception as e:
                print(f"Warning: Could not log metrics to W&B ({e}).")

    # Compute mean and standard deviation across runs
    accs = [r["accuracy"] for r in run_results]
    recs = [r["recall"] for r in run_results]
    precs = [r["precision"] for r in run_results]
    f1s = [r["macro_f1"] for r in run_results]
    aucs = [r["roc_auc"] for r in run_results]
    fprs = [r["fpr"] for r in run_results]

    agg_metrics = {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "precision": float(np.mean(precs)),
        "precision_std": float(np.std(precs)),
        "recall": float(np.mean(recs)),
        "recall_std": float(np.std(recs)),
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "fpr": float(np.mean(fprs)),
        "fpr_std": float(np.std(fprs)),
        "confusion_matrix": run_results[0]["confusion_matrix"]
    }

    print(f"\n--- Aggregated Multi-Run ({num_runs} Runs) LOSO Results for {rep_key.upper()} ---")
    print(f"Test Acc: {agg_metrics['accuracy']*100:.2f}% ± {agg_metrics['accuracy_std']*100:.2f}% | "
          f"Test Rec: {agg_metrics['recall']*100:.2f}% ± {agg_metrics['recall_std']*100:.2f}% | "
          f"Test Macro F1: {agg_metrics['macro_f1']*100:.2f}% ± {agg_metrics['macro_f1_std']*100:.2f}% | "
          f"Test ROC-AUC: {agg_metrics['roc_auc']:.4f} ± {agg_metrics['roc_auc_std']:.4f}")

    results = {
        "representation": rep_key,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "seeds": seeds,
        "accuracy": agg_metrics["accuracy"],
        "accuracy_std": agg_metrics["accuracy_std"],
        "precision": agg_metrics["precision"],
        "precision_std": agg_metrics["precision_std"],
        "recall": agg_metrics["recall"],
        "recall_std": agg_metrics["recall_std"],
        "macro_f1": agg_metrics["macro_f1"],
        "macro_f1_std": agg_metrics["macro_f1_std"],
        "roc_auc": agg_metrics["roc_auc"],
        "roc_auc_std": agg_metrics["roc_auc_std"],
        "fpr": agg_metrics["fpr"],
        "fpr_std": agg_metrics["fpr_std"],
        "confusion_matrix": agg_metrics["confusion_matrix"],
        "test_metrics": agg_metrics,
        "runs": run_results
    }

    if save_benchmark:
        BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
        if output_json_path is None:
            rep_clean = rep_key.lower().replace("rep1_doppler", "rep1_doppler").replace("rep2_projections", "rep2_projections").replace("rep3_pointset", "rep3_pointset")
            if not rep_clean.endswith("_benchmark.json"):
                output_json_path = BENCHMARKS_DIR / f"{rep_clean}_benchmark.json"
            else:
                output_json_path = BENCHMARKS_DIR / rep_clean
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Automatically saved benchmark JSON to: {output_json_path}")

    return results


if __name__ == "__main__":
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    all_res = {}
    for rep in ["rep1_doppler", "rep2_projections", "rep3_pointset"]:
        res = run_loso_cross_validation(rep_key=rep, epochs=20, lr=5e-4, batch_size=32, num_runs=3, save_benchmark=True)
        all_res[rep] = res

    out_path = BENCHMARKS_DIR / "loso_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=4)
    print(f"\nSaved master LOSO benchmark results to {out_path}")

