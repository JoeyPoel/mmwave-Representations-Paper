"""
Rule-Based Non-AI Physical Baseline Module for TI mmWave Radar Fall Detection.

Implements interpretable, non-trainable physics-based heuristic detectors using:
1. Vertical Height Drop (delta_z / drop_z): Height reduction during postural descent.
2. Radial Doppler Speed (max_abs_v / min_v): Sudden downward kinetic acceleration.
3. Post-Motion Floor Height (final_z): Settlement near floor level (z <= threshold).
4. Spatial Ground Dispersion (xy_span): Ground plane point expansion upon impact.
5. Composite Physics Score: Weighted combination of drop_z, max_abs_v, and final_z.

Evaluated strictly using 3-Fold Leave-One-Subject-Out (LOSO) cross-validation across subjects Areeb, Raffay, Towsif.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from .dataset_parser import get_loso_splits, load_recording_df, extract_label, extract_subject

def extract_window_physical_features(
    df_clean: pd.DataFrame,
    window_size_frames: int = 10,
    stride_frames: int = 5,
    min_pts_per_win: int = 5
) -> List[Dict[str, float]]:
    """
    Extracts physical kinematic and spatial features from sliding temporal windows of a recording.

    Args:
        df_clean (pd.DataFrame): Cleaned radar points dataframe with columns ['frame', 'x', 'y', 'z', 'v', 'snr'].
        window_size_frames (int): Temporal window width in frames (default: 10 frames = 1.0s at 10 Hz).
        stride_frames (int): Window stride (default: 5 frames).
        min_pts_per_win (int): Minimum points required to keep a window.

    Returns:
        List[Dict[str, float]]: List of feature dictionaries per window.
    """
    if len(df_clean) == 0:
        return []

    all_frames = np.sort(df_clean["frame"].unique())
    max_frame = int(all_frames.max())
    frame_groups = {f_id: group for f_id, group in df_clean.groupby("frame")}

    windows = []
    for start_f in range(0, max_frame - (window_size_frames - 1) + 1, stride_frames):
        end_f = start_f + (window_size_frames - 1)
        win_dfs = [frame_groups[fi] for fi in range(start_f, end_f + 1) if fi in frame_groups]

        if len(win_dfs) == 0:
            continue

        df_win = pd.concat(win_dfs, ignore_index=True)
        if len(df_win) < min_pts_per_win or df_win["frame"].nunique() < 4:
            continue

        max_z = float(df_win["z"].max())
        min_z = float(df_win["z"].min())
        delta_z = max_z - min_z

        # Start vs end height drop
        first_f = df_win["frame"].min()
        last_f = df_win["frame"].max()
        df_first = df_win[df_win["frame"] == first_f]
        df_last = df_win[df_win["frame"] == last_f]

        start_z = float(df_first["z"].median()) if len(df_first) > 0 else max_z
        final_z = float(df_last["z"].median()) if len(df_last) > 0 else float(df_win["z"].median())
        drop_z = start_z - final_z

        max_abs_v = float(df_win["v"].abs().max())
        min_v = float(df_win["v"].min())

        x_span = float(df_win["x"].max() - df_win["x"].min())
        y_span = float(df_win["y"].max() - df_win["y"].min())
        xy_span = float(np.sqrt(x_span**2 + y_span**2))

        windows.append({
            "start_frame": start_f,
            "end_frame": end_f,
            "delta_z": delta_z,
            "drop_z": drop_z,
            "max_z": max_z,
            "min_z": min_z,
            "final_z": final_z,
            "max_abs_v": max_abs_v,
            "min_v": min_v,
            "xy_span": xy_span,
            "num_points": len(df_win)
        })

    return windows


def extract_dataset_rule_features(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Parses all 102 CSV files in the dataset and computes file-level aggregated physical features.

    Args:
        data_dir (Optional[Path]): Dataset root directory. If None, uses default TI dataset path.

    Returns:
        pd.DataFrame: DataFrame containing file path, subject, ground-truth label, and aggregated physical features.
    """
    splits = get_loso_splits(data_dir)
    all_files = set()
    for fold in splits:
        all_files.update(fold["train_files"])
        all_files.update(fold["test_files"])
    all_files = sorted(list(all_files))

    records = []
    for fpath in all_files:
        df_clean = load_recording_df(fpath)
        label = extract_label(fpath)
        subject = extract_subject(fpath)
        win_feats = extract_window_physical_features(df_clean)

        if len(win_feats) == 0:
            records.append({
                "file": fpath.name,
                "path": str(fpath.resolve()),
                "subject": subject,
                "label": label,
                "drop_z": 0.0,
                "delta_z": 0.0,
                "max_abs_v": 0.0,
                "min_v": 0.0,
                "final_z": 1.0,
                "xy_span": 0.0,
                "max_z": 0.0,
                "num_windows": 0
            })
            continue

        df_w = pd.DataFrame(win_feats)
        records.append({
            "file": fpath.name,
            "path": str(fpath.resolve()),
            "subject": subject,
            "label": label,
            "drop_z": float(df_w["drop_z"].max()),
            "delta_z": float(df_w["delta_z"].max()),
            "max_abs_v": float(df_w["max_abs_v"].max()),
            "min_v": float(df_w["min_v"].min()),
            "final_z": float(df_w["final_z"].min()),
            "xy_span": float(df_w["xy_span"].max()),
            "max_z": float(df_w["max_z"].max()),
            "num_windows": len(df_w)
        })

    return pd.DataFrame(records)


class RuleBasedFallDetector:
    """
    Non-AI Physical Heuristic Detector for mmWave Radar Fall Detection.
    Calibrates thresholds on training subjects and evaluates on held-out test subjects.
    """

    def __init__(self, rule_type: str = "drop_z"):
        """
        Args:
            rule_type (str): Rule strategy identifier. Options:
                - 'drop_z': Single height drop threshold (drop_z >= thr)
                - 'max_abs_v': Single Doppler speed threshold (max_abs_v >= thr)
                - 'dual': Dual heuristic (drop_z >= thr_z AND max_abs_v >= thr_v)
                - 'multi': Multi-criteria (drop_z >= thr_z AND max_abs_v >= thr_v AND final_z <= thr_fz)
                - 'composite': Physics composite index score (drop_z + w_v*max_abs_v - w_fz*final_z >= thr)
        """
        valid_rules = {"drop_z", "max_abs_v", "dual", "multi", "composite"}
        if rule_type not in valid_rules:
            raise ValueError(f"Invalid rule_type '{rule_type}'. Choose from {valid_rules}")
        self.rule_type = rule_type
        self.calibrated_params: Dict[str, float] = {}

    def fit(self, df_train: pd.DataFrame) -> Dict[str, float]:
        """
        Calibrates threshold parameters strictly on training set to maximize F1-score.

        Args:
            df_train (pd.DataFrame): Training subset with features and binary 'label' column.

        Returns:
            Dict[str, float]: Calibrated parameter dictionary.
        """
        y_tr = df_train["label"].values
        best_f1 = -1.0
        best_params = {}

        if self.rule_type == "drop_z":
            for thr in np.linspace(0.05, 1.5, 146):
                pred = (df_train["drop_z"].values >= thr).astype(int)
                f1 = f1_score(y_tr, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"thr": float(thr)}

        elif self.rule_type == "max_abs_v":
            for thr in np.linspace(0.1, 3.0, 146):
                pred = (df_train["max_abs_v"].values >= thr).astype(int)
                f1 = f1_score(y_tr, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"thr": float(thr)}

        elif self.rule_type == "dual":
            for thr_z in np.linspace(0.05, 1.2, 24):
                for thr_v in np.linspace(0.2, 2.5, 24):
                    pred = ((df_train["drop_z"].values >= thr_z) & (df_train["max_abs_v"].values >= thr_v)).astype(int)
                    f1 = f1_score(y_tr, pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {"thr_z": float(thr_z), "thr_v": float(thr_v)}

        elif self.rule_type == "multi":
            for thr_z in np.linspace(0.05, 1.0, 20):
                for thr_v in np.linspace(0.2, 2.2, 20):
                    for thr_fz in np.linspace(0.1, 1.2, 12):
                        pred = ((df_train["drop_z"].values >= thr_z) & (df_train["max_abs_v"].values >= thr_v) & (df_train["final_z"].values <= thr_fz)).astype(int)
                        f1 = f1_score(y_tr, pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = {"thr_z": float(thr_z), "thr_v": float(thr_v), "thr_fz": float(thr_fz)}

        elif self.rule_type == "composite":
            for w_v in np.linspace(0.1, 1.0, 10):
                for w_fz in np.linspace(0.1, 1.0, 10):
                    score_tr = df_train["drop_z"].values + w_v * df_train["max_abs_v"].values - w_fz * df_train["final_z"].values
                    for thr in np.linspace(score_tr.min(), score_tr.max(), 50):
                        pred = (score_tr >= thr).astype(int)
                        f1 = f1_score(y_tr, pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = {"w_v": float(w_v), "w_fz": float(w_fz), "thr": float(thr)}

        self.calibrated_params = best_params
        return best_params

    def predict_score(self, df_eval: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts binary fall labels and continuous decision confidence scores on evaluation set.

        Args:
            df_eval (pd.DataFrame): Evaluation subset with physical features.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (binary predictions [0/1], continuous scores)
        """
        if not self.calibrated_params:
            raise RuntimeError("Detector must be fitted via fit() before calling predict_score()")

        if self.rule_type == "drop_z":
            thr = self.calibrated_params["thr"]
            scores = df_eval["drop_z"].values
            preds = (scores >= thr).astype(int)

        elif self.rule_type == "max_abs_v":
            thr = self.calibrated_params["thr"]
            scores = df_eval["max_abs_v"].values
            preds = (scores >= thr).astype(int)

        elif self.rule_type == "dual":
            thr_z = self.calibrated_params["thr_z"]
            thr_v = self.calibrated_params["thr_v"]
            scores = df_eval["drop_z"].values + 0.5 * df_eval["max_abs_v"].values
            preds = ((df_eval["drop_z"].values >= thr_z) & (df_eval["max_abs_v"].values >= thr_v)).astype(int)

        elif self.rule_type == "multi":
            thr_z = self.calibrated_params["thr_z"]
            thr_v = self.calibrated_params["thr_v"]
            thr_fz = self.calibrated_params["thr_fz"]
            scores = df_eval["drop_z"].values + 0.5 * df_eval["max_abs_v"].values - 0.5 * df_eval["final_z"].values
            preds = ((df_eval["drop_z"].values >= thr_z) & (df_eval["max_abs_v"].values >= thr_v) & (df_eval["final_z"].values <= thr_fz)).astype(int)

        elif self.rule_type == "composite":
            w_v = self.calibrated_params["w_v"]
            w_fz = self.calibrated_params["w_fz"]
            thr = self.calibrated_params["thr"]
            scores = df_eval["drop_z"].values + w_v * df_eval["max_abs_v"].values - w_fz * df_eval["final_z"].values
            preds = (scores >= thr).astype(int)

        return preds, scores


def evaluate_loso_rule_baseline(
    df_features: pd.DataFrame,
    rule_type: str = "drop_z",
    data_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes complete 3-Fold Leave-One-Subject-Out (LOSO) evaluation for a specified rule detector.

    Args:
        df_features (pd.DataFrame): Aggregated feature DataFrame from extract_dataset_rule_features().
        rule_type (str): Rule strategy identifier.
        data_dir (Optional[Path]): Dataset root directory.

    Returns:
        Dict[str, Any]: Consolidated Out-of-Fold (OOF) evaluation metrics and per-fold results.
    """
    splits = get_loso_splits(data_dir)
    oof_y_true = []
    oof_y_pred = []
    oof_y_score = []
    fold_details = []

    for fold_idx, fold in enumerate(splits):
        test_subj = fold["test_subject"]
        train_paths = [str(Path(p).resolve()) for p in fold["train_files"]]
        test_paths = [str(Path(p).resolve()) for p in fold["test_files"]]

        df_tr = df_features[df_features["path"].isin(train_paths)].copy()
        df_te = df_features[df_features["path"].isin(test_paths)].copy()

        detector = RuleBasedFallDetector(rule_type=rule_type)
        calibrated_params = detector.fit(df_tr)
        preds, scores = detector.predict_score(df_te)

        y_te = df_te["label"].values
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, zero_division=0)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)

        fold_details.append({
            "fold": fold_idx,
            "test_subject": test_subj,
            "calibrated_params": calibrated_params,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "macro_f1": float(f1)
        })

        oof_y_true.extend(y_te)
        oof_y_pred.extend(preds)
        oof_y_score.extend(scores)

    acc = accuracy_score(oof_y_true, oof_y_pred)
    prec = precision_score(oof_y_true, oof_y_pred, zero_division=0)
    rec = recall_score(oof_y_true, oof_y_pred, zero_division=0)
    f1 = f1_score(oof_y_true, oof_y_pred, zero_division=0)
    auc = roc_auc_score(oof_y_true, oof_y_score) if len(set(oof_y_true)) > 1 else 0.5

    cm = confusion_matrix(oof_y_true, oof_y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "rule_type": rule_type,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "macro_f1": float(f1),
        "roc_auc": float(auc),
        "fpr": float(fpr),
        "confusion_matrix": cm.tolist(),
        "fold_details": fold_details,
        "oof_y_true": oof_y_true,
        "oof_y_pred": oof_y_pred,
        "oof_y_score": oof_y_score
    }
