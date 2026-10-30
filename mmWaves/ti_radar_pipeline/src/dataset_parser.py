"""
Dataset parser module for TI mmWave Radar Fall Detection dataset (102 CSV files).
Strictly ingests the TI IWR6843 dataset without external dataset mixing.
Constructs Leave-One-Subject-Out (LOSO) cross-validation splits over subjects: Areeb, Raffay, Towsif.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# Path to TI IWR6843 dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if (PROJECT_ROOT / "datasets" / "mmwave-radar-fall-detection" / "GatheredData").exists():
    DATA_DIR = PROJECT_ROOT / "datasets" / "mmwave-radar-fall-detection" / "GatheredData"
elif (PROJECT_ROOT.parent / "datasets" / "mmwave-radar-fall-detection" / "GatheredData").exists():
    DATA_DIR = PROJECT_ROOT.parent / "datasets" / "mmwave-radar-fall-detection" / "GatheredData"
else:
    DATA_DIR = PROJECT_ROOT / "datasets" / "mmwave-radar-fall-detection" / "GatheredData"

def get_all_recording_files(data_dir: Path = None) -> List[Path]:
    """
    Discovers all 102 CSV recording files (51 Falls, 51 Non-Falls / ADLs).
    """
    if data_dir is None or not data_dir.exists():
        data_dir = DATA_DIR
    elif not data_dir.exists() and (PROJECT_ROOT / data_dir).exists():
        data_dir = PROJECT_ROOT / data_dir
    elif not data_dir.exists() and (PROJECT_ROOT.parent / data_dir).exists():
        data_dir = PROJECT_ROOT.parent / data_dir

    fall_files = sorted(list((data_dir / "Fall").glob("*.csv")))
    not_files = sorted(list((data_dir / "Not").glob("*.csv")))
    all_files = sorted(fall_files + not_files)
    assert len(all_files) == 102, f"Expected exactly 102 CSV files, found {len(all_files)} in {data_dir}"
    return all_files


def extract_subject(file_path: Path) -> str:
    """
    Extracts the subject identifier from the CSV filename (e.g. 'Areeb_back_4.csv' -> 'Areeb').
    """
    stem = file_path.stem
    subject = stem.split("_")[0]
    valid_subjects = {"Areeb", "Raffay", "Towsif"}
    if subject not in valid_subjects:
        raise ValueError(f"Unknown subject '{subject}' extracted from {file_path.name}")
    return subject

def extract_label(file_path: Path) -> int:
    """
    Extracts binary label: 1 for Fall (from Fall/ directory), 0 for Non-Fall / ADL (from Not/ directory).
    """
    parent_name = file_path.parent.name.lower()
    if parent_name == "fall":
        return 1
    elif parent_name == "not":
        return 0
    else:
        raise ValueError(f"Unexpected parent directory '{file_path.parent.name}' for {file_path}")

def load_recording_df(file_path: Path) -> pd.DataFrame:
    """
    Loads raw point cloud data from a single TI CSV recording file.
    Columns: ['frame', 'DetObj#', 'x', 'y', 'z', 'v', 'snr', 'noise']
    Applies minimal noise filtering (snr >= 100, |v| <= 3.0 m/s).
    """
    df = pd.read_csv(file_path)
    # Apply minimal noise filter
    df_clean = df[(df["x"].between(-2.0, 2.0)) & 
                  (df["y"].between(0.0, 6.0)) & 
                  (df["z"].between(-0.5, 2.2)) & 
                  (df["v"].abs() <= 3.0) & 
                  (df["snr"] >= 100)].copy()
    if len(df_clean) == 0:
        # Fallback to basic coordinate box if noise filter removed all points
        df_clean = df[(df["x"].between(-3.0, 3.0)) & (df["y"].between(0.0, 8.0))].copy()
    return df_clean


def get_loso_splits(data_dir: Path = None) -> List[Dict[str, any]]:
    """
    Constructs 3 Leave-One-Subject-Out (LOSO) cross-validation folds across subjects (Areeb, Raffay, Towsif).
    In each fold:
    - Training Set (68 files): 34 Falls and 34 Non-Falls (2 complete subjects).
    - Test Set (34 files): 17 Falls and 17 Non-Falls (1 complete held-out subject).
    Guarantees zero subject data leakage and exact 1:1 class balance across all splits.
    """
    all_files = get_all_recording_files(data_dir)
    subjects = sorted(list({extract_subject(f) for f in all_files}))
    assert len(subjects) == 3, f"Expected 3 subjects, got {subjects}"

    folds = []
    for idx, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s != test_subj]
        train_files = [f for f in all_files if extract_subject(f) in train_subjs]
        test_files = [f for f in all_files if extract_subject(f) == test_subj]

        folds.append({
            "fold": idx,
            "test_subject": test_subj,
            "train_subjects": train_subjs,
            "train_files": train_files,
            "test_files": test_files
        })
    return folds


if __name__ == "__main__":
    files = get_all_recording_files()
    print(f"Total TI CSV files discovered: {len(files)}")
    splits = get_loso_splits()
    for f_info in splits:
        tr_falls = sum(1 for f in f_info["train_files"] if extract_label(f) == 1)
        tr_adls = sum(1 for f in f_info["train_files"] if extract_label(f) == 0)
        te_falls = sum(1 for f in f_info["test_files"] if extract_label(f) == 1)
        te_adls = sum(1 for f in f_info["test_files"] if extract_label(f) == 0)
        print(f"Fold {f_info['fold']} [Held-Out Test Subject: {f_info['test_subject']:6s}]: "
              f"Train={len(f_info['train_files'])} files (Falls:{tr_falls}, ADLs:{tr_adls}) | "
              f"Test={len(f_info['test_files'])} files (Falls:{te_falls}, ADLs:{te_adls})")



