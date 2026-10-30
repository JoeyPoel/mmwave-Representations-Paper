"""
Transforms module for mmWave Radar Fall Detection.
Implements target-relative spatial centering, leakage-free feature scaling,
and representation generation for:
1. Representation 1: Kinematic / Doppler-Time Map (ResNet-18)
2. Representation 2: Spatial / 2D Orthogonal Projections (ResNet-18)
3. Representation 3: Geometric / Native 3D Point Tensor (PointNet++)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def center_target_relative(df_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Performs target-relative spatial centering frame-by-frame:
    x_c = median(x), y_c = median(y)
    delta_x = x - x_c, delta_y = y - y_c
    Preserves vertical height z to track posture descent while eliminating absolute X, Y room position.
    """
    if len(df_frame) == 0:
        df_out = df_frame.copy()
        df_out["delta_x"] = 0.0
        df_out["delta_y"] = 0.0
        return df_out

    x_c = df_frame["x"].median()
    y_c = df_frame["y"].median()

    df_out = df_frame.copy()
    df_out["delta_x"] = df_frame["x"] - x_c
    df_out["delta_y"] = df_frame["y"] - y_c
    return df_out

def uniform_oversample_frame_points(points: np.ndarray, target_n: int = 32, rng=None) -> np.ndarray:
    """
    Standardizes point count per frame to target_n points using variance-preserving oversampling / sampling.
    Preserves spatial covariance of point attributes [delta_x, delta_y, z, v].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    M = len(points)
    if M == 0:
        return np.zeros((target_n, points.shape[1] if points.ndim > 1 else 4), dtype=np.float32)
    if M < target_n:
        idx = rng.choice(M, size=target_n, replace=True)
    else:
        idx = rng.choice(M, size=target_n, replace=False)
    return points[idx].astype(np.float32)

def extract_raw_windows_from_df(df: pd.DataFrame, window_size_frames: int = 10, stride: int = 5, min_pts_per_win: int = 5) -> List[np.ndarray]:
    """
    Extracts sliding temporal windows (10 frames each) from a recording DataFrame.
    Precomputes frame grouping and target-relative median centering for high-speed performance.
    Returns list of window arrays, each of shape (10, 32, 4) with columns [delta_x, delta_y, z, v].
    """
    if len(df) == 0:
        return []

    # Pre-group frames once by frame ID and compute median centering
    frame_dict = {}
    for f_id, df_f in df.groupby("frame"):
        x = df_f["x"].values
        y = df_f["y"].values
        z = df_f["z"].values
        v = df_f["v"].values
        if len(x) > 0:
            x_c = np.median(x)
            y_c = np.median(y)
            dx = x - x_c
            dy = y - y_c
            frame_dict[f_id] = np.column_stack([dx, dy, z, v]).astype(np.float32)
        else:
            frame_dict[f_id] = np.zeros((0, 4), dtype=np.float32)

    if len(frame_dict) == 0:
        return []

    max_frame = int(max(frame_dict.keys()))
    rng = np.random.default_rng(42)
    windows = []

    for start_f in range(0, max_frame - (window_size_frames - 1) + 1, stride):
        end_f = start_f + (window_size_frames - 1)
        win_pts_count = sum(len(frame_dict.get(fi, np.zeros((0, 4)))) for fi in range(start_f, end_f + 1))
        if win_pts_count < min_pts_per_win:
            continue

        win_frames = []
        for fi in range(start_f, end_f + 1):
            pts = frame_dict.get(fi, np.zeros((0, 4), dtype=np.float32))
            pts_sampled = uniform_oversample_frame_points(pts, target_n=32, rng=rng)
            win_frames.append(pts_sampled)

        windows.append(np.array(win_frames, dtype=np.float32)) # (10, 32, 4)

    return windows


class FoldScaler:
    """
    Leakage-free feature scaler computed strictly on active training fold data.
    """
    def __init__(self):
        self.mean_v = 0.0
        self.std_v = 1.0
        self.mean_z = 0.0
        self.std_z = 1.0

    def fit(self, X_train_raw: np.ndarray):
        """
        X_train_raw shape: (N, 10, 32, 4) with [delta_x, delta_y, z, v]
        """
        zs = X_train_raw[:, :, :, 2].ravel()
        vs = X_train_raw[:, :, :, 3].ravel()
        self.mean_z = float(np.mean(zs))
        self.std_z = float(np.std(zs)) + 1e-6
        self.mean_v = float(np.mean(vs))
        self.std_v = float(np.std(vs)) + 1e-6

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Scales z and v features while keeping delta_x, delta_y centered.
        """
        X_out = X_raw.copy()
        X_out[:, :, :, 2] = (X_out[:, :, :, 2] - self.mean_z) / self.std_z
        X_out[:, :, :, 3] = (X_out[:, :, :, 3] - self.mean_v) / self.std_v
        return X_out


def transform_rep1_doppler_time(X_windows: np.ndarray, grid_size: int = 32) -> np.ndarray:
    """
    Representation 1: Kinematic / Micro-Doppler Time Map.
    Discretizes Doppler velocity v across 10 temporal frames into 2D intensity grid (32x32).
    Input X_windows shape: (N, 10, 32, 4) where col 3 is v.
    Returns: (N, 3, 32, 32) for ResNet-18 input.
    """
    X_windows = np.asarray(X_windows)
    N, T, P, C = X_windows.shape
    v_bins = np.linspace(-3.0, 3.0, grid_size + 1)
    t_bins = np.linspace(0, T, grid_size + 1)
    t_vals = np.repeat(np.arange(T) + 0.5, P)

    rep1_images = []
    for i in range(N):
        v_vals = X_windows[i, :, :, 3].ravel()
        img, _, _ = np.histogram2d(v_vals, t_vals, bins=[v_bins, t_bins])
        if img.max() > 0:
            img = img / img.max()
        img_3ch = np.stack([img, img, img], axis=0) # (3, 32, 32)
        rep1_images.append(img_3ch)

    return np.array(rep1_images, dtype=np.float32)


def transform_rep2_orthogonal_projections(X_windows: np.ndarray, grid_size: int = 32) -> np.ndarray:
    """
    Representation 2: Spatial / 2D Orthogonal Projections.
    Accumulates relative coordinates into orthogonal 2D occupancy grids:
    Channel 0: Elevation Grid (delta_x vs z)
    Channel 1: Ground Grid (delta_x vs delta_y)
    Channel 2: Doppler-Height Grid (z vs v)
    Input X_windows shape: (N, 10, 32, 4)
    Returns: (N, 3, 32, 32) for ResNet-18 input.
    """
    X_windows = np.asarray(X_windows)
    N, T, P, C = X_windows.shape
    x_bins = np.linspace(-1.5, 1.5, grid_size + 1)
    y_bins = np.linspace(-1.5, 1.5, grid_size + 1)
    z_bins = np.linspace(-0.5, 2.2, grid_size + 1)
    v_bins = np.linspace(-3.0, 3.0, grid_size + 1)

    rep2_images = []
    for i in range(N):
        win = X_windows[i].reshape(-1, 4) # (320, 4)
        dx = win[:, 0]
        dy = win[:, 1]
        z = win[:, 2]
        v = win[:, 3]

        ch0, _, _ = np.histogram2d(dx, z, bins=[x_bins, z_bins])
        ch1, _, _ = np.histogram2d(dx, dy, bins=[x_bins, y_bins])
        ch2, _, _ = np.histogram2d(z, v, bins=[z_bins, v_bins])

        # Normalize each channel independently
        for ch in [ch0, ch1, ch2]:
            cmax = ch.max()
            if cmax > 0:
                ch /= cmax

        img_3ch = np.stack([ch0, ch1, ch2], axis=0) # (3, 32, 32)
        rep2_images.append(img_3ch)

    return np.array(rep2_images, dtype=np.float32)


def transform_rep3_pointnet(X_windows: np.ndarray) -> np.ndarray:
    """
    Representation 3: Geometric / Native 3D Point Tensor.
    Formats continuous point vectors [delta_x, delta_y, z, v] per frame.
    Applies height delta computation z - centroid_z for explicit posture tracking.
    Input X_windows shape: (N, 10, 32, 4)
    Returns: (N, 5, 320) for PointNet++ input.
    """
    X_windows = np.asarray(X_windows)
    N, T, P, C = X_windows.shape
    rep3_tensors = []
    for i in range(N):
        win = X_windows[i] # (10, 32, 4)
        win_feat = []
        for t_idx in range(T):
            frame_pts = win[t_idx] # (32, 4)
            cz = frame_pts[:, 2].mean() if len(frame_pts) > 0 else 0.0
            dz = frame_pts[:, 2] - cz
            # Combine [delta_x, delta_y, z, v, delta_z] -> (32, 5)
            pts_5d = np.column_stack([frame_pts, dz])
            win_feat.append(pts_5d)
            
        win_arr = np.concatenate(win_feat, axis=0) # (320, 5)
        win_tensor = win_arr.T # (5, 320)
        rep3_tensors.append(win_tensor)

    return np.array(rep3_tensors, dtype=np.float32)


def transform_representation(X_windows: np.ndarray, rep_key: str) -> np.ndarray:
    """
    Dynamic representation selector accepting key names:
    - 'rep1_doppler' / 'rep1': Representation 1 (Micro-Doppler Time Map)
    - 'rep2_projections' / 'rep2': Representation 2 (Spatial 2D Orthogonal Projections)
    - 'rep3_pointset' / 'rep3': Representation 3 (Native 3D Point Tensor)
    """
    key_clean = str(rep_key).lower().strip()
    if key_clean in ("rep1_doppler", "rep1"):
        return transform_rep1_doppler_time(X_windows)
    elif key_clean in ("rep2_projections", "rep2"):
        return transform_rep2_orthogonal_projections(X_windows)
    elif key_clean in ("rep3_pointset", "rep3"):
        return transform_rep3_pointnet(X_windows)
    else:
        raise ValueError(f"Unknown representation key '{rep_key}'. Expected one of: 'rep1_doppler', 'rep2_projections', 'rep3_pointset'")

