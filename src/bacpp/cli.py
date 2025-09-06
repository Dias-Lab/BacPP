from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from importlib import resources as pkg_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import matplotlib.patches as mpatches
import sys
import plotly.graph_objects as go
import plotly.io as pio
import os
import subprocess
import shlex
import shutil

# Fixed constants for calculating spectrum amplitute (SA) - Arakawa et al., 2009
K3 = 600.0
K4 = 40.0
ALPHA = 0.4
# ---------- DATA PREPROCESSING ----------
def read_first_fasta_sequence(path: Path) -> str:
    """Read the first sequence from a FASTA/FA/FNA file."""
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        in_seq = False
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                if in_seq: 
                    break
                in_seq = True
                continue
            if in_seq:
                seq_parts.append(line.strip())
    return "".join(seq_parts)

def gc_skew_vectorized(seq: str, num_windows: int = 4096) -> np.ndarray:
    """
    Fully vectorized GC skew over num_windows:
      window_size = floor(len(seq)/num_windows), last window takes remainder.
      skew = (C - G) / (C + G), 0 if denom==0.
    """
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")

    s_bytes = seq.upper().encode("ascii", errors="ignore")
    n = len(s_bytes)
    if n == 0:
        return np.zeros(num_windows, dtype=float)

    win = n // num_windows
    starts = win * np.arange(num_windows, dtype=np.int64)
    ends = np.minimum(starts + win, n)
    ends[-1] = n

    arr = np.frombuffer(s_bytes, dtype="S1")
    is_c = (arr == b"C")
    is_g = (arr == b"G")

    c_cum = np.concatenate(([0], np.cumsum(is_c, dtype=np.int64)))
    g_cum = np.concatenate(([0], np.cumsum(is_g, dtype=np.int64)))

    c_counts = c_cum[ends] - c_cum[starts]
    g_counts = g_cum[ends] - g_cum[starts]
    denom = c_counts + g_counts

    skews = np.zeros(num_windows, dtype=float)
    nz = denom != 0
    skews[nz] = (c_counts[nz] - g_counts[nz]) / denom[nz]
    return skews

def at_skew_vectorized(seq: str, num_windows: int = 4096) -> np.ndarray:
    
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")

    s_bytes = seq.upper().encode("ascii", errors="ignore")
    n = len(s_bytes)
    if n == 0:
        return np.zeros(num_windows, dtype=float)

    win = n // num_windows
    starts = win * np.arange(num_windows, dtype=np.int64)
    ends = np.minimum(starts + win, n)
    ends[-1] = n

    arr = np.frombuffer(s_bytes, dtype="S1")
    is_a = (arr == b"A")
    is_t = (arr == b"T")

    a_cum = np.concatenate(([0], np.cumsum(is_a, dtype=np.int64)))
    t_cum = np.concatenate(([0], np.cumsum(is_t, dtype=np.int64)))

    a_counts = a_cum[ends] - a_cum[starts]
    t_counts = t_cum[ends] - t_cum[starts]
    denom = a_counts + t_counts

    skews = np.zeros(num_windows, dtype=float)
    nz = denom != 0
    skews[nz] = (a_counts[nz] - t_counts[nz]) / denom[nz]
    return skews

def gcsi_features_from_gcskew(
    gc_skew: np.ndarray,
    k3: float = 600.0,
    k4: float = 40.0,
    alpha: float = 0.4,
) -> Tuple[float, float, float, int]:
    """
    FFT power spectrum features + cumulative GC skew geometry.
      - sr  = power at index 1 / mean(power at indices 2..N-1)
      - sa  = k4 * (k3 * power_at_1Hz) ** alpha
      - peak.dist  = max(cumsum(gc_skew)) - min(cumsum(gc_skew))
      - index.dist = circular distance between argmax and argmin of cumsum
    """
    x = np.asarray(gc_skew, dtype=float)
    N = x.size

    ps = np.abs(np.fft.fft(x)) ** 2
    power_at_1Hz = float(ps[1]) if N > 1 else 0.0
    avg_other = float(ps[2:N].mean()) if N > 2 else 0.0

    gc_sr = (power_at_1Hz / avg_other) if avg_other > 0 else 0.0
    gc_sa = k4 * (k3 * power_at_1Hz) ** alpha

    cum = np.cumsum(x)
    gc_peak_dist = float(cum.max() - cum.min())

    argmax = int(np.argmax(cum))
    argmin = int(np.argmin(cum))
    idx_gap = abs(argmax - argmin)
    circ_gap = min(idx_gap, N - idx_gap)
    gc_index_dist = int(circ_gap)

    return gc_sr, gc_sa, gc_peak_dist, gc_index_dist

def atsi_features_from_atskew(
    at_skew: np.ndarray,
    k3: float = 600.0,
    k4: float = 40.0,
    alpha: float = 0.4,
) -> Tuple[float, float, float, int]:
    
    x = np.asarray(at_skew, dtype=float)
    N = x.size

    ps = np.abs(np.fft.fft(x)) ** 2
    power_at_1Hz = float(ps[1]) if N > 1 else 0.0
    avg_other = float(ps[2:N].mean()) if N > 2 else 0.0

    at_sr = (power_at_1Hz / avg_other) if avg_other > 0 else 0.0
    at_sa = k4 * (k3 * power_at_1Hz) ** alpha

    cum = np.cumsum(x)
    at_peak_dist = float(cum.max() - cum.min())

    argmax = int(np.argmax(cum))
    argmin = int(np.argmin(cum))
    idx_gap = abs(argmax - argmin)
    circ_gap = min(idx_gap, N - idx_gap)
    at_index_dist = int(circ_gap)

    return at_sr, at_sa, at_peak_dist, at_index_dist

## ---------- Interaction terms ----------
def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    mask = (b != 0)
    out[mask] = a[mask] / b[mask]
    return out

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms using legacy naming:
      - Base GC cols: sr.gc, sa.gc, peak.dist.gc, index.dist.gc
      - Base AT cols: sr.at, sa.at, peak.dist.at, index.dist.at
      - Multiplication: A X B  (e.g., srXsa.gc)
      - Division:       A I B  (e.g., srIsa.gc)  (safe; NaN if denom == 0)
    """
    # Pull base features in legacy naming
    sr_gc = df["sr.gc"].to_numpy(float)
    sa_gc = df["sa.gc"].to_numpy(float)
    pk_gc = df["peak.dist.gc"].to_numpy(float)
    idx_gc = df["index.dist.gc"].to_numpy(float)

    sr_at = df["sr.at"].to_numpy(float)
    sa_at = df["sa.at"].to_numpy(float)
    pk_at = df["peak.dist.at"].to_numpy(float)
    idx_at = df["index.dist.at"].to_numpy(float)

    # ---------- GC group interactions ----------
    # Multiplications
    df["srXsa.gc"] = sr_gc * sa_gc
    df["srXpeak.dist.gc"] = sr_gc * pk_gc
    df["srXindex.dist.gc"] = sr_gc * idx_gc
    df["saXpeak.dist.gc"] = sa_gc * pk_gc
    df["saXindex.dist.gc"] = sa_gc * idx_gc
    df["peak.distXindex.dist.gc"] = pk_gc * idx_gc

    # Divisions (A I B == A / B)
    df["srIsa.gc"] = _safe_div(sr_gc,  sa_gc)
    df["srIpeak.dist.gc"] = _safe_div(sr_gc,  pk_gc)
    df["srIindex.dist.gc"] = _safe_div(sr_gc,  idx_gc)

    df["saIsr.gc"] = _safe_div(sa_gc,  sr_gc)
    df["saIpeak.dist.gc"] = _safe_div(sa_gc,  pk_gc)
    df["saIindex.dist.gc"] = _safe_div(sa_gc,  idx_gc)

    df["peak.distIsr.gc"] = _safe_div(pk_gc,  sr_gc)
    df["peak.distIsa.gc"] = _safe_div(pk_gc,  sa_gc)
    df["peak.distIindex.dist.gc"] = _safe_div(pk_gc,  idx_gc)

    df["index.distIsr.gc"] = _safe_div(idx_gc, sr_gc)
    df["index.distIsa.gc"] = _safe_div(idx_gc, sa_gc)
    df["index.distIpeak.dist.gc"] = _safe_div(idx_gc, pk_gc)

    # ---------- AT group interactions ----------
    # Multiplications
    df["srXsa.at"] = sr_at * sa_at
    df["srXpeak.dist.at"] = sr_at * pk_at
    df["srXindex.dist.at"] = sr_at * idx_at
    df["saXpeak.dist.at"] = sa_at * pk_at
    df["saXindex.dist.at"] = sa_at * idx_at
    df["peak.distXindex.dist.at"] = pk_at * idx_at

    # Divisions (A I B == A / B)
    df["srIsa.at"] = _safe_div(sr_at,  sa_at)
    df["srIpeak.dist.at"] = _safe_div(sr_at,  pk_at)
    df["srIindex.dist.at"] = _safe_div(sr_at,  idx_at)

    df["saIsr.at"] = _safe_div(sa_at,  sr_at)
    df["saIpeak.dist.at"] = _safe_div(sa_at,  pk_at)
    df["saIindex.dist.at"] = _safe_div(sa_at,  idx_at)

    df["peak.distIsr.at"] = _safe_div(pk_at,  sr_at)
    df["peak.distIsa.at"] = _safe_div(pk_at,  sa_at)
    df["peak.distIindex.dist.at"] = _safe_div(pk_at,  idx_at)

    df["index.distIsr.at"] = _safe_div(idx_at, sr_at)
    df["index.distIsa.at"] = _safe_div(idx_at, sa_at)
    df["index.distIpeak.dist.at"] = _safe_div(idx_at, pk_at)

    return df

## ---------- Batch runner ----------
def run_folder(
    folder: str | Path,
    num_windows: int = 4096,
    patterns: Tuple[str, ...] = ("*.fasta", "*.fa", "*.fna"),
    add_interaction_terms: bool = True,
    cpus: int = 1,
) -> pd.DataFrame:
    """
    Scan folder for FASTA/FA/FNA, compute features, and return a DataFrame:
      columns = [file, sr, sa, peak.dist, index.dist, ...interactions*]
    """
    folder = Path(folder)
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(folder.glob(pat)))
    
    if cpus is None or cpus < 1:
        cpus = 1
    
    if cpus == 1 or len(files) <= 1:
        rows = [_compute_features_for_file(fp, num_windows) for fp in files]
    else:
        # Preserve input order by using ex.map with itertools.repeat
        with ProcessPoolExecutor(max_workers=cpus) as ex:
            rows = list(ex.map(_compute_features_for_file, files, repeat(num_windows), chunksize=1))

    df = pd.DataFrame(rows)
    if add_interaction_terms and not df.empty:
        df = add_interactions(df)
        # If you want a specific order, use iloc with indices
        new_column_order = [
            0, 1, 2, 3, 4, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
            5, 6, 7, 8, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
        ]
        df = df.iloc[:, new_column_order]
    return df

## ---------- Batch runner ----------
def compute_window_skews(seq: str, num_windows: int = 4096):
    """Return (gc_skew, at_skew, cum_gc, cum_at) as NumPy arrays of length num_windows."""
    s_bytes = seq.upper().encode("ascii", errors="ignore")
    n = len(s_bytes)
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")
    if n == 0:
        z = np.zeros(num_windows, dtype=float)
        return z, z, z, z

    win = n // num_windows
    starts = win * np.arange(num_windows, dtype=np.int64)
    ends = np.minimum(starts + win, n)
    ends[-1] = n

    arr = np.frombuffer(s_bytes, dtype="S1")
    is_a, is_t, is_g, is_c = (arr == b"A"), (arr == b"T"), (arr == b"G"), (arr == b"C")

    def rngsum(mask):
        cum = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
        return cum[ends] - cum[starts]

    a, t, g, c = rngsum(is_a), rngsum(is_t), rngsum(is_g), rngsum(is_c)
    at_den = a + t
    gc_den = g + c

    at_skew = np.zeros_like(at_den, dtype=float)
    gc_skew = np.zeros_like(gc_den, dtype=float)
    nz_at = at_den != 0
    nz_gc = gc_den != 0
    at_skew[nz_at] = (a[nz_at] - t[nz_at]) / at_den[nz_at]
    gc_skew[nz_gc] = (c[nz_gc] - g[nz_gc]) / gc_den[nz_gc]

    return gc_skew, at_skew, np.cumsum(gc_skew), np.cumsum(at_skew)

def plot_linear_skews(gc_skew: np.ndarray, at_skew: np.ndarray, title: str, out_path: Path):
    """Linear plot of cumulative GC and AT skew vs window index."""
    x = np.arange(gc_skew.size)
    plt.figure(figsize=(9, 3))
    plt.plot(x, np.cumsum(gc_skew), label="Cumulative GC skew")
    plt.plot(x, np.cumsum(at_skew), label="Cumulative AT skew")
    plt.xlabel("Window index"); plt.ylabel("Cumulative skew"); plt.title(title)
    plt.legend(loc="best"); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_circular_skews(gc_skew: np.ndarray, at_skew: np.ndarray, title:str, out_path):
    N = gc_skew.size
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)

    # Manually set up the max spike length for aesthetic purpose
    max_len = 0.10

    # Robust per-series normalization
    gc_mag = np.percentile(np.abs(gc_skew), 99);  gc_mag = gc_mag if gc_mag else 1.0
    at_mag = np.percentile(np.abs(at_skew), 99);  at_mag = at_mag if at_mag else 1.0

    gc_spikes = np.clip((gc_skew / gc_mag) * max_len, -max_len, max_len)
    at_spikes = np.clip((at_skew / at_mag) * max_len, -max_len, max_len)

    gc_base_radius = 2.0
    at_base_radius = 1.8

    # ---- Plotting parameters ----
    r_pad = 0.05
    r_max = max(gc_base_radius, at_base_radius) + max_len + r_pad
    r_min = 0.0 
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    # Apply fixed limits before plotting spikes
    ax.set_ylim(r_min, r_max)

    # Baselines
    ax.plot(theta, np.full(N, gc_base_radius), linewidth=1, alpha=0.6)
    ax.plot(theta, np.full(N, at_base_radius), linewidth=1, alpha=0.6)

    # GC spikes: + outward (orange), − inward (purple)
    gc_pos = gc_spikes >= 0; gc_neg = ~gc_pos
    if np.any(gc_pos):
        ax.vlines(theta[gc_pos], gc_base_radius, gc_base_radius + gc_spikes[gc_pos],
                  colors="orange", linewidth=0.6)
    if np.any(gc_neg):
        ax.vlines(theta[gc_neg], gc_base_radius + gc_spikes[gc_neg], gc_base_radius,
                  colors="purple", linewidth=0.6)

    # AT spikes: + outward (olive), − inward (gray)
    at_pos = at_spikes >= 0; at_neg = ~at_pos
    if np.any(at_pos):
        ax.vlines(theta[at_pos], at_base_radius, at_base_radius + at_spikes[at_pos],
                  colors="olive", linewidth=0.6)
    if np.any(at_neg):
        ax.vlines(theta[at_neg], at_base_radius + at_spikes[at_neg], at_base_radius,
                  colors="gray", linewidth=0.6)

    ax.set_rticks([]); ax.set_xticks([]); ax.text(0.5, 0.5, title, transform=ax.transAxes, va='center',ha='center',fontsize=12)
    plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()

def visualize_genome_skews(fasta_path: Path, out_dir: Path, num_windows: int = 4096):
    """Create linear and circular GC/AT skew images for one genome."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = read_first_fasta_sequence(fasta_path)
    gc_skew, at_skew, _, _ = compute_window_skews(seq, num_windows=num_windows)
    base = fasta_path.stem
    plot_linear_skews(gc_skew, at_skew, f"{base} — cumulative GC&AT skew",
                      out_dir / f"{base}_linear_skew.png")
    plot_circular_skews(gc_skew, at_skew, f"{base} — circular GC&AT skew",
                        out_dir / f"{base}_circular_skew.png")

def batch_visualize(fasta_files: List[str], out_root: Path, num_windows: int = 4096):
    """Generate images for all genomes under out_root / 'image'."""
    img_dir = IMAGES_DIR; img_dir.mkdir(parents=True, exist_ok=True)
    for f in fasta_files:
        fp = Path(f)
        if fp.exists():
            visualize_genome_skews(fp, img_dir, num_windows=num_windows)
    return img_dir

# ---------- PREDICTION ----------
## Optional import for XGBoost-only path
try:
    import xgboost as xgb
except Exception:
    xgb = None

def _err(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(1)

def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def _ensure_columns(df: pd.DataFrame, needed_cols):
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        _err(f"Input is missing {len(missing)} required columns (first 10 shown): {missing[:10]}")
    return df[needed_cols].copy()

def _apply_standard_scaler(block: dict, X: np.ndarray) -> np.ndarray:
    mean_ = np.asarray(block["mean_"], dtype=float)
    scale_ = np.asarray(block["scale_"], dtype=float)
    if X.shape[1] != mean_.shape[0]:
        _err(f"Scaler n_features mismatch: X has {X.shape[1]}, scaler expects {mean_.shape[0]}")
    return (X - mean_) / scale_

def _apply_pca(block: dict, X_std: np.ndarray) -> np.ndarray:
    comps = np.asarray(block["components_"], dtype=float) 
    mean = np.asarray(block["mean_"], dtype=float)
    return (X_std - mean) @ comps.T

def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def _predict_with_knnpc(model_json: dict, feats_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Uses saved StandardScaler + PCA(3) + stored 3D training embedding for kNN voting.
    Output: DataFrame[id_col, 'polyploidy_pred'] with 0/1 predictions.
    """
    feature_cols = model_json["feature_cols"]
    X_df = _ensure_columns(feats_df, feature_cols)
    ids = feats_df[id_col].astype(str).values

    X = X_df.to_numpy(dtype=float)
    X_std = _apply_standard_scaler(model_json["scaler"], X)
    X_pc = _apply_pca(model_json["pca"], X_std)

    train_pc = np.asarray(model_json["training_embedding"]["X_pc"], dtype=float)
    train_y = np.asarray(model_json["training_embedding"]["y"], dtype=int)

    n_neighbors = int(model_json["knn"]["n_neighbors"])
    metric = model_json["knn"]["metric"]
    p = model_json["knn"]["p"]
    weights = model_json["knn"]["weights"]

    preds = []
    for x in X_pc:
        if metric == "minkowski" and (p is None or int(p) == 2):
            d = np.sqrt(((train_pc - x) ** 2).sum(axis=1))
        elif metric == "minkowski" and int(p) == 1:
            d = np.abs(train_pc - x).sum(axis=1)
        else:
            _err(f"kNN metric '{metric}' with p={p} is not supported in this loader.")

        nn_idx = np.argpartition(d, n_neighbors)[:n_neighbors]
        nn_dist = d[nn_idx]
        nn_lab = train_y[nn_idx]

        if weights == "distance":
            if np.any(nn_dist == 0):
                vote = nn_lab[nn_dist == 0][0]
            else:
                w = 1.0 / nn_dist
                s0 = w[nn_lab == 0].sum()
                s1 = w[nn_lab == 1].sum()
                vote = 1 if s1 >= s0 else 0
        else:
            c0 = np.sum(nn_lab == 0)
            c1 = np.sum(nn_lab == 1)
            vote = 1 if c1 >= c0 else 0

        preds.append(vote)

    return pd.DataFrame({id_col: ids, "polyploidy_pred": np.array(preds, dtype=int)})

def plot_pca3_knnpc_ref3_plotly(input_csv: str,
                                model_path: str | Path,
                                id_col: str = "file",
                                out_dir: str | Path = "image",
                                base_name: str = "pca3_positions",
                                point_size_ref: int = 6,
                                point_size_new: int = 5,
                                annotate_new: bool = False):
    """
    Project samples into the saved 3D PCA space and write an interactive HTML plot:
      - Reference points colored by y_multi ∈ {1,2,3}.
      - New samples in gray.
      - Click-and-drag to rotate; scroll to zoom; hover for details.

    Output: <out_dir>/<base_name>.html
    """
    feats_df = pd.read_csv(input_csv)
    if id_col not in feats_df.columns:
        _err(f"ID column '{id_col}' not found in input CSV.")

    mdl_json = _load_json(Path(model_path))

    # Ensure feature order matches training
    feature_cols = mdl_json["feature_cols"]
    X_df = _ensure_columns(feats_df, feature_cols)
    X = X_df.to_numpy(dtype=float)

    # Standardize + PCA using saved params
    X_std = _apply_standard_scaler(mdl_json["scaler"], X)
    X_pc_new = _apply_pca(mdl_json["pca"], X_std)

    # Reference embedding + labels (multiclass for coloring)
    train_pc = np.asarray(mdl_json["training_embedding"]["X_pc"], dtype=float)
    y_multi = mdl_json["training_embedding"].get("y_multi", None)
    if y_multi is None:
        _err("kNNPC.json missing 'training_embedding.y_multi'. "
             "Regenerate kNNPC.json to include both y (binary) and y_multi (1/2/3).")
    y_multi = np.asarray(y_multi, dtype=int)

    # Colors for reference groups
    color_map = {1: "orange", 2: "limegreen", 3: "skyblue"}

    # Build Plotly traces
    traces = []
    for g in (1, 2, 3):
        mask = (y_multi == g)
        if np.any(mask):
            traces.append(
                go.Scatter3d(
                    x=train_pc[mask, 0],
                    y=train_pc[mask, 1],
                    z=train_pc[mask, 2],
                    mode="markers",
                    name=f"Group {g}",
                    marker=dict(size=point_size_ref, color=color_map[g], opacity=0.7),
                    hovertemplate=("Reference<br>"
                                   "Group: " + str(g) + "<br>"
                                   "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>")
                )
            )

    new_ids = feats_df[id_col].astype(str).values
    traces.append(
        go.Scatter3d(
            x=X_pc_new[:, 0],
            y=X_pc_new[:, 1],
            z=X_pc_new[:, 2],
            mode="markers+text" if annotate_new else "markers",
            name="New samples",
            marker=dict(size=point_size_new, color="gray", opacity=0.95),
            text=new_ids,
            textposition="top center" if annotate_new else None,
            hovertemplate=(
                "New sample<br>"
                f"{id_col}: %{{text}}<br>"
                "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D PCA — Reference vs New Samples",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False),
            aspectmode="data"
        )
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = Path(out_dir) / f"{base_name}.html"
                                    
    pio.write_html(fig, file=str(out_html), full_html=True, include_plotlyjs="cdn")
    print(f"[OK] Saved interactive 3D PCA plot → {out_html}")

def plot_within_group_distance_hist(input_csv: str,
                                    model_path: str | Path,
                                    id_col: str = "file",
                                    out_png: str | Path = "distance_confidence.png",
                                    bins: int = 50,
                                    annotate_new: bool = False):
    """
    Plot mirrored histograms of reference pairwise distances in 3D PCA:
      - Within-group distribution on TOP.
      - Between-group distribution on BOTTOM (mirrored).
      - Overlay vertical dashed lines for NN distances of new samples.
    """
    feats_df = pd.read_csv(input_csv)
    if id_col not in feats_df.columns:
        _err(f"ID column '{id_col}' not found in input CSV.")

    mdl_json = _load_json(Path(model_path))

    # Ensure feature order matches training
    feature_cols = mdl_json["feature_cols"]
    X_df = _ensure_columns(feats_df, feature_cols)
    X = X_df.to_numpy(dtype=float)

    # Standardize + PCA using saved params
    X_std = _apply_standard_scaler(mdl_json["scaler"], X)
    X_pc_new = _apply_pca(mdl_json["pca"], X_std)  # (m, 3)

    # Reference embedding + labels
    train_pc = np.asarray(mdl_json["training_embedding"]["X_pc"], dtype=float)
    y_multi  = mdl_json["training_embedding"].get("y_multi", None)
    if y_multi is None:
        _err("kNNPC.json missing 'training_embedding.y_multi'. Regenerate with y_multi.")
    y_multi = np.asarray(y_multi, dtype=int)

    # --- Pairwise distances
    def _pairwise_upper(X3: np.ndarray) -> np.ndarray:
        n = X3.shape[0]
        if n < 2: return np.array([], dtype=float)
        out = []
        for i in range(n - 1):
            diff = X3[i+1:] - X3[i]
            out.append(np.sqrt(np.sum(diff * diff, axis=1)))
        return np.concatenate(out) if out else np.array([], dtype=float)

    # --- Within-group distances
    within_all = []
    for g in (1, 2, 3):
        grp = train_pc[y_multi == g]
        if grp.shape[0] >= 2:
            within_all.append(_pairwise_upper(grp))
    within_all = np.concatenate(within_all) if within_all else np.array([], dtype=float)

    # --- Between-group distances
    idx1, idx2, idx3 = np.where(y_multi == 1)[0], np.where(y_multi == 2)[0], np.where(y_multi == 3)[0]
    between_chunks = []
    def _cross_pairs(A, B):
        if A.size == 0 or B.size == 0: return
        XA, XB = train_pc[A], train_pc[B]
        d = np.sqrt(np.sum((XA[:, None, :] - XB[None, :, :])**2, axis=2)).ravel()
        between_chunks.append(d)
    _cross_pairs(idx1, idx2)
    _cross_pairs(idx1, idx3)
    _cross_pairs(idx2, idx3)
    between_all = np.concatenate(between_chunks) if between_chunks else np.array([], dtype=float)

    # --- Nearest-neighbor distances for new samples
    ids_new = feats_df[id_col].astype(str).values
    nn_dists_new = (
        np.sqrt(np.sum((X_pc_new[:, None, :] - train_pc[None, :, :])**2, axis=2)).min(axis=1)
        if train_pc.size else np.full(X_pc_new.shape[0], np.nan, dtype=float)
    )

    # --- Shared histogram range
    all_vals = []
    if within_all.size: all_vals.append(within_all)
    if between_all.size: all_vals.append(between_all)
    hist_range = (min(np.min(v) for v in all_vals), max(np.max(v) for v in all_vals)) if all_vals else None

    # --- Plot mirrored histogram
    plt.figure(figsize=(12, 6))
    if within_all.size:
        plt.hist(within_all, bins=bins, range=hist_range, alpha=0.7,
                 color="gray", edgecolor="black", label="Within-group")
    if between_all.size:
        plt.hist(between_all, bins=bins, range=hist_range, alpha=0.6,
                 color="skyblue", edgecolor="black", label="Between-group",
                 orientation="vertical", weights=-np.ones_like(between_all))

    # --- Draw PED for new samples
    for d0 in nn_dists_new:
        if np.isfinite(d0):
            plt.axvline(d0, color="black", linestyle="--", alpha=0.9, linewidth=1)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Euclidean distance in 3D PC space (PC1–PC3)")
    plt.ylabel("Count (top=within, bottom=between)")
    plt.title("Reference distance distributions + NN distances of new samples")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_png = IMAGES_DIR / Path(out_png).name
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Saved distance-confidence histogram → {out_png}")
                                        
def _ped_confidences_knnpc(mdl_json: dict, feats_df: pd.DataFrame, id_col: str) -> np.ndarray:
    """
    Compute PED.confidence per row in feats_df using kNNPC space.

    Steps
      1) Build distributions of pairwise Euclidean distances among reference points
         in 3D PCA space:
           - within-group: pairs with the same y_multi ∈ {1,2,3}
           - between-group: pairs with different y_multi
      2) For each new sample, compute its nearest-reference distance (in 3D PCA).
      3) Convert that distance to ECDF percentiles:
           p_w = F_within(d_nn),  p_b = F_between(d_nn)
      4) Define confidence = (1 - p_w) * (1 - p_b) ∈ [0,1].
         (small d relative to within AND between → confidence near 1)
    """
    # Ensure feature order matches training
    feature_cols = mdl_json["feature_cols"]
    X_df = _ensure_columns(feats_df, feature_cols)
    X = X_df.to_numpy(dtype=float)

    # Standardize + PCA using saved params
    X_std = _apply_standard_scaler(mdl_json["scaler"], X)
    X_pc_new = _apply_pca(mdl_json["pca"], X_std)  # (m, 3)

    # Reference embedding + labels
    train_pc = np.asarray(mdl_json["training_embedding"]["X_pc"], dtype=float)   # (n_ref, 3)
    y_multi = mdl_json["training_embedding"].get("y_multi", None)
    if y_multi is None:
        return np.full(X_pc_new.shape[0], np.nan, dtype=float)
    y_multi = np.asarray(y_multi, dtype=int)

    def _pairwise_upper(X3: np.ndarray) -> np.ndarray:
        n = X3.shape[0]
        if n < 2: return np.array([], dtype=float)
        out = []
        for i in range(n - 1):
            diff = X3[i+1:] - X3[i]
            out.append(np.sqrt(np.sum(diff * diff, axis=1)))
        return np.concatenate(out) if out else np.array([], dtype=float)

    # --- Build within-group distribution
    within_all = []
    for g in (1, 2, 3):
        grp = train_pc[y_multi == g]
        if grp.shape[0] >= 2:
            within_all.append(_pairwise_upper(grp))
    within_all = np.concatenate(within_all) if within_all else np.array([], dtype=float)

    # --- Build between-group distribution
    idx1 = np.where(y_multi == 1)[0]
    idx2 = np.where(y_multi == 2)[0]
    idx3 = np.where(y_multi == 3)[0]
    between_chunks = []
    def _cross_pairs(A, B):
        if A.size == 0 or B.size == 0: return
        XA, XB = train_pc[A], train_pc[B]
        d = np.sqrt(np.sum((XA[:, None, :] - XB[None, :, :])**2, axis=2)).ravel()
        between_chunks.append(d)
    _cross_pairs(idx1, idx2)
    _cross_pairs(idx1, idx3)
    _cross_pairs(idx2, idx3)
    between_all = np.concatenate(between_chunks) if between_chunks else np.array([], dtype=float)

    # If either distribution is missing, we can't compute the combined score reliably
    if within_all.size == 0 or between_all.size == 0:
        return np.full(X_pc_new.shape[0], np.nan, dtype=float)

    # Sort once for fast ECDF lookups
    w_sorted = np.sort(within_all)
    b_sorted = np.sort(between_all)
    Nw, Nb = w_sorted.size, b_sorted.size

    # Nearest-reference distance for each new sample
    nn_dists = np.sqrt(np.sum((X_pc_new[:, None, :] - train_pc[None, :, :])**2, axis=2)).min(axis=1)

    # ECDF percentiles
    ranks_w = np.searchsorted(w_sorted, nn_dists, side="right")
    p_w = ranks_w.astype(float) / float(Nw)

    ranks_b = np.searchsorted(b_sorted, nn_dists, side="right")
    p_b = ranks_b.astype(float) / float(Nb)

    # Combined confidence: smaller is better vs both distributions
    conf = (1.0 - p_w) * (1.0 - p_b)
    return conf

def _predict_with_lg(model_json: dict, feats_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Rebuilds binary LogisticRegression prediction from saved scaler stats + coefficients.
    """
    feature_cols = model_json["feature_cols"]
    X_df = _ensure_columns(feats_df, feature_cols)
    ids = feats_df[id_col].astype(str).values

    X = X_df.to_numpy(dtype=float)
    X_std = _apply_standard_scaler(model_json["scaler"], X)

    coef = np.asarray(model_json["logreg"]["coef_"], dtype=float)
    inter = np.asarray(model_json["logreg"]["intercept_"], dtype=float)
    z = X_std @ coef.T + inter
    p1 = _sigmoid(z).reshape(-1)
    yhat = (p1 >= 0.5).astype(int)
    return pd.DataFrame({id_col: ids, "polyploidy_pred": yhat, "predicted.probability": p1})

def _predict_with_xgb(model_path: Path, feats_df: pd.DataFrame, id_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Loads native XGBoost JSON model and predicts P(class=1), threshold 0.5.
    Requires the exact feature order used in training.
    """
    if xgb is None:
        _err("xgboost is not installed. Please install it to use the XGB model path.")
    ids = feats_df[id_col].astype(str).values
    X_df = _ensure_columns(feats_df, feature_cols)
    X = X_df.to_numpy(dtype=float)
    mdl = xgb.XGBClassifier()
    mdl.load_model(str(model_path))
    p1 = mdl.predict_proba(X)[:, 1]
    yhat = (p1 >= 0.5).astype(int)
    return pd.DataFrame({id_col: ids, "polyploidy_pred": yhat, "predicted.probability": p1})

def run_prediction(
    input_csv: str,
    output_csv: str,
    model: str = "knn",
    id_col: str = "file",
    model_path: str | None = None,
):
    """
    Predict polyploidy (0/1) from a feature CSV using one of three models:
      - 'knn' (default): kNN with StandardScaler + PCA(3) using kNNPC.json
      - 'lg': Logistic Regression with StandardScaler using MLG.json
      - 'xgb': XGBoost native booster using XGBoost.json
    Writes a 2-column CSV: [id_col, polyploidy_pred]
    """
    in_path = Path(input_csv)
    if not in_path.exists():
        _err(f"Input file not found: {in_path}")

    feats = pd.read_csv(in_path)
    if id_col not in feats.columns:
        _err(f"ID column '{id_col}' not found in input CSV.")

    # Default model file names if not provided
    MODELS_DIR = Path(__file__).resolve().parent / "models"
    if model_path is None:
        default_map = {
            "knn": MODELS_DIR / "kNNPC.json",
            "lg":   MODELS_DIR / "MLG.json",
            "xgb":   MODELS_DIR / "XGBoost.json",
        }
        model_path = default_map.get(model.lower())
    mdl_path = Path(model_path)

    if not mdl_path.exists():
        _err(f"Model file not found: {mdl_path}")

    # Dispatch per model
    model = model.lower()
    if model == "knn":
        mdl_json = _load_json(mdl_path)
        out = _predict_with_knnpc(mdl_json, feats, id_col)

        ped_conf = _ped_confidences_knnpc(mdl_json, feats, id_col)
        out["PED.confidence"] = ped_conf

    elif model == "lg":
        mdl_json = _load_json(mdl_path)
        out = _predict_with_lg(mdl_json, feats, id_col)

    elif model == "xgb":
        feature_cols = None
        for companion in ["MLG.json", "kNNPC.json"]:
            cand = mdl_path.parent / companion
            if cand.exists():
                try:
                    feature_cols = _load_json(cand)["feature_cols"]
                    break
                except Exception:
                    pass
        if feature_cols is None:
            _err("Could not infer feature order for XGBoost. Place MLG.json or kNNPC.json next to XGBoost.json, "
                 "or pass model_path to a JSON that includes 'feature_cols'.")
        out = _predict_with_xgb(mdl_path, feats, id_col, feature_cols)

    else:
        _err(f"Unknown model '{model}'. Choose from ['knn','lg','xgb'].")

    cols = [id_col, "polyploidy_pred"]
    if "PED.confidence" in out.columns:
        cols.append("PED.confidence")
    if "predicted.probability" in out.columns:
        cols.append("predicted.probability")
    out = out[cols]
    output_csv = OUTPUTS_DIR / Path(output_csv).name
    out.to_csv(output_csv, index=False)
    print(f"[OK] Wrote predictions → {output_csv}")

def _compute_features_for_file(fp: Path, num_windows: int) -> dict:
    seq = read_first_fasta_sequence(fp)
    gc_skew = gc_skew_vectorized(seq, num_windows=num_windows)
    gc_sr, gc_sa, gc_peak_dist, gc_index_dist = gcsi_features_from_gcskew(
        gc_skew, k3=K3, k4=K4, alpha=ALPHA
    )
    at_skew = at_skew_vectorized(seq, num_windows=num_windows)
    at_sr, at_sa, at_peak_dist, at_index_dist = atsi_features_from_atskew(
        at_skew, k3=K3, k4=K4, alpha=ALPHA
    )
    return {
        "file": fp.name,
        "sr.gc": gc_sr,
        "sa.gc": gc_sa,
        "peak.dist.gc": gc_peak_dist,
        "index.dist.gc": gc_index_dist,
        "sr.at": at_sr,
        "sa.at": at_sa,
        "peak.dist.at": at_peak_dist,
        "index.dist.at": at_index_dist,
    }

# ---------- OPTIONAL: estimate genome assembly and contamination using CheckM2 ---------- #
def _run_checkm2_multi(input_dir: Path, threads: int, outputs_dir: Path) -> list[Path]:
    """
    Run CheckM2 once per detected extension among {fasta, fa, fna}.
    Results are written under: output_dir / 'checkm2-result' / <ext> /
    Returns a list of result directories that were produced.
    """
    result_root = outputs_dir / "checkm2-result"
    result_root.mkdir(parents=True, exist_ok=True)

    exts = ["fasta", "fa", "fna"]
    produced = []

    for ext in exts:
        if not list(input_dir.glob(f"*.{ext}")):
            continue
        out_dir = result_root / ext
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = (
            f"checkm2 predict --threads {threads} "
            f"-x {ext} "
            f"--input {shlex.quote(str(input_dir))} "
            f"--output-directory {shlex.quote(str(out_dir))} --force"
        )
        print(f"[checkm2] Running: {cmd}")
        try:
            subprocess.run(shlex.split(cmd), check=True)
            produced.append(out_dir)
        except subprocess.CalledProcessError as e:
            print(f"[checkm2] ERROR: CheckM2 failed for -x {ext}. Command:\n{cmd}\n{e}")
            sys.exit(1)

    if not produced:
        print(f"[checkm2] WARNING: No *.fasta/*.fa/*.fna files found in {input_dir}. Skipping CheckM2.")
    else:
        print(f"[checkm2] Finished. Reports under: {result_root}")

    return produced


def _load_checkm2_results_multi(result_dirs: list[Path]) -> pd.DataFrame:
    """
    Read all quality_report.tsv files from result_dirs and return a DataFrame:
      columns: ['file','completeness','contamination']
    We try to be robust to column naming in CheckM2 outputs and to whether 'Name'
    includes an extension or not. If no extension is present, we emit three
    candidates: name.fasta, name.fa, name.fna (the merge will pick the right one).
    """
    frames = []
    for d in result_dirs:
        tsv = d / "quality_report.tsv"
        if not tsv.exists():
            alt = d / "quality_report.txt"
            if alt.exists():
                tsv = alt
            else:
                print(f"[checkm2] WARNING: Missing quality report in {d}")
                continue

        df = pd.read_csv(tsv, sep="\t")

        # Try to find the identifier + metrics columns regardless of capitalization
        name_col = next((c for c in ["Name", "Bin Id", "Bin", "Genome", "ID", "Sample Id"] if c in df.columns), None)
        comp_col = next((c for c in ["Completeness", "completeness", "CheckM2 completeness", "CheckM2_completeness"] if c in df.columns), None)
        cont_col = next((c for c in ["Contamination", "contamination", "CheckM2 contamination", "CheckM2_contamination"] if c in df.columns), None)

        if name_col is None or comp_col is None or cont_col is None:
            print(f"[checkm2] WARNING: Could not locate expected columns in {tsv}. Found: {list(df.columns)}")
            continue

        sub = df[[name_col, comp_col, cont_col]].copy()
        sub.columns = ["name", "completeness", "contamination"]

        for col in ["completeness", "contamination"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce") / 100.0
        
        records = []
        for _, r in sub.iterrows():
            name = str(r["name"])
            comp = r["completeness"]
            cont = r["contamination"]

            # If the name already has an extension, use as-is
            if any(name.endswith(f".{e}") for e in ["fasta", "fa", "fna"]):
                records.append({"file": name, "completeness": comp, "contamination": cont})
            else:
                # Emit candidates with each extension; the merge will keep matches only
                for e in ["fasta", "fa", "fna"]:
                    records.append({"file": f"{name}.{e}", "completeness": comp, "contamination": cont})

        if records:
            frames.append(pd.DataFrame.from_records(records))

    if not frames:
        return pd.DataFrame(columns=["file", "completeness", "contamination"])

    out = pd.concat(frames, ignore_index=True)
    # If duplicates exist (e.g., both name.fasta and name.fa appeared), keep the first
    out = out.drop_duplicates(subset=["file"])
    return out

def _merge_checkm2_into_predictions(pred_csv: Path, checkm2_dir: Path):
    """
    Append 'completeness' and 'contamination' to predictions.csv by matching file stems.
    """
    if not pred_csv.exists():
        _err(f"predictions CSV not found: {pred_csv}")

    pred = pd.read_csv(pred_csv)
    if "file" not in pred.columns:
        _err(f"'file' column not found in predictions CSV: {list(pred.columns)}")

    pred = pred.copy()
    pred["stem"] = pred["file"].map(lambda s: Path(str(s)).stem)

    qc = _load_checkm2_summary(checkm2_dir)
    merged = pred.merge(qc, how="left", on="stem")

    # Put new columns near the end (after existing ones)
    cols = [c for c in merged.columns if c != "stem"]
    merged = merged[cols]
    merged.to_csv(pred_csv, index=False)
    print(f"[checkm2] Appended completeness & contamination → {pred_csv}")

def copy_examples_to(dest: str | Path) -> Path:
    """Copy packaged examples to a destination directory and return the path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # Use importlib.resources for robust access whether installed as wheel/sdist
    with pkg_resources.as_file((pkg_resources.files("bacpp") / "examples")) as src_dir:
        shutil.copytree(src_dir, dest, dirs_exist_ok=True)
    print(f"[examples] Copied packaged examples → {dest}")
    return dest

# ---------- CLI ----------
OUTPUTS_DIR = None
IMAGES_DIR = None

def main():
    global OUTPUTS_DIR, IMAGES_DIR

    p = argparse.ArgumentParser(
        description="BPP: Bactererial Ploidy Predictor to identify bacteiral polyploidy based on global genomic architecture."
    )
    p.add_argument("folder", nargs="?", default=None, help="Input folder containing FASTA/FA/FNA files")
    p.add_argument("--predict", action="store_true", help="After feature extraction, run polyploidy prediction using a trained model.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: <folder>/outputs)")
    p.add_argument("--cpus", type=int, default=min(4, os.cpu_count() or 1), help="Number of CPU cores (1=serial).")
    p.add_argument("--images", action="store_true", help="Generate GC/AT skew images into <output folder>/image")
    p.add_argument("--checkm2", action="store_true", help="Run CheckM2 and append completeness/contamination to predictions.csv.")
    p.add_argument("--copy-examples", metavar="DIR", help="Copy packaged examples into DIR and exit.")
    p.add_argument("--model", default="knn", choices=["knn", "lg", "xgb"], help="Model to use for prediction if --predict is set. Default: knn")
    p.add_argument("--model-path", default=None, help="Path to model file (defaults to ./models/kNNPC.json / ./models/MLG.json / ./models/XGBoost.json).")
    p.add_argument("--num-windows", type=int, default=4096, help="Number of windows for extracting global genomic architecture (default: 4096)")
    p.add_argument("--no-interactions", action="store_true", help="Do not add interaction terms")
    p.add_argument("--id-col", default="file", help="ID column name in the features CSV for prediction. Default: file")
    p.add_argument("--pred-input", default=None, help="Optional: features CSV to use for prediction (overrides --out).")
    p.add_argument("--pred-output", default=None,
                   help="Optional: predictions CSV path (2 columns: ID, polyploidy_pred). "
                        "Default: <features_csv_dir>/predictions.csv")

    args = p.parse_args()

    # --- input folder not required when creating example files
    if args.copy_examples:
        copy_examples_to(args.copy_examples)
        return
    
    if not args.folder:
        p.error("the following arguments are required: folder (unless --copy-examples is used)")
    
    # --- Normalize paths (accept with/without trailing slash) ---
    args.folder = str(Path(args.folder).resolve())

    # --- If user didn't provide --out, set it to <folder>/outputs ---
    if args.out is None:
        OUTPUTS_DIR = Path(args.folder) / "outputs"
    else:
        OUTPUTS_DIR = Path(args.out).resolve()

    # Set args.out to the resolved path for downstream consistency
    args.out = str(OUTPUTS_DIR)

    # Always define IMAGES_DIR under the chosen OUTPUTS_DIR
    IMAGES_DIR = OUTPUTS_DIR / "images"

    # Create directories
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Outputs → {OUTPUTS_DIR}")
    print(f"[OK] Images  → {IMAGES_DIR}")

    # Optional: normalize any optional paths if provided
    if args.pred_input is not None:
        args.pred_input  = str(Path(args.pred_input).resolve())
    if args.pred_output is not None:
        args.pred_output = str(Path(args.pred_output).resolve())
    if args.model_path is not None:
        args.model_path  = str(Path(args.model_path).resolve())

    # ---- Feature extraction ----
    df = run_folder(
        args.folder,
        num_windows=args.num_windows,
        add_interaction_terms=not args.no_interactions,
        cpus=args.cpus,
    )

    if args.images:
        fasta_files = [
            str(p)
            for pat in ("*.fasta", "*.fa", "*.fna")
            for p in Path(args.folder).glob(pat)
        ]
        img_dir = batch_visualize(
            fasta_files=fasta_files,
            out_root=Path(args.folder),
            num_windows=args.num_windows
        )
        print(f"Saved images to {img_dir}")

    # Ensure output directory exists
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define default extracted features path
    features_path = OUTPUTS_DIR / "extracted_features.csv"
    df.to_csv(features_path, index=False)
    print(f"Wrote {features_path} with {len(df)} rows")

    # ---- prediction ----
    pred_csv_path = None
    if args.predict:
        features_path = Path(args.pred_input) if args.pred_input else out_dir / "extracted_features.csv"
        feats_csv = str(features_path)
        feats_path = Path(feats_csv)
        pred_out = args.pred_output if args.pred_output else str(feats_path.with_name("predictions.csv"))

        print(f"Running prediction using model='{args.model}' on {feats_csv} ...")
        run_prediction(
            input_csv=feats_csv,
            output_csv=pred_out,
            model=args.model,
            id_col=args.id_col,
            model_path=args.model_path,
        )
        print(f"Prediction written to {pred_out}")
        pred_csv_path = Path(pred_out)

    # ---- optional CheckM2 (after predictions so we can merge into predictions.csv) ----
    if getattr(args, "checkm2", False):
        if pred_csv_path is None:
            print("[checkm2] ERROR: No predictions file to annotate. Run with --predict (or supply --pred-output).")
            sys.exit(1)

        input_dir = Path(args.folder)
        outputs_dir = Path(args.out)

        # Run once per extension present (fasta/fa/fna)
        result_dirs = _run_checkm2_multi(input_dir=input_dir, threads=args.cpus, outputs_dir=outputs_dir)

        # Load and merge into predictions.csv
        try:
            pred_df = pd.read_csv(pred_csv_path)
        except Exception as e:
            print(f"[checkm2] ERROR: Could not read predictions at {pred_csv_path}: {e}")
            sys.exit(1)

        chk_df = _load_checkm2_results_multi(result_dirs)

        if chk_df.empty:
            print("[checkm2] WARNING: No completeness/contamination parsed; predictions.csv left unchanged.")
        else:
            merged = pred_df.merge(chk_df, on="file", how="left")
            merged.to_csv(pred_csv_path, index=False)
            print(f"[checkm2] Appended completeness/contamination → {pred_csv_path}")

    # ---- Generate multi-view 3D PCA plots when kNNPC.json is the active model ----
    using_knnpc = (args.model.lower() == "knn")
    if args.model_path:
        using_knnpc = using_knnpc or (Path(args.model_path).name.lower() == "knnpc.json")

    if using_knnpc:
        feats_csv = args.pred_input if args.pred_input else str(Path(args.out) / "extracted_features.csv")

        MODELS_DIR = Path(__file__).resolve().parent / "models"
        knnpc_path = MODELS_DIR / "kNNPC.json"
        if args.model_path:
            knnpc_path = Path(args.model_path)

        image_dir = IMAGES_DIR
        image_dir.mkdir(parents=True, exist_ok=True)

        if knnpc_path.exists():
            try:
                plot_pca3_knnpc_ref3_plotly(
                    input_csv=feats_csv,
                    model_path=knnpc_path,
                    id_col=args.id_col,
                    out_dir=image_dir,
                    base_name="pca3_positions_interactive",
                    point_size_ref=6,
                    point_size_new=5,
                    annotate_new=False
                )

                # Keep distance histogram (matplotlib PNG)
                plot_within_group_distance_hist(
                    input_csv=feats_csv,
                    model_path=knnpc_path,
                    id_col=args.id_col,
                    out_png=str(image_dir / "distance_confidence.png"),
                    bins=50,
                    annotate_new=False
                )
            except SystemExit:
                pass
        else:
            print(f"[info] kNNPC model file not found for 3D plots: {knnpc_path}")

if __name__ == "__main__":
    main()
