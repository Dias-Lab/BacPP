from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json
import matplotlib.patches as mpatches
import sys
from matplotlib.lines import Line2D

# ---------- DATA PREPROCESSING ----------
## ---------- Core utilities ----------

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
    # (these columns will be created in run_folder below)
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
    k3: float = 600.0,
    k4: float = 40.0,
    alpha: float = 0.4,
    patterns: Tuple[str, ...] = ("*.fasta", "*.fa", "*.fna"),
    add_interaction_terms: bool = True,
) -> pd.DataFrame:
    """
    Scan folder for FASTA/FA/FNA, compute features, and return a DataFrame:
      columns = [file, sr, sa, peak.dist, index.dist, ...interactions*]
    """
    folder = Path(folder)
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(folder.glob(pat)))

    rows = []
    for fp in files:
        seq = read_first_fasta_sequence(fp)
        gc_skew = gc_skew_vectorized(seq, num_windows=num_windows)
        gc_sr, gc_sa, gc_peak_dist, gc_index_dist = gcsi_features_from_gcskew(
            gc_skew, k3=k3, k4=k4, alpha=alpha
        )
        at_skew = at_skew_vectorized(seq, num_windows=num_windows)
        at_sr, at_sa, at_peak_dist, at_index_dist = atsi_features_from_atskew(
            at_skew, k3=k3, k4=k4, alpha=alpha
        )
        rows.append(
    {
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
)
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

    # Fixed spike length in data units
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
    # ----------------------------------------------------------------------

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
    img_dir = out_root / "image"; img_dir.mkdir(parents=True, exist_ok=True)
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
    # return in training order
    return df[needed_cols].copy()

def _apply_standard_scaler(block: dict, X: np.ndarray) -> np.ndarray:
    mean_ = np.asarray(block["mean_"], dtype=float)
    scale_ = np.asarray(block["scale_"], dtype=float)
    if X.shape[1] != mean_.shape[0]:
        _err(f"Scaler n_features mismatch: X has {X.shape[1]}, scaler expects {mean_.shape[0]}")
    return (X - mean_) / scale_

def _apply_pca(block: dict, X_std: np.ndarray) -> np.ndarray:
    comps = np.asarray(block["components_"], dtype=float)  # (k, d)
    mean = np.asarray(block["mean_"], dtype=float)         # (d,)
    # sklearn PCA transform: (X - mean) @ components_.T
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
    X_pc = _apply_pca(model_json["pca"], X_std)  # (m, 3)

    train_pc = np.asarray(model_json["training_embedding"]["X_pc"], dtype=float)
    train_y = np.asarray(model_json["training_embedding"]["y"], dtype=int)

    n_neighbors = int(model_json["knn"]["n_neighbors"])
    metric = model_json["knn"]["metric"]
    p = model_json["knn"]["p"]
    weights = model_json["knn"]["weights"]

    preds = []
    for x in X_pc:
        if metric == "minkowski" and (p is None or int(p) == 2):
            d = np.sqrt(((train_pc - x) ** 2).sum(axis=1))     # Euclidean
        elif metric == "minkowski" and int(p) == 1:
            d = np.abs(train_pc - x).sum(axis=1)               # Manhattan
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

def plot_pca3_knnpc_ref3_multi(input_csv: str,
                               model_path: str | Path,
                               id_col: str = "file",
                               out_dir: str | Path = "image",
                               base_name: str = "pca3_positions",
                               point_size_ref: int = 18,
                               point_size_new: int = 8,
                               annotate_new: bool = False):
    """
    Project samples into the 3D PCA space (from kNNPC.json) and save multiple views:
      - Reference (training) points colored by 3-class labels (y_multi ∈ {1,2,3})
          Group 1 -> orange, Group 2 -> limegreen, Group 3 -> skyblue
      - New samples: small gray points (no hollow circles)
    Saves files:
      <out_dir>/<base_name>_view01.png ... _view06.png
    """
    import matplotlib.patches as mpatches

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

    # Reference embedding + labels (multiclass for coloring in the plot)
    train_pc = np.asarray(mdl_json["training_embedding"]["X_pc"], dtype=float)
    y_multi = mdl_json["training_embedding"].get("y_multi", None)
    if y_multi is None:
        _err("kNNPC.json missing 'training_embedding.y_multi'. "
             "Regenerate kNNPC.json to include both y (binary) and y_multi (1/2/3).")
    y_multi = np.asarray(y_multi, dtype=int)

    # Colors for reference groups
    color_map = {1: "orange", 2: "limegreen", 3: "skyblue"}
    colors_ref = [color_map.get(int(lbl), "gray") for lbl in y_multi]

    # Ensure output dir exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define 6 diverse viewpoints (azimuth, elevation)
    views = [
        (  0, 20),
        ( 45, 25),
        ( 90, 30),
        (135, 35),
        (180, 20),
        (225, 25),
    ]

    # Common legend handles
    leg_items = [
        mpatches.Patch(color=color_map[1], label="Group 1"),
        mpatches.Patch(color=color_map[2], label="Group 2"),
        mpatches.Patch(color=color_map[3], label="Group 3"),
        mpatches.Patch(facecolor="gray", edgecolor="none", label="New samples"),
    ]

    # Make one plot per view
    for i, (az, el) in enumerate(views, start=1):
        fig = plt.figure(figsize=(7.6, 6.6))
        ax = fig.add_subplot(111, projection="3d")

        # Reference cloud
        ax.scatter(train_pc[:, 0], train_pc[:, 1], train_pc[:, 2],
                   s=point_size_ref, c=colors_ref, alpha=0.5,
                   depthshade=False, label="Reference")

        # New samples: small gray points
        ax.scatter(X_pc_new[:, 0], X_pc_new[:, 1], X_pc_new[:, 2],
                   s=point_size_new, c="gray", alpha=0.9,
                   depthshade=False, label="New samples")

        if annotate_new:
            for j, sid in enumerate(feats_df[id_col].astype(str).values):
                ax.text(X_pc_new[j, 0], X_pc_new[j, 1], X_pc_new[j, 2],
                        f"{sid}", fontsize=8, color="black", ha="center", va="bottom")

        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.view_init(elev=el, azim=az)
        ax.set_title(f"3D PCA (view {i}: az={az}, el={el})")
        ax.legend(handles=leg_items, loc="best")
        plt.tight_layout()

        out_png = out_dir / f"{base_name}_view{i:02d}.png"
        plt.savefig(out_png, dpi=220)
        plt.close()

    print(f"[OK] Saved multi-view 3D PCA plots → {out_dir}/"
          f"{base_name}_view01.png ... _view{len(views):02d}.png")

def plot_within_group_distance_hist(input_csv: str,
                                    model_path: str | Path,
                                    id_col: str = "file",
                                    out_png: str | Path = "distance_confidence.png",
                                    bins: int = 50,
                                    annotate_new: bool = False):
    """
    Build a histogram of within-group pairwise Euclidean distances among reference samples
    in 3D PCA space (PC1-3), and overlay vertical lines for each new sample equal to its
    distance to the closest reference point (nearest neighbor in 3D).

    Saves a single image to out_png.
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

    # Reference embedding + 3-class labels for "within-group" distances
    train_pc = np.asarray(mdl_json["training_embedding"]["X_pc"], dtype=float)   # (n_ref, 3)
    y_multi  = mdl_json["training_embedding"].get("y_multi", None)
    if y_multi is None:
        _err("kNNPC.json missing 'training_embedding.y_multi'. "
             "Regenerate it to include both y (binary) and y_multi (1/2/3).")
    y_multi = np.asarray(y_multi, dtype=int)

    def _upper_triangle_pairwise(X3: np.ndarray) -> np.ndarray:
        """Return Euclidean distances for upper triangle pairs (i<j) for 3D points."""
        n = X3.shape[0]
        if n < 2:
            return np.array([], dtype=float)
        chunks = []
        for i in range(n - 1):
            diff = X3[i + 1:] - X3[i]
            di = np.sqrt(np.sum(diff * diff, axis=1))
            chunks.append(di)
        return np.concatenate(chunks) if chunks else np.array([], dtype=float)

    # Pool within-group distances across all three groups
    within_all = []
    for g in (1, 2, 3):
        grp = train_pc[y_multi == g]
        if grp.shape[0] >= 2:
            within_all.append(_upper_triangle_pairwise(grp))
    within_all = np.concatenate(within_all) if len(within_all) else np.array([], dtype=float)

    # For each new sample, get NN distance to ANY reference (closest ref, regardless of group)
    nn_dists_new = []
    ids_new = feats_df[id_col].astype(str).values
    for x in X_pc_new:
        d = np.sqrt(np.sum((train_pc - x) ** 2, axis=1))
        nn_dists_new.append(float(d.min()) if d.size else np.nan)
    nn_dists_new = np.array(nn_dists_new, dtype=float)

    # ---- Plot ----
    plt.figure(figsize=(9, 4.5))
    # Histogram of within-group ref-ref distances
    plt.hist(within_all, bins=bins, alpha=0.6, color="gray", edgecolor="black")
    plt.xlabel("Euclidean distance in 3D PC space")
    plt.ylabel("Count")
    plt.title("Within-group reference distances (hist) + new-sample NN distances (vertical lines)")

    # Overlay vertical lines for each new sample
    for i, d0 in enumerate(nn_dists_new):
        if np.isfinite(d0):
            plt.axvline(d0, color="black", linestyle="--", alpha=0.85, linewidth=1)
            if annotate_new:
                plt.text(d0, plt.ylim()[1] * 0.95, ids_new[i],
                         rotation=90, va="top", ha="right", fontsize=7)

    plt.tight_layout()
    out_png = str(out_png)
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Saved distance-confidence histogram → {out_png}")

def _ped_confidences_knnpc(mdl_json: dict, feats_df: pd.DataFrame, id_col: str) -> np.ndarray:
    """
    Compute PED.confidence for each row in feats_df using the kNNPC space:
      1) Build the distribution of within-group (1,2,3) pairwise distances among
         reference points in 3D PCA space.
      2) For each new sample, compute its nearest-reference distance (in 3D PCA).
      3) Convert that distance to a percentile in the within-group distribution
         and return confidence = 1 - percentile (so smaller distance -> higher confidence).
    """
    # Ensure feature order matches training
    feature_cols = mdl_json["feature_cols"]
    X_df = _ensure_columns(feats_df, feature_cols)
    X = X_df.to_numpy(dtype=float)

    # Standardize + PCA using saved params
    X_std = _apply_standard_scaler(mdl_json["scaler"], X)
    X_pc_new = _apply_pca(mdl_json["pca"], X_std)  # (m, 3)

    # Reference embedding + labels for within-group pairs
    train_pc = np.asarray(mdl_json["training_embedding"]["X_pc"], dtype=float)   # (n_ref, 3)
    y_multi = mdl_json["training_embedding"].get("y_multi", None)
    if y_multi is None:
        # If no multiclass labels are available, we can't form the within-group distribution
        # Return NaNs and let the caller handle messaging.
        return np.full(X_pc_new.shape[0], np.nan, dtype=float)
    y_multi = np.asarray(y_multi, dtype=int)

    # Pool within-group pairwise distances across groups 1/2/3
    def _upper_triangle_pairwise(X3: np.ndarray) -> np.ndarray:
        n = X3.shape[0]
        if n < 2:
            return np.array([], dtype=float)
        acc = []
        for i in range(n - 1):
            diff = X3[i + 1:] - X3[i]
            di = np.sqrt(np.sum(diff * diff, axis=1))
            acc.append(di)
        return np.concatenate(acc) if acc else np.array([], dtype=float)

    within_all = []
    for g in (1, 2, 3):
        grp = train_pc[y_multi == g]
        if grp.shape[0] >= 2:
            within_all.append(_upper_triangle_pairwise(grp))
    within_all = np.concatenate(within_all) if within_all else np.array([], dtype=float)

    # If we still don't have a distribution, return NaNs
    if within_all.size == 0:
        return np.full(X_pc_new.shape[0], np.nan, dtype=float)

    # Sort once for fast percentile lookup
    w_sorted = np.sort(within_all)
    N = w_sorted.size

    # Nearest-reference distance for each new sample
    nn_dists = np.sqrt(np.sum((X_pc_new[:, None, :] - train_pc[None, :, :])**2, axis=2)).min(axis=1)

    # Percentile p = ECDF(d) = (# within <= d) / N → confidence = 1 - p
    ranks = np.searchsorted(w_sorted, nn_dists, side="right")
    percentiles = ranks.astype(float) / float(N)
    confidences = 1.0 - percentiles
    return confidences

def _predict_with_mlg(model_json: dict, feats_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Rebuilds binary LogisticRegression prediction from saved scaler stats + coefficients.
    """
    feature_cols = model_json["feature_cols"]
    X_df = _ensure_columns(feats_df, feature_cols)
    ids = feats_df[id_col].astype(str).values

    X = X_df.to_numpy(dtype=float)
    X_std = _apply_standard_scaler(model_json["scaler"], X)

    coef = np.asarray(model_json["logreg"]["coef_"], dtype=float)        # (1, d)
    inter = np.asarray(model_json["logreg"]["intercept_"], dtype=float)  # (1,)
    z = X_std @ coef.T + inter                                           # (m,1)
    p1 = _sigmoid(z).reshape(-1)
    yhat = (p1 >= 0.5).astype(int)
    return pd.DataFrame({id_col: ids, "polyploidy_pred": yhat})

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
    return pd.DataFrame({id_col: ids, "polyploidy_pred": yhat})

def run_prediction(
    input_csv: str,
    output_csv: str,
    model: str = "knnpc",
    id_col: str = "file",
    model_path: str | None = None,
):
    """
    Predict polyploidy (0/1) from a feature CSV using one of three models:
      - 'knnpc' (default): kNN with StandardScaler + PCA(3) using kNNPC.json
      - 'mlg': Logistic Regression with StandardScaler using MLG.json
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
            "knnpc": MODELS_DIR / "kNNPC.json",
            "mlg":   MODELS_DIR / "MLG.json",
            "xgb":   MODELS_DIR / "XGBoost.json",
        }
        model_path = default_map.get(model.lower())
    mdl_path = Path(model_path)

    if not mdl_path.exists():
        _err(f"Model file not found: {mdl_path}")

    # Dispatch per model
    model = model.lower()
    if model == "knnpc":
        mdl_json = _load_json(mdl_path)
        out = _predict_with_knnpc(mdl_json, feats, id_col)

        ped_conf = _ped_confidences_knnpc(mdl_json, feats, id_col)
        out["PED.confidence"] = ped_conf

    elif model == "mlg":
        mdl_json = _load_json(mdl_path)
        out = _predict_with_mlg(mdl_json, feats, id_col)

    elif model == "xgb":
        # Need the exact training feature order; read from a sibling JSON that has 'feature_cols'
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
        _err(f"Unknown model '{model}'. Choose from ['knnpc','mlg','xgb'].")

    cols = [id_col, "polyploidy_pred"]
    if "PED.confidence" in out.columns:
        cols.append("PED.confidence")
    out = out[cols]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"[OK] Wrote predictions → {output_csv}")
# ---------- Optional CLI ----------

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Compute GC-skew and AT-skew FFT features for FASTA files.")
    p.add_argument("folder", type=str, help="Folder containing FASTA/FA/FNA files")
    p.add_argument("--num-windows", type=int, default=4096, help="Number of windows (default: 4096)")
    p.add_argument("--k3", type=float, default=600.0)
    p.add_argument("--k4", type=float, default=40.0)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--no-interactions", action="store_true", help="Do not add interaction terms")
    p.add_argument("--out", type=str, default=".", help="Output directory (default: current dir)")
    p.add_argument("--images", action="store_true", help="Generate GC/AT skew images into ./image")

    p.add_argument("--predict", action="store_true",
                   help="After feature extraction, run polyploidy prediction using a trained model.")
    p.add_argument("--model", default="knnpc", choices=["knnpc", "mlg", "xgb"],
                   help="Model to use for prediction if --predict is set. Default: knnpc")
    p.add_argument("--model-path", default=None,
                   help="Path to model file (defaults to ./models/kNNPC.json / ./models/MLG.json / ./models/XGBoost.json).")
    p.add_argument("--id-col", default="file",
                   help="ID column name in the features CSV for prediction. Default: file")
    p.add_argument("--pred-input", default=None,
                   help="Optional: features CSV to use for prediction (overrides --out).")
    p.add_argument("--pred-output", default=None,
                   help="Optional: predictions CSV path (2 columns: ID, polyploidy_pred). "
                        "Default: <features_csv_dir>/predictions.csv")

    args = p.parse_args()

    # --- Normalize paths (accept with/without trailing slash) ---
    args.folder = str(Path(args.folder).resolve())
    args.out    = str(Path(args.out).resolve())

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
        k3=args.k3,
        k4=args.k4,
        alpha=args.alpha,
        add_interaction_terms=not args.no_interactions,
    )

    if args.images:
        img_dir = batch_visualize([str(pth) for pth in (Path(args.folder).glob('*.fa*'))],
                                  out_root=Path(args.folder),
                                  num_windows=args.num_windows)
        print(f"Saved images to {img_dir}")

    
    # Ensure output directory exists
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define default extracted features path
    features_path = out_dir / "extracted_features.csv"
    df.to_csv(features_path, index=False)
    print(f"Wrote {features_path} with {len(df)} rows")

    # ---- prediction ----
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
    # ---- Generate multi-view 3D PCA plots when kNNPC.json is the active model ----
    using_knnpc = (args.model.lower() == "knnpc")
    if args.model_path:
        using_knnpc = using_knnpc or (Path(args.model_path).name.lower() == "knnpc.json")

    if using_knnpc:
        feats_csv = args.pred_input if args.pred_input else str(Path(args.out) / "extracted_features.csv")

        MODELS_DIR = Path(__file__).resolve().parent / "models"
        knnpc_path = MODELS_DIR / "kNNPC.json"
        if args.model_path:
            knnpc_path = Path(args.model_path)

        # Put PCA view images into the same /image directory you already use
        image_dir = out_dir / "image"
        image_dir.mkdir(parents=True, exist_ok=True)

        if knnpc_path.exists():
            try:
                plot_pca3_knnpc_ref3_multi(
                    input_csv=feats_csv,
                    model_path=knnpc_path,
                    id_col=args.id_col,
                    out_dir=image_dir,
                    base_name="pca3_positions",   # will create pca3_positions_view01..06.png
                    point_size_ref=18,
                    point_size_new=8,
                    annotate_new=False
                )
                
                plot_within_group_distance_hist(
                    input_csv=feats_csv,
                    model_path=knnpc_path,
                    id_col=args.id_col,
                    out_png=str(image_dir / "distance_confidence.png"),
                    bins=50,
                    annotate_new=False  # flip to True if you want sample IDs printed on the lines
                )
            except SystemExit:
                # _err() already printed a helpful message; continue gracefully
                pass
        else:
            print(f"[info] kNNPC model file not found for 3D plots: {knnpc_path}")
    
