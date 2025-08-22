from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

# ---------- Core utilities ----------

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

# ---------- Interaction terms ----------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    mask = (b != 0)
    out[mask] = a[mask] / b[mask]
    return out

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    file = df["file"]
    """
    Add interaction terms:
      X -> multiplication, I -> division (safe; NaN if denominator == 0).
    """
    gc_sr = df["gc_sr"].to_numpy(float)
    gc_sa = df["gc_sa"].to_numpy(float)
    gc_pk = df["gc_peak.dist"].to_numpy(float)
    gc_idx = df["gc_index.dist"].to_numpy(float)
    at_sr = df["at_sr"].to_numpy(float)
    at_sa = df["at_sa"].to_numpy(float)
    at_pk = df["at_peak.dist"].to_numpy(float)
    at_idx = df["at_index.dist"].to_numpy(float)

    # Multiplications
    df["gc_srXsa"] = gc_sr * gc_sa
    df["gc_srXpeak.dist"] = gc_sr * gc_pk
    df["gc_srXindex.dist"] = gc_sr * gc_idx
    df["gc_saXpeak.dist"] = gc_sa * gc_pk
    df["gc_saXindex.dist"] = gc_sa * gc_idx
    df["gc_peak.distXindex.dist"] = gc_pk * gc_idx

    # Divisions (A I B == A / B)
    df["gc_srIsa"] = _safe_div(gc_sr, gc_sa)
    df["gc_srIpeak.dist"] = _safe_div(gc_sr, gc_pk)
    df["gc_srIindex.dist"] = _safe_div(gc_sr, gc_idx)

    df["gc_saIsr"] = _safe_div(gc_sa, gc_sr)
    df["gc_saIpeak.dist"] = _safe_div(gc_sa, gc_pk)
    df["gc_saIindex.dist"] = _safe_div(gc_sa, gc_idx)

    df["gc_peak.distIsr"] = _safe_div(gc_pk, gc_sr)
    df["gc_peak.distIsa"] = _safe_div(gc_pk, gc_sa)
    df["gc_peak.distIindex.dist"] = _safe_div(gc_pk, gc_idx)

    df["gc_index.distIsr"] = _safe_div(gc_idx, gc_sr)
    df["gc_index.distIsa"] = _safe_div(gc_idx, gc_sa)
    df["gc_index.distIpeak.dist"] = _safe_div(gc_idx, gc_pk)

    # Multiplications
    df["at_srXsa"] = at_sr * at_sa
    df["at_srXpeak.dist"] = at_sr * at_pk
    df["at_srXindex.dist"] = at_sr * at_idx
    df["at_saXpeak.dist"] = at_sa * at_pk
    df["at_saXindex.dist"] = at_sa * at_idx
    df["at_peak.distXindex.dist"] = at_pk * at_idx

    # Divisions (A I B == A / B)
    df["at_srIsa"] = _safe_div(at_sr, at_sa)
    df["at_srIpeak.dist"] = _safe_div(at_sr, at_pk)
    df["at_srIindex.dist"] = _safe_div(at_sr, at_idx)

    df["at_saIsr"] = _safe_div(at_sa, at_sr)
    df["at_saIpeak.dist"] = _safe_div(at_sa, at_pk)
    df["at_saIindex.dist"] = _safe_div(at_sa, at_idx)

    df["at_peak.distIsr"] = _safe_div(at_pk, at_sr)
    df["at_peak.distIsa"] = _safe_div(at_pk, at_sa)
    df["at_peak.distIindex.dist"] = _safe_div(at_pk, at_idx)

    df["at_index.distIsr"] = _safe_div(at_idx, at_sr)
    df["at_index.distIsa"] = _safe_div(at_idx, at_sa)
    df["at_index.distIpeak.dist"] = _safe_div(at_idx, at_pk)
    return df

# ---------- Batch runner ----------

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
                "gc_sr": gc_sr,
                "gc_sa": gc_sa,
                "gc_peak.dist": gc_peak_dist,
                "gc_index.dist": gc_index_dist,
                "at_sr": at_sr,
                "at_sa": at_sa,
                "at_peak.dist": at_peak_dist,
                "at_index.dist": at_index_dist,
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

# ---------- Batch runner ----------
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
    r_min = 0.0   # or a positive value if you want a fixed inner hole, e.g. 0.6
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
    """Generate images for many genomes under out_root / 'image'."""
    img_dir = out_root / "image"; img_dir.mkdir(parents=True, exist_ok=True)
    for f in fasta_files:
        fp = Path(f)
        if fp.exists():
            visualize_genome_skews(fp, img_dir, num_windows=num_windows)
    return img_dir

# ---------- Optional CLI ----------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compute GC-skew and AT-skew FFT features for FASTA files.")
    p.add_argument("folder", type=str, help="Folder containing FASTA/FA/FNA files")
    p.add_argument("--num-windows", type=int, default=4096,
                   help="Number of windows (default: 4096)")
    p.add_argument("--k3", type=float, default=600.0)
    p.add_argument("--k4", type=float, default=40.0)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--no-interactions", action="store_true",
                   help="Do not add interaction terms")
    p.add_argument("--out", type=str, default="extracted_features.csv", help="Output CSV path")
    p.add_argument("--images", action="store_true",
               help="Generate GC/AT skew images into ./image")

    args = p.parse_args()
    df = run_folder(
        args.folder,
        num_windows=args.num_windows,
        k3=args.k3,
        k4=args.k4,
        alpha=args.alpha,
        add_interaction_terms=not args.no_interactions,
    )
    if args.images:
        img_dir = batch_visualize([str(p) for p in (Path(args.folder).glob('*.fa*'))],
                              out_root=Path(args.folder),
                              num_windows=args.num_windows)
        print(f"Saved images to {img_dir}")
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} rows")
