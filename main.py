# gcsi_batch.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# ---------- Core utilities ----------

def read_first_fasta_sequence(path: Path) -> str:
    """Read the first sequence from a FASTA/FA/FNA file (dependency-free)."""
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        in_seq = False
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                if in_seq:  # already read first record
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

    sr = (power_at_1Hz / avg_other) if avg_other > 0 else 0.0
    sa = k4 * (k3 * power_at_1Hz) ** alpha

    cum = np.cumsum(x)
    peak_dist = float(cum.max() - cum.min())

    argmax = int(np.argmax(cum))
    argmin = int(np.argmin(cum))
    idx_gap = abs(argmax - argmin)
    circ_gap = min(idx_gap, N - idx_gap)
    index_dist = int(circ_gap)

    return sr, sa, peak_dist, index_dist

# ---------- Interaction terms ----------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    mask = (b != 0)
    out[mask] = a[mask] / b[mask]
    return out

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms:
      X -> multiplication, I -> division (safe; NaN if denominator == 0).
    """
    sr = df["sr"].to_numpy(float)
    sa = df["sa"].to_numpy(float)
    pk = df["peak.dist"].to_numpy(float)
    idx = df["index.dist"].to_numpy(float)

    # Multiplications
    df["srXsa"] = sr * sa
    df["srXpeak.dist"] = sr * pk
    df["srXindex.dist"] = sr * idx
    df["saXpeak.dist"] = sa * pk
    df["saXindex.dist"] = sa * idx
    df["peak.distXindex.dist"] = pk * idx

    # Divisions (A I B == A / B)
    df["srIsa"] = _safe_div(sr, sa)
    df["srIpeak.dist"] = _safe_div(sr, pk)
    df["srIindex.dist"] = _safe_div(sr, idx)

    df["saIsr"] = _safe_div(sa, sr)
    df["saIpeak.dist"] = _safe_div(sa, pk)
    df["saIindex.dist"] = _safe_div(sa, idx)

    df["peak.distIsr"] = _safe_div(pk, sr)
    df["peak.distIsa"] = _safe_div(pk, sa)
    df["peak.distIindex.dist"] = _safe_div(pk, idx)

    df["index.distIsr"] = _safe_div(idx, sr)
    df["index.distIsa"] = _safe_div(idx, sa)
    df["index.distIpeak.dist"] = _safe_div(idx, pk)

    return df

# ---------- Batch runner ----------

def run_folder(
    folder: str | Path,
    num_windows: int = 4096,
    k3: float = 600.0,
    k4: float = 40.0,
    alpha: float = 0.4,
    patterns: Tuple[str, ...] = ("*.fasta", "*.fa", "*.fna"),
    add_interaction_terms: bool = True,   # enabled by default
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
        sr, sa, peak_dist, index_dist = gcsi_features_from_gcskew(
            gc_skew, k3=k3, k4=k4, alpha=alpha
        )
        rows.append(
            {
                "file": fp.name,
                "sr": sr,
                "sa": sa,
                "peak.dist": peak_dist,
                "index.dist": index_dist,
            }
        )

    df = pd.DataFrame(rows)
    if add_interaction_terms and not df.empty:
        df = add_interactions(df)
    return df

# ---------- Optional CLI ----------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compute GC-skew FFT features for FASTA files.")
    p.add_argument("folder", type=str, help="Folder containing FASTA/FA/FNA files")
    p.add_argument("--num-windows", type=int, default=4096,
                   help="Number of windows (default: 4096)")
    p.add_argument("--k3", type=float, default=600.0)
    p.add_argument("--k4", type=float, default=40.0)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--no-interactions", action="store_true",
                   help="Do not add interaction terms")
    p.add_argument("--out", type=str, default="gcsi_features.csv", help="Output CSV path")

    args = p.parse_args()
    df = run_folder(
        args.folder,
        num_windows=args.num_windows,
        k3=args.k3,
        k4=args.k4,
        alpha=args.alpha,
        add_interaction_terms=not args.no_interactions,
    )
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} rows")
