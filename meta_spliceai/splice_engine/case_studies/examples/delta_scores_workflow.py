"""
MetaSpliceAI – delta_scores_workflow.py

A reference implementation of SpliceAI/OpenSpliceAI-style delta scores for variant effect prediction.
It constructs WT/ALT windows, runs a provided model to get per-base splice probabilities, computes
four event-specific deltas (donor gain/loss, acceptor gain/loss), and returns both structured annotations
and visualizations. It also includes utilities for ROC/PR evaluation on labeled variants (e.g., ClinVar).

Design goals
------------
• Model-agnostic: you pass in a callable that maps a DNA string (or one-hot) → per-base probs.
• VCF-ready: returns DS_* (delta scores) and DP_* (relative positions) suitable for VCF INFO fields.
• Robust to +/- strand and small indels; warns on large events crossing window bounds.
• Evaluation: ROC/PR with bootstrapped CIs; case-study plots; cohort stratification helpers.

Assumptions
-----------
• The model callable returns arrays shaped [L] for donor and acceptor probabilities (neither optional).
• Coordinates are 1-based chromosomal; windowing is symmetric around the VCF POS by default.
• For indels, ALT sequence is constructed in-place and trimmed/padded back to fixed window length.

Key outputs
-----------
- annotate_variant(...): returns dict with DS_AG, DS_AL, DS_DG, DS_DL and DP_* positions (relative to variant)
- annotate_vcf(...): iterates VCF, writes annotated VCF (or TSV)
- evaluate_variants(...): computes ROC/PR given labels and a chosen score (e.g., max(|Δ|))
- plot_variant_tracks(...): WT/ALT probability tracks + delta tracks with flagged peak positions

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class WindowConfig:
    flank: int = 50            # +/- bp around the variant
    max_indel: int = 200       # max absolute length change tolerated within window
    strand: str = '+'          # '+' or '-'; if '-', reverse-complement windows for scoring

@dataclass
class DeltaConfig:
    return_delta_arrays: bool = True  # include full Δ arrays in the return dict
    compute_ds_max: bool = True       # include SpliceAI-style DS_MAX = max among the 4 events

@dataclass
class EvalConfig:
    positive_label: str = 'pathogenic'  # value in y labels considered positive
    score_name: str = 'DS_MAX'          # which score to evaluate by default

# -----------------------------
# DNA helpers
# -----------------------------

_RC = str.maketrans('ACGTNacgtn', 'TGCANtgcan')

def revcomp(seq: str) -> str:
    return seq.translate(_RC)[::-1]

# -----------------------------
# Core scoring API
# -----------------------------

# The model callable signature expected by this module.
#   model_predict(seq: str) -> Dict[str, np.ndarray]
# Must return keys: 'donor', 'acceptor' (and optionally 'neither').
ModelFn = Callable[[str], Dict[str, np.ndarray]]


def _fix_window_len(seq: str, target_len: int) -> str:
    if len(seq) == target_len:
        return seq
    if len(seq) > target_len:
        # center trim
        extra = len(seq) - target_len
        left_trim = extra // 2
        right_trim = extra - left_trim
        return seq[left_trim: len(seq) - right_trim]
    # pad with N evenly on both sides
    missing = target_len - len(seq)
    left_pad = missing // 2
    right_pad = missing - left_pad
    return ('N' * left_pad) + seq + ('N' * right_pad)


def build_wt_alt_windows(ref_context: str, pos_in_context: int, ref: str, alt: str, cfg: WindowConfig) -> Tuple[str, str, int]:
    """Build fixed-length WT/ALT windows given a reference context string that already
    spans +/- cfg.flank around the variant genomic POS.
    - pos_in_context is a 0-based index of the variant position in ref_context.
    - ref, alt are VCF REF/ALT strings (post-normalization ideal).
    Returns (wt, alt, center_idx) where center_idx is the variant index in the returned strings.
    """
    L = 2*cfg.flank + 1
    # WT is the context itself
    wt = ref_context
    # Apply REF→ALT at the center; handle length change
    alt_seq = wt[:pos_in_context] + alt + wt[pos_in_context+len(ref):]
    if abs(len(alt_seq) - len(wt)) > cfg.max_indel:
        warnings.warn(f"Indel length change {len(alt_seq)-len(wt)} exceeds max_indel={cfg.max_indel}.")
    # Force equal length windows for scoring/Δ alignment
    wt = _fix_window_len(wt, L)
    alt_seq = _fix_window_len(alt_seq, L)
    center_idx = cfg.flank  # by construction
    # Strand handling: reverse-complement if gene is on '-'
    if cfg.strand == '-':
        wt = revcomp(wt)
        alt_seq = revcomp(alt_seq)
        center_idx = len(wt) - 1 - center_idx
    return wt, alt_seq, center_idx


def score_sequence(seq: str, model: ModelFn) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    out = model(seq)
    donor = np.asarray(out['donor'], dtype=float)
    acceptor = np.asarray(out['acceptor'], dtype=float)
    neither = np.asarray(out['neither'], dtype=float) if 'neither' in out else None
    assert donor.shape == acceptor.shape, 'donor/acceptor length mismatch'
    return donor, acceptor, neither


def compute_delta_arrays(d_ref: np.ndarray, a_ref: np.ndarray,
                         d_alt: np.ndarray, a_alt: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        'Δdonor': d_alt - d_ref,
        'Δacceptor': a_alt - a_ref,
    }


def summarize_delta_events(delta: Dict[str, np.ndarray], center_idx: int, window: int) -> Dict[str, Tuple[float, int]]:
    """Return maxima for 4 events within +/- window of the center index.
    Keys:
      DS_DG (donor gain) = max(Δdonor)
      DS_DL (donor loss) = max(-Δdonor)
      DS_AG (acceptor gain) = max(Δacceptor)
      DS_AL (acceptor loss) = max(-Δacceptor)
    Also return DP_* = arg where the max occurs, as a signed offset relative to center.
    """
    lo = max(0, center_idx - window)
    hi = min(len(delta['Δdonor']), center_idx + window + 1)
    dd = delta['Δdonor'][lo:hi]
    da = delta['Δacceptor'][lo:hi]

    def _max_with_pos(arr: np.ndarray, sign: int = +1) -> Tuple[float, int]:
        z = arr if sign > 0 else -arr
        idx = int(np.argmax(z))
        val = float(z[idx])
        # convert back from clipped interval to full-series offset; sign keeps gain/loss semantics
        rel = (lo + idx) - center_idx
        return (val if sign > 0 else val, rel)

    DS_DG, DP_DG = _max_with_pos(dd, +1)
    DS_DL, DP_DL = _max_with_pos(dd, -1)
    DS_AG, DP_AG = _max_with_pos(da, +1)
    DS_AL, DP_AL = _max_with_pos(da, -1)

    return {
        'DS_DG': (DS_DG, DP_DG),
        'DS_DL': (DS_DL, DP_DL),
        'DS_AG': (DS_AG, DP_AG),
        'DS_AL': (DS_AL, DP_AL),
    }


def annotate_variant(ref_context: str, pos_in_context: int, ref: str, alt: str,
                     model: ModelFn, wcfg: WindowConfig = WindowConfig(), dcfg: DeltaConfig = DeltaConfig()) -> Dict:
    wt, alt_seq, center_idx = build_wt_alt_windows(ref_context, pos_in_context, ref, alt, wcfg)
    d_ref, a_ref, _ = score_sequence(wt, model)
    d_alt, a_alt, _ = score_sequence(alt_seq, model)
    delta = compute_delta_arrays(d_ref, a_ref, d_alt, a_alt)
    summary = summarize_delta_events(delta, center_idx, wcfg.flank)

    # Flatten outputs
    out = {
        'WT_seq': wt,
        'ALT_seq': alt_seq,
        'center_idx': center_idx,
        'DS_DG': summary['DS_DG'][0], 'DP_DG': summary['DS_DG'][1],
        'DS_DL': summary['DS_DL'][0], 'DP_DL': summary['DS_DL'][1],
        'DS_AG': summary['DS_AG'][0], 'DP_AG': summary['DS_AG'][1],
        'DS_AL': summary['DS_AL'][0], 'DP_AL': summary['DS_AL'][1],
    }
    if dcfg.compute_ds_max:
        out['DS_MAX'] = max(out['DS_DG'], out['DS_DL'], out['DS_AG'], out['DS_AL'])
    if dcfg.return_delta_arrays:
        out['Δdonor'] = delta['Δdonor']
        out['Δacceptor'] = delta['Δacceptor']
    return out

# -----------------------------
# Evaluation
# -----------------------------

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve


def evaluate_variants(df: pd.DataFrame, cfg: EvalConfig = EvalConfig()) -> Dict[str, float]:
    """Compute ROC-AUC and PR-AUC using cfg.score_name as the decision score.
    Expects df to contain columns [cfg.score_name, 'label'] where label is e.g. 'pathogenic'/'benign'.
    """
    y_true = (df['label'].astype(str).str.lower() == cfg.positive_label).astype(int).to_numpy()
    y_score = df[cfg.score_name].astype(float).to_numpy()
    out = {
        'roc_auc': float(roc_auc_score(y_true, y_score)),
        'pr_auc': float(average_precision_score(y_true, y_score)),
    }
    # Curves for optional plotting
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    out['roc_curve'] = (fpr, tpr)
    out['pr_curve'] = (rec, prec)
    return out

# -----------------------------
# Plotting
# -----------------------------


def plot_variant_tracks(var_id: str,
                        wt_d: np.ndarray, wt_a: np.ndarray,
                        alt_d: np.ndarray, alt_a: np.ndarray,
                        delta_d: np.ndarray, delta_a: np.ndarray,
                        center_idx: int,
                        summary: Dict[str, Tuple[float, int]],
                        out_png: Optional[str] = None) -> None:
    """Render a compact figure similar to OpenSpliceAI cryptic-variant panels:
    Row 1-2: WT donor/acceptor; Row 3-4: ALT donor/acceptor; Row 5-6: Δ donor/acceptor with DS_* peak markers.
    """
    x = np.arange(len(wt_d))
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), dpi=150, sharex=True)
    axes[0].plot(x, wt_d, linewidth=0.8);       axes[0].set_ylabel('WT Donor')
    axes[1].plot(x, wt_a, linewidth=0.8);       axes[1].set_ylabel('WT Acceptor')
    axes[2].plot(x, alt_d, linewidth=0.8);      axes[2].set_ylabel('ALT Donor')
    axes[3].plot(x, alt_a, linewidth=0.8);      axes[3].set_ylabel('ALT Acceptor')
    axes[4].plot(x, delta_d, linewidth=0.8);    axes[4].set_ylabel('Δ Donor')
    axes[5].plot(x, delta_a, linewidth=0.8);    axes[5].set_ylabel('Δ Acceptor')

    for ax in axes:
        ax.axvline(center_idx, linestyle='--', linewidth=0.6)

    # Mark peak offsets relative to center
    for k in ['DS_DG', 'DS_DL']:
        val, off = summary[k]
        axes[4].scatter([center_idx + off], [delta_d[center_idx + off]], s=22, marker='o')
        axes[4].text(center_idx + off, delta_d[center_idx + off], f"{k}={val:.3f}", fontsize=8, ha='left', va='bottom')
    for k in ['DS_AG', 'DS_AL']:
        val, off = summary[k]
        axes[5].scatter([center_idx + off], [delta_a[center_idx + off]], s=22, marker='o')
        axes[5].text(center_idx + off, delta_a[center_idx + off], f"{k}={val:.3f}", fontsize=8, ha='left', va='bottom')

    axes[-1].set_xlabel('Position (index in window)')
    fig.suptitle(var_id, y=0.995)
    fig.tight_layout(h_pad=0.6)
    if out_png:
        fig.savefig(out_png)
        plt.close(fig)
    else:
        plt.show()

# -----------------------------
# Batch annotation (skeleton)
# -----------------------------


def annotate_vcf(records: Iterable[Dict], fetch_ref_context: Callable[[Dict, int], Tuple[str, int]],
                 model: ModelFn, wcfg: WindowConfig = WindowConfig(), dcfg: DeltaConfig = DeltaConfig()) -> pd.DataFrame:
    """Annotate an iterable of VCF-like records with delta scores.
    - records: iterator of dicts with keys {chrom, pos, ref, alt, strand, id, info}
    - fetch_ref_context(record, flank) -> (ref_context_str, pos_in_context)
    Returns a DataFrame with one row per (record, alt), including DS_* and DP_*.
    """
    rows: List[Dict] = []
    for rec in records:
        strand = rec.get('strand', wcfg.strand)
        wcfg2 = WindowConfig(flank=wcfg.flank, max_indel=wcfg.max_indel, strand=strand)
        ref_context, pos_in_context = fetch_ref_context(rec, wcfg2.flank)
        for alt in (rec['alt'] if isinstance(rec['alt'], (list, tuple)) else [rec['alt']]):
            ann = annotate_variant(ref_context, pos_in_context, rec['ref'], alt, model, wcfg2, dcfg)
            rows.append({
                'id': rec.get('id'), 'chrom': rec['chrom'], 'pos': rec['pos'],
                'ref': rec['ref'], 'alt': alt, 'strand': strand,
                **{k: v for k, v in ann.items() if not k.startswith('WT_') and not k.startswith('ALT_')},
            })
    return pd.DataFrame(rows)


# -----------------------------
# Example (pseudo) model adapter
# -----------------------------

class ExampleFlatModel:
    """A toy adapter that pretends to be a SpliceAI-like model.
    Replace with an adapter that calls OpenSpliceAI/SpliceAI and returns dict with donor/acceptor arrays.
    """
    def __init__(self, p_d: float = 0.1, p_a: float = 0.1):
        self.pd = p_d; self.pa = p_a
    def __call__(self, seq: str) -> Dict[str, np.ndarray]:
        L = len(seq)
        # Fake a peak near the middle to illustrate deltas
        x = np.linspace(-2, 2, L)
        donor = 1/(1+np.exp(-3*(x-0.2))) * self.pd
        acceptor = 1/(1+np.exp(-3*(-x-0.2))) * self.pa
        return {'donor': donor, 'acceptor': acceptor}


# -----------------------------
# Minimal smoke test (optional)
# -----------------------------
if __name__ == '__main__':
    ref_context = 'A'*45 + 'G' + 'A'*55  # 101bp context with center G
    rec = {'chrom': '1', 'pos': 1000, 'ref': 'G', 'alt': 'A', 'strand': '+', 'id': 'rsDemo'}
    model = ExampleFlatModel(0.4, 0.5)
    df = annotate_vcf([rec], lambda r, f: (ref_context, 45), model)
    print(df.head())
"""
