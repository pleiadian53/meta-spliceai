"""Splice-site evaluation for meta-model adjusted probabilities.

Generates per-gene donor/acceptor performance using the same evaluator that
produces *full_splice_performance.tsv* for the raw SpliceAI model.

Usage (module):

>>> python -m meta_spliceai.splice_engine.meta_models.training.eval_meta_splice \
        --dataset  data/train_pc_1000/master \
        --run-dir  runs/loco_cv_cpu \
        --annotations  data/ensembl/spliceai_eval/splice_sites.parquet \
        --out-tsv  runs/loco_cv_cpu/full_splice_performance_meta.tsv \
        --threshold 0.90 --consensus-window 2 --sample 200000
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils

# Import utility functions from the new meta_evaluation_utils module
# These are re-exported here for backward compatibility with existing code
from .meta_evaluation_utils import (
    meta_splice_performance_correct,
    meta_splice_performance_argmax
)

################################################################################
# Helpers
################################################################################

import math


def _abs_positions(idx_arr: np.ndarray, gene_start: int, gene_end: int, strand: str) -> np.ndarray:
    """Convert relative indices to absolute genomic coordinates."""
    if strand == "+":
        return gene_start + idx_arr
    else:
        # On the minus strand the relative index counts from the *end*
        return gene_end - idx_arr


def _score_gene_sites(pred_pos: np.ndarray, truth_pos: np.ndarray, window: int) -> tuple[int, int, int]:
    """Return TP, FP, FN counts given arrays of predicted and truth positions.

    A prediction counts as TP if any truth site lies within ±*window* nt; each truth
    site can be matched at most once (greedy matching).
    """
    tp = 0
    fp = 0
    fn = len(truth_pos)
    if len(pred_pos) == 0:
        return tp, fp, fn
    truth_remaining = set(truth_pos.tolist())
    for p in pred_pos:
        matched = None
        for t in truth_remaining:
            if abs(p - t) <= window:
                matched = t
                break
        if matched is not None:
            tp += 1
            truth_remaining.remove(matched)
            fn -= 1
        else:
            fp += 1
    return tp, fp, fn


def _vectorised_site_metrics(
    ann_df: pl.DataFrame | None,
    pred_dict: Dict[str, Dict[str, Any]],
    *,
    threshold: float = 0.9,
    threshold_donor: float | None = None,
    threshold_acceptor: float | None = None,
    window: int = 2,
    missing_policy: str = "skip",
    include_tns: bool = False,
) -> pd.DataFrame:
    """Compute per-gene donor / acceptor precision-recall metrics (vectorised)."""
    # Build truth lookup: gene_id -> {"donor": set(abs_pos), "acceptor": set(abs_pos)}
    truth_lookup: Dict[str, Dict[str, set[int]]] = defaultdict(lambda: {"donor": set(), "acceptor": set()})
    if ann_df is not None:
        # Build truth lookup from annotation DataFrame
        for row in ann_df.iter_rows(named=True):
            gid = row["gene_id"]
            site_type = row["site_type"].lower()  # SpliceAI: acceptor/donor
            if site_type in {"donor", "acceptor"}:
                abs_pos = int(row["position"])
                truth_lookup[gid][site_type].add(abs_pos)
    else:
        total_donor_sites = 0
        total_acceptor_sites = 0
        for idx, (gid, info) in enumerate(pred_dict.items()):
            # If missing_policy="predict", use eval_donor and eval_acceptor instead of truth_donor and truth_acceptor
            # These are created by merging truth sets with predicted sets
            if missing_policy == "predict" and "eval_donor" in info and "eval_acceptor" in info:
                donor_sites = info.get("eval_donor", set())
                acceptor_sites = info.get("eval_acceptor", set())
                if idx < 3: 
                    print(f"[DIAGNOSTIC] Using evaluation sets for prediction policy")
            else:
                donor_sites = info.get("truth_donor", set())
                acceptor_sites = info.get("truth_acceptor", set())
                
            truth_lookup[gid]["donor"].update(donor_sites)
            truth_lookup[gid]["acceptor"].update(acceptor_sites)
            total_donor_sites += len(donor_sites)
            total_acceptor_sites += len(acceptor_sites)
        print(f"[DIAGNOSTIC] Found {total_donor_sites} donor sites and {total_acceptor_sites} acceptor sites from dataset")
        
        # Print more detailed sample gene truth sets for verification
        sample_size = min(3, len(pred_dict))
        sample_genes = sorted(list(pred_dict.keys()))[:sample_size]
        print("\n[DIAGNOSTIC] Sample of truth sets from dataset:")
        for gid in sample_genes:
            donor_count = len(pred_dict[gid]["truth_donor"])
            acceptor_count = len(pred_dict[gid]["truth_acceptor"])
            print(f"  Gene {gid}: {donor_count} donor sites, {acceptor_count} acceptor sites")
            if donor_count > 0:
                print(f"    Donor positions: {sorted(pred_dict[gid]['truth_donor'])[:5]}")
            if acceptor_count > 0:
                print(f"    Acceptor positions: {sorted(pred_dict[gid]['truth_acceptor'])[:5]}")
            if donor_count > 0 or acceptor_count > 0:
                print(f"    Donor positions: {sorted(list(truth_lookup[gid]['donor']))[:5]}{'...' if donor_count > 5 else ''}")
                print(f"    Acceptor positions: {sorted(list(truth_lookup[gid]['acceptor']))[:5]}{'...' if acceptor_count > 5 else ''}")

    records: List[Dict[str, Any]] = []
    # Thresholds are already resolved by the caller.
    if threshold_donor is None:
        threshold_donor = threshold
    if threshold_acceptor is None:
        threshold_acceptor = threshold

    include_tns_flag = include_tns
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for gid, info in pred_dict.items():
        strand = info["strand"]
        gstart = info["gene_start"]
        gend = info["gene_end"]
        for stype, prob_arr in (("donor", info["donor_prob"]), ("acceptor", info["acceptor_prob"])):
            # Identify local maxima ≥ threshold
            prob_arr = prob_arr.astype(np.float32)
            if stype == "donor":
                thr = threshold_donor
            else:
                thr = threshold_acceptor
            above = np.where(prob_arr >= thr)[0]
            # Collapse contiguous above-threshold stretches to a single peak (highest score)
            peaks: List[int] = []
            if above.size:
                current_group: List[int] = [int(above[0])]
                for idx in above[1:]:
                    if idx - current_group[-1] <= window:
                        current_group.append(int(idx))
                    else:
                        # Flush previous group – keep the index with the highest score
                        group_scores = prob_arr[current_group]
                        best_idx = current_group[int(np.nanargmax(group_scores))]
                        peaks.append(best_idx)
                        current_group = [int(idx)]
                # flush last group
                if current_group:
                    group_scores = prob_arr[current_group]
                    best_idx = current_group[int(np.nanargmax(group_scores))]
                    peaks.append(best_idx)
            if peaks:
                peaks_idx = np.array(peaks, dtype=int)
                pred_abs = _abs_positions(peaks_idx, gstart, gend, strand)
            else:
                pred_abs = np.empty(0, dtype=int)

            # Apply missing-policy filtering for truth positions
            if missing_policy == "skip":
                covered_abs: set[int] | None = info.get("covered_abs")
                if covered_abs is not None:
                    truth_set = truth_lookup.get(gid, {}).get(stype, set())
                    truth_filtered = (pos for pos in truth_set if pos in covered_abs)
                    truth_abs = np.fromiter(truth_filtered, dtype=int)
                else:
                    truth_abs = np.fromiter(truth_lookup.get(gid, {}).get(stype, []), dtype=int)
            else:
                truth_abs = np.fromiter(truth_lookup.get(gid, {}).get(stype, []), dtype=int)

            # ------------------------------------------------------------------
            # Window-tolerant matching: allow ±window nt between pred & truth
            # ------------------------------------------------------------------
            tp, fp, fn = _score_gene_sites(pred_abs, truth_abs, window)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            
            # **Enhanced diagnostics for position coordinate debugging**
            if stype == "donor" and (len(pred_abs) > 0 or len(truth_abs) > 0) and tp_sum + fp_sum < 5:  # Limit to first few genes
                print(f"\n[POSITION DEBUG] Gene {gid}, {stype}:")
                print(f"  Gene bounds: start={gstart}, end={gend}, strand={strand}")
                print(f"  Relative peaks found: {peaks if 'peaks' in locals() else 'None'}")
                print(f"  Pred positions (abs): {sorted(pred_abs) if len(pred_abs) > 0 else 'None'}")
                print(f"  Truth positions (abs): {sorted(truth_abs) if len(truth_abs) > 0 else 'None'}")
                if len(pred_abs) > 0 and len(truth_abs) > 0:
                    # Calculate all pairwise distances to identify systematic offsets
                    min_distances = []
                    for pred_pos in pred_abs:
                        distances = [abs(pred_pos - truth_pos) for truth_pos in truth_abs]
                        min_distances.append(min(distances))
                    print(f"  Minimum distances: {min_distances}")
                    if min_distances:
                        median_offset = np.median(min_distances)
                        print(f"  Median offset: {median_offset}bp")
                        if median_offset > 100:
                            print(f"  ⚠️  LARGE SYSTEMATIC OFFSET DETECTED: {median_offset}bp")
                            print(f"     This suggests coordinate system mismatch!")
                print(f"  Matching results: TP={tp}, FP={fp}, FN={fn}")
                print(f"  Window tolerance: ±{window}bp")

            # Debug zero metrics issue - print details for a limited number of genes with issues
            # (to avoid overwhelming the output)
            if stype == "donor" and (len(pred_abs) > 0 or len(truth_abs) > 0) and (tp == 0 or fp > 10):
                # Limit diagnostic output to first 10 genes
                if tp_sum + fp_sum < 10:
                    print(f"[DIAGNOSTIC] Gene {gid}, {stype}: TP={tp}, FP={fp}, FN={fn}")
                    print(f"  Pred sites: {len(pred_abs)}, Truth sites: {len(truth_abs)}")
                    if len(pred_abs) > 0 and len(truth_abs) > 0:
                        print(f"  Sample pred positions: {sorted(pred_abs)[:5]}")
                        print(f"  Sample truth positions: {sorted(truth_abs)[:5]}")

            # Optionally record TP/FP/FN sets if detailed analysis needed later
            # (not currently used downstream, but we keep the arrays for potential
            # debugging/visualisation)
            tp_set = None
            fp_set = None
            fn_set = None

            # Optional TN metrics
            if include_tns_flag:
                valid_positions = np.sum(~np.isnan(prob_arr))
                tn = int(valid_positions - (tp + fp + fn))
            else:
                tn = None

            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0

            if include_tns_flag and tn is not None:
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                balanced_acc = (recall_val + specificity) / 2.0
                denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
                # top-k accuracy
                k = len(truth_lookup[gid][stype])
                if k > 0:
                    # sort descending, ignore NaN
                    sorted_idx = np.argsort(np.nan_to_num(prob_arr, nan=-1.0))[::-1]
                    sorted_idx = sorted_idx[~np.isnan(prob_arr[sorted_idx])][:k]
                    topk_abs_arr = _abs_positions(sorted_idx.astype(int), gstart, gend, strand)
                    topk_pos = set(topk_abs_arr.tolist())
                    correct_topk = len(topk_pos & truth_lookup[gid][stype])
                    topk_acc = correct_topk / k
                else:
                    topk_acc = np.nan
            else:
                specificity = accuracy = balanced_acc = mcc = topk_acc = np.nan
            rec_entry = {
                "gene_id": gid,
                "site_type": stype,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision_val,
                "recall": recall_val,
                "f1_score": f1_val,
            }
            if include_tns_flag:
                rec_entry.update({
                    "TN": tn,
                    "specificity": specificity,
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_acc,
                    "mcc": mcc,
                    "topk_acc": topk_acc,
                })
            records.append(rec_entry)
    return pd.DataFrame.from_records(records)


def _infer_rel_pos(df: pl.DataFrame) -> pl.Series:
    """Best-effort inference of relative position column name."""
    for cand in ("rel_position", "rel_pos", "position"):
        if cand in df.columns:
            return df[cand]
    raise KeyError("Dataset must contain a relative position column (rel_position / rel_pos / position)")


def _build_pred_results(
    df: pd.DataFrame,
    proba: np.ndarray,
    *,
    missing_policy: str = "skip",
) -> Dict[str, Dict[str, Any]]:
    """Convert per-row DataFrame + meta probabilities into pred_results dict required
    by the existing evaluation helpers.
    """
    pred_dict: Dict[str, Dict[str, Any]] = {}

    # Attach meta probabilities to df
    df = df.copy()
    # Class order: 0=neither, 1=donor, 2=acceptor
    df["donor_meta"] = proba[:, 1]
    df["acceptor_meta"] = proba[:, 2]

    # Ensure necessary metadata columns are present
    required_meta = {"gene_id", "strand", "gene_start", "gene_end"}
    missing = required_meta - set(df.columns)
    if missing:
        raise KeyError(f"Dataset missing columns required for meta evaluation: {missing}")

    # Relative position
    rel_pos = _infer_rel_pos(pl.from_pandas(df))
    df["rel_pos"] = rel_pos.to_numpy()

    # Build per-gene arrays
    for gid, sub in df.groupby("gene_id", sort=False):
        first_row = sub.iloc[0]
        gene_len = int(first_row["gene_end"] - first_row["gene_start"] + 1)
        if missing_policy == "zero":
            donor_arr = np.zeros(gene_len, dtype=np.float32)
            accept_arr = np.zeros(gene_len, dtype=np.float32)
        else:  # "skip" or "predict"
            donor_arr = np.full(gene_len, np.nan, dtype=np.float32)
            accept_arr = np.full(gene_len, np.nan, dtype=np.float32)

        # ------------------------------------------------------------------
        # Some datasets store *rel_pos* as 1-based inclusive coordinates.
        # Guard against out-of-bounds by detecting that pattern and shifting.
        # ------------------------------------------------------------------
        rel_idx = sub["rel_pos"].astype(int).to_numpy()
        if rel_idx.max() >= gene_len:
            # Off-by-one – convert to 0-based by subtracting 1
            rel_idx = rel_idx - 1
        donor_arr[rel_idx] = sub["donor_meta"].values.astype(np.float32)
        accept_arr[rel_idx] = sub["acceptor_meta"].values.astype(np.float32)

        # Absolute positions covered by the dataset rows (needed for missing-policy "skip")
        abs_idx = _abs_positions(rel_idx, int(first_row["gene_start"]), int(first_row["gene_end"]), str(first_row["strand"]))

        # Collect truth splice-site positions from dataset itself (optional)
        truth_donor: set[int] = set()
        truth_acceptor: set[int] = set()
        if "site_type" in sub.columns or "splice_type" in sub.columns:
            _stype_col = "site_type" if "site_type" in sub.columns else "splice_type"
            for pos, st in zip(abs_idx, sub[_stype_col].astype(str).tolist()):
                if st == "donor":
                    truth_donor.add(int(pos))
                elif st == "acceptor":
                    truth_acceptor.add(int(pos))
        pred_dict[gid] = {
            "strand": first_row["strand"],
            "gene_start": int(first_row["gene_start"]),
            "gene_end": int(first_row["gene_end"]),
            "donor_prob": donor_arr,
            "acceptor_prob": accept_arr,
            "covered_abs": set(abs_idx.tolist()),
            "truth_donor": truth_donor,
            "truth_acceptor": truth_acceptor,
        }
    return pred_dict


################################################################################
# Public API
################################################################################

def meta_splice_performance(
    dataset_path: str | Path,
    run_dir: str | Path,
    annotations_path: str | Path | None = None,
    *,
    threshold: float | None = None,
    threshold_donor: float | None = None,
    threshold_acceptor: float | None = None,
    consensus_window: int = 2,
    sample: int | None = None,
    out_tsv: str | Path | None = None,
    base_tsv: str | Path | None = None,
    error_artifact: str | Path | None = None,
    errors_only: bool = False,
    include_tns: bool = False,
    missing_policy: str = "skip",
    verbose: int = 1,
    recompute_meta_scores: bool = False,
    # Thresholds are handled via threshold_donor and threshold_acceptor
) -> Path:
    """Compute donor/acceptor evaluation for meta-model probabilities and compare to base model results (if provided).

    Parameters
    ----------
    dataset_path : Parquet file or directory of Parquet shards containing the
        feature matrix rows.
    run_dir : directory that holds the trained meta-model (model_multiclass.*)
    annotations_path : file containing true splice-site annotations; must be
        loadable by Polars (tsv/csv/parquet).
    threshold : fallback probability threshold if per-class values are not provided (default 0.9).
    threshold_donor / threshold_acceptor : optional explicit per-class thresholds; if omitted they are auto-loaded from threshold_suggestion.txt or fall back to *threshold*.
    consensus_window : ± window when matching predictions to truth (default 2).
    sample : optional row sample to speed up evaluation.
    errors_only : if True, restrict evaluation to rows where the base model was FP or FN (requires artifacts with pred_type).
    out_tsv : optional path; if given, results are saved there (TSV).
    base_tsv : optional TSV file containing base model performance to compare against. If not provided, script looks for *full_splice_performance.tsv* in run_dir.
    """

    dataset_path = Path(dataset_path)
    run_dir = Path(run_dir)
    annotations_path = Path(annotations_path) if annotations_path is not None else None

    # --------------------------------------------------------------
    # Resolve per-class thresholds (auto-load from suggestion file)
    # --------------------------------------------------------------
    from . import classifier_utils as _cutils
    thr_map: dict[str, float] = {}
    try:
        thr_map = _cutils.load_thresholds(run_dir)
    except Exception:
        thr_map = {}
    if threshold_donor is None:
        threshold_donor = thr_map.get("threshold_donor")
    if threshold_acceptor is None:
        threshold_acceptor = thr_map.get("threshold_acceptor")
    
    # **CRITICAL FIX**: Use much more reasonable fallback thresholds
    # The original 0.9 default was causing systematic evaluation failures
    if threshold_donor is None or threshold_acceptor is None:
        fallback = threshold if threshold is not None else thr_map.get("threshold_global", 0.5)  # Changed from 0.9 to 0.5
        threshold_donor = threshold_donor if threshold_donor is not None else fallback
        threshold_acceptor = threshold_acceptor if threshold_acceptor is not None else fallback
    
    # Add threshold diagnostics
    if verbose:
        print(f"[meta_splice_eval] Using thresholds: donor={threshold_donor:.3f}, acceptor={threshold_acceptor:.3f}")
        if threshold_donor >= 0.9 or threshold_acceptor >= 0.9:
            print(f"[meta_splice_eval] ⚠️  CRITICAL WARNING: Very high thresholds detected (≥0.9)!")
            print(f"[meta_splice_eval] This will likely cause systematic evaluation failures if meta-model probabilities are not highly concentrated near 1.0.")
            print(f"[meta_splice_eval] Consider using thresholds around 0.5-0.7 for more realistic evaluation.")
        elif threshold_donor >= 0.8 or threshold_acceptor >= 0.8:
            print(f"[meta_splice_eval] WARNING: High thresholds detected (≥0.8). Verify these are appropriate for your meta-model's probability distribution.")
        else:
            print(f"[meta_splice_eval] Using reasonable thresholds for evaluation.")

    predict_fn, feature_names = _cutils._load_model_generic(run_dir)

    cols = [
        "donor_meta",
        "acceptor_meta",
        "gene_id",
        "strand",
        "gene_start",
        "gene_end",
        "neither_score",
        "donor_score",
        "acceptor_score",
        "rel_position",
        "rel_pos",
        "position",
        # Add important label columns
        "splice_type", 
        "site_type",
        "mapped_type",
        "true_position",
        "predicted_position",
        "pred_type",
        "absolute_position",
    ] + feature_names

    # Load rows lazily
    if dataset_path.is_dir():
        lf = pl.scan_parquet(str(dataset_path / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert")

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_cols = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    # Get column names using collect_schema() to avoid performance warning
    available_cols = set(lf.collect_schema().names())
    lf = lf.select([c for c in unique_cols if c in available_cols])

    # ------------------------------------------------------------------
    # Load annotations *early* so per-gene bounds are ready for batching
    # ------------------------------------------------------------------
    gene_bounds: Dict[str, tuple[int, int]] = {}
    ann_df: pl.DataFrame | None = None
    if annotations_path is not None:
        annotations_path = Path(annotations_path)
        if annotations_path.exists():
            ann_ext = annotations_path.suffix.lower()
            if ann_ext == ".parquet":
                ann_df = pl.read_parquet(annotations_path)
            elif ann_ext in (".tsv", ".csv"):
                sep = "\t" if ann_ext == ".tsv" else ","
                ann_df = pl.read_csv(annotations_path, separator=sep, schema_overrides={"chrom": pl.Utf8})
            else:
                raise ValueError(f"Unsupported annotation file format: {annotations_path.name}")
            if "position" in ann_df.columns:
                _bounds_df = ann_df.select(["gene_id", "position"]).group_by("gene_id").agg([
                    pl.col("position").min().alias("_min"),
                    pl.col("position").max().alias("_max"),
                ])
                for gid, pmin, pmax in zip(_bounds_df["gene_id"], _bounds_df["_min"], _bounds_df["_max"]):
                    gene_bounds[gid] = (int(pmin), int(pmax))
        else:
            print(f"[meta_splice_eval] WARNING: annotation file {annotations_path} missing – falling back to dataset-internal truth columns.")

    # Track genes with FP/FN for errors_only display purposes
    err_gene_set: set[str] | None = None

    # ------------------------------------------------------------------
    # Optional errors_only path: filter LF to FP/FN rows from artifacts
    # ------------------------------------------------------------------
    if errors_only:
        candidate_paths = []
        # 1) User-supplied path takes absolute precedence
        if error_artifact is not None:
            candidate_paths.append(Path(error_artifact))
        # 2) Legacy search order inside run_dir
        candidate_paths.extend([
            run_dir / "analysis_sequences_details.parquet",
            run_dir / "analysis_sequences_details.tsv",
            run_dir / "analysis_sequences_full.tsv",
            run_dir / "full_splice_positions_enhanced.tsv",
        ])
        # 3) Global evaluation directory under project data/
        project_root = Path(__file__).resolve().parents[4]
        candidate_paths.append(project_root / "data/ensembl/spliceai_eval/full_splice_positions_enhanced.tsv")
        err_path = next((p for p in candidate_paths if p is not None and p.exists()), None)
        if err_path is None:
            raise FileNotFoundError("--errors-only was set but no error artifact could be located. "
                                    "Provide a valid path via --error-artifact.")
        else:
            print(f"[meta_splice_eval] errors_only mode – filtering using {err_path.name}")
            if err_path.suffix == ".parquet":
                err_df = pl.read_parquet(err_path, columns=None)
            else:
                err_df = pl.read_csv(
                    err_path, 
                    separator="\t", 
                    schema_overrides={"chrom": pl.Utf8},
                )
            if "pred_type" not in err_df.columns:
                print("[meta_splice_eval] error artifact lacks 'pred_type' column – falling back to full dataset")
            else:
                err_df = err_df.filter(pl.col("pred_type").is_in(["FP", "FN"]))
                err_gene_set = set(err_df["gene_id"].unique())  # genes that contained errors
                # We no longer filter the lazyframe; keep full dataset so that TPs can change status.
                print(f"[meta_splice_eval] errors_only will DISPLAY only genes with FP/FN (count={len(err_gene_set)}) but evaluation uses full dataset")
    if sample is not None:
        lf = _cutils._lazyframe_sample(lf, sample, seed=42)
    ######################################################################
    # Chunked row iteration to avoid OOM
    ######################################################################

    # Helper to update pred_dict incrementally --------------------------------
    def _update_pred_dict(pred_dict: Dict[str, Dict[str, Any]], batch_df: pd.DataFrame, proba_batch: np.ndarray, *, verbose: int = 0, missing_policy: str = "skip", batch_count: int = 0) -> None:
        """Update the prediction dictionary with a batch of predictions and dataset rows.
        
        This is a batched update function that adds model probabilities and truth sites
        to the prediction dictionary.
        """
        
        # Debug print when first called
        if verbose >= 2 and batch_count <= 1:
            print("\n[DEBUG] Starting _update_pred_dict")
            print(f"[DEBUG] Batch shape: {batch_df.shape}")
            # Only print key columns of interest
            key_columns = [
                'gene_id', 'strand', 'position', 'donor_score', 'acceptor_score', 'neither_score',
                'splice_type', 'pred_type', 'true_position', 'predicted_position', 'absolute_position'
            ]
            present_columns = [col for col in key_columns if col in batch_df.columns]
            print(f"[DEBUG] Key columns present: {present_columns}")
            
            if "splice_type" in batch_df.columns:
                print(f"[DEBUG] Unique splice types: {sorted(batch_df['splice_type'].unique())}")
                print(f"[DEBUG] First 5 rows with splice_type:\n{batch_df[['gene_id', 'splice_type', 'position'] + [c for c in ['true_position', 'pred_type'] if c in batch_df.columns]].head(5)}")
            else:
                print("[DEBUG] No 'splice_type' column found in the dataset!")

        
        # Attach probs to batch_df
        batch_df = batch_df.copy()
        # Class order: 0=neither, 1=donor, 2=acceptor
        batch_df["donor_meta"] = proba_batch[:, 1]
        batch_df["acceptor_meta"] = proba_batch[:, 2]

        # Infer relative position column once per batch
        # Handle the case where training data uses 'position' and not 'rel_pos'
        if "position" in batch_df.columns and "rel_pos" not in batch_df.columns:
            #  print("[meta_splice_eval] Using 'position' column as relative position")
            batch_df["rel_pos"] = batch_df["position"]
        else:
            rel_pos_series = _infer_rel_pos(pl.from_pandas(batch_df))
            batch_df["rel_pos"] = rel_pos_series.to_numpy()

        for gid, sub in batch_df.groupby("gene_id", sort=False):
            first_row = sub.iloc[0]

            # ------------------------------------------------------------------
            # Determine gene bounds using annotation (fall back to row values)
            # ------------------------------------------------------------------
            gstart, gend = gene_bounds.get(
                gid,
                (int(first_row["gene_start"]), int(first_row["gene_end"]))
            )
            gene_len = int(gend - gstart + 1)

            if gid not in pred_dict:
                # Initialise arrays for this gene
                if missing_policy == "zero":
                    donor_arr = np.zeros(gene_len, dtype=np.float32)
                    accept_arr = np.zeros(gene_len, dtype=np.float32)
                else:  # skip / predict
                    donor_arr = np.full(gene_len, np.nan, dtype=np.float32)
                    accept_arr = np.full(gene_len, np.nan, dtype=np.float32)
                pred_dict[gid] = {
                    "strand": first_row["strand"],
                    "gene_start": gstart,
                    "gene_end": gend,
                    "donor_prob": donor_arr,
                    "acceptor_prob": accept_arr,
                    "covered_abs": set() if missing_policy == "skip" else None,
                    "truth_donor": set(),
                    "truth_acceptor": set(),
                }
            else:
                donor_arr = pred_dict[gid]["donor_prob"]
                accept_arr = pred_dict[gid]["acceptor_prob"]

            # ------------------------------------------------------------------
            # Expand arrays if we encounter positions outside current bounds
            #  • left side (rel_idx < 0)
            #  • right side (rel_idx ≥ current length)
            # ------------------------------------------------------------------
            min_rel = int(sub["rel_pos"].min())
            max_rel = int(sub["rel_pos"].max())
            fill_val = 0.0 if missing_policy == "zero" else np.nan
            # Prepend if needed (positions upstream of gene_start on + strand or downstream on – strand)
            if min_rel < 0:
                prepend = -min_rel
                donor_arr = np.concatenate([np.full(prepend, fill_val, dtype=np.float32), donor_arr])
                accept_arr = np.concatenate([np.full(prepend, fill_val, dtype=np.float32), accept_arr])
                # Shift existing indices so they align after the prepend
                sub["rel_pos"] = sub["rel_pos"] + prepend
                # Update stored bounds
                if pred_dict[gid]["strand"] == "+":
                    pred_dict[gid]["gene_start"] -= prepend
                else:
                    pred_dict[gid]["gene_end"] += prepend
            # Append if needed
            curr_len = donor_arr.size
            if max_rel >= curr_len:
                extend = max_rel - curr_len + 1
                donor_arr = np.concatenate([donor_arr, np.full(extend, fill_val, dtype=np.float32)])
                accept_arr = np.concatenate([accept_arr, np.full(extend, fill_val, dtype=np.float32)])
                if pred_dict[gid]["strand"] == "+":
                    pred_dict[gid]["gene_end"] += extend
                else:
                    pred_dict[gid]["gene_start"] -= extend
            # Persist the possibly resized arrays back
            pred_dict[gid]["donor_prob"] = donor_arr
            pred_dict[gid]["acceptor_prob"] = accept_arr

            rel_idx = sub["rel_pos"].astype(int).to_numpy()
            if rel_idx.max() >= gene_len:
                rel_idx = rel_idx - 1  # off-by-one safeguard

            donor_arr[rel_idx] = sub["donor_meta"].values.astype(np.float32)
            accept_arr[rel_idx] = sub["acceptor_meta"].values.astype(np.float32)

            # Update truth sets if present in batch
            if "site_type" in sub.columns or "splice_type" in sub.columns or "mapped_type" in sub.columns:
                _stype_col = "site_type" if "site_type" in sub.columns else "splice_type" if "splice_type" in sub.columns else "mapped_type"
                
                # Show detailed debug info about the splice types
                if verbose >= 2 and batch_count <= 1:
                    print(f"\n===== Gene {gid} DEBUG START =====")  # banner
                if verbose >= 3 and batch_count <= 1:
                    print(f"[DEBUG] Gene {gid}: Processing")
                    print(f"  Total rows for gene: {len(sub)}")
                    if len(sub) > 0:
                        print(f"  First row splice_type: '{sub[_stype_col].iloc[0]}' (type: {type(sub[_stype_col].iloc[0]).__name__})")
                    print(f"  Unique {_stype_col} values: {sorted(sub[_stype_col].unique())}")
                    print(f"  {_stype_col} value counts: {sub[_stype_col].value_counts().to_dict()}")
                    print(f"  {_stype_col} data type: {sub[_stype_col].dtype}")
                
                # Filter for actual splice sites (donor or acceptor)
                # Different datasets use different formats for splice types:
                # - Text format: "donor"/"acceptor" (case-insensitive)
                # - Numeric format: typically "0" for donor, "1" for acceptor, "2" for neither
                
                # Check if we have numeric splice types
                unique_types = sorted(sub[_stype_col].astype(str).unique())
                # We now expect only string labels plus optional '0' / 'neither' for non-splice.
                
                # ------------------------------------------------------------------
                # Build donor / acceptor masks that work with **either** string labels
                # ('donor', 'acceptor') **or** legacy numeric encodings (0/1/2).
                # Canonical numeric mapping we assume:
                #   0 = neither, 1 = donor, 2 = acceptor
                # Legacy alt-mapping that we still support:
                #   0 = donor, 1 = acceptor, 2 = neither
                # ------------------------------------------------------------------
                col_as_str = sub[_stype_col].astype(str)

                # Unified label handling:
                #   donor    → 'donor' (case-insensitive)
                #   acceptor → 'acceptor' (case-insensitive)
                #   neither  → '0' or 'neither'
                donor_mask = (col_as_str.str.lower() == "donor")
                acceptor_mask = (col_as_str.str.lower() == "acceptor")

                # '0' or 'neither' denote non-splice positions; no legacy numeric remapping.

                splice_mask = donor_mask | acceptor_mask
                
                if verbose >= 2 and batch_count <= 1:
                    # Show filtering results
                    print(f"  Donor matches: {donor_mask.sum()}")
                    print(f"  Acceptor matches: {acceptor_mask.sum()}")
                    print(f"  Combined splice sites: {splice_mask.sum()}")
                    
                    if splice_mask.sum() > 0:
                        # Show some examples of matching rows
                        print("\n[DEBUG] Sample matching splice site rows:")
                        sample_rows = sub[splice_mask].head(3)
                        for i, row in sample_rows.iterrows():
                            print(f"  Row {i}: {_stype_col}={row.get(_stype_col)}, position={row.get('position')}, " 
                                  f"true_position={row.get('true_position', 'N/A')}, strand={row.get('strand')}")
                    else:
                        # If no splice sites found, check sample raw values
                        if verbose >= 2:
                            print("\n[DEBUG] No splice sites matched using standard filters! Sample raw values:")
                            for i in range(min(5, len(sub))):
                                print(f"  Row {i} {_stype_col}: '{sub[_stype_col].iloc[i]}' (type: {type(sub[_stype_col].iloc[i]).__name__})")
                        
                        # Try alternative matching approaches
                        if verbose >= 2:
                            print("\n[DEBUG] Trying alternative match techniques...")
                        alt_matches = sub[sub[_stype_col].astype(str).str.contains('donor|acceptor', case=False, na=False)]
                        print(f"  String contains match found: {len(alt_matches)} rows")
                        
                        # Try exact match with sample values
                        test_vals = ['donor', 'acceptor', 'Donor', 'Acceptor']
                        for val in test_vals:
                            exact_match = (sub[_stype_col] == val).sum()
                            print(f"  Exact match for '{val}': {exact_match} rows")
                            
                        # Check if using string casts helps
                        for val in test_vals:
                            str_match = (sub[_stype_col].astype(str) == val).sum()
                            print(f"  String cast match for '{val}': {str_match} rows")
                
                # Only proceed if we found splice sites
                if splice_mask.sum() > 0:
                    # Process only the rows that are actual splice sites
                    splice_sites = sub[splice_mask].copy()
                    
                    # In case 'position' is being used as rel_pos, make sure it's available
                    if "rel_pos" not in splice_sites.columns and "position" in splice_sites.columns:
                        splice_sites["rel_pos"] = splice_sites["position"]
                    
                    # Use true_position if available (more accurate than rel_pos)
                    if "true_position" in splice_sites.columns:
                        splice_pos = splice_sites["true_position"].astype(int).to_numpy()
                        if verbose >= 2 and batch_count <= 1:
                            print(f"[DEBUG] Using true_position column for splice sites")
                    else:
                        splice_pos = splice_sites["rel_pos"].astype(int).to_numpy()
                        if verbose >= 2 and batch_count <= 1:
                            print(f"[DEBUG] Using rel_pos column for splice sites")
                    
                    # Normalise splice type values to canonical strings so that the
                    # downstream truth-set update works consistently
                    # Harmonise splice_type column for downstream logic
                    _norm_map = {
                        "donor": "donor",
                        "acceptor": "acceptor",
                        "0": "neither",
                        "neither": "neither",
                    }
                    splice_sites[_stype_col] = splice_sites[_stype_col].astype(str).str.lower().map(_norm_map)
                    splice_types = splice_sites[_stype_col].astype(str).tolist()
                    
                    # Absolute genomic coordinate: prefer explicit 'position' column (annotation-derived);
                    # otherwise fall back to converting relative indices.
                    if "position" in splice_sites.columns:
                        abs_positions = splice_sites["position"].astype(int).to_numpy()
                    else:
                        rel_arr = splice_sites["rel_pos"].astype(int).to_numpy()
                        abs_positions = _abs_positions(
                            rel_arr,
                            int(first_row["gene_start"]),
                            int(first_row["gene_end"]),
                            str(first_row["strand"]),
                        )
                    
                    # Additional debug info for first few entries
                    if verbose >= 2 and batch_count <= 1 and len(splice_sites) > 0:
                        print("\n[DEBUG] Sample splice site data:")
                        sample_size = min(5, len(splice_sites))
                        for i in range(sample_size):
                            print(f"  Row {i+1}: {_stype_col}={splice_sites[_stype_col].iloc[i]}, "
                                  f"position={splice_sites['position'].iloc[i]}, "
                                  f"abs_position={abs_positions[i]}, "
                                  f"true_position={splice_sites['true_position'].iloc[i] if 'true_position' in splice_sites.columns else 'N/A'}, "
                                  f"strand={splice_sites['strand'].iloc[i]}")
                    
                    # Update truth sets for donor and acceptor positions
                    updated_donor_count = 0
                    updated_acceptor_count = 0
                    for abs_pos, st in zip(abs_positions, splice_types):
                        if not pd.isna(abs_pos) and abs_pos > 0:  # Skip NaN or invalid positions
                            # Handle splice types - '0' is 'neither' so we ignore it
                            if st == "donor":
                                pred_dict[gid]["truth_donor"].add(int(abs_pos))
                                updated_donor_count += 1
                            elif st == "acceptor":
                                pred_dict[gid]["truth_acceptor"].add(int(abs_pos))
                                updated_acceptor_count += 1
                    
                    if verbose >= 2 and batch_count <= 1:
                        print(f"[DEBUG] Added {updated_donor_count} donor sites and {updated_acceptor_count} acceptor sites to truth sets for gene {gid}")
                        print(f"[DEBUG] Truth sets now contain: {len(pred_dict[gid]['truth_donor'])} donor, {len(pred_dict[gid]['truth_acceptor'])} acceptor sites")
                        print(f"===== Gene {gid} DEBUG END =====\n")
            
            if missing_policy == "skip":
                abs_idx = _abs_positions(rel_idx, int(first_row["gene_start"]), int(first_row["gene_end"]), str(first_row["strand"]))
                pred_dict[gid]["covered_abs"].update(abs_idx.tolist())

    # ------------------------------------------------------------------
    # Iterate over parquet shards (each file) to keep memory bounded
    # ------------------------------------------------------------------
    pred_results: Dict[str, Dict[str, Any]] = {}

    if dataset_path.is_dir():
        parquet_paths = sorted(dataset_path.glob("*.parquet"))
    else:
        parquet_paths = [dataset_path]

    BATCH_SIZE = 100_000  # rows – adjust if necessary

    for pq_path in parquet_paths:
        # Lazily stream the shard (Polars <0.20 does not support `columns=`)
        scan = pl.scan_parquet(str(pq_path))
        # Optional projection to a subset of columns if supported by Polars version
        # Try to project only the columns we actually use; fall back gracefully if
        # the Polars version exposes a different API.
        try:
            schema = scan.collect_schema()
            names_attr = getattr(schema, "names", None)
            column_names = names_attr() if callable(names_attr) else names_attr  # noqa: E501
            if column_names is None:
                # Polars < 0.18: schema is a list of (name, dtype) tuples
                if isinstance(schema, list):
                    column_names = [x[0] if isinstance(x, tuple) else x for x in schema]
                else:
                    column_names = []
            scan = scan.select([c for c in unique_cols if c in set(column_names)])
        except Exception:
            # Projection failed (unsupported Polars); continue without it.
            pass
        if sample is not None:
            # Polars LazyFrame.sample is only available in newer versions. Fall back to
            # the hash-based sampling helper if the method is absent.
            try:
                scan = scan.sample(n=sample, seed=42)
            except AttributeError:
                # Use classifier_utils helper (hash-based) – guarantees deterministic subset
                from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
                scan = _cutils._lazyframe_sample(scan, sample, seed=42)

        # Select streaming iterator – fallback to full collect if Polars lacks iter_batches
        if hasattr(scan, "iter_batches"):
            batch_iter = scan.iter_batches(batch_size=BATCH_SIZE, rechunk=False)
        else:
            # Older Polars: materialise DataFrame (streaming) then slice manually
            try:                                 # Polars ≥ 1.25
                full_pl = scan.collect(engine="streaming")
            except TypeError:                    # Polars < 1.25
                full_pl = scan.collect(streaming=True)
            if sample is not None and full_pl.height > sample:
                full_pl = full_pl.sample(n=sample, seed=42)
            batch_iter = (full_pl.slice(i, BATCH_SIZE) for i in range(0, full_pl.height, BATCH_SIZE))

        # For debugging, limit the number of batches processed
        debug_mode = verbose and (sample is None or sample > 1000)
        debug_sample_genes = set() if debug_mode else None
        debug_batch_limit = 4 if debug_mode else None
        batch_count = 0
        
        for batch_pl in batch_iter:
            batch_count += 1
            if debug_batch_limit is not None and batch_count > debug_batch_limit:
                if verbose >= 1:
                    print(f"[DEBUG] Stopping after {batch_count-1} batches for targeted debugging")
                break
                
            # Drop rows with missing values among *present* feature columns
            present_feats = [f for f in feature_names if f in batch_pl.columns]
            if present_feats:
                batch_pl = batch_pl.drop_nulls(subset=present_feats)
            if batch_pl.height == 0:
                continue

            batch_df = batch_pl.to_pandas()
            # Fill any feature completely missing from this batch with zeros
            for f in feature_names:
                if f not in batch_df.columns:
                    batch_df[f] = 0.0
                    
            if debug_mode and "splice_type" in batch_df.columns:
                # Sample a few genes for deeper analysis
                sample_genes = batch_df["gene_id"].unique()[:5]
                debug_sample_genes.update(sample_genes)
                
                # Show splice_type distribution
                slice_mask = batch_df["gene_id"].isin(sample_genes)
                if slice_mask.sum() > 0:
                    sample_df = batch_df[slice_mask]
                    if "splice_type" in sample_df.columns:
                        if verbose >= 2:
                            print(f"\n[DEBUG] Splice type values and counts in sample genes:")
                            print(sample_df["splice_type"].value_counts().to_dict())

            use_cached = (not recompute_meta_scores and "donor_meta" in batch_df.columns and "acceptor_meta" in batch_df.columns)
            if use_cached:
                donor_arr_cached = batch_df["donor_meta"].to_numpy(np.float32)
                accept_arr_cached = batch_df["acceptor_meta"].to_numpy(np.float32)
                # Reconstruct 3-column proba array in the same class order
                neither_cached = np.clip(1.0 - donor_arr_cached - accept_arr_cached, 0.0, 1.0)
                proba_batch = np.stack([
                    neither_cached,
                    donor_arr_cached,
                    accept_arr_cached,
                ], axis=1)
            else:
                # Apply the same preprocessing as training to handle chromosome encoding
                feature_batch = batch_df[feature_names].copy()
                
                # Handle non-numeric columns (especially chromosome) with proper encoding
                for col in feature_names:
                    if col in feature_batch.columns:
                        try:
                            # Try to convert to numeric first
                            feature_batch[col] = pd.to_numeric(feature_batch[col], errors='raise')
                        except (ValueError, TypeError):
                            if verbose >= 2:
                                print(f"Converting non-numeric column '{col}' to float with numeric encoding")
                            # Apply the same encoding as training: map unique values to integers
                            unique_vals = feature_batch[col].dropna().unique()
                            val_map = {val: float(idx) for idx, val in enumerate(sorted(unique_vals))}
                            feature_batch[col] = feature_batch[col].map(val_map).fillna(-1.0).astype(np.float32)
                
                X_batch = feature_batch.to_numpy(np.float32)
                proba_batch = predict_fn(X_batch)
            _update_pred_dict(
                pred_results,
                batch_df,
                proba_batch,
                verbose=verbose,
                missing_policy=missing_policy,
                batch_count=batch_count,
            )


            # Normalise numeric splice_type labels (legacy 0/1/2 encoding) ------
            # ------------------------------------------------------------------
            if ann_df is not None and "site_type" in ann_df.columns:
                # Robust normalisation that handles mixed encodings (e.g. "0", "donor", "acceptor")
                try:
                    from meta_spliceai.splice_engine.meta_models.training.label_utils import encode_labels as _enc, INT_TO_LABEL as _INT_TO_LABEL
                    labels_int = _enc(ann_df["site_type"].to_list())
                    ann_df = ann_df.with_columns(
                        pl.Series("site_type", [_INT_TO_LABEL[int(x)] for x in labels_int])
                    )
                except Exception as _exc:
                    # Fallback: replace plain "0" with "neither"
                    ann_df = ann_df.with_columns(
                        pl.when(pl.col("site_type") == "0").then("neither").otherwise(pl.col("site_type")).alias("site_type")
                    )


    # If missing_policy="predict", predict for positions not included in training data
    if missing_policy == "predict":
        if verbose >= 1:
            print("\n[meta_splice_eval] Predicting for unseen & ambiguous positions via enhanced inference workflow …")

        # ------------------------------------------------------------------
        # Build *covered_pos* mapping from existing predictions -------------
        # ------------------------------------------------------------------
        covered_pos: Dict[str, Set[int]] = {}
        for gid, info in pred_results.items():
            donor_arr = info["donor_prob"]
            # Positions with *any* prediction already present in training dataset
            present_idx = np.where(~np.isnan(donor_arr))[0]
            if len(present_idx):
                covered_pos[gid] = set(present_idx.astype(int).tolist())

        # ------------------------------------------------------------------
        # Run the lightweight inference workflow ---------------------------
        # ------------------------------------------------------------------
        from meta_spliceai.splice_engine.meta_models.workflows.splice_inference_workflow import (
            run_enhanced_splice_inference_workflow,
        )

        artefact_dir = run_enhanced_splice_inference_workflow(
            covered_pos=covered_pos,
            t_low=0.02,
            t_high=0.80,
            verbosity=max(0, verbose - 1),
        )
        feature_master_dir = Path(artefact_dir) / "features" / "master"
        if not feature_master_dir.exists():
            raise FileNotFoundError("Inference feature dataset not found at " + str(feature_master_dir))

        if verbose >= 1:
            print(f"[meta_splice_eval] Streaming predictions over inference feature dataset ({feature_master_dir}) …")

        # ------------------------------------------------------------------
        # Helper – iterate dataset in batches and update pred_dict ---------
        # ------------------------------------------------------------------
        BATCH_SIZE = 100_000

        def _stream_predict_dataset(ds_path: Path) -> None:
            """Stream rows from *ds_path* (Parquet shards) and update *pred_results*."""
            pq_paths = sorted(ds_path.glob("*.parquet")) if ds_path.is_dir() else [ds_path]
            for pq_path in pq_paths:
                scan = pl.scan_parquet(str(pq_path))
                # Project to available columns to speed up scan
                try:
                    schema = scan.collect_schema()
                    names_attr = getattr(schema, "names", None)
                    col_names = names_attr() if callable(names_attr) else names_attr
                    if col_names is None:
                        if isinstance(schema, list):
                            col_names = [x[0] if isinstance(x, tuple) else x for x in schema]
                        else:
                            col_names = []
                    scan = scan.select([c for c in unique_cols if c in set(col_names)])
                except Exception:
                    pass
                if sample is not None:
                    try:
                        scan = scan.sample(n=sample, seed=42)
                    except AttributeError:
                        from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
                        scan = _cutils._lazyframe_sample(scan, sample, seed=42)
                if hasattr(scan, "iter_batches"):
                    batch_iter = scan.iter_batches(batch_size=BATCH_SIZE, rechunk=False)
                else:
                    try:
                        full_pl = scan.collect(engine="streaming")
                    except TypeError:
                        full_pl = scan.collect(streaming=True)
                    if sample is not None and full_pl.height > sample:
                        full_pl = full_pl.sample(n=sample, seed=42)
                    batch_iter = (
                        full_pl.slice(i, BATCH_SIZE) for i in range(0, full_pl.height, BATCH_SIZE)
                    )
                batch_ix = 0
                for batch_pl in batch_iter:
                    batch_ix += 1
                    present_feats = [f for f in feature_names if f in batch_pl.columns]
                    if present_feats:
                        batch_pl = batch_pl.drop_nulls(subset=present_feats)
                    if batch_pl.height == 0:
                        continue
                    batch_df = batch_pl.to_pandas()
                    for f in feature_names:
                        if f not in batch_df.columns:
                            batch_df[f] = 0.0
                    
                    # Apply the same preprocessing as training to handle chromosome encoding
                    feature_batch = batch_df[feature_names].copy()
                    
                    # Handle non-numeric columns (especially chromosome) with proper encoding
                    for col in feature_names:
                        if col in feature_batch.columns:
                            try:
                                # Try to convert to numeric first
                                feature_batch[col] = pd.to_numeric(feature_batch[col], errors='raise')
                            except (ValueError, TypeError):
                                if verbose >= 2:
                                    print(f"Converting non-numeric column '{col}' to float with numeric encoding")
                                # Apply the same encoding as training: map unique values to integers
                                unique_vals = feature_batch[col].dropna().unique()
                                val_map = {val: float(idx) for idx, val in enumerate(sorted(unique_vals))}
                                feature_batch[col] = feature_batch[col].map(val_map).fillna(-1.0).astype(np.float32)
                    
                    X_batch = feature_batch.to_numpy(np.float32)
                    proba_batch = predict_fn(X_batch)
                    _update_pred_dict(
                        pred_results,
                        batch_df,
                        proba_batch,
                        verbose=verbose,
                        missing_policy=missing_policy,
                        batch_count=batch_ix,
                    )

        _stream_predict_dataset(feature_master_dir)

        # ------------------------------------------------------------------
        # Merge prediction results with truth sets (optional) --------------
        # ------------------------------------------------------------------
        try:
            meta_splice_predict.merge_prediction_results(pred_results, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"[meta_splice_eval] Warning: merge_prediction_results failed – {exc}")
    
    # ------------------------------------------------------------------
    # Probability-score distribution summary (for calibration diagnostics)
    # ------------------------------------------------------------------
    if verbose:
        if pred_results:
            donor_flat = np.concatenate([v["donor_prob"] for v in pred_results.values()])
            accept_flat = np.concatenate([v["acceptor_prob"] for v in pred_results.values()])
            # Remove NaNs (may appear if some positions were masked or not scored)
            d_nan = np.isnan(donor_flat).sum()
            a_nan = np.isnan(accept_flat).sum()
            donor_flat = donor_flat[~np.isnan(donor_flat)]
            accept_flat = accept_flat[~np.isnan(accept_flat)]
            if donor_flat.size and accept_flat.size:
                def _stats(arr):
                    return float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr)), float(np.nanmedian(arr))
                d_min, d_max, d_mean, d_med = _stats(donor_flat)
                a_min, a_max, a_mean, a_med = _stats(accept_flat)
                print(f"[meta_splice_eval] Donor prob   (n={donor_flat.size:,}, NaNs={d_nan:,}): "
                      f"min={d_min:.3f} max={d_max:.3f} mean={d_mean:.3f} median={d_med:.3f}")
                print(f"[meta_splice_eval] Acceptor prob (n={accept_flat.size:,}, NaNs={a_nan:,}): "
                      f"min={a_min:.3f} max={a_max:.3f} mean={a_mean:.3f} median={a_med:.3f}")
                
                # Add calibration diagnostics
                high_conf_donor = (donor_flat > 0.95).sum() / len(donor_flat) * 100
                high_conf_acceptor = (accept_flat > 0.95).sum() / len(accept_flat) * 100
                
                print(f"[meta_splice_eval] High confidence predictions (>0.95): donor={high_conf_donor:.1f}%, acceptor={high_conf_acceptor:.1f}%")
                
                if high_conf_donor > 70 or high_conf_acceptor > 70:
                    print(f"[meta_splice_eval] WARNING: Meta-model shows very high confidence (>70% predictions >0.95).")
                    print(f"[meta_splice_eval] This may indicate overfitting or calibration issues.")
                    print(f"[meta_splice_eval] Consider using lower thresholds (e.g., 0.5-0.7) for more realistic performance evaluation.")
                    if threshold_donor >= 0.9 or threshold_acceptor >= 0.9:
                        print(f"[meta_splice_eval] Current thresholds (donor={threshold_donor:.3f}, acceptor={threshold_acceptor:.3f}) are too high for this model.")
                        print(f"[meta_splice_eval] Suggest using thresholds around 0.5-0.7 for better discrimination.")
                
            else:
                print("[meta_splice_eval] Probability summary skipped – all values were NaN.")
        else:
            print("[meta_splice_eval] No prediction results to summarise.")

    # ------------------------------------------------------------------
    # Truth-site score distribution summary (annotated splice positions)
    # ------------------------------------------------------------------
    if verbose and ann_df is not None and pred_results:
        donor_truth_scores: list[float] = []
        accept_truth_scores: list[float] = []
        total_truth_donor = 0
        total_truth_acceptor = 0
        for gid, info in pred_results.items():
            gstart = info["gene_start"]
            # Collect donor truth scores
            if info["truth_donor"]:
                total_truth_donor += len(info["truth_donor"])
                rel_idx = np.fromiter(
                    ((pos - gstart) if info["strand"] == "+" else (info["gene_end"] - pos) for pos in info["truth_donor"]),
                    dtype=int,
                )
                valid_mask = (rel_idx >= 0) & (rel_idx < info["donor_prob"].size)
                if not np.all(valid_mask) and verbose >= 2:
                    invalid_count = (~valid_mask).sum()
                    print(f"[meta_splice_eval] Warning: {invalid_count} donor truth sites fell outside gene bounds for {gid}; skipped.")
                vals = info["donor_prob"][rel_idx[valid_mask]]
                donor_truth_scores.extend(vals[~np.isnan(vals)])
            # Collect acceptor truth scores
            if info["truth_acceptor"]:
                total_truth_acceptor += len(info["truth_acceptor"])
                rel_idx = np.fromiter(
                    ((pos - gstart) if info["strand"] == "+" else (info["gene_end"] - pos) for pos in info["truth_acceptor"]),
                    dtype=int,
                )
                valid_mask = (rel_idx >= 0) & (rel_idx < info["acceptor_prob"].size)
                if not np.all(valid_mask) and verbose >= 2:
                    invalid_count = (~valid_mask).sum()
                    print(f"[meta_splice_eval] Warning: {invalid_count} acceptor truth sites fell outside gene bounds for {gid}; skipped.")
                vals = info["acceptor_prob"][rel_idx[valid_mask]]
                accept_truth_scores.extend(vals[~np.isnan(vals)])
        # Print statistics if any scores collected
        if donor_truth_scores:
            d_arr = np.asarray(donor_truth_scores, dtype=float)
            print(
                f"[meta_splice_eval] TRUE donor site prob   (n={d_arr.size:,}/{total_truth_donor:,}): "
                f"min={np.nanmin(d_arr):.3f} max={np.nanmax(d_arr):.3f} "
                f"mean={np.nanmean(d_arr):.3f} median={np.nanmedian(d_arr):.3f}"
            )
        if accept_truth_scores:
            a_arr = np.asarray(accept_truth_scores, dtype=float)
            print(
                f"[meta_splice_eval] TRUE acceptor site prob (n={a_arr.size:,}/{total_truth_acceptor:,}): "
                f"min={np.nanmin(a_arr):.3f} max={np.nanmax(a_arr):.3f} "
                f"mean={np.nanmean(a_arr):.3f} median={np.nanmedian(a_arr):.3f}"
            )

    # Vectorised metrics computation (much faster, fewer deps)
    res_df = _vectorised_site_metrics(
        ann_df,
        pred_results,
        threshold=threshold if threshold is not None else 0.9,
        threshold_donor=threshold_donor,
        threshold_acceptor=threshold_acceptor,
        window=consensus_window,
        missing_policy=missing_policy,
        include_tns=include_tns,
    )

    # Persist
    if out_tsv is None:
        out_tsv = run_dir / "full_splice_performance_meta.tsv"
    out_tsv = Path(out_tsv)
    res_df.to_csv(out_tsv, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Console summary – show key metrics for quick inspection
    # ------------------------------------------------------------------
    summary_cols = [
        "gene_id",
        "site_type",
        "precision",
        "recall",
        "f1_score",
        "TP",
        "FP",
        "FN",
    ]
    if include_tns:
        summary_cols.extend([
            "TN",
            "specificity",
            "accuracy",
            "balanced_accuracy",
            "mcc",
            "topk_acc",
        ])
    printable_df = res_df[summary_cols].copy()
    if errors_only and err_gene_set is not None:
        printable_df = printable_df[printable_df["gene_id"].isin(err_gene_set)]

    # Limit display rows to avoid flooding terminal
    max_rows = 20
    pd.set_option("display.max_rows", max_rows)
    print("\n[meta_splice_eval] Sample performance summary (first " f"{max_rows} rows):")
    print(printable_df.head(max_rows).to_string(index=False, float_format="{:0.3f}".format))

    # ------------------ Base vs Meta comparison ----------------------
    if base_tsv is None:
        # 1) Look in run_dir
        candidate = run_dir / "full_splice_performance.tsv"
        if candidate.exists():
            base_tsv = candidate
        else:
            # 2) Fall back to shared evaluation directory (e.g. data/ensembl/spliceai_eval/full_splice_performance.tsv)
            project_root = Path(__file__).resolve().parents[4]
            shared_candidate = project_root / "data/ensembl/spliceai_eval/full_splice_performance.tsv"
            if shared_candidate.exists():
                base_tsv = shared_candidate
    if base_tsv is not None and Path(base_tsv).exists():
        base_df = pd.read_csv(base_tsv, sep="\t")
        if "site_type" not in base_df.columns and "splice_type" in base_df.columns:
            base_df = base_df.rename(columns={"splice_type": "site_type"})
        needed_cols = ["gene_id", "site_type", "f1_score", "precision", "recall", "TP", "FP", "FN"]
        if include_tns:
            needed_cols.extend(["TN", "specificity", "accuracy", "balanced_accuracy", "mcc", "topk_acc"])

        missing = set(needed_cols) - set(base_df.columns)
        if not missing:
            merged = base_df[needed_cols].merge(
                res_df[needed_cols],
                on=["gene_id", "site_type"],
                suffixes=("_base", "_meta"),
            )
            # Compute deltas
            metrics_for_delta = ["f1_score", "precision", "recall", "TP", "FP", "FN"]
            if include_tns:
                metrics_for_delta.extend(["TN", "specificity", "accuracy", "balanced_accuracy", "mcc", "topk_acc"])
            for m in metrics_for_delta:
                merged[f"{m}_delta"] = merged[f"{m}_meta"] - merged[f"{m}_base"]

            mean_f1_base = merged["f1_score_base"].mean()
            mean_f1_meta = merged["f1_score_meta"].mean()
            print(f"\n[meta_splice_eval] Global F1: base {mean_f1_base:.3f} → meta {mean_f1_meta:.3f} (Δ {mean_f1_meta - mean_f1_base:+.3f})")

            tp_delta = merged["TP_delta"].sum()
            fp_delta = merged["FP_delta"].sum()
            fn_delta = merged["FN_delta"].sum()
            print(f"[meta_splice_eval] Aggregate TP Δ {tp_delta:+}, FP Δ {fp_delta:+}, FN Δ {fn_delta:+}")

            improved = (merged["f1_score_delta"] > 0).sum()
            worsened = (merged["f1_score_delta"] < 0).sum()
            unchanged = (merged["f1_score_delta"] == 0).sum()
            print(f"[meta_splice_eval] Genes improved: {improved}, worsened: {worsened}, unchanged: {unchanged}")

            # ---------------- Sample before/after table ----------------
            summary_cols_ba = [
                "gene_id",
                "site_type",
                "TP_base", "TP_meta",
                "FP_base", "FP_meta",
                "FN_base", "FN_meta",
            ]
            if include_tns:
                summary_cols_ba.extend([
                    "TN_base", "TN_meta",
                    "specificity_meta", "accuracy_meta", "balanced_accuracy_meta", "mcc_meta", "topk_acc_meta",
                ])
            summary_cols_ba.extend(["precision_meta", "recall_meta", "f1_score_meta"])
            merged_display = merged[summary_cols_ba].copy()
            if errors_only and err_gene_set is not None:
                merged_display = merged_display[merged_display["gene_id"].isin(err_gene_set)]
            merged_display = merged_display.rename(columns={
                "TP_base": "TP0",
                "TP_meta": "TP",
                "FP_base": "FP0",
                "FP_meta": "FP",
                "FN_base": "FN0",
                "FN_meta": "FN",
                "precision_meta": "precision",
                "recall_meta": "recall",
                "f1_score_meta": "f1_score",
            })
            print("\n[meta_splice_eval] Sample before/after summary (first " f"{max_rows} rows):")
            print(merged_display.head(max_rows).to_string(index=False, float_format="{:0.3f}".format))
        else:
            print("[meta_splice_eval] Base performance file missing required columns – recomputing from raw SpliceAI scores …")
            try:
                donor_col, accept_col, neither_col = "donor_score", "acceptor_score", "neither_score"
                pred_base: Dict[str, Dict[str, Any]] = {}
                BATCH_SIZE = 100_000
                # Stream through the same feature dataset
                pq_paths = sorted(dataset_path.glob("*.parquet")) if dataset_path.is_dir() else [dataset_path]
                for pq in pq_paths:
                    scan = pl.scan_parquet(str(pq))
                    full_pl = scan.collect(streaming=True)
                    for i in range(0, full_pl.height, BATCH_SIZE):
                        batch_pl = full_pl.slice(i, BATCH_SIZE)
                        batch_df = batch_pl.to_pandas()
                        # Ensure raw score columns exist
                        if donor_col not in batch_df.columns or accept_col not in batch_df.columns:
                            raise KeyError(f"Raw score columns '{donor_col}', '{accept_col}' not found in dataset")
                        
                        # Raw score columns should already be numeric, but ensure proper type
                        donor_arr = batch_df[donor_col].astype(np.float32).to_numpy()
                        accept_arr = batch_df[accept_col].astype(np.float32).to_numpy()
                        if neither_col in batch_df.columns:
                            neither_arr = batch_df[neither_col].astype(np.float32).to_numpy()
                        else:
                            neither_arr = np.clip(1.0 - donor_arr - accept_arr, 0.0, 1.0)
                        proba_b = np.stack([neither_arr, donor_arr, accept_arr], axis=1)
                        _update_pred_dict(pred_base, batch_df, proba_b, verbose=0, missing_policy=missing_policy, batch_count=0)
                base_df_re = _vectorised_site_metrics(
                    ann_df,
                    pred_base,
                    threshold=threshold if threshold is not None else 0.9,
                    threshold_donor=threshold_donor,
                    threshold_acceptor=threshold_acceptor,
                    window=consensus_window,
                    missing_policy=missing_policy,
                    include_tns=include_tns,
                )
                base_out_path = run_dir / "full_splice_performance.tsv"
                base_df_re.to_csv(base_out_path, sep="\t", index=False)
                print(f"[meta_splice_eval] Recomputed base performance saved to {base_out_path}")
            except Exception as exc:
                print(f"[meta_splice_eval] Fallback base performance computation failed: {exc}")
    else:
        print("[meta_splice_eval] No base performance TSV found – skipping comparison.")

    return out_tsv


################################################################################
# CLI
################################################################################


################################################################################
# Simplified evaluator – *label-only*, no coordinates needed
################################################################################
from pathlib import Path
import pandas as _pd
import numpy as _np


def meta_splice_performance_simple(
    dataset_path: str | Path,
    run_dir: str | Path,
    annotations_path: str | Path | None = None,
    *,
    threshold: float | None = None,
    threshold_donor: float | None = None,
    threshold_acceptor: float | None = None,
    consensus_window: int | None = None,  # kept for API compatibility
    sample: int | None = None,
    out_tsv: str | Path | None = None,
    base_tsv: str | Path | None = None,
    error_artifact: str | Path | None = None,
    errors_only: bool = False,
    include_tns: bool = False,
    missing_policy: str = "skip",
    verbose: int = 1,
    recompute_meta_scores: bool = False,
    donor_score_col: str = "donor_score",
    acceptor_score_col: str = "acceptor_score",
    donor_meta_col: str = "donor_meta",
    acceptor_meta_col: str = "acceptor_meta",
    gene_col: str = "gene_id",
    label_col: str = "splice_type",
    **kwargs,
) -> Path:
    """Compute per-gene splice-site metrics for the *meta* model and compare them to the *base* model.

    This streamlined version uses **only the columns already present in the training parquet**:
        – true label column (``splice_type``)
        – base model raw probabilities (``donor_score``, ``acceptor_score``)
        – meta model probabilities (``donor_meta``, ``acceptor_meta``)

    All coordinate / annotation logic has been removed.
    The function signature remains unchanged so callers such as
    ``run_gene_cv_sigmoid.py`` keep working.  Unused parameters are accepted
    but ignored.
    """

    # Normalise paths
    run_dir = Path(run_dir)
    dataset_path = Path(dataset_path)

    # ------------------------------------------------------------------
    # 0. Resolve thresholds with better defaults ------------------------
    # ------------------------------------------------------------------
    # Try to load thresholds from run directory first
    from . import classifier_utils as _cutils
    thr_map: dict[str, float] = {}
    try:
        thr_map = _cutils.load_thresholds(run_dir)
    except Exception:
        thr_map = {}
    
    # Use provided thresholds or fall back to loaded ones or reasonable defaults
    if threshold_donor is None:
        threshold_donor = thr_map.get("threshold_donor")
    if threshold_acceptor is None:
        threshold_acceptor = thr_map.get("threshold_acceptor")
    
    # **CRITICAL FIX**: Use 0.5 as default instead of 0.9 to avoid systematic evaluation failures
    fallback_threshold = threshold if threshold is not None else thr_map.get("threshold_global", 0.5)  # Changed from 0.9
    thr_d = threshold_donor if threshold_donor is not None else fallback_threshold
    thr_a = threshold_acceptor if threshold_acceptor is not None else fallback_threshold

    if verbose:
        print(f"[meta_splice_eval] Using thresholds: donor={thr_d:.3f}, acceptor={thr_a:.3f}")
        if thr_d >= 0.9 or thr_a >= 0.9:
            print(f"[meta_splice_eval] ⚠️  CRITICAL WARNING: Very high thresholds detected (≥0.9)!")
            print(f"[meta_splice_eval] This will likely cause systematic evaluation failures.")
        elif thr_d >= 0.8 or thr_a >= 0.8:
            print(f"[meta_splice_eval] WARNING: High thresholds detected. Verify these are appropriate.")
        else:
            print(f"[meta_splice_eval] Using reasonable thresholds for evaluation.")

    # ------------------------------------------------------------------
    # 1. Load dataset --------------------------------------------------------
    # ------------------------------------------------------------------
    
    # Use memory-optimized loading for large datasets
    try:
        from meta_spliceai.splice_engine.meta_models.training.memory_optimized_datasets import (
            load_dataset_with_memory_management,
            estimate_dataset_size_efficiently
        )
        
        # Estimate size to decide loading strategy
        estimated_rows, file_count = estimate_dataset_size_efficiently(dataset_path)
        
        if estimated_rows > 2_000_000 or file_count > 10:
            if verbose:
                print(f"[meta_splice_eval] Large dataset detected ({estimated_rows:,} rows), using memory-optimized loading")
            df_pl = load_dataset_with_memory_management(
                dataset_path,
                max_memory_gb=12.0,
                fallback_to_standard=True
            )
            df = df_pl.to_pandas()
        else:
            df = _pd.read_parquet(dataset_path)
            
    except ImportError:
        if verbose:
            print(f"[meta_splice_eval] Memory optimization not available, using standard loading")
        df = _pd.read_parquet(dataset_path)
    except Exception as e:
        if verbose:
            print(f"[meta_splice_eval] Memory optimization failed ({e}), using standard loading")
        df = _pd.read_parquet(dataset_path)
    # Apply aggressive sampling for large datasets to prevent OOM
    if len(df) > 100_000:
        # For very large datasets, always sample to prevent OOM
        effective_sample = sample if sample is not None else 50_000
        effective_sample = min(effective_sample, 100_000)  # Cap at 100k for memory safety
        
        if verbose:
            print(f"[meta_splice_eval] Large dataset detected ({len(df):,} rows)")
            print(f"[meta_splice_eval] Applying sampling to {effective_sample:,} positions for memory safety")
        
        df = df.sample(n=effective_sample, random_state=0)
        df = df.reset_index(drop=True)
    elif sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=0)
        # Reset index after sampling to avoid IndexError
        df = df.reset_index(drop=True)
    
    if verbose:
        print(f"[meta_splice_eval] Processing {len(df):,} rows from {dataset_path}")

    required = {gene_col, label_col, donor_score_col, acceptor_score_col, donor_meta_col, acceptor_meta_col}
    missing = required - set(df.columns)
    # If meta columns are missing we will predict them on the fly
    need_meta = {donor_meta_col, acceptor_meta_col} - set(df.columns)
    if missing - {donor_meta_col, acceptor_meta_col}:
        raise KeyError(f"Dataset is missing required columns: {', '.join(sorted(missing - {donor_meta_col, acceptor_meta_col}))}")

    if need_meta or recompute_meta_scores:
        if verbose:
            print("[meta_splice_eval] Computing meta-model probabilities …")
        predict_fn, feature_names = _cutils._load_model_generic(run_dir)
        missing_feat = [c for c in feature_names if c not in df.columns]
        if missing_feat:
            raise KeyError(f"Dataset is missing features required by the meta model: {', '.join(missing_feat)}")
        
        # Apply the same preprocessing as training to handle chromosome encoding
        feature_df = df[feature_names].copy()
        
        # Handle non-numeric columns (especially chromosome) with proper encoding
        for col in feature_names:
            if col in feature_df.columns:
                try:
                    # Try to convert to numeric first
                    feature_df[col] = _pd.to_numeric(feature_df[col], errors='raise')
                except (ValueError, TypeError):
                    if verbose:
                        print(f"Converting non-numeric column '{col}' to float with numeric encoding")
                    # Apply the same encoding as training: map unique values to integers
                    unique_vals = feature_df[col].dropna().unique()
                    val_map = {val: float(idx) for idx, val in enumerate(sorted(unique_vals))}
                    feature_df[col] = feature_df[col].map(val_map).fillna(-1.0).astype(_np.float32)
        
        proba = predict_fn(feature_df.to_numpy(dtype=_np.float32))
        # Prob order: 0=neither,1=donor,2=acceptor
        df[donor_meta_col] = proba[:, 1]
        df[acceptor_meta_col] = proba[:, 2]
        df["neither_meta"] = proba[:, 0]

    # Ensure neither_meta exists (can be derived if independent sigmoids)
    if "neither_meta" not in df.columns:
        df["neither_meta"] = 1.0 - df[donor_meta_col] - df[acceptor_meta_col]

    # **Early probability distribution diagnostics to identify threshold issues**
    if verbose:
        donor_probs = df[donor_meta_col].dropna()
        acceptor_probs = df[acceptor_meta_col].dropna()
        
        print(f"\n[meta_splice_eval] 📊 Meta-model probability distribution summary:")
        for name, probs in [("Donor", donor_probs), ("Acceptor", acceptor_probs)]:
            if len(probs) > 0:
                mean_val = probs.mean()
                median_val = probs.median()
                max_val = probs.max()
                std_val = probs.std()
                above_50 = (probs > 0.5).sum() / len(probs) * 100
                above_80 = (probs > 0.8).sum() / len(probs) * 100
                above_95 = (probs > 0.95).sum() / len(probs) * 100
                
                print(f"  {name:8}: mean={mean_val:.3f}, median={median_val:.3f}, max={max_val:.3f}, std={std_val:.3f}")
                print(f"           >0.5: {above_50:5.1f}%, >0.8: {above_80:5.1f}%, >0.95: {above_95:5.1f}%")
        
        # Threshold appropriateness check
        thr_check_d = (donor_probs > thr_d).sum() / len(donor_probs) * 100 if len(donor_probs) > 0 else 0
        thr_check_a = (acceptor_probs > thr_a).sum() / len(acceptor_probs) * 100 if len(acceptor_probs) > 0 else 0
        
        print(f"\n[meta_splice_eval] 🎯 Threshold appropriateness check:")
        print(f"  Predictions above donor threshold ({thr_d:.3f}): {thr_check_d:.1f}%")
        print(f"  Predictions above acceptor threshold ({thr_a:.3f}): {thr_check_a:.1f}%")
        
        if thr_check_d < 1.0 or thr_check_a < 1.0:
            print(f"  ⚠️  WARNING: Very few predictions above threshold (<1%)!")
            print(f"     Consider lowering thresholds to ~0.1-0.3 for better sensitivity.")
        elif thr_check_d < 5.0 or thr_check_a < 5.0:
            print(f"  ⚠️  WARNING: Few predictions above threshold (<5%)!")
            print(f"     Consider lowering thresholds for better sensitivity.")
        elif thr_check_d > 50.0 or thr_check_a > 50.0:
            print(f"  ℹ️  High fraction above threshold (>50%) - may need higher threshold.")
        else:
            print(f"  ✅ Reasonable fraction of predictions above threshold.")

    # ------------------------------------------------------------------
    # 2. Base & meta predictions -------------------------------------------
    # ------------------------------------------------------------------
    df["pred_base_donor"] = df[donor_score_col] >= thr_d
    df["pred_base_acceptor"] = df[acceptor_score_col] >= thr_a
    df["pred_meta_donor"] = df[donor_meta_col] >= thr_d
    df["pred_meta_acceptor"] = df[acceptor_meta_col] >= thr_a

    # ------------------------------------------------------------------
    # 2b. Five-number summary of meta scores per true splice type ------------
    # ------------------------------------------------------------------
    if verbose:
        label_norm = df[label_col].astype(str).str.replace("0", "neither")
        df["_label_norm"] = label_norm
        stats_out = []
        for stype, col in (("donor", donor_meta_col), ("acceptor", acceptor_meta_col), ("neither", "neither_meta")):
            vals = df.loc[df["_label_norm"] == stype, col].dropna().to_numpy()
            if vals.size:
                stats_out.append((stype,
                                  float(_np.min(vals)),
                                  float(_np.max(vals)),
                                  float(_np.mean(vals)),
                                  float(_np.median(vals)),
                                  float(_np.std(vals))))
        if stats_out:
            print("\n[meta_splice_eval] Meta-model probability 5-number summary (by true site type):")
            print("type     n        min    max    mean   median   std")
            for stype, mn, mx, mean, med, std in stats_out:
                n = int((df["_label_norm"] == stype).sum())
                print(f"{stype:<9}{n:8d}  {mn:5.3f}  {mx:5.3f}  {mean:5.3f}  {med:5.3f}  {std:5.3f}")
        df.drop(columns=["_label_norm"], inplace=True)

    # ------------------------------------------------------------------
    # 3. Per-gene, per-site-type metrics ------------------------------------
    # ------------------------------------------------------------------
    records: list[dict[str, _np.int64]] = []
    for gene_id, gdf in df.groupby(gene_col):
        for stype in ("donor", "acceptor"):
            truth_mask = gdf[label_col] == stype
            base_pred = gdf[f"pred_base_{stype}"]
            meta_pred = gdf[f"pred_meta_{stype}"]

            tp0 = int((base_pred & truth_mask).sum())
            fp0 = int((base_pred & ~truth_mask).sum())
            fn0 = int((~base_pred & truth_mask).sum())

            tp = int((meta_pred & truth_mask).sum())
            fp = int((meta_pred & ~truth_mask).sum())
            fn = int((~meta_pred & truth_mask).sum())

            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

            records.append({
                gene_col: gene_id,
                "site_type": stype,
                "TP0": tp0,
                "TP": tp,
                "FP0": fp0,
                "FP": fp,
                "FN0": fn0,
                "FN": fn,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1_score": round(f1, 3),
            })

    out_df = _pd.DataFrame.from_records(records)
    # ------------------------------------------------------------------
    # 4. Save & report -------------------------------------------------------
    # ------------------------------------------------------------------
    run_dir = Path(run_dir)
    out_path = Path(out_tsv) if out_tsv is not None else run_dir / "meta_vs_base_performance.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)

    if verbose:
        print("[meta_splice_eval] Sample performance summary (first 20 rows):")
        print(out_df[[gene_col, "site_type", "precision", "recall", "f1_score", "TP", "FP", "FN"]].head(20).to_string(index=False))

        print("\n[meta_splice_eval] Sample before/after summary (first 20 rows):")
        print(out_df.head(20).to_string(index=False))

        global_base_f1 = _np.nan_to_num(2 * out_df["TP0"].sum() / (2 * out_df["TP0"].sum() + out_df["FP0"].sum() + out_df["FN0"].sum()))
        global_meta_f1 = _np.nan_to_num(2 * out_df["TP"].sum() / (2 * out_df["TP"].sum() + out_df["FP"].sum() + out_df["FN"].sum()))
        print(f"\n[meta_splice_eval] Global F1: base {global_base_f1:.3f} → meta {global_meta_f1:.3f} (Δ {global_meta_f1 - global_base_f1:+.3f})")

    return out_path

###############################################################################
# CLI -------------------------------------------------------------------------
###############################################################################

def _main(argv: List[str] | None = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Evaluate meta-model splice-site performance.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-dir", required=True, help="Directory containing trained meta-model artefacts")
    p.add_argument("--annotations", default=None, help="Optional splice-site annotation file (parquet/tsv/csv). If omitted, ground-truth sites are inferred from the dataset's own site_type / position columns.")
    p.add_argument("--out-tsv", default=None, help="Output TSV path (default: run-dir/full_splice_performance_meta.tsv)")
    p.add_argument("--threshold", type=float, default=None, help="Fallback threshold if per-class values are not found (default 0.9)")
    p.add_argument("--threshold-donor", type=float, default=None, help="Explicit donor-class threshold (overrides auto)")
    p.add_argument("--threshold-acceptor", type=float, default=None, help="Explicit acceptor-class threshold (overrides auto)")
    p.add_argument("--consensus-window", type=int, default=2)
    p.add_argument("--sample", type=int, default=None, help="Optional row sample for speed")
    p.add_argument("--verbose", dest='verbose', type=int, nargs='?', const=1, default=0, 
                   help="Set verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Use without value for level 1.")

    p.add_argument("--base-tsv", default=None, help="Path to base model full_splice_performance.tsv for comparison")
    p.add_argument("--missing-policy", choices=["skip", "zero", "predict"], default="skip",
                    help="How to handle positions missing from the input dataset: "
                         "'skip' (ignore; default), 'zero' (fill with 0.0), 'predict' (rebuild features and predict)")
    # Note: threshold-donor and threshold-acceptor are used both for evaluation metrics
    # and for prediction confidence thresholds when missing_policy="predict"
    p.add_argument("--include-tns", action="store_true", help="Include TN counts and derived metrics")
    p.add_argument("--errors-only", action="store_true", help="Restrict evaluation to FP/FN rows when displaying metrics (evaluation still uses full dataset)")
    p.add_argument("--error-artifact", default=None, help="Explicit path to FP/FN artifact when --errors-only is set")
    p.add_argument("--recompute-meta-scores", action="store_true", help="Force recomputation of meta-model probabilities even if donor_meta/acceptor_meta columns are present")
    args = p.parse_args(argv)

    out_path = meta_splice_performance(
        # ------------------------------------------------------------------
        # Primary inputs -----------------------------------------------------
        # ------------------------------------------------------------------
        dataset_path=args.dataset,         # Feature dataset (Parquet/TSV) with per-position meta-model scores
        run_dir=args.run_dir,              # Directory holding run artefacts (base TSV, logs, plots)
        annotations_path=args.annotations, # Splice-site truth table (`splice_sites.tsv`) with absolute coords
        # ------------------------------------------------------------------
        # Evaluation controls -------------------------------------------------
        # ------------------------------------------------------------------
        threshold=args.threshold,
        consensus_window=args.consensus_window,
        sample=args.sample,                # Randomly evaluate only N rows (None ⇒ full dataset)
        out_tsv=args.out_tsv,              # Path for the per-gene metrics TSV (defaults inside run_dir)
        base_tsv=args.base_tsv,
        error_artifact=args.error_artifact, # Optional FP/FN artefact to augment metrics
        errors_only=args.errors_only,
        missing_policy=args.missing_policy,
        include_tns=args.include_tns,
        verbose=args.verbose,
        threshold_donor=args.threshold_donor,
        threshold_acceptor=args.threshold_acceptor,
        recompute_meta_scores=args.recompute_meta_scores,  # Ignore cached *_meta columns and recalc

    )
    n_rows = pd.read_csv(out_path, sep="\t").shape[0]
    print(f"[meta_splice_eval] wrote {n_rows:,} rows → {out_path}")


if __name__ == "__main__":  # pragma: no cover
    _main()


