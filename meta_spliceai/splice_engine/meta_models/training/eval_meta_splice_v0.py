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

from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils

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
        if "position" not in ann_df.columns:
            raise KeyError("Annotation file must contain a 'position' column with genomic coordinates")
        gene_col = "gene_id" if "gene_id" in ann_df.columns else "gene"
        for row in ann_df.select([gene_col, "site_type", "position"]).iter_rows():
            gid, stype, pos = row
            if stype not in ("donor", "acceptor"):
                continue
            truth_lookup[gid][stype].add(int(pos))
    else:
        total_donor_sites = 0
        total_acceptor_sites = 0
        for gid, info in pred_dict.items():
            donor_sites = info.get("truth_donor", set())
            acceptor_sites = info.get("truth_acceptor", set())
            truth_lookup[gid]["donor"].update(donor_sites)
            truth_lookup[gid]["acceptor"].update(acceptor_sites)
            total_donor_sites += len(donor_sites)
            total_acceptor_sites += len(acceptor_sites)
        
        # Diagnostic logging to verify we're collecting truth sites
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
            # keep only peaks
            peaks: List[int] = []
            for idx in above:
                lo = max(0, idx - window)
                hi = min(len(prob_arr), idx + window + 1)
                if prob_arr[idx] >= np.nanmax(prob_arr[lo:hi]):
                    peaks.append(idx)
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
    if missing_policy == "predict":
        raise NotImplementedError("missing_policy='predict' is not yet implemented – choose 'skip' or 'zero' for now.")

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
    # Fallbacks
    if threshold_donor is None or threshold_acceptor is None:
        fallback = threshold if threshold is not None else thr_map.get("threshold_global", 0.9)
        threshold_donor = threshold_donor if threshold_donor is not None else fallback
        threshold_acceptor = threshold_acceptor if threshold_acceptor is not None else fallback

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

    lf = lf.select([c for c in unique_cols if c in lf.columns])

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
    def _update_pred_dict(pred_dict: Dict[str, Dict[str, Any]], batch_df: pd.DataFrame, proba_batch: np.ndarray, *, verbose: bool = False, missing_policy: str = "skip", batch_count: int = 0) -> None:
        """Update the prediction dictionary with a batch of predictions and dataset rows.
        
        This is a batched update function that adds model probabilities and truth sites
        to the prediction dictionary.
        """
        
        # Debug print when first called
        if verbose and batch_count <= 1:
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
        batch_df["donor_meta"] = proba_batch[:, 1]
        batch_df["acceptor_meta"] = proba_batch[:, 2]

        # Infer relative position column once per batch
        # Handle the case where training data uses 'position' and not 'rel_pos'
        if "position" in batch_df.columns and "rel_pos" not in batch_df.columns:
            print("[meta_splice_eval] Using 'position' column as relative position")
            batch_df["rel_pos"] = batch_df["position"]
        else:
            rel_pos_series = _infer_rel_pos(pl.from_pandas(batch_df))
            batch_df["rel_pos"] = rel_pos_series.to_numpy()

        for gid, sub in batch_df.groupby("gene_id", sort=False):
            first_row = sub.iloc[0]
            gene_len = int(first_row["gene_end"] - first_row["gene_start"] + 1)
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
                    "gene_start": int(first_row["gene_start"]),
                    "gene_end": int(first_row["gene_end"]),
                    "donor_prob": donor_arr,
                    "acceptor_prob": accept_arr,
                    "covered_abs": set() if missing_policy == "skip" else None,
                    "truth_donor": set(),
                    "truth_acceptor": set(),
                }
            else:
                donor_arr = pred_dict[gid]["donor_prob"]
                accept_arr = pred_dict[gid]["acceptor_prob"]

            rel_idx = sub["rel_pos"].astype(int).to_numpy()
            if rel_idx.max() >= gene_len:
                rel_idx = rel_idx - 1  # off-by-one safeguard

            donor_arr[rel_idx] = sub["donor_meta"].values.astype(np.float32)
            accept_arr[rel_idx] = sub["acceptor_meta"].values.astype(np.float32)

            # Update truth sets if present in batch
            if "site_type" in sub.columns or "splice_type" in sub.columns or "mapped_type" in sub.columns:
                _stype_col = "site_type" if "site_type" in sub.columns else "splice_type" if "splice_type" in sub.columns else "mapped_type"
                
                # Show detailed debug info about the splice types
                if verbose and batch_count <= 1:
                    print(f"\n[DEBUG] Gene {gid}: Processing")
                    print(f"  Total rows for gene: {len(sub)}")
                    if len(sub) > 0:
                        print(f"  First row splice_type: '{sub[_stype_col].iloc[0]}' (type: {type(sub[_stype_col].iloc[0]).__name__})")
                    print(f"  Unique {_stype_col} values: {sorted(sub[_stype_col].unique())}")
                    print(f"  {_stype_col} value counts: {sub[_stype_col].value_counts().to_dict()}")
                    print(f"  {_stype_col} data type: {sub[_stype_col].dtype}")
                
                # Filter for actual splice sites (donor or acceptor)
                # Try both exact string match and case-insensitive matching
                donor_mask = (sub[_stype_col] == "donor") | (sub[_stype_col].astype(str).str.lower() == "donor")
                acceptor_mask = (sub[_stype_col] == "acceptor") | (sub[_stype_col].astype(str).str.lower() == "acceptor")
                splice_mask = donor_mask | acceptor_mask
                
                if verbose and batch_count <= 1:
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
                        print("\n[DEBUG] No splice sites matched using standard filters! Sample raw values:")
                        for i in range(min(5, len(sub))):
                            print(f"  Row {i} {_stype_col}: '{sub[_stype_col].iloc[i]}' (type: {type(sub[_stype_col].iloc[i]).__name__})")
                        
                        # Try alternative matching approaches
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
                        if verbose and batch_count <= 1:
                            print(f"[DEBUG] Using true_position column for splice sites")
                    else:
                        splice_pos = splice_sites["rel_pos"].astype(int).to_numpy()
                        if verbose and batch_count <= 1:
                            print(f"[DEBUG] Using rel_pos column for splice sites")
                    
                    # Get the splice types (donor/acceptor)
                    splice_types = splice_sites["splice_type"].astype(str).tolist()
                    
                    # If absolute_position is already available, use it directly
                    if "absolute_position" in splice_sites.columns:
                        abs_positions = splice_sites["absolute_position"].astype(float).to_numpy()
                        if verbose and batch_count <= 1:
                            print(f"[DEBUG] Using absolute_position column directly")
                    else:
                        # Convert relative positions to absolute positions
                        abs_positions = _abs_positions(
                            splice_pos,
                            strand=sub["strand"].iloc[0],
                            gene_start=gene_starts.get(gid, None),
                            transcript_start=tx_starts.get(gid, None),
                            window_start=window_starts.get(gid, None),
                        )
                    
                    # Additional debug info for first few entries
                    if verbose and batch_count <= 1 and len(splice_sites) > 0:
                        print("\n[DEBUG] Sample splice site data:")
                        sample_size = min(5, len(splice_sites))
                        for i in range(sample_size):
                            print(f"  Row {i+1}: splice_type={splice_sites['splice_type'].iloc[i]}, "
                                  f"position={splice_sites['position'].iloc[i]}, "
                                  f"abs_position={abs_positions[i]}, "
                                  f"true_position={splice_sites['true_position'].iloc[i] if 'true_position' in splice_sites.columns else 'N/A'}, "
                                  f"strand={splice_sites['strand'].iloc[i]}")
                    
                    # Update truth sets for donor and acceptor positions
                    updated_donor_count = 0
                    updated_acceptor_count = 0
                    for abs_pos, st in zip(abs_positions, splice_types):
                        if not pd.isna(abs_pos) and abs_pos > 0:  # Skip NaN or invalid positions
                            if st == "donor":
                                pred_dict[gid]["truth_donor"].add(int(abs_pos))
                                updated_donor_count += 1
                            elif st == "acceptor":
                                pred_dict[gid]["truth_acceptor"].add(int(abs_pos))
                                updated_acceptor_count += 1
                    
                    if verbose and batch_count <= 1:
                        print(f"[DEBUG] Added {updated_donor_count} donor sites and {updated_acceptor_count} acceptor sites to truth sets for gene {gid}")
                        print(f"[DEBUG] Truth sets now contain: {len(pred_dict[gid]['truth_donor'])} donor, {len(pred_dict[gid]['truth_acceptor'])} acceptor sites")
            
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
            scan = scan.sample(n=sample, seed=42)

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
                        print(f"\n[DEBUG] Splice type values and counts in sample genes:")
                        print(sample_df["splice_type"].value_counts().to_dict())

            use_cached = (not recompute_meta_scores and "donor_meta" in batch_df.columns and "acceptor_meta" in batch_df.columns)
            if use_cached:
                donor_arr_cached = batch_df["donor_meta"].to_numpy(np.float32)
                accept_arr_cached = batch_df["acceptor_meta"].to_numpy(np.float32)
                proba_batch = np.stack([
                    np.zeros_like(donor_arr_cached),
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
                            # Apply the same encoding as training: map unique values to integers
                            unique_vals = feature_batch[col].dropna().unique()
                            val_map = {val: float(idx) for idx, val in enumerate(sorted(unique_vals))}
                            feature_batch[col] = feature_batch[col].map(val_map).fillna(-1.0).astype(np.float32)
                
                X_batch = feature_batch.to_numpy(np.float32)
                proba_batch = predict_fn(X_batch)
            _update_pred_dict(pred_results, batch_df, proba_batch, verbose=verbose, missing_policy=missing_policy, batch_count=batch_count)

    # ------------------------------------------------------------------
    # At this point *pred_results* has full per-gene arrays; proceed as usual
    # ------------------------------------------------------------------

    # Load annotations (optional)
    ann_df: pl.DataFrame | None = None
    if annotations_path is not None:
        annotations_path = Path(annotations_path) if annotations_path is not None else None
        if not annotations_path.exists():
            print(f"[meta_splice_eval] WARNING: annotation file {annotations_path} missing – falling back to dataset-internal truth columns.")
        else:
            ann_ext = annotations_path.suffix.lower()
            if ann_ext == ".parquet":
                ann_df = pl.read_parquet(annotations_path)
            elif ann_ext in (".tsv", ".csv"):
                sep = "\t" if ann_ext == ".tsv" else ","
                try:
                    ann_df = pl.read_csv(annotations_path, separator=sep)
                except Exception:
                    # Mixed dtypes (chrom) are common – force Utf8
                    ann_df = pl.read_csv(annotations_path, separator=sep, dtypes={"chrom": pl.Utf8})
            else:
                raise ValueError("Unsupported annotation file format: " + annotations_path.name)

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
            print("[meta_splice_eval] Base performance file missing required columns – skipping per-gene comparison table")
    else:
        print("[meta_splice_eval] No base performance TSV found – skipping comparison.")

    return out_tsv


################################################################################
# CLI
################################################################################


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
    p.add_argument("--verbose", action="store_true", help="Enable verbose output")
    p.add_argument("--base-tsv", default=None, help="Path to base model full_splice_performance.tsv for comparison")
    p.add_argument("--missing-policy", choices=["skip", "zero", "predict"], default="skip",
                    help="How to handle positions missing from the input dataset: "
                         "'skip' (ignore; default), 'zero' (fill with 0.0), 'predict' (rebuild features and predict)")
    p.add_argument("--include-tns", action="store_true", help="Include TN counts and derived metrics")
    p.add_argument("--errors-only", action="store_true", help="Restrict evaluation to FP/FN rows when displaying metrics (evaluation still uses full dataset)")
    p.add_argument("--error-artifact", default=None, help="Explicit path to FP/FN artifact when --errors-only is set")
    p.add_argument("--recompute-meta-scores", action="store_true", help="Force recomputation of meta-model probabilities even if donor_meta/acceptor_meta columns are present")
    args = p.parse_args(argv)

    out_path = meta_splice_performance(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        annotations_path=args.annotations,
        threshold=args.threshold,
        consensus_window=args.consensus_window,
        sample=args.sample,
        out_tsv=args.out_tsv,
        base_tsv=args.base_tsv,
        error_artifact=args.error_artifact,
        errors_only=args.errors_only,
        missing_policy=args.missing_policy,
        include_tns=args.include_tns,
        verbose=args.verbose,
        threshold_donor=args.threshold_donor,
        threshold_acceptor=args.threshold_acceptor,
        recompute_meta_scores=args.recompute_meta_scores,
    )
    n_rows = pd.read_csv(out_path, sep="\t").shape[0]
    print(f"[meta_splice_eval] wrote {n_rows:,} rows → {out_path}")


if __name__ == "__main__":  # pragma: no cover
    _main()
