#!/usr/bin/env python3
"""Leave-One-Chromosome-Out (LOCO-CV) driver for the 3-class meta-model.

This script automates full-genome chromosome-aware evaluation:
    • Iterates over *chromosome groups* produced by
      ``chromosome_split.group_chromosomes`` (small chromosomes are bucketed so
      each test fold has ≥ ``--min-rows-test`` rows).
    • Trains an XGBoost multiclass classifier on the remaining chromosomes.
    • Reports accuracy & macro-F1 on the held-out group.
    • Saves per-fold JSON + a CSV summarising all folds.

Example (GPU histogram):
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass \
    --dataset train_pc_1000/master \
    --out-dir runs/loco_cv_gpu \
    --tree-method gpu_hist --max-bin 256 \
    --n-estimators 1200
```

The aggregated CSV at *out-dir*/``loco_metrics.csv`` contains one row per held-out
chromosome group.  The mean of the metric columns gives the overall LOCO-CV
score.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import gc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
from xgboost import XGBClassifier

from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training import chromosome_split as csplit
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels, _preprocess_features_for_model

# Import new scalability utilities
try:
    from meta_spliceai.splice_engine.meta_models.training import scalability_utils
    SCALABILITY_UTILS_AVAILABLE = True
except ImportError:
    SCALABILITY_UTILS_AVAILABLE = False
    print("Warning: scalability_utils module not available. Running in compatibility mode.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="LOCO-CV for the 3-way meta-classifier.")
    p.add_argument("--dataset", required=True, help="Dataset directory or single Parquet file")
    p.add_argument("--out-dir", required=True, help="Directory to save fold metrics & aggregates")

    p.add_argument("--group-col", default="chrom", help="Column with chromosome labels (default: chrom)")
    p.add_argument("--gene-col", default="gene_id", help="Column with gene IDs for group split")

    p.add_argument("--base-tsv", help="Path to base model full_splice_performance.tsv for comparison")
    p.add_argument("--errors-only", action="store_true", help="Evaluate only rows where base model was FP/FN (uses artifacts pred_type)")

    p.add_argument("--row-cap", type=int, default=100_000, help="Row cap to fit in memory (0 = disabled)")
    p.add_argument("--valid-size", type=float, default=0.15, help="Fraction of full dataset for validation")
    p.add_argument("--min-rows-test", type=int, default=1_000, help="Minimum rows per test fold after grouping small chromosomes")
    p.add_argument("--heldout-chroms", default="", help="Comma-separated list of chromosomes to reserve as a single test set (e.g. '1,3,5,7,9'). When provided, overrides LOCO-CV and runs one fixed split like SpliceAI.")
    p.add_argument("--diag-sample", type=int, default=25_000, help="Rows to sample for richer_metrics diagnostics (0 = full)")
    p.add_argument("--annotations", default=None, help="[DEPRECATED] Use --splice-sites-path instead. Kept for backward compatibility.")
    p.add_argument("--splice-sites-path", default="data/ensembl/splice_sites.tsv", 
                   help="Path to splice site annotations file (parquet/csv/tsv)")
    p.add_argument("--neigh-sample", type=int, default=0, help="If >0 run neighbour-window diagnostic with this many random genes")
    p.add_argument("--neigh-window", type=int, default=10, help="Neighbourhood size for neighbour-window diagnostic")
    p.add_argument("--transcript-topk", action="store_true", help="Calculate transcript-level top-k accuracy")
    p.add_argument("--no-transcript-cache", action="store_true", help="Disable caching for transcript-level accuracy calculations")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--tree-method", default="hist", choices=["hist", "gpu_hist", "approx"], help="Underlying XGBoost algorithm")
    p.add_argument("--max-bin", type=int, default=256, help="Max bin (hist algorithms only)")
    p.add_argument("--device", default="auto", help="XGBoost device parameter (cuda|cpu|auto)")
    p.add_argument("--n-estimators", type=int, default=800)

    p.add_argument("--leakage-probe", action="store_true", help="Run leakage correlation probe after training")

    p.add_argument("--feature-selection", action="store_true", help="Enable feature selection to reduce dimensionality")
    p.add_argument("--max-features", type=int, default=1000, help="Maximum number of features to select when feature selection is enabled")
    p.add_argument("--selection-method", type=str, default="model", choices=["model", "mutual_info"], help="Method to use for feature selection")
    p.add_argument("--exclude-features", type=str, help="Path to a file with features to exclude, one per line")
    p.add_argument("--force-features", type=str, help="Path to a file with features to always include, one per line")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Honour row cap via env var understood by datasets.load_dataset
    import os
    if args.row_cap > 0 and not os.getenv("SS_MAX_ROWS"):
        os.environ["SS_MAX_ROWS"] = str(args.row_cap)

    # ---------------------------------------------------------------------
    # 1. Data loading and preprocessing
    # ---------------------------------------------------------------------
    df, feature_names = datasets.load_dataset(
        args.dataset, row_cap=args.row_cap, random_state=args.seed
    )
    # Assign columns to X, y, and other arrays
    y = df["label"].values  # Use "label" as the standard column name
    chrom_array = df[args.group_col].values
    gene_array = df[args.gene_col].values if args.gene_col in df.columns else None

    # Preprocess features
    # First convert any object columns to a numeric representation
    df, feature_names = _preprocess_features_for_model(df, feature_names)

    # Use chromosome as a feature or not
    if args.group_col != "chrom" or args.group_col not in feature_names:
        logger.info("Note: group column is not used as a training feature")
        X = df[feature_names]
    else:
        logger.info(f"Using '{args.group_col}' as a training feature with numeric encoding")
        X = df[feature_names]

    # ---------------------------------------------------------------------
    # 1.5 Feature selection (if enabled)
    # ---------------------------------------------------------------------
    feature_selection_info = {}
    
    if args.feature_selection and SCALABILITY_UTILS_AVAILABLE:
        # Load exclude/include feature lists if provided
        exclude_features = []
        force_include_features = []
        
        if args.exclude_features and os.path.exists(args.exclude_features):
            with open(args.exclude_features, 'r') as f:
                exclude_features = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(exclude_features)} features to exclude")
        
        if args.force_features and os.path.exists(args.force_features):
            with open(args.force_features, 'r') as f:
                force_include_features = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(force_include_features)} features to force include")
        
        # Run feature selection
        logger.info(f"Running feature selection using '{args.selection_method}' method")
        logger.info(f"Selecting up to {args.max_features} features from {len(feature_names)}")
        
        start_time = pd.Timestamp.now()
        X_selected, selected_features, feature_info = scalability_utils.select_features(
            X, y,
            max_features=args.max_features,
            method=args.selection_method,
            random_state=args.seed,
            exclude_features=exclude_features,
            force_include_features=force_include_features
        )
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Update feature information
        logger.info(f"Feature selection completed in {duration:.2f} seconds")
        logger.info(f"Selected {len(selected_features)} features out of {len(feature_names)}")
        logger.info(f"  - K-mer features: {feature_info['selected_kmer_features']} (was {feature_info['original_kmer_features']})")
        logger.info(f"  - Non-k-mer features: {feature_info['selected_non_kmer_features']} (was {feature_info['original_non_kmer_features']})")
        
        # Update X and feature_names
        X = X_selected
        feature_names = selected_features
        feature_selection_info = feature_info
        
        # Save feature selection information
        with open(out_dir / "feature_selection_info.json", "w") as f:
            json.dump(feature_info, f, indent=2)
        
        # Save selected feature list
        with open(out_dir / "selected_features.txt", "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
    else:
        # Convert to numpy array for models that need it
        X = X.values

    # Basic feature stats reporting
    logger.info(f"[LOCO] Feature matrix: {len(feature_names)} total columns")
    non_kmer_features = [f for f in feature_names if not f.startswith("6mer_")]
    kmer_features = [f for f in feature_names if f.startswith("6mer_")]
    kmer_examples = sorted(kmer_features)[:3] if kmer_features else []
    
    # Print feature stats
    logger.info(f"       Non-k-mer features ({len(non_kmer_features)}):", ", ".join(non_kmer_features))
    if kmer_features:
        logger.info(f"       Example k-mer features ({len(kmer_features)} total):", ", ".join(kmer_examples))

    # Always save transcript mapping columns if transcript-topk is enabled
    transcript_columns = {}
    if args.transcript_topk:
        # Always get the original string version of chrom from the input dataframe
        # This ensures we have it for transcript mapping regardless of encoding
        # Handle both pandas and polars dataframes
        if hasattr(df, 'to_pandas'):
            # It's a polars DataFrame
            pandas_df = df.select(['chrom', 'position']).to_pandas()
            transcript_columns['chrom'] = pandas_df['chrom']
            if 'position' in pandas_df.columns:
                transcript_columns['position'] = pandas_df['position']
        else:
            # It's already a pandas DataFrame
            transcript_columns['chrom'] = df['chrom'].copy()
            if 'position' in df.columns:
                transcript_columns['position'] = df['position'].copy()
        
        # Handle position if it's in X_df but not yet saved
        if 'position' in X_df.columns and 'position' not in transcript_columns:
            transcript_columns['position'] = X_df['position'].copy()
            X_df = X_df.drop(columns=['position'])
    
    # Preprocess features to handle non-numeric columns (k-mers, categorical features, etc.)
    feature_names = list(X_df.columns)
    processed_df = _preprocess_features_for_model(X_df, feature_names)
    X = processed_df.values
    y = _encode_labels(y_series)
    
    # Update feature names to reflect possibly modified column set
    feature_names = list(processed_df.columns)

    # ---------------------------------------------------------------------
    # Inspect feature names for potentially leaky or non-sequence columns
    # ---------------------------------------------------------------------
    import random
    from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature as _is_kmer

    non_kmer_feats = [f for f in feature_names if not _is_kmer(f)]
    kmer_feats = [f for f in feature_names if _is_kmer(f)]

    print(f"[LOCO] Feature matrix: {len(feature_names)} total columns")
    print(
        f"       Non-k-mer features ({len(non_kmer_feats)}): {', '.join(non_kmer_feats) if non_kmer_feats else '—'}"
    )
    if kmer_feats:
        sample_kmers = random.sample(kmer_feats, k=min(3, len(kmer_feats)))
        print(
            f"       Example k-mer features ({len(kmer_feats)} total): {', '.join(sample_kmers)}"
        )

    chrom = df[args.group_col].to_numpy()
    genes = df[args.gene_col].to_numpy()
    # Keep original chromosome array for fold diagnostics
    chrom_dataset = df["chrom"].to_numpy() if "chrom" in df.columns else None

    # ---------------------------------------------------------------------
    # 2. Either fixed chromosome hold-out or LOCO folds
    # ---------------------------------------------------------------------
    fold_rows: list[Dict[str, object]] = []
    last_model = None

    if args.heldout_chroms:
        heldout_list = [c.strip() for c in args.heldout_chroms.split(',') if c.strip()]
        tr_idx, val_idx, te_idx, *_ = csplit.holdout_split(
            X, y,
            chrom_array=chrom,
            holdout_chroms=heldout_list,
            valid_size=args.valid_size,
            gene_array=genes,
            seed=args.seed,
        )
        held_out = ",".join(heldout_list)
        print(f"[HOLDOUT] Test chroms: {held_out}  (rows={len(te_idx)})")
        if chrom_dataset is not None:
            test_chroms = np.unique(chrom_dataset[te_idx])
            train_chroms = np.unique(chrom_dataset[tr_idx])
            preview = ",".join(train_chroms[:5]) + (" ..." if len(train_chroms) > 5 else "")
            print(f"  test chroms: {','.join(test_chroms)}")
            print(f"  train chroms: {preview} total {len(train_chroms)}")
        # Reuse the same training code path via a helper
        loops = [(held_out, tr_idx, val_idx, te_idx)]
    else:
        loops = csplit.loco_cv_splits(
            X, y,
            chrom_array=chrom,
            gene_array=genes,
            valid_size=args.valid_size,
            min_rows=args.min_rows_test,
            seed=args.seed,
        )

    for held_out, tr_idx, val_idx, te_idx in loops:
        print(f"[LOCO] Held-out test set = {held_out}  (rows={len(te_idx)})")
        if chrom_dataset is not None:
            test_chroms = np.unique(chrom_dataset[te_idx])
            train_chroms = np.unique(chrom_dataset[tr_idx])
            preview = ",".join(train_chroms[:5]) + (" ..." if len(train_chroms) > 5 else "")
            print(f"  test chroms: {','.join(test_chroms)}")
            print(f"  train chroms: {preview} total {len(train_chroms)}")

        model = XGBClassifier(
            n_estimators=args.n_estimators,
            tree_method=args.tree_method,
            max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=args.seed,
            n_jobs=-1,
            device=args.device if args.device != "auto" else None,
        )

        model.fit(
            X[tr_idx],
            y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )

        proba = model.predict_proba(X[te_idx])
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y[te_idx], pred)
        macro_f1 = f1_score(y[te_idx], pred, average="macro")

        splice_mask = y[te_idx] != 0
        if splice_mask.any():
            splice_acc = accuracy_score(y[te_idx][splice_mask], pred[splice_mask])
            splice_macro_f1 = f1_score(y[te_idx][splice_mask], pred[splice_mask], average="macro")
        else:
            splice_acc = np.nan
            splice_macro_f1 = np.nan

        # Top-K accuracy
        k = int(splice_mask.sum())
        if k > 0:
            splice_prob = proba[:, 1] + proba[:, 2]
            top_idx = np.argsort(-splice_prob)[:k]
            top_k_correct = (y[te_idx][top_idx] != 0).sum()
            top_k_acc = top_k_correct / k
        else:
            top_k_acc = np.nan

        row = {
            "held_out": held_out,
            "test_rows": len(te_idx),
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
            "splice_accuracy": splice_acc,
            "splice_macro_f1": splice_macro_f1,
            "top_k_accuracy": top_k_acc,
        }
        fold_rows.append(row)

        with open(out_dir / f"metrics_{held_out}.json", "w") as fh:
            json.dump(row, fh, indent=2)

        cm = confusion_matrix(y[val_idx], model.predict(X[val_idx]), labels=[0, 1, 2])
        label_names = ["neither", "donor", "acceptor"]
        import pandas as _pd
        print("Class distribution (true labels):", {
            name: int((y[val_idx] == i).sum()) for i, name in enumerate(label_names)
        })
        print(_pd.DataFrame(cm, index=label_names, columns=label_names))

        last_model = model  # capture for saving after loop

    # ---------------------------------------------------------------------
    # 2b. Save final model & feature manifest for diagnostics
    # ---------------------------------------------------------------------
    # Train final model on *all* rows for diagnostics
    final_model = XGBClassifier(
        n_estimators=args.n_estimators,
        tree_method=args.tree_method,
        max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
        device=args.device if args.device != "auto" else None,
    )
    final_model.fit(X, y)

    # initial pickling so diagnostics can load the model
    import pickle
    with open(out_dir / "model_multiclass.pkl", "wb") as fh:
        pickle.dump(final_model, fh)
    # we will overwrite later after attaching optimal_threshold
    pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_manifest.csv", index=False)

    # ---------------------------------------------------------------------
    # 3. Output aggregated results
    # ---------------------------------------------------------------------
    df_metrics = pd.DataFrame(fold_rows)
    
    if len(fold_rows) == 0:
        print("\nWARNING: No valid folds were found with the current sample size.")
        print("Please try increasing the row-cap or reducing min-rows-test.")
        return
        
    with open(out_dir / "loco_metrics.csv", "w") as fh:
        df_metrics.to_csv(fh, index=False)

    print("\nLOCO-CV results by held-out chromosome group:\n")
    print(df_metrics)

    metric_columns = [
        "test_accuracy", "test_macro_f1", "splice_accuracy", "splice_macro_f1",
        "top_k_accuracy"
    ]
    # Ensure all required columns exist in the dataframe
    mean_metrics = df_metrics[metric_columns].mean()
    print("\nAverage across folds:")
    for name, val in mean_metrics.items():
        print(f"{name:>16s} {val:.6f}")

    # Save overall summary
    with open(out_dir / "metrics_aggregate.json", "w") as fh:
        json.dump(mean_metrics.to_dict(), fh, indent=2)

    # 5. Diagnostics & Post-training analysis
    diag_sample = None if args.diag_sample == 0 else args.diag_sample
    _cutils.richer_metrics(args.dataset, out_dir, sample=diag_sample)
    _cutils.gene_score_delta(args.dataset, out_dir, sample=diag_sample)
    _cutils.shap_importance(args.dataset, out_dir, sample=diag_sample)
    _cutils.probability_diagnostics(args.dataset, out_dir, sample=diag_sample)
    _cutils.base_vs_meta(args.dataset, out_dir, sample=diag_sample)

    # --- meta splice-site evaluation ---
    thresh = 0.9  # default threshold
    th_path = out_dir / "threshold_suggestion.txt"
    if th_path.exists():
        try:
            import pandas as _pd
            ts_df = _pd.read_csv(th_path, sep="\t", header=None, names=["key", "value"])
            if "threshold_global" in ts_df["key"].values:
                thresh = float(ts_df.loc[ts_df["key"]=="threshold_global", "value"].iloc[0])
                print(f"[LOCO-CV] Using suggested threshold {thresh:.3f} from probability_diagnostics")
            elif "best_threshold" in ts_df["key"].values:
                thresh = float(ts_df.loc[ts_df["key"]=="best_threshold", "value"].iloc[0])
                print(f"[LOCO-CV] Using suggested threshold {thresh:.3f} from probability_diagnostics")
            else:
                print(f"[LOCO-CV] Using default threshold {thresh:.3f}")
            # capture per-class thresholds if present
            for _cls, attr in [("threshold_donor", "threshold_donor"), ("threshold_acceptor", "threshold_acceptor")]:
                if _cls in ts_df["key"].values:
                    try:
                        setattr(final_model, attr, float(ts_df.loc[ts_df["key"]==_cls, "value"].iloc[0]))
                    except Exception:
                        pass
        except Exception as _e:
            print("[LOCO-CV] Failed to parse threshold_suggestion.txt:", _e)

    # ------------------------------------------------------------------
    # Persist final model with attached optimal threshold for inference
    # ------------------------------------------------------------------
    try:
        final_model.optimal_threshold = thresh  # binary cutoff
        final_model.threshold_neither = thresh  # for completeness
        # donor/acceptor thresholds set earlier if available
    except Exception:
        pass  # guard: attribute injection should never fail
    import pickle
    with open(out_dir / "model_multiclass.pkl", "wb") as fh:
        pickle.dump(final_model, fh)
        
    from pathlib import Path as _Path
    base_tsv = _Path(args.base_tsv) if args.base_tsv else None
    if base_tsv and not base_tsv.exists():
        base_tsv = None

    # Perform full evaluation
    try:
        # Default evaluation at dataset level
        _cutils.meta_splice_performance(
            dataset_path=args.dataset,
            run_dir=out_dir,
            annotations_path=args.annotations,
            threshold=thresh,
            base_tsv=base_tsv,
            errors_only=args.errors_only,
            sample=None,
            verbose=1,
        )
        
        # Optional transcript-level top-k accuracy
        if args.transcript_topk:
            # Import here to avoid circular dependencies
            from meta_spliceai.splice_engine.meta_models.evaluation.transcript_mapping import (
                calculate_transcript_level_top_k,
                report_transcript_top_k
            )
            print("[LOCO-CV] Calculating transcript-level top-k accuracy...")
            try:
                # Check if we have the necessary transcript mapping columns
                if 'position' in transcript_columns and 'chrom' in transcript_columns:
                    # Create a dedicated DataFrame for transcript mapping
                    test_indices = np.where(chr_targets == hold_out_chr)[0]
                    
                    # Prepare the test data for transcript mapping
                    test_tx_df = pd.DataFrame({
                        'position': transcript_columns['position'].iloc[test_indices].values,
                        'chrom': transcript_columns['chrom'].iloc[test_indices].values,
                        'label': y[test_indices],
                        'prob_donor': y_proba_meta[test_indices, 0],      # Meta-model probs
                        'prob_acceptor': y_proba_meta[test_indices, 1],  # Meta-model probs 
                    })
                    
                    # Calculate transcript-level metrics
                    transcript_top_k_metrics = calculate_transcript_level_top_k(
                        df=test_tx_df,
                        splice_sites_path=args.splice_sites_path,
                        transcript_features_path=args.transcript_features_path,
                        gene_features_path=args.gene_features_path,
                        donor_label=0,     # Meta-model uses 0=donor
                        acceptor_label=1,  # 1=acceptor
                        neither_label=2,   # 2=neither
                        position_col="position",
                        chrom_col="chrom",
                        label_col="label",  # Using label column for site types
                        use_cache=not args.no_transcript_cache
                    )
                    
                    # Print and record transcript-level metrics
                    print("\n" + report_transcript_top_k(transcript_top_k_metrics))
                    with open(out_dir / "transcript_topk_metrics.json", "w") as f:
                        json.dump(transcript_top_k_metrics, f, indent=2)
                else:
                    # Provide detailed feedback on missing columns
                    missing = []
                    if 'position' not in transcript_columns:
                        missing.append("'position'")
                    if 'chrom' not in transcript_columns:
                        missing.append("'chrom'")
                    print(f"[warning] Skipping transcript-level metrics: Missing required columns {', '.join(missing)}")
                    print("Make sure both 'chrom' and 'position' are preserved for transcript mapping")
            except Exception as e:
                print(f"[warning] Transcript-level top-k calculation failed: {e}")
        
        # NEW: meta vs base comparison table
        try:
            meta_tsv_path = _Path(out_dir) / "full_splice_performance_meta.tsv"
            base_tsv_candidate = _Path(base_tsv) if base_tsv else (_Path(out_dir) / "full_splice_performance.tsv")
            cmp_out = _Path(out_dir) / "perf_meta_vs_base.tsv"
            if meta_tsv_path.exists() and base_tsv_candidate.exists():
                print(f"[LOCO-CV] Generating performance comparison → {cmp_out}")
                try:
                    _cutils.compare_splice_performance(
                        meta_tsv=meta_tsv_path,
                        base_tsv=base_tsv_candidate,
                        out_tsv=cmp_out,
                    )
                    print(f"[LOCO-CV] Saved performance comparison to {cmp_out}")
                except Exception as exc:
                    print(f"[LOCO-CV] compare_splice_performance failed: {exc}")
            else:
                print(f"[LOCO-CV] Skipping compare_splice_performance – TSV(s) missing (meta_tsv: {meta_tsv_path.exists()}, base_tsv: {base_tsv_candidate.exists()})")
                # Show current expected paths
                print(f"  Meta TSV: {meta_tsv_path}")
                print(f"  Base TSV: {base_tsv_candidate}")
        except Exception as e:
            print("[warning] compare_splice_performance failed:", e)
            
    except Exception as e:
        print("[warning] meta_splice_performance failed:", e)

    # Optional neighbour-window diagnostic
    if args.neigh_sample > 0:
        try:
            _cutils.neighbour_window_diagnostics(
                dataset_path=args.dataset,
                run_dir=out_dir,
                annotations_path=args.annotations,  # Now optional - can be None
                n_sample=args.neigh_sample,
                window=args.neigh_window,
            )
        except Exception as e:
            print("[warning] neighbour_window_diagnostics failed:", e)

    # Optional leakage probe
    if args.leakage_probe:
        try:
            # Use smaller sample size to avoid OOM
            _cutils.leakage_probe(args.dataset, out_dir, sample=10_000)
        except Exception as e:
            print("[warning] leakage_probe failed:", e)
            
    print("[LOCO-CV] Diagnostics complete – see generated CSV/JSON artefacts in run-dir.")


if __name__ == "__main__":  # pragma: no cover
    main()
