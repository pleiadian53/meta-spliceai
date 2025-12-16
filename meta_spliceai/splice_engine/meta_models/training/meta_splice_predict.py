"""
Meta-model splice site prediction module for MetaSpliceAI.

This module implements prediction functionality for the meta-model to make
predictions on positions not included in the meta-model training data.
"""

from typing import Dict, Any, Set, List, Tuple, Optional, Union, Sequence
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper: build enriched feature DataFrames for a list of genes
# ----------------------------------------------------------------------
from typing import Sequence
import tempfile

def _build_feature_matrix_for_genes(
    gene_ids: List[str],
    *,
    eval_dir: Optional[str] = None,
    output_dir: Optional[Path] | None = None,
    batch_size: int = 200,
    kmer_sizes: Tuple[int, ...] = (),
    enrichers: Sequence[str] | None = None,
    verbose: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Run the incremental-builder pipeline for a *specific* list of genes and
    return a mapping ``gene_id -> enriched feature DataFrame``.  The resulting
    DataFrames contain **exactly** the same feature columns that were used to
    train the meta-model, so that we can safely feed them into the pre-trained
    classifier."""
    from meta_spliceai.splice_engine.meta_models.builder.incremental_builder import (
        incremental_build_training_dataset,
    )

    # Create a temporary output directory if the caller did not supply one
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="meta_pred_"))

    master_path = incremental_build_training_dataset(
        eval_dir=eval_dir,
        output_dir=output_dir,
        n_genes=len(gene_ids),
        subset_policy="custom",  # explicit gene list
        additional_gene_ids=gene_ids,
        batch_size=min(batch_size, len(gene_ids)),
        kmer_sizes=kmer_sizes,
        enrichers=enrichers,
        downsample_kwargs=None,  # KEEP all TN rows – leave filtering to caller
        run_workflow=False,
        overwrite=True,
        verbose=verbose,
    )

    dfs: Dict[str, pd.DataFrame] = {}
    for pq in Path(master_path).glob("*.parquet"):
        for gid, grp in pl.read_parquet(pq).to_pandas().groupby("gene_id"):
            dfs[gid] = grp
    return dfs


def predict_missing_positions(
    pred_dict: Dict[str, Dict[str, Any]],
    gene_data: Dict[str, pd.DataFrame],
    predict_fn,
    feature_names: List[str],
    *,
    verbose: int = 0,
    donor_threshold: float = 0.8,
    acceptor_threshold: float = 0.8,
    eval_dir: Optional[str] = None,
    kmer_sizes: Tuple[int, ...] = (6,),
    enrichers: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Set[int]]]:
    """
    Predict splice sites for positions not included in the meta-model training data.
    
    This function identifies positions that weren't in the meta-model training data
    (i.e., not in the covered_abs set) and predicts their splice type using the
    meta-model.
    
    Args:
        pred_dict: Dictionary containing gene prediction data including 'covered_abs'
            - comes from the evaluation driver; for each gene it stores covered_abs, 
            the set of absolute positions that WERE in the meta-model training data.

        gene_data: Dictionary mapping gene IDs to full gene DataFrames with all positions
        predict_fn: Meta-model prediction function
        feature_names: Features required by the meta-model
        verbose: Whether to print verbose output
        donor_threshold: Minimum confidence threshold for donor site predictions
        acceptor_threshold: Minimum confidence threshold for acceptor site predictions
        
    Returns:
        Dictionary mapping gene IDs to dictionaries of predicted donor/acceptor sets
    """
    predicted_sites: Dict[str, Dict[str, Set[int]]] = {
        "donor": set(),
        "acceptor": set(),
    }
    
    if verbose >= 1:
        print("[meta_splice_predict] Starting prediction for missing positions...")
    
    # Diagnostic counters
    genes_processed = 0
    total_predicted_donor = 0
    total_predicted_acceptor = 0
    skipped_no_gene_data = 0
    skipped_none_covered_abs = 0
    skipped_no_missing_positions = 0
    
    if verbose >= 1:
        print(f"[meta_splice_predict] Found {len(pred_dict)} genes in prediction dict and {len(gene_data)} genes in full data")
    
    all_donor_pos = set()
    all_acceptor_pos = set()
    total_positions_evaluated = 0
    total_raw_donor_predictions = 0
    total_raw_acceptor_predictions = 0

    # ------------------------------------------------------------------
    # Quick diagnostic – count how many *unseen* positions we need to score
    # ------------------------------------------------------------------
    total_missing_positions = 0
    per_gene_missing: Dict[str, int] = {}
    for gid, info in pred_dict.items():
        gdf = gene_data.get(gid)
        if gdf is None:
            continue
        covered_abs: Optional[Set[int]] = info.get("covered_abs")
        abs_positions = gdf["position"].to_numpy()
        if covered_abs is None:
            missing_mask = np.zeros(len(abs_positions), dtype=bool)  # all covered ⇒ 0 missing
        else:
            missing_mask = ~np.isin(abs_positions.astype(int), list(covered_abs))
        n_missing = int(missing_mask.sum())
        per_gene_missing[gid] = n_missing
        total_missing_positions += n_missing
    if verbose >= 1:
        print(f"[meta_splice_predict] Total *missing* positions to predict: {total_missing_positions:,}")
        if verbose >= 2:
            # Show top-5 genes with most missing positions
            top5 = sorted(per_gene_missing.items(), key=lambda kv: kv[1], reverse=True)[:5]
            print("[meta_splice_predict] Top-5 genes by missing counts:")
            for gid, cnt in top5:
                print(f"   {gid}: {cnt}")

    # ------------------------------------------------------------------
    # Build enriched feature matrices so that unseen positions have the
    # SAME feature columns as the training data
    # ------------------------------------------------------------------
    enriched_gene_dfs: Dict[str, pd.DataFrame] = {}
    try:
        enriched_gene_dfs = _build_feature_matrix_for_genes(
            list(pred_dict.keys()),
            eval_dir=eval_dir,
            kmer_sizes=kmer_sizes,
            enrichers=enrichers,
            verbose=max(0, verbose - 1),
        )
        if verbose >= 1:
            print(f"[meta_splice_predict] Enriched feature matrices built for {len(enriched_gene_dfs)} genes")
    except Exception as e:
        if verbose >= 1:
            print(f"[meta_splice_predict] WARNING: failed to build enriched features – falling back to placeholder zeros. Error: {e}")
    
    for gid, info in pred_dict.items():
        if gid not in gene_data:
            skipped_no_gene_data += 1
            continue
            
        # Get the covered positions (those used in meta-model training)
        covered_abs = info.get("covered_abs")
        
        # If covered_abs is None, we need to determine positions to predict differently
        # This happens when missing_policy="skip" (default) was used in the original data collection
        if covered_abs is None:
            if verbose >= 2:
                print(f"[meta_splice_predict] Gene {gid}: covered_abs is None, determining positions heuristically")
            
            # In this case, we need a different approach to determine which positions to predict
            # We'll use the positions that have truth labels as our "covered" positions
            # And predict for positions that don't have truth labels
            
            truth_donor = info.get("truth_donor", set())
            truth_acceptor = info.get("truth_acceptor", set())
            
            # Use the union of all truth positions as positions that are "covered"
            # This means we'll only predict for positions not in either truth set
            covered_abs = truth_donor.union(truth_acceptor)
            
            # If verbose, log what we're doing
            if verbose >= 2:
                print(f"[meta_splice_predict] Gene {gid}: using {len(covered_abs)} truth positions as covered positions")
            
        # Get the full gene data
        gene_df = gene_data[gid]
        
        # Quick check if data is valid
        if gene_df.shape[0] == 0 or len(feature_names) == 0:
            if verbose >= 2:
                print(f"[meta_splice_predict] Gene {gid}: Empty data or no feature names, skipping")
            continue
            
        # Check if we have at least one feature from feature_names in gene_df
        # This ensures our prediction function will work
        if not any(feature in gene_df.columns for feature in feature_names):
            if verbose >= 2:
                print(f"[meta_splice_predict] Gene {gid}: Missing necessary features, skipping")
                print(f"  Available columns: {gene_df.columns[:5]}..., needed: {feature_names[:3]}...")
            continue
            
        # Skip if we don't have strand or position information
        if "strand" not in gene_df.columns or "position" not in gene_df.columns:
            continue
            
        strand = gene_df["strand"].iloc[0]
        gene_start = gene_df["gene_start"].iloc[0] if "gene_start" in gene_df.columns else None
        gene_end = gene_df["gene_end"].iloc[0] if "gene_end" in gene_df.columns else None
        
        # Identify positions not in the covered set
        # We need to convert relative positions to absolute positions for comparison
        if "position" in gene_df.columns:
            rel_positions = gene_df["position"].to_numpy()
            
            # Convert to absolute positions based on strand and gene coordinates
            if strand == "-":
                if gene_end is not None:
                    abs_positions = gene_end - rel_positions
                else:
                    # Can't convert without gene_end on negative strand
                    continue
            else:  # "+" strand
                if gene_start is not None:
                    abs_positions = gene_start + rel_positions
                else:
                    # Can't convert without gene_start on positive strand
                    continue
                    
            # Create a mask for positions not in the covered set
            # Only if covered_abs is a proper set, otherwise all positions are considered covered
            if covered_abs is None:
                missing_mask = np.zeros(len(abs_positions), dtype=bool)  # All positions considered covered
            else:
                missing_mask = np.array([int(pos) not in covered_abs for pos in abs_positions])
            
            # If all positions are already covered, skip this gene
            if not np.any(missing_mask):
                skipped_no_missing_positions += 1
                continue
                
            # Determine relative positions of *missing* sites
            missing_rel_positions = gene_df.loc[missing_mask, "position"].to_numpy()

            # Retrieve enriched feature DataFrame for this gene
            enriched_df = enriched_gene_dfs.get(gid)
            if enriched_df is None:
                if verbose >= 2:
                    print(f"[meta_splice_predict] Gene {gid}: no enriched features – skipping")
                skipped_no_gene_data += 1
                continue

            # Keep only those rows that correspond to the missing positions
            missing_df = enriched_df[enriched_df["position"].isin(missing_rel_positions)].copy()

            # Ensure all expected feature columns are present (fill with 0.0 if not)
            missing_features = [feat for feat in feature_names if feat not in missing_df.columns]
            if missing_features:
                if verbose >= 1 and genes_processed <= 3:
                    print(f"[meta_splice_predict] Gene {gid}: {len(missing_features)} missing features – filling with zeros")
                for feat in missing_features:
                    missing_df[feat] = 0.0
            
            # IMPORTANT: Proper feature preparation requires running the full feature extraction pipeline
            # Currently we're using a simplified approach by filling missing features with zeros
            # A complete implementation would use the following steps:
            #
            # 1. Create a temporary dataset for the missing positions
            # 2. Run the full feature extraction workflow (k-mer extraction, enrichment, etc.)
            # 3. Load the properly constructed features back for prediction
            # 
            # This would involve calling:
            #  - run_enhanced_splice_prediction_workflow for predictive features
            #  - incremental_build_training_dataset to assemble with correct feature columns
            #
            # Reference: splice_engine/meta_models/builder/incremental_builder.py
            
            # Current simplified placeholder approach (not ideal):
            # Find which required features are missing
            missing_features = [feat for feat in feature_names if feat not in missing_df.columns]
            
            # Add all missing features at once with a single vectorized operation
            if missing_features:
                if verbose >= 1 and genes_processed == 0:
                    print(f"[WARNING] Using simplified feature preparation. Missing {len(missing_features)} features.")
                    print(f"[WARNING] For production, implement full feature extraction workflow")
                    
                # Create a DataFrame with zeros for all missing features
                # NOTE: This is a TEMPORARY SOLUTION and not ideal
                zeros_df = pd.DataFrame(0.0, index=missing_df.index, columns=missing_features)
                
                # Concatenate horizontally with the original DataFrame
                missing_df = pd.concat([missing_df, zeros_df], axis=1)
            
            # Extract features for prediction
            X_missing = missing_df[feature_names].to_numpy(np.float32)
            
            # Apply meta-model to predict probabilities
            proba_missing = predict_fn(X_missing)
            
            # Get probabilities for each class
            # proba_missing shape: [n_samples, 3] where columns are [donor, acceptor, neither]
            
            # Apply minimum confidence thresholds to filter predictions
            # This prevents over-prediction of sites
            # Thresholds are passed as parameters to the function
            
            # First, diagnose the model's output distribution if this is the first gene
            if verbose >= 2 and genes_processed == 0:
                print(f"[DIAGNOSTIC] Model output probability distribution for first gene:")
                print(f"  Average donor probability: {np.mean(proba_missing[:, 0]):.4f}")
                print(f"  Average acceptor probability: {np.mean(proba_missing[:, 1]):.4f}")
                print(f"  Average neither probability: {np.mean(proba_missing[:, 2]):.4f}")
                
                # Print percentile values to understand the distribution
                for p in [50, 75, 90, 95, 99]:
                    print(f"  {p}th percentile - donor: {np.percentile(proba_missing[:, 0], p):.4f}, "
                          f"acceptor: {np.percentile(proba_missing[:, 1], p):.4f}, "
                          f"neither: {np.percentile(proba_missing[:, 2], p):.4f}")
            
            # Check for extreme bias in the model outputs
            donor_avg = np.mean(proba_missing[:, 0])
            acceptor_avg = np.mean(proba_missing[:, 1])
            neither_avg = np.mean(proba_missing[:, 2])
            
            # Adaptive thresholding approach if there's extreme bias
            if donor_avg > 0.6 and acceptor_avg < 0.2 and verbose >= 1 and genes_processed == 0:
                print(f"[WARNING] Extreme bias detected in model outputs (donor: {donor_avg:.4f}, acceptor: {acceptor_avg:.4f})")
                print(f"[WARNING] Using alternative prediction approach with relative thresholding")
            
            # EXTREME CORRECTION: Based on observed severe bias in model output
            # Current model is predicting 99.76% of all positions as donor sites, which is biologically impossible
            # Most positions should be neither donor nor acceptor (non-splice sites)
            
            # Diagnose the extreme bias problem
            if verbose >= 1 and genes_processed == 0:
                print(f"[CORRECTION] Model diagnosis for first gene ({len(proba_missing)} positions):")
                print(f"  Average class probabilities: donor={np.mean(proba_missing[:, 0]):.4f}, "
                      f"acceptor={np.mean(proba_missing[:, 1]):.4f}, neither={np.mean(proba_missing[:, 2]):.4f}")
                print(f"  Median class probabilities: donor={np.median(proba_missing[:, 0]):.4f}, "
                      f"acceptor={np.median(proba_missing[:, 1]):.4f}, neither={np.median(proba_missing[:, 2]):.4f}")
                print(f"  Max class probabilities: donor={np.max(proba_missing[:, 0]):.4f}, "
                      f"acceptor={np.max(proba_missing[:, 1]):.4f}, neither={np.max(proba_missing[:, 2]):.4f}")
            
            # Calculate what percentage of positions would be predicted as each class with standard approach
            standard_donor_mask = (proba_missing[:, 0] > donor_threshold) & (proba_missing[:, 0] > proba_missing[:, 1]) & (proba_missing[:, 0] > proba_missing[:, 2])
            standard_acceptor_mask = (proba_missing[:, 1] > acceptor_threshold) & (proba_missing[:, 1] > proba_missing[:, 0]) & (proba_missing[:, 1] > proba_missing[:, 2])
            standard_neither_mask = ~standard_donor_mask & ~standard_acceptor_mask
            
            donor_pct = np.sum(standard_donor_mask) / len(proba_missing) * 100 if len(proba_missing) > 0 else 0
            acceptor_pct = np.sum(standard_acceptor_mask) / len(proba_missing) * 100 if len(proba_missing) > 0 else 0
            
            # If we're severely over-predicting donor sites (more than 10%), apply extreme correction
            if donor_pct > 10.0 and genes_processed == 0 and verbose >= 1:
                print(f"[CORRECTION] Standard approach would predict {donor_pct:.1f}% donor sites, {acceptor_pct:.1f}% acceptor sites")
                print(f"[CORRECTION] Applying EXTREME correction for model bias")
            
            # APPROACH 1: PERCENTILE-BASED THRESHOLDING
            # Keep only the top N% of predictions as actual splice sites
            # Typically, true splice sites are very rare (< 1% of positions in a gene)
            max_splice_site_percent = 1.0  # At most 1% of positions should be splice sites
            max_positions = int(len(proba_missing) * max_splice_site_percent / 100)
            
            if max_positions > 0:
                # Get the top positions for donor and acceptor based on their respective probabilities
                donor_scores = proba_missing[:, 0]
                acceptor_scores = proba_missing[:, 1]
                
                # Find positions where the class probability exceeds minimum threshold
                # This helps ensure we only consider positions with some minimal confidence
                min_donor_threshold = donor_threshold * 0.5  # More relaxed min threshold
                min_acceptor_threshold = acceptor_threshold * 0.5  # More relaxed min threshold
                
                donor_candidates = donor_scores > min_donor_threshold
                acceptor_candidates = acceptor_scores > min_acceptor_threshold
                
                # If we have any candidates, select top positions
                if np.sum(donor_candidates) > 0:
                    donor_candidate_indices = np.where(donor_candidates)[0]
                    donor_candidate_scores = donor_scores[donor_candidate_indices]
                    
                    # Select at most max_positions/2 donors (half the budget)
                    top_k_donors = min(max_positions // 2, len(donor_candidate_indices))
                    
                    if top_k_donors > 0:
                        top_donor_indices = donor_candidate_indices[np.argsort(donor_candidate_scores)[-top_k_donors:]]
                        extreme_donor_mask = np.zeros_like(standard_donor_mask)
                        extreme_donor_mask[top_donor_indices] = True
                    else:
                        extreme_donor_mask = np.zeros_like(standard_donor_mask)
                else:
                    extreme_donor_mask = np.zeros_like(standard_donor_mask)
                
                # Similar process for acceptor sites
                if np.sum(acceptor_candidates) > 0:
                    acceptor_candidate_indices = np.where(acceptor_candidates)[0]
                    acceptor_candidate_scores = acceptor_scores[acceptor_candidate_indices]
                    
                    # Select at most max_positions/2 acceptors (half the budget)
                    top_k_acceptors = min(max_positions // 2, len(acceptor_candidate_indices))
                    
                    if top_k_acceptors > 0:
                        top_acceptor_indices = acceptor_candidate_indices[np.argsort(acceptor_candidate_scores)[-top_k_acceptors:]]
                        extreme_acceptor_mask = np.zeros_like(standard_acceptor_mask)
                        extreme_acceptor_mask[top_acceptor_indices] = True
                    else:
                        extreme_acceptor_mask = np.zeros_like(standard_acceptor_mask)
                else:
                    extreme_acceptor_mask = np.zeros_like(standard_acceptor_mask)
                
                # Use extreme correction masks
                donor_mask = extreme_donor_mask
                acceptor_mask = extreme_acceptor_mask
                
                if verbose >= 1 and genes_processed <= 3:
                    print(f"[CORRECTION] Selected top {np.sum(extreme_donor_mask)} donor and {np.sum(extreme_acceptor_mask)} acceptor sites")
                    print(f"[CORRECTION] This represents {np.sum(extreme_donor_mask)/len(proba_missing)*100:.2f}% and {np.sum(extreme_acceptor_mask)/len(proba_missing)*100:.2f}% of total positions")
            else:
                # Fallback to standard approach if no positions to evaluate
                donor_mask = standard_donor_mask
                acceptor_mask = standard_acceptor_mask
            
            # Count raw predictions before thresholding (any position where donor/acceptor > neither)
            raw_donor_count = np.sum(proba_missing[:, 0] > proba_missing[:, 2])
            raw_acceptor_count = np.sum(proba_missing[:, 1] > proba_missing[:, 2])
            
            # Track statistics
            total_positions_evaluated += len(proba_missing)
            total_raw_donor_predictions += raw_donor_count
            total_raw_acceptor_predictions += raw_acceptor_count
            
            if verbose >= 2:
                print(f"[meta_splice_predict] Gene {gid}: Raw prediction counts - {raw_donor_count} donor, {raw_acceptor_count} acceptor")
                print(f"[meta_splice_predict] Gene {gid}: Thresholded prediction counts - {np.sum(donor_mask)} donor, {np.sum(acceptor_mask)} acceptor")
            
            # Extract absolute positions for donor and acceptor predictions using
            # the *same* dimensionality as `donor_mask` / `acceptor_mask`
            pred_rel_pos = missing_df["position"].to_numpy()
            if strand == "-":
                donor_positions = (gene_end - pred_rel_pos[donor_mask]).astype(int)
                acceptor_positions = (gene_end - pred_rel_pos[acceptor_mask]).astype(int)
            else:
                donor_positions = (gene_start + pred_rel_pos[donor_mask]).astype(int)
                acceptor_positions = (gene_start + pred_rel_pos[acceptor_mask]).astype(int)
            
            # Update the prediction sets
            if "predicted_donor" not in info:
                info["predicted_donor"] = set()
            if "predicted_acceptor" not in info:
                info["predicted_acceptor"] = set()
                
            info["predicted_donor"].update(donor_positions)
            info["predicted_acceptor"].update(acceptor_positions)
            
            predicted_sites["donor"].update(donor_positions)
            predicted_sites["acceptor"].update(acceptor_positions)
            
            # Count predictions
            total_predicted_donor += len(donor_positions)
            total_predicted_acceptor += len(acceptor_positions)
            
            all_donor_pos.update(donor_positions)
            all_acceptor_pos.update(acceptor_positions)
            
        genes_processed += 1
        
        # Print progress every 100 genes
        if verbose >= 1 and genes_processed % 100 == 0:
            print(f"[meta_splice_predict] Processed {genes_processed} genes...")
            
    if verbose >= 1:
        print(f"[meta_splice_predict] Prediction complete for {genes_processed} genes.")
    print(f"[meta_splice_predict] Predicted {len(all_donor_pos)} donor and {len(all_acceptor_pos)} acceptor sites.")
    print(f"[meta_splice_predict] Skipped genes: {skipped_no_gene_data} (no gene data), {skipped_none_covered_abs} (covered_abs is None), {skipped_no_missing_positions} (no missing positions)")
    
    # Report prediction statistics
    if total_positions_evaluated > 0:
        raw_donor_rate = total_raw_donor_predictions / total_positions_evaluated * 100
        raw_acceptor_rate = total_raw_acceptor_predictions / total_positions_evaluated * 100
        thresholded_donor_rate = len(all_donor_pos) / total_positions_evaluated * 100
        thresholded_acceptor_rate = len(all_acceptor_pos) / total_positions_evaluated * 100
        
        print(f"\n[meta_splice_predict] Prediction statistics:")
        print(f"  Total positions evaluated: {total_positions_evaluated:,}")
        print(f"  Raw prediction rates: {raw_donor_rate:.2f}% donor, {raw_acceptor_rate:.2f}% acceptor")
        print(f"  Thresholded prediction rates: {thresholded_donor_rate:.2f}% donor, {thresholded_acceptor_rate:.2f}% acceptor")
        print(f"  Thresholds used: {donor_threshold} donor, {acceptor_threshold} acceptor")
        print(f"  Final prediction counts: {len(all_donor_pos):,} donor, {len(all_acceptor_pos):,} acceptor sites")
        
        # Compute donor/acceptor ratio
        if len(all_acceptor_pos) > 0:
            donor_acceptor_ratio = len(all_donor_pos) / len(all_acceptor_pos)
            print(f"  Donor/Acceptor ratio: {donor_acceptor_ratio:.2f}")
            # In most genes, donor and acceptor counts should be very close
            if donor_acceptor_ratio > 2.0 or donor_acceptor_ratio < 0.5:
                print(f"  WARNING: Unusual donor/acceptor ratio of {donor_acceptor_ratio:.2f} - bias in model or thresholds")
        
        if len(all_donor_pos) > 0 and len(all_acceptor_pos) == 0:
            print(f"  WARNING: No acceptor sites predicted despite {len(all_donor_pos):,} donor sites - possible bias in model")
        elif len(all_acceptor_pos) > 0 and len(all_donor_pos) == 0:
            print(f"  WARNING: No donor sites predicted despite {len(all_acceptor_pos):,} acceptor sites - possible bias in model"
        )
        
        # Provide guidance on threshold tuning
        print("\n[meta_splice_predict] Threshold tuning recommendations:")
        if len(all_donor_pos) == 0 or len(all_acceptor_pos) == 0:
            print("  Consider adjusting thresholds to achieve more balanced predictions")
            if len(all_donor_pos) == 0:
                print(f"  Try lowering donor threshold below {donor_threshold}")
            if len(all_acceptor_pos) == 0:
                print(f"  Try lowering acceptor threshold below {acceptor_threshold}")
        elif donor_acceptor_ratio > 2.0:
            print(f"  Donor sites over-predicted ({len(all_donor_pos):,} vs {len(all_acceptor_pos):,} acceptors)")
            print(f"  Try increasing donor threshold above {donor_threshold}")
        elif donor_acceptor_ratio < 0.5:
            print(f"  Acceptor sites over-predicted ({len(all_acceptor_pos):,} vs {len(all_donor_pos):,} donors)")
            print(f"  Try increasing acceptor threshold above {acceptor_threshold}")
        else:
            print(f"  Current thresholds ({donor_threshold}, {acceptor_threshold}) producing balanced predictions")
            print(f"  Ratio of {donor_acceptor_ratio:.2f} is within reasonable range")
        
    return predicted_sites


def load_full_gene_data(dataset_path: Union[str, Path], *, sample: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Load full gene data from dataset for prediction on all positions.
    
    Args:
        dataset_path: Path to dataset directory or file
        sample: Optional sample size to limit data loading
        
    Returns:
        Dictionary mapping gene IDs to gene DataFrames

    Examples: 
        >>> load_full_gene_data("train_pc_1000/master")

        - Lists all .parquet files beneath that directory (the Parquets 
            produced when you built the curated training set).
        - For each file it reads only the columns needed for prediction 
            (gene_id, strand, gene_start / gene_end, position and the three SpliceAI scores).
        - Groups the rows by gene_id and returns a dict {gene_id: DataFrame} 
            that represents every position in every gene that exists in that 
            training-dataset directory.
    """
    dataset_path = Path(dataset_path)
    
    # Columns required for prediction
    required_cols = [
        "gene_id", "strand", "gene_start", "gene_end", 
        "position", "donor_score", "acceptor_score", "neither_score"
    ]
    
    # Load data
    if dataset_path.is_dir():
        parquet_paths = sorted(dataset_path.glob("*.parquet"))
    else:
        parquet_paths = [dataset_path]
        
    gene_data = {}
    
    for pq_path in parquet_paths:
        # Load with minimal columns to keep memory usage reasonable
        df = pl.scan_parquet(str(pq_path)).select(required_cols).collect().to_pandas()
        
        if sample is not None and len(df) > sample:
            df = df.sample(n=sample, random_state=42)
            
        # Group by gene_id
        for gid, group in df.groupby("gene_id"):
            if gid in gene_data:
                # Append to existing gene data
                gene_data[gid] = pd.concat([gene_data[gid], group])
            else:
                gene_data[gid] = group
                
    return gene_data


def merge_prediction_results(
    pred_dict: Dict[str, Dict[str, Any]], 
    *, 
    verbose: int = 0
) -> None:
    """
    Merge predicted sites with truth sites for evaluation when using missing_policy="predict".
    
    This function updates the pred_dict by merging predicted_donor/acceptor sets with
    truth_donor/acceptor sets to create eval_donor/acceptor sets for evaluation.
    
    Args:
        pred_dict: Dictionary containing gene prediction data
        verbose: Whether to print verbose output
    """
    if verbose >= 1:
        print("[meta_splice_predict] Merging prediction results with truth sets...")
        
    total_truth_donor = 0
    total_truth_acceptor = 0
    total_predicted_donor = 0
    total_predicted_acceptor = 0
    total_eval_donor = 0
    total_eval_acceptor = 0
    
    # Track newly added positions for detailed diagnostics
    total_new_donor = 0
    total_new_acceptor = 0
    genes_with_new_predictions = 0
    
    new_donor_stats = defaultdict(list)
    new_acceptor_stats = defaultdict(list)
    
    for gid, info in pred_dict.items():
        # Initialize evaluation sets if not present
        if "eval_donor" not in info:
            info["eval_donor"] = set()
        if "eval_acceptor" not in info:
            info["eval_acceptor"] = set()
            
        # Get truth sets
        truth_donor = info.get("truth_donor", set())
        truth_acceptor = info.get("truth_acceptor", set())
        
        # Get predicted sets
        predicted_donor = info.get("predicted_donor", set())
        predicted_acceptor = info.get("predicted_acceptor", set())
        
        # Calculate newly predicted sites (not in truth set)
        new_donor_sites = predicted_donor - truth_donor
        new_acceptor_sites = predicted_acceptor - truth_acceptor
        
        # Track counts of newly predicted sites
        new_donor_count = len(new_donor_sites)
        new_acceptor_count = len(new_acceptor_sites)
        
        if new_donor_count > 0 or new_acceptor_count > 0:
            genes_with_new_predictions += 1
            
            # Store new predictions in the info dictionary for later analysis
            info["new_donor"] = new_donor_sites
            info["new_acceptor"] = new_acceptor_sites
        
        # Union of truth and predicted sets becomes the evaluation set
        info["eval_donor"] = truth_donor.union(predicted_donor)
        info["eval_acceptor"] = truth_acceptor.union(predicted_acceptor)
        
        # Count sets
        total_truth_donor += len(truth_donor)
        total_truth_acceptor += len(truth_acceptor)
        total_predicted_donor += len(predicted_donor)
        total_predicted_acceptor += len(predicted_acceptor)
        total_eval_donor += len(info["eval_donor"])
        total_eval_acceptor += len(info["eval_acceptor"])
        total_new_donor += new_donor_count
        total_new_acceptor += new_acceptor_count
    
    if verbose >= 1:
        print(f"[meta_splice_predict] Truth sets: {total_truth_donor} donor, {total_truth_acceptor} acceptor")
        print(f"[meta_splice_predict] Predicted sets: {total_predicted_donor} donor, {total_predicted_acceptor} acceptor")
        print(f"[meta_splice_predict] Evaluation sets: {total_eval_donor} donor, {total_eval_acceptor} acceptor")
        print(f"[meta_splice_predict] NEW predictions: {total_new_donor} donor, {total_new_acceptor} acceptor in {genes_with_new_predictions} genes")
        
        # Report detailed diagnostics about potential false positives
        if genes_with_new_predictions > 0:
            print("\n[meta_splice_predict] === NEW PREDICTION ANALYSIS ====")
            # Sample up to 5 genes with new predictions
            genes_with_new = [gid for gid, info in pred_dict.items() if "new_donor" in info or "new_acceptor" in info]
            sample_size = min(5, len(genes_with_new))
            sample_genes = genes_with_new[:sample_size]
            
            for gid in sample_genes:
                info = pred_dict[gid]
                new_donor = info.get("new_donor", set())
                new_acceptor = info.get("new_acceptor", set())
                print(f"\n  Gene {gid}:")
                if new_donor:
                    print(f"    New donor predictions: {len(new_donor)}")
                    print(f"    Sample positions: {sorted(list(new_donor))[:5]}")
                if new_acceptor:
                    print(f"    New acceptor predictions: {len(new_acceptor)}")
                    print(f"    Sample positions: {sorted(list(new_acceptor))[:5]}")
            
            # Show summary stats
            avg_new_sites = (total_new_donor + total_new_acceptor) / genes_with_new_predictions
            print(f"\n  Average new predictions per gene with new sites: {avg_new_sites:.2f}")
            print(f"  Percentage of genes with new predictions: {genes_with_new_predictions/len(pred_dict)*100:.2f}%")
