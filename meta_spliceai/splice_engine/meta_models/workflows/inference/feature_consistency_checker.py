"""
Feature Consistency Checker for Meta-Model Inference

This module ensures that feature representations are consistent between training and inference,
preventing catastrophic prediction failures due to feature misalignment.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json

logger = logging.getLogger(__name__)


class FeatureConsistencyChecker:
    """
    Validates feature consistency between training and inference.
    
    This checker ensures that:
    1. Feature names match between training and inference
    2. Feature ordering is preserved
    3. Feature value distributions are reasonable
    4. No unexpected features appear during inference
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        verbose: int = 1
    ):
        """
        Initialize the feature consistency checker.
        
        Parameters
        ----------
        model_path : str or Path
            Path to the trained model directory containing feature manifest
        verbose : int
            Verbosity level (0=silent, 1=normal, 2=detailed)
        """
        self.model_path = Path(model_path)
        self.verbose = verbose
        
        # Load training feature manifest
        self.training_features = self._load_training_features()
        self.training_feature_stats = self._load_feature_statistics()
        
    def _load_training_features(self) -> List[str]:
        """Load the ordered list of training features."""
        manifest_path = self.model_path / "feature_manifest.csv"
        
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Feature manifest not found at {manifest_path}. "
                "Cannot ensure feature consistency without training manifest."
            )
        
        df = pd.read_csv(manifest_path)
        features = df['feature'].tolist()
        
        if self.verbose >= 1:
            print(f"[FeatureChecker] Loaded {len(features)} training features")
            if self.verbose >= 2:
                print(f"  First 5 features: {features[:5]}")
        
        return features
    
    def _load_feature_statistics(self) -> Optional[Dict]:
        """Load training feature statistics if available."""
        stats_path = self.model_path / "feature_statistics.json"
        
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            if self.verbose >= 2:
                print(f"[FeatureChecker] Loaded feature statistics from training")
            return stats
        
        return None
    
    def check_feature_consistency(
        self,
        inference_features: pd.DataFrame,
        early_stop: bool = True,
        check_distributions: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Check if inference features are consistent with training features.
        
        Parameters
        ----------
        inference_features : pd.DataFrame
            Feature matrix from inference (should have feature columns)
        early_stop : bool
            If True, stop at first inconsistency (faster for chunked processing)
        check_distributions : bool
            If True, also check if feature value distributions are reasonable
            
        Returns
        -------
        is_consistent : bool
            True if features are consistent
        issues : List[str]
            List of inconsistency issues found
        """
        issues = []
        
        # Get inference feature names (excluding metadata columns)
        metadata_cols = {'gene_id', 'position', 'chrom', 'strand'}
        inference_feature_names = [
            col for col in inference_features.columns 
            if col not in metadata_cols
        ]
        
        # 1. Check feature count
        if len(inference_feature_names) != len(self.training_features):
            issues.append(
                f"Feature count mismatch: training={len(self.training_features)}, "
                f"inference={len(inference_feature_names)}"
            )
            if early_stop:
                return False, issues
        
        # 2. Check feature names match
        training_set = set(self.training_features)
        inference_set = set(inference_feature_names)
        
        missing_features = training_set - inference_set
        if missing_features:
            issues.append(
                f"Missing {len(missing_features)} training features: "
                f"{list(missing_features)[:5]}..."
            )
            if early_stop:
                return False, issues
        
        extra_features = inference_set - training_set
        if extra_features:
            issues.append(
                f"Unexpected {len(extra_features)} features in inference: "
                f"{list(extra_features)[:5]}..."
            )
            if early_stop:
                return False, issues
        
        # 3. Check feature ordering (CRITICAL!)
        for i, (train_feat, inf_feat) in enumerate(
            zip(self.training_features, inference_feature_names)
        ):
            if train_feat != inf_feat:
                issues.append(
                    f"Feature order mismatch at position {i}: "
                    f"training='{train_feat}', inference='{inf_feat}'"
                )
                if early_stop:
                    return False, issues
        
        # 4. Check feature value distributions (optional)
        if check_distributions and self.training_feature_stats:
            for feat in self.training_features:
                if feat not in inference_features.columns:
                    continue
                    
                inf_values = inference_features[feat].values
                
                # Check for NaN/Inf values
                if np.any(np.isnan(inf_values)):
                    issues.append(f"Feature '{feat}' contains NaN values")
                if np.any(np.isinf(inf_values)):
                    issues.append(f"Feature '{feat}' contains Inf values")
                
                # Check value ranges if stats available
                if feat in self.training_feature_stats:
                    stats = self.training_feature_stats[feat]
                    inf_min, inf_max = np.min(inf_values), np.max(inf_values)
                    train_min = stats.get('min', -np.inf)
                    train_max = stats.get('max', np.inf)
                    
                    # Allow some tolerance for continuous features
                    tolerance = 0.1 * (train_max - train_min) if train_max > train_min else 1.0
                    
                    if inf_min < train_min - tolerance:
                        issues.append(
                            f"Feature '{feat}' has values below training range: "
                            f"min={inf_min:.3f} < train_min={train_min:.3f}"
                        )
                    if inf_max > train_max + tolerance:
                        issues.append(
                            f"Feature '{feat}' has values above training range: "
                            f"max={inf_max:.3f} > train_max={train_max:.3f}"
                        )
        
        is_consistent = len(issues) == 0
        
        if self.verbose >= 1:
            if is_consistent:
                print("[FeatureChecker] ✅ Feature consistency check PASSED")
            else:
                print(f"[FeatureChecker] ❌ Feature consistency check FAILED with {len(issues)} issues")
                if self.verbose >= 2:
                    for issue in issues[:5]:  # Show first 5 issues
                        print(f"  - {issue}")
        
        return is_consistent, issues
    
    def reorder_features(
        self,
        inference_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Reorder inference features to match training feature order.
        
        Parameters
        ----------
        inference_features : pd.DataFrame
            Feature matrix with potentially misordered columns
            
        Returns
        -------
        pd.DataFrame
            Feature matrix with columns in correct order
        """
        # Identify metadata columns to preserve
        metadata_cols = {'gene_id', 'position', 'chrom', 'strand'}
        metadata_present = [col for col in metadata_cols if col in inference_features.columns]
        
        # Ensure all training features are present
        missing_features = set(self.training_features) - set(inference_features.columns)
        if missing_features:
            if self.verbose >= 1:
                print(f"[FeatureChecker] Adding {len(missing_features)} missing features with default values")
            for feat in missing_features:
                inference_features[feat] = 0.0
        
        # Reorder columns: metadata first, then features in training order
        ordered_cols = metadata_present + self.training_features
        
        # Only include columns that exist
        ordered_cols = [col for col in ordered_cols if col in inference_features.columns]
        
        reordered_df = inference_features[ordered_cols]
        
        if self.verbose >= 2:
            print(f"[FeatureChecker] Reordered features from {len(inference_features.columns)} to {len(ordered_cols)} columns")
        
        return reordered_df


def check_early_chunk_consistency(
    feature_df: pd.DataFrame,
    model_path: Union[str, Path],
    chunk_idx: int = 0,
    max_checks: int = 2,
    verbose: int = 1,
    strict_mode: bool = False
) -> bool:
    """
    Quick consistency check for early chunks in chunked processing.
    
    This is optimized for the meta_only mode where we process many chunks,
    and we want to fail fast if features are inconsistent.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature matrix for current chunk
    model_path : str or Path
        Path to trained model directory
    chunk_idx : int
        Current chunk index (0-based)
    max_checks : int
        Only check first N chunks (for efficiency)
    verbose : int
        Verbosity level
    strict_mode : bool
        If False, allows missing features (filled with defaults) for unseen genes
        
    Returns
    -------
    bool
        True if consistent or past check limit, False if inconsistent
    """
    # Only check first few chunks
    if chunk_idx >= max_checks:
        return True
    
    if verbose >= 1:
        print(f"[EarlyCheck] Checking feature consistency for chunk {chunk_idx}")
    
    try:
        checker = FeatureConsistencyChecker(model_path, verbose=verbose)
        
        # Get feature columns (excluding metadata)
        metadata_cols = {'gene_id', 'position', 'chrom', 'strand'}
        feature_cols = [col for col in feature_df.columns if col not in metadata_cols]
        
        # Check if we have missing features (common for unseen genes)
        training_features = checker.training_features
        missing_features = set(training_features) - set(feature_cols)
        
        if missing_features and not strict_mode:
            # Allow missing features for unseen genes
            if verbose >= 1:
                print(f"[EarlyCheck] ⚠️ {len(missing_features)} features missing - will use defaults")
                if verbose >= 2:
                    print(f"    Missing: {sorted(missing_features)[:5]}...")
            
            # Check that the features we DO have are in the correct order
            # This is critical - wrong ordering would cause catastrophic predictions
            training_order = {feat: idx for idx, feat in enumerate(training_features)}
            current_positions = []
            
            for feat in feature_cols:
                if feat in training_order:
                    current_positions.append((training_order[feat], feat))
            
            # Sort by training position
            current_positions.sort(key=lambda x: x[0])
            expected_order = [feat for _, feat in current_positions]
            
            # Check if features are in the expected relative order
            for i, feat in enumerate(expected_order):
                if feature_cols[i] != feat:
                    print(f"[EarlyCheck] ❌ CRITICAL: Feature ordering mismatch!")
                    print(f"    Expected '{feat}' at position {i}, got '{feature_cols[i]}'")
                    raise ValueError(
                        "Feature ordering is incorrect! This would cause catastrophic predictions. "
                        "The model expects features in a specific order."
                    )
            
            return True  # Missing features OK, ordering is correct
        
        # Do full consistency check
        is_consistent, issues = checker.check_feature_consistency(
            feature_df,
            early_stop=True,  # Stop at first issue for speed
            check_distributions=(chunk_idx == 0)  # Only check distributions on first chunk
        )
        
        if not is_consistent:
            # Filter out missing feature issues if not in strict mode
            if not strict_mode:
                critical_issues = [
                    issue for issue in issues 
                    if 'Feature count mismatch' not in issue and 'Missing' not in issue
                ]
                if not critical_issues:
                    # Only missing features, which we allow in non-strict mode
                    return True
                issues = critical_issues
            
            print(f"[EarlyCheck] ❌ CRITICAL: Feature inconsistency detected in chunk {chunk_idx}!")
            for issue in issues[:3]:
                print(f"    {issue}")
            raise ValueError(
                f"Feature inconsistency detected! This would lead to incorrect predictions. "
                f"Issues: {issues[0]}"
            )
        
        return True
        
    except FileNotFoundError as e:
        if verbose >= 1:
            print(f"[EarlyCheck] ⚠️ Warning: Cannot check consistency - {e}")
        return True  # Continue without check if manifest missing


def save_feature_statistics(
    training_features: pd.DataFrame,
    output_path: Union[str, Path],
    feature_names: Optional[List[str]] = None
) -> None:
    """
    Save feature statistics from training data for later validation.
    
    Parameters
    ----------
    training_features : pd.DataFrame
        Training feature matrix
    output_path : str or Path
        Where to save the statistics JSON file
    feature_names : List[str], optional
        Specific features to compute stats for (default: all numeric columns)
    """
    output_path = Path(output_path)
    
    if feature_names is None:
        # Use all numeric columns except metadata
        metadata_cols = {'gene_id', 'position', 'chrom', 'strand', 'label', 'splice_type'}
        feature_names = [
            col for col in training_features.columns 
            if col not in metadata_cols and training_features[col].dtype in [np.float32, np.float64, np.int32, np.int64]
        ]
    
    stats = {}
    for feat in feature_names:
        if feat in training_features.columns:
            values = training_features[feat].values
            stats[feat] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'has_nan': bool(np.any(np.isnan(values))),
                'has_inf': bool(np.any(np.isinf(values)))
            }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[FeatureStats] Saved statistics for {len(stats)} features to {output_path}")
