"""Utility helpers for the splice inference workflow."""

from __future__ import annotations
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict

__all__ = [
    "validate_score_columns",
    "load_model_with_calibration",
    "perform_neighborhood_analysis",
    "standardize_label_encoding",
    "diagnostic_sampling"
]

# -------------------------------------------------------------------
# Diagnostics helper -------------------------------------------------
# -------------------------------------------------------------------
def validate_score_columns(df: pl.DataFrame, src: Path, *, eps: float = 1e-9, verbose: int = 0) -> None:
    """Validate donor/acceptor/neither score columns.

    Ensures columns are float, non-null, and within [0,1]. If
    `verbose >= 3` also checks that their sum ≈ 1.0.
    """
    cols = ["donor_score", "acceptor_score", "neither_score"]
    # dtype check
    bad_dtype = [c for c in cols if df[c].dtype not in (pl.Float32, pl.Float64)]
    if bad_dtype:
        raise ValueError(f"{src.name}: non-float dtype for {bad_dtype}")
    # nulls – iterate per column to avoid DataFrame shape issues
    if any(df[c].null_count() > 0 for c in cols):
        raise ValueError(f"{src.name}: null values detected in score columns")
    # range
    range_bad = (
        ((df["donor_score"] < -eps) | (df["donor_score"] > 1 + eps))
        | ((df["acceptor_score"] < -eps) | (df["acceptor_score"] > 1 + eps))
        | ((df["neither_score"] < -eps) | (df["neither_score"] > 1 + eps))
    ).any()
    if range_bad:
        raise ValueError(f"{src.name}: score values outside [0,1] range")
    if verbose >= 3:
        incoherent = (
            df.with_columns((pl.col("donor_score") + pl.col("acceptor_score") + pl.col("neither_score")).alias("__sum"))
                .select(((pl.col("__sum") - 1.0).abs() > 1e-3).any()).item()
        )
        if incoherent:
            raise ValueError(f"{src.name}: donor+acceptor+neither not ≈ 1.0 for some rows")


# -------------------------------------------------------------------
# Calibrated model loading helpers ---------------------------------
# -------------------------------------------------------------------
def load_model_with_calibration(model_path: Path, use_calibration: bool = True) -> Any:
    """Load a model with optional calibration layer.

    This function handles multiple model formats and calibration types:
    - Standard XGBoost .json models (uncalibrated)
    - Pickle files containing calibrated ensembles
      - CalibratedSigmoidEnsemble (binary calibration)
      - PerClassCalibratedSigmoidEnsemble (per-class calibration)

    Parameters
    ----------
    model_path : Path
        Path to the model file (.json, .pkl)
    use_calibration : bool, default=True
        Whether to use calibration when available
        If False, will attempt to extract the base model from calibrated ensembles
    
    Returns
    -------
    Any
        Loaded model object (XGBClassifier or calibrated ensemble)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Handle XGBoost JSON models
    if model_path.suffix.lower() == '.json':
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(model_path))
        return model
        
    # Handle pickled models (including calibrated ensembles)
    elif model_path.suffix.lower() == '.pkl':
        # Pickle compatibility: Handle models trained with old package name
        # This allows loading models pickled as 'splice_surveyor' after package rename to 'meta_spliceai'
        import sys
        import meta_spliceai
        sys.modules['splice_surveyor'] = meta_spliceai
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Handle calibrated ensembles
        if not use_calibration:
            # Extract base model from calibrated ensemble if possible
            if hasattr(model, 'models') and len(model.models) > 0:
                return model.models[0]  # Return the base model
        return model
            
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """Get information about a model file.
    
    Parameters
    ----------
    model_path : Path
        Path to model file
        
    Returns
    -------
    Dict
        Dictionary with model information
        - type: 'xgboost', 'calibrated_binary', 'calibrated_per_class'
        - has_calibration: bool
        - calibration_method: 'platt', 'isotonic', None
    """
    info = {
        "type": None,
        "has_calibration": False,
        "calibration_method": None
    }
    
    try:
        # Check file extension
        if model_path.suffix.lower() == '.json':
            info["type"] = "xgboost"
            return info
            
        # Handle pickled models
        elif model_path.suffix.lower() == '.pkl':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            # Determine model type
            if hasattr(model, '__class__'):
                class_name = model.__class__.__name__
                if class_name == "CalibratedSigmoidEnsemble":
                    info["type"] = "calibrated_binary"
                    info["has_calibration"] = True
                    
                    # Determine calibration method
                    if hasattr(model, 'calibrator'):
                        if model.calibrator.__class__.__name__ == "LogisticRegression":
                            info["calibration_method"] = "platt"
                        elif model.calibrator.__class__.__name__ == "IsotonicRegression":
                            info["calibration_method"] = "isotonic"
                            
                elif class_name == "PerClassCalibratedSigmoidEnsemble":
                    info["type"] = "calibrated_per_class"
                    info["has_calibration"] = True
                    
                    # Check first non-None calibrator
                    if hasattr(model, 'calibrators') and model.calibrators:
                        for cal in model.calibrators:
                            if cal is not None:
                                if cal.__class__.__name__ == "LogisticRegression":
                                    info["calibration_method"] = "platt"
                                    break
                                elif cal.__class__.__name__ == "IsotonicRegression":
                                    info["calibration_method"] = "isotonic"
                                    break
                else:
                    info["type"] = "unknown"
    except Exception as e:
        info["type"] = "error"
        info["error"] = str(e)
        
    return info


# -------------------------------------------------------------------
# Label encoding standardization ------------------------------------
# -------------------------------------------------------------------
def standardize_label_encoding(labels) -> np.ndarray:
    """Standardize label encoding to ensure consistency.
    
    MetaSpliceAI uses the following encoding:
    - 0: Donor sites
    - 1: Acceptor sites
    - 2 or '2' or 'neither' or 0 (int): Neither (non-splice sites)
    
    This function ensures consistent encoding regardless of input format.
    
    Parameters
    ----------
    labels : array-like
        Input labels in various formats
        
    Returns
    -------
    np.ndarray
        Standardized label array with values 0, 1, or 2
    """
    if isinstance(labels, (pd.Series, pd.DataFrame)):
        labels = labels.values
    elif isinstance(labels, pl.Series):
        labels = labels.to_numpy()
    elif isinstance(labels, pl.DataFrame):
        if len(labels.columns) != 1:
            raise ValueError("DataFrame must have exactly one column")
        labels = labels.to_numpy().flatten()
    
    # Convert to numpy array if not already
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
        
    # Handle string labels
    if labels.dtype.kind in ['U', 'S']:  # Unicode or byte strings
        # Create a new array for numeric labels
        numeric_labels = np.zeros(labels.shape, dtype=np.int32)
        
        # Map string values to numeric
        for i, label in enumerate(labels):
            label_str = str(label).strip().lower()
            if label_str in ['0', 'donor']:
                numeric_labels[i] = 0
            elif label_str in ['1', 'acceptor']:
                numeric_labels[i] = 1
            elif label_str in ['2', 'neither', '0']:
                numeric_labels[i] = 2
            else:
                raise ValueError(f"Unknown label: {label}")
                
        return numeric_labels
    
    # Handle numeric labels directly
    return labels.astype(np.int32)


# -------------------------------------------------------------------
# Neighborhood analysis --------------------------------------------
# -------------------------------------------------------------------
def perform_neighborhood_analysis(
    model: Any, 
    X: Union[np.ndarray, pl.DataFrame, pd.DataFrame],
    df: Union[pl.DataFrame, pd.DataFrame],
    chrom_col: str, 
    pos_col: str, 
    sample_count: int, 
    window_size: int,
    out_dir: Path,
    plot_title: str = "Neighborhood Probability Patterns",
):
    """Analyze model predictions in neighborhoods around sampled positions.
    
    Parameters
    ----------
    model : classifier
        Trained model to generate predictions
    X : array-like
        Feature matrix for prediction
    df : pd.DataFrame or pl.DataFrame
        Original data frame with position information
    chrom_col : str
        Column name for chromosome
    pos_col : str
        Column name for position
    sample_count : int
        Number of positions to sample
    window_size : int
        Window size around sampled positions
    out_dir : Path
        Output directory for analysis results
    plot_title : str, default="Neighborhood Probability Patterns"
        Title for the visualization
        
    Returns
    -------
    dict
        Analysis results
    """
    if sample_count <= 0 or window_size <= 0:
        return None
    
    # Ensure output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
        
    # Convert polars DataFrame to pandas if needed
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    # Sample positions to analyze
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(df), min(sample_count, len(df)), replace=False)
    
    # Extract chromosome and position for each sample
    sample_chroms = df[chrom_col].iloc[sample_indices].values
    sample_positions = df[pos_col].iloc[sample_indices].values
    
    # Group samples by chromosome
    chrom_samples = defaultdict(list)
    for idx, (chrom, pos) in enumerate(zip(sample_chroms, sample_positions)):
        chrom_samples[chrom].append((idx, pos))
    
    # Results container
    neighborhood_results = []
    
    # Process each chromosome
    for chrom, samples in chrom_samples.items():
        # Get all positions for this chromosome
        chrom_mask = df[chrom_col] == chrom
        chrom_df = df[chrom_mask]
        
        # Get features for this chromosome
        if isinstance(X, np.ndarray):
            chrom_X = X[chrom_mask]
        elif isinstance(X, pd.DataFrame):
            chrom_X = X.loc[chrom_mask].values
        elif isinstance(X, pl.DataFrame):
            chrom_X = X.filter(pl.col(chrom_col) == chrom).to_pandas().values
        else:
            raise TypeError(f"Unsupported type for X: {type(X)}")
        
        # Process each sampled position
        for sample_idx, center_pos in samples:
            # Find positions within the window
            window_mask = (chrom_df[pos_col] >= center_pos - window_size) & \
                          (chrom_df[pos_col] <= center_pos + window_size)
            window_df = chrom_df[window_mask]
            
            if len(window_df) == 0:
                continue
                
            # Get indices in the chromosome subset
            window_indices = window_df.index
            window_X = chrom_X[np.where(np.isin(chrom_df.index, window_indices))[0]]
                
            # Get predictions for positions in the window
            window_probs = model.predict_proba(window_X)
            
            # Record results
            for i, (idx, probs) in enumerate(zip(window_indices, window_probs)):
                result = {
                    "sample_idx": sample_idx,
                    "chrom": chrom,
                    "center_pos": center_pos,
                    "pos": chrom_df.loc[idx, pos_col],
                    "rel_pos": chrom_df.loc[idx, pos_col] - center_pos,
                    "prob_donor": probs[0],  # Assuming 0=donor
                    "prob_acceptor": probs[1],  # Assuming 1=acceptor
                    "prob_neither": probs[2],  # Assuming 2=neither
                }
                
                # Add true label if available
                if "label" in chrom_df.columns:
                    result["true_label"] = chrom_df.loc[idx, "label"]
                    
                neighborhood_results.append(result)
    
    if not neighborhood_results:
        print("No neighborhood results generated. Check window size and sample count.")
        return None
        
    # Convert to DataFrame
    neigh_df = pd.DataFrame(neighborhood_results)
    
    # Save results
    neigh_path = out_dir / "neighborhood_analysis.csv"
    neigh_df.to_csv(neigh_path, index=False)
    print(f"Saved neighborhood analysis to {neigh_path}")
    
    # Create visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Group by relative position and calculate average probabilities
        pos_probs = neigh_df.groupby("rel_pos").agg({
            "prob_donor": "mean",
            "prob_acceptor": "mean",
            "prob_neither": "mean"
        }).reset_index()
        
        plt.plot(pos_probs["rel_pos"], pos_probs["prob_donor"], label="Donor")
        plt.plot(pos_probs["rel_pos"], pos_probs["prob_acceptor"], label="Acceptor")
        plt.plot(pos_probs["rel_pos"], pos_probs["prob_neither"], label="Neither")
        
        plt.xlabel("Distance from sampled position")
        plt.ylabel("Average probability")
        plt.title(plot_title)
        plt.axvline(x=0, color="gray", linestyle="--")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(out_dir / "neighborhood_analysis.png")
        plt.close()
        print(f"Created neighborhood visualization at {out_dir / 'neighborhood_analysis.png'}")
        
    except Exception as e:
        print(f"Failed to create neighborhood visualization: {e}")
        
    return {"neighborhood_samples": len(neigh_df)}


# -------------------------------------------------------------------
# Unseen position analysis ----------------------------------------
# -------------------------------------------------------------------
def analyze_unseen_neighborhood(
    model_path: Path,
    center_positions: Dict[str, List[int]],  # Map of chrom/gene_id -> positions
    window_size: int = 50,
    output_dir: Optional[Path] = None,
    use_calibration: bool = True,
    covered_pos: Optional[Dict[str, Set[int]]] = None,
    t_low: float = 0.02,
    t_high: float = 0.80,
    verbosity: int = 1,
) -> Dict[str, Any]:
    """
    Analyze neighborhoods around specific positions, including unseen positions.
    
    This function integrates the inference workflow to generate features for positions
    that were not included in the training data, then performs neighborhood analysis.
    
    Parameters
    ----------
    model_path : Path
        Path to the model file for prediction
    center_positions : Dict[str, List[int]]
        Mapping of chromosome/gene_id to a list of positions to analyze neighborhoods around
    window_size : int, default=50
        Size of window (in nucleotides) around each center position
    output_dir : Path, optional
        Output directory for results. If None, a temporary directory will be created
    use_calibration : bool, default=True
        Whether to use calibration if the model supports it
    covered_pos : Dict[str, Set[int]], optional
        Positions already covered in training data
    t_low, t_high : float
        Thresholds for the ambiguous score zone in the base model
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with analysis results
    """
    from meta_spliceai.splice_engine.meta_models.workflows.splice_inference_workflow import (
        run_enhanced_splice_inference_workflow
    )
    import tempfile
    import shutil
    
    # Create temporary output directory if none provided
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="neighborhood_analysis_")
        output_dir = Path(temp_dir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare target genes from center_positions
        target_genes = list(center_positions.keys())
        
        if verbosity >= 1:
            print(f"[neighborhood-analysis] Running inference workflow for {len(target_genes)} genes")
            print(f"[neighborhood-analysis] Analyzing neighborhoods around {sum(len(pos) for pos in center_positions.values())} positions")
        
        # Run the inference workflow to generate features for positions
        result = run_enhanced_splice_inference_workflow(
            target_genes=target_genes,
            covered_pos=covered_pos,
            t_low=t_low,
            t_high=t_high,
            model_path=model_path,
            use_calibration=use_calibration,
            neigh_sample=0,  # We'll do our own neighborhood analysis
            output_dir=output_dir,
            verbosity=verbosity
        )
        
        # Extract paths from result
        feature_dir = result.get("feature_dir")
        if not feature_dir:
            raise RuntimeError("Inference workflow didn't return a valid feature directory")
        
        # Load feature matrix
        import glob
        import pandas as pd
        
        parquet_files = list(glob.glob(str(feature_dir / "*.parquet")))
        if not parquet_files:
            raise RuntimeError(f"No feature files found in {feature_dir}")
        
        # Combine all parquet files
        feature_dfs = [pd.read_parquet(pf) for pf in parquet_files]
        if not feature_dfs:
            raise RuntimeError("Failed to load any feature data from parquet files")
            
        feature_df = pd.concat(feature_dfs, ignore_index=True)
        
        # Extract metadata and features
        meta_cols = ["gene_id", "chrom", "position", "strand"]
        meta_data = {}
        
        for col in meta_cols:
            if col in feature_df.columns:
                meta_data[col] = feature_df[col]
                feature_df = feature_df.drop(columns=[col])
        
        # Get label if available
        if "label" in feature_df.columns:
            meta_data["label"] = feature_df["label"]
            feature_df = feature_df.drop(columns=["label"])
        
        # Convert to numpy for prediction
        X = feature_df.values
        
        # Load model
        model = load_model_with_calibration(model_path, use_calibration=use_calibration)
        
        # Create dataframe from metadata
        meta_df = pd.DataFrame(meta_data)
        
        # Determine chromosome/gene column
        id_col = "gene_id" if "gene_id" in meta_df.columns else "chrom"
        
        # Perform neighborhood analysis for each center position
        all_results = []
        
        for chrom_or_gene, positions in center_positions.items():
            # Filter data for this chromosome/gene
            chrom_mask = meta_df[id_col] == chrom_or_gene
            chrom_df = meta_df[chrom_mask]
            chrom_X = X[chrom_mask]
            
            if len(chrom_df) == 0:
                if verbosity >= 1:
                    print(f"[warning] No data found for {id_col}={chrom_or_gene}")
                continue
            
            # Process each center position
            for center_pos in positions:
                # Find positions within the window
                window_mask = (chrom_df["position"] >= center_pos - window_size) & \
                              (chrom_df["position"] <= center_pos + window_size)
                window_df = chrom_df[window_mask]
                
                if len(window_df) == 0:
                    if verbosity >= 2:
                        print(f"[warning] No positions found in window around {chrom_or_gene}:{center_pos}")
                    continue
                
                # Get window data indices
                window_indices = window_df.index
                window_X = chrom_X[np.where(np.isin(chrom_df.index, window_indices))[0]]
                
                # Make predictions
                probs = model.predict_proba(window_X)
                
                # Record results
                for i, (idx, prob) in enumerate(zip(window_indices, probs)):
                    result = {
                        "id": chrom_or_gene,
                        "center_pos": center_pos,
                        "pos": chrom_df.loc[idx, "position"],
                        "rel_pos": chrom_df.loc[idx, "position"] - center_pos,
                        "prob_donor": prob[0],
                        "prob_acceptor": prob[1],
                        "prob_neither": prob[2],
                        "strand": chrom_df.loc[idx, "strand"] if "strand" in chrom_df.columns else None
                    }
                    
                    # Add label if available
                    if "label" in chrom_df.columns:
                        result["true_label"] = chrom_df.loc[idx, "label"]
                        
                    all_results.append(result)
        
        # Convert to dataframe
        results_df = pd.DataFrame(all_results)
        
        if len(results_df) == 0:
            if verbosity >= 1:
                print("[warning] No neighborhood results generated")
            return {"success": False, "error": "No results generated"}
        
        # Save results
        results_path = output_dir / "unseen_neighborhood_analysis.csv"
        results_df.to_csv(results_path, index=False)
        
        # Create visualization
        try:
            plt.figure(figsize=(12, 8))
            
            # Group by relative position and calculate average probabilities
            pos_probs = results_df.groupby("rel_pos").agg({
                "prob_donor": "mean",
                "prob_acceptor": "mean",
                "prob_neither": "mean"
            }).reset_index()
            
            plt.plot(pos_probs["rel_pos"], pos_probs["prob_donor"], label="Donor")
            plt.plot(pos_probs["rel_pos"], pos_probs["prob_acceptor"], label="Acceptor")
            plt.plot(pos_probs["rel_pos"], pos_probs["prob_neither"], label="Neither")
            
            plt.xlabel("Distance from center position")
            plt.ylabel("Average probability")
            plt.title(f"Neighborhood Analysis (window={window_size})")
            plt.axvline(x=0, color="gray", linestyle="--")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plt.savefig(output_dir / "unseen_neighborhood_analysis.png")
            plt.close()
        except Exception as e:
            if verbosity >= 1:
                print(f"[warning] Failed to create visualization: {e}")
        
        return {
            "success": True,
            "results_path": results_path,
            "output_dir": output_dir,
            "num_samples": len(results_df),
            "center_positions": center_positions
        }
        
    except Exception as e:
        if verbosity >= 1:
            print(f"[error] Failed to analyze unseen neighborhoods: {e}")
        return {"success": False, "error": str(e)}
        
    finally:
        # Clean up temporary directory if we created one
        if temp_dir is not None:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


# -------------------------------------------------------------------
# Diagnostic sampling ----------------------------------------------
# -------------------------------------------------------------------
def diagnostic_sampling(
    X: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series, pl.Series]] = None,
    meta_data: Optional[Dict[str, Union[np.ndarray, pd.Series, pl.Series]]] = None,
    sample_size: int = 5000,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[Union[np.ndarray, pd.DataFrame, pl.DataFrame], Optional[np.ndarray], Optional[Dict]]:
    """Sample a subset of data for detailed diagnostics.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like, optional
        Labels
    meta_data : dict, optional
        Additional metadata columns like chromosome, position
    sample_size : int, default=5000
        Number of samples to return
    stratify : bool, default=True
        Whether to stratify sampling by label
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple
        (X_sample, y_sample, meta_data_sample)
    """
    if sample_size <= 0 or sample_size >= len(X):
        # No sampling needed
        return X, y, meta_data
    
    # Set random seed
    np.random.seed(random_state)
    
    # Determine sampling indices
    if stratify and y is not None:
        # Convert y to numpy array if needed
        if isinstance(y, (pd.Series, pl.Series)):
            y_np = y.to_numpy()
        else:
            y_np = np.array(y)
        
        # Get unique labels and their frequencies
        unique_labels, counts = np.unique(y_np, return_counts=True)
        
        # Calculate samples per class
        samples_per_class = {}
        for label, count in zip(unique_labels, counts):
            # Proportional allocation with minimum of 1 sample per class
            samples_per_class[label] = max(1, int(sample_size * (count / len(y_np))))
        
        # Adjust for rounding errors
        total = sum(samples_per_class.values())
        if total < sample_size:
            # Add remaining samples to the largest class
            largest_class = max(samples_per_class, key=lambda k: samples_per_class[k])
            samples_per_class[largest_class] += sample_size - total
        
        # Sample from each class
        indices = []
        for label, n_samples in samples_per_class.items():
            class_indices = np.where(y_np == label)[0]
            if len(class_indices) > n_samples:
                sampled = np.random.choice(class_indices, n_samples, replace=False)
                indices.extend(sampled)
            else:
                # Take all if not enough samples
                indices.extend(class_indices)
    else:
        # Simple random sampling
        indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
    
    # Extract sampled data
    if isinstance(X, np.ndarray):
        X_sample = X[indices]
    elif isinstance(X, pd.DataFrame):
        X_sample = X.iloc[indices]
    elif isinstance(X, pl.DataFrame):
        X_sample = X.select(pl.all().take(indices))
    else:
        raise TypeError(f"Unsupported type for X: {type(X)}")
    
    # Extract labels if provided
    y_sample = None
    if y is not None:
        if isinstance(y, np.ndarray):
            y_sample = y[indices]
        elif isinstance(y, pd.Series):
            y_sample = y.iloc[indices]
        elif isinstance(y, pl.Series):
            y_sample = y.take(indices)
    
    # Extract metadata if provided
    meta_data_sample = None
    if meta_data is not None:
        meta_data_sample = {}
        for key, value in meta_data.items():
            if isinstance(value, np.ndarray):
                meta_data_sample[key] = value[indices]
            elif isinstance(value, pd.Series):
                meta_data_sample[key] = value.iloc[indices]
            elif isinstance(value, pl.Series):
                meta_data_sample[key] = value.take(indices)
    
    return X_sample, y_sample, meta_data_sample