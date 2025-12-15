"""
Utility functions for the error model module.

This module provides helper functions for the error model package,
such as output verification and result processing utilities.
"""

import os
from typing import List, Dict, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt


def compare_directories(dir1, dir2, prefix="", ignore_patterns=None):
    """
    Compare two directory structures recursively and report differences.
    
    Parameters:
    - dir1 (str): Path to the first directory for comparison
    - dir2 (str): Path to the second directory for comparison
    - prefix (str): String prefix to use for indentation in output
    - ignore_patterns (list): List of regex patterns to ignore in comparison
    
    Returns:
    - bool: True if directories match after ignoring expected differences
    """
    if ignore_patterns is None:
        # Default patterns to ignore - sample-specific files and status files
        ignore_patterns = [
            r'.*-sample\d+\.pdf$',  # Sample-specific PDFs (random samples)
            r'.*_done\.txt$'         # Status files
        ]
    
    # List all files in both directories
    dir1_items = set(os.listdir(dir1))
    dir2_items = set(os.listdir(dir2))
    
    # Check common items, items only in dir1, and items only in dir2
    common_items = dir1_items.intersection(dir2_items)
    dir1_only = dir1_items - dir2_items
    dir2_only = dir2_items - dir1_items
    
    print(f"{prefix}Dir1 ({dir1}) has {len(dir1_items)} items")
    print(f"{prefix}Dir2 ({dir2}) has {len(dir2_items)} items")
    print(f"{prefix}Items in common: {len(common_items)}")
    print(f"{prefix}Items only in Dir1: {len(dir1_only)}")
    print(f"{prefix}Items only in Dir2: {len(dir2_only)}")
    
    # Filter out items that match ignore patterns
    filtered_dir1_only = []
    filtered_dir2_only = []
    
    import re
    
    for item in dir1_only:
        should_ignore = any(re.match(pattern, item) for pattern in ignore_patterns)
        if not should_ignore:
            filtered_dir1_only.append(item)
    
    for item in dir2_only:
        should_ignore = any(re.match(pattern, item) for pattern in ignore_patterns)
        if not should_ignore:
            filtered_dir2_only.append(item)
    
    # Now we'll print both the full lists and the filtered lists
    if dir1_only:
        print(f"{prefix}  Items only in Dir1 (full list): {sorted(dir1_only)}")
        if filtered_dir1_only:
            print(f"{prefix}  Non-ignorable items only in Dir1: {sorted(filtered_dir1_only)}")
        else:
            print(f"{prefix}  All differences in Dir1 are expected/ignorable")
    
    if dir2_only:
        print(f"{prefix}  Items only in Dir2 (full list): {sorted(dir2_only)}")
        if filtered_dir2_only:
            print(f"{prefix}  Non-ignorable items only in Dir2: {sorted(filtered_dir2_only)}")
        else:
            print(f"{prefix}  All differences in Dir2 are expected/ignorable")
    
    # Recursively compare subdirectories
    subdirs_match = True
    for item in common_items:
        path1 = os.path.join(dir1, item)
        path2 = os.path.join(dir2, item)
        
        if os.path.isdir(path1) and os.path.isdir(path2):
            # This is a directory, compare recursively
            print(f"\n{prefix}Comparing subdirectory: {item}")
            subdir_match = compare_directories(path1, path2, prefix + "  ", ignore_patterns)
            subdirs_match = subdirs_match and subdir_match
    
    # Determine if the directories match after ignoring expected differences
    dirs_match = (len(filtered_dir1_only) == 0 and len(filtered_dir2_only) == 0 and subdirs_match)
    
    if dirs_match:
        print(f"\n{prefix}✅ Directories match after ignoring expected differences!")
    else:
        print(f"\n{prefix}❌ Directories have meaningful differences (excluding expected variations)")
    
    return dirs_match


def verify_error_model_outputs(
    experiment_dir: str,
    splice_types: List[str] = None,
    error_models: List[str] = None
) -> bool:
    """
    Verify that expected output files exist with correct naming conventions.
    
    Parameters
    ----------
    experiment_dir : str
        Full path to the experiment directory containing outputs
    splice_types : list, default=None
        List of splice types to check, if None checks all ["donor", "acceptor", "any"]
    error_models : list, default=None
        List of error models to check, if None checks all ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
        
    Returns
    -------
    bool
        True if any expected files exist, False otherwise
    """
    if splice_types is None:
        splice_types = ["donor", "acceptor", "any"]
    
    if error_models is None:
        error_models = ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
    
    # Error label mapping
    error_label_map = {
        "fp_vs_tp": ("FP", "TP"),
        "fn_vs_tp": ("FN", "TP"),
        "fn_vs_tn": ("FN", "TN"),
        "fp_vs_tn": ("FP", "TN")  # Not biologically meaningful
    }
    
    # Build a list of expected status files
    expected_files = []
    for splice_type in splice_types:
        for error_model in error_models:
            # Skip FP vs TN as it's not biologically meaningful
            if error_model == "fp_vs_tn":
                continue
                
            # Convert model name format
            if error_model in error_label_map:
                error_label, correct_label = error_label_map[error_model]
            else:
                continue  # Skip invalid error models
            
            # Expected status file - use lowercase for directory names
            expected_file = os.path.join(
                experiment_dir, 
                splice_type,  # Keep original case for splice types
                error_model,  # This is already lowercase: fp_vs_tp, fn_vs_tp, fn_vs_tn
                "xgboost",
                f"{splice_type}_{error_label}_vs_{correct_label}_done.txt"
            )
            expected_files.append(expected_file)
    
    # Check if files exist
    found_files = [f for f in expected_files if os.path.exists(f)]
    
    if not found_files:
        print("\nNo expected output files were found.")
        return False
    else:
        print(f"\nFound {len(found_files)} out of {len(expected_files)} expected output files.")
        print("Files found:")
        for f in found_files:
            print(f"  • {os.path.basename(f)}")
        
        # Show missing files if any
        if len(found_files) < len(expected_files):
            missing_files = [f for f in expected_files if f not in found_files]
            print("\nFiles missing:")
            
            # Group missing files by splice type and error model for better readability
            missing_by_group = {}
            for f in missing_files:
                # Extract splice type and error model from path
                parts = f.split(os.sep)
                if len(parts) >= 4:
                    splice_type = parts[-4]  # Splice type directory
                    error_model = parts[-3]  # Error model directory (already lowercase)
                    
                    key = f"{splice_type}: {error_model}"
                    if key not in missing_by_group:
                        missing_by_group[key] = []
                    missing_by_group[key].append(os.path.basename(f))
            
            # Print missing files by group
            for group, files in missing_by_group.items():
                print(f"  Missing in {group}:")
                for f in files:
                    print(f"    - {f}")
            
            print("\nPossible reasons for missing files:")
            print("  • Dataset not found for this combination")
            print("  • Error during processing of this combination")
            print("  • Check the log output above for specific errors")
        
        return True


def get_model_status(
    experiment_dir: str,
    splice_types: List[str] = None,
    error_models: List[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Get the status of error models across all combinations.
    
    Parameters
    ----------
    experiment_dir : str
        Full path to the experiment directory containing outputs
    splice_types : list, default=None
        List of splice types to check, if None checks all ["donor", "acceptor", "any"]
    error_models : list, default=None
        List of error models to check, if None checks all ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
        
    Returns
    -------
    dict
        Nested dictionary with structure: {splice_type: {error_model: status}}
        where status is one of "completed", "missing", or "error"
    """
    if splice_types is None:
        splice_types = ["donor", "acceptor", "any"]
    
    if error_models is None:
        error_models = ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
    
    # Error label mapping
    error_label_map = {
        "fp_vs_tp": ("FP", "TP"),
        "fn_vs_tp": ("FN", "TP"),
        "fn_vs_tn": ("FN", "TN")
    }
    
    # Initialize status dictionary
    status_dict = {splice_type: {error_model: "missing" for error_model in error_models} 
                   for splice_type in splice_types}
    
    # Check each combination
    for splice_type in splice_types:
        for error_model in error_models:
            # Skip FP vs TN as it's not biologically meaningful
            if error_model == "fp_vs_tn":
                status_dict[splice_type][error_model] = "skipped"
                continue
                
            # Convert model name format
            error_label, correct_label = error_label_map[error_model]
            
            # Expected status file
            status_file = os.path.join(
                experiment_dir, 
                splice_type,
                error_model,
                "xgboost",
                f"{splice_type}_{error_label}_vs_{correct_label}_done.txt"
            )
            
            # Expected error log file
            error_file = os.path.join(
                experiment_dir, 
                splice_type,
                error_model,
                "xgboost",
                f"{splice_type}_{error_label}_vs_{correct_label}_error.log"
            )
            
            # Check status
            if os.path.exists(status_file):
                status_dict[splice_type][error_model] = "completed"
            elif os.path.exists(error_file):
                status_dict[splice_type][error_model] = "error"
    
    return status_dict


def get_model_file_paths(
    experiment_dir: str,
    splice_type: str,
    error_model: str,
    file_types: List[str] = None
) -> Dict[str, str]:
    """
    Get paths to output files for a specific error model.
    
    Parameters
    ----------
    experiment_dir : str
        Full path to the experiment directory containing outputs
    splice_type : str
        Type of splice site ("donor", "acceptor", or "any")
    error_model : str
        Error model name (e.g., "fp_vs_tp")
    file_types : list, default=None
        List of file types to find, if None returns paths for all known types
        
    Returns
    -------
    dict
        Dictionary mapping file types to absolute file paths
    """
    if file_types is None:
        file_types = [
            "feature_distributions", "feature_importance_comparison", 
            "global_importance", "shap_summary", "shap_beeswarm",
            "roc_curve", "pr_curve", "local_shap", 
            "importance_files", "importance_plots", "done"
        ]
    
    # Error label mapping
    error_label_map = {
        "fp_vs_tp": ("FP", "TP"),
        "fn_vs_tp": ("FN", "TP"),
        "fn_vs_tn": ("FN", "TN"),
        # "fp_vs_tn": ("FP", "TN")
    }
    
    # Convert model name format
    error_label, correct_label = error_label_map.get(error_model, ("", ""))
    if not error_label:
        return {}
    
    # Create the base file prefix for this model
    model_prefix = f"{splice_type}_{error_label.lower()}_vs_{correct_label.lower()}"
    
    # File paths can be either in the experiment directory or in the xgboost subdirectory
    base_dir = os.path.join(experiment_dir)
    xgboost_dir = os.path.join(base_dir, "xgboost")
    
    # Initialize results
    file_paths = {}
    
    # Status file is always in the base directory
    done_file = os.path.join(base_dir, f"{splice_type}_{error_label}_vs_{correct_label}_done.txt")
    if os.path.exists(done_file) and "done" in file_types:
        file_paths["done"] = done_file
    
    # Common file patterns to search for
    file_patterns = {
        "feature_distributions": [
            f"{model_prefix}-feature-distributions.pdf",
            f"{model_prefix}-feature-distributions-advanced*.pdf",  # Handles versioned variants
        ],
        "feature_importance_comparison": [
            f"{model_prefix}-feature-importance-comparison.pdf",
            f"{model_prefix}-xgboost-total_gain-comparison.pdf"
        ],
        "global_importance": [
            f"{model_prefix}-global_importance-barplot.pdf",
            f"{model_prefix}-global_shap_importance-meta.csv"
        ],
        "shap_summary": [
            f"{model_prefix}-shap_summary*-meta.pdf",
            f"{model_prefix}-shap_summary_with_margin.pdf"
        ],
        "shap_beeswarm": [
            f"{model_prefix}-shap_beeswarm-meta.pdf"
        ],
        "roc_curve": [
            f"{model_prefix}-xgboost-ROC-CV-*folds.pdf",  # Pattern for CV-based ROC with any fold count
            f"{model_prefix}-xgboost-roc.pdf"
        ],
        "pr_curve": [
            f"{model_prefix}-xgboost-PRC-CV-*folds.pdf",  # Pattern for CV-based PRC with any fold count
            f"{model_prefix}-xgboost-prc.pdf"
        ],
        "local_shap": [
            f"{model_prefix}-sample*.pdf",  # Pattern for individual sample SHAP plots with any ID
            f"{model_prefix}-local-shap-frequency-comparison-meta.pdf",
            f"{model_prefix}-local_top*_freq-meta.csv"  # Pattern for top-k frequency with any k
        ],
        "importance_files": [
            # XGBoost importance TSVs
            f"{model_prefix}-xgboost-importance-*.tsv"
        ],
        "importance_plots": [
            # XGBoost importance barplots
            f"{model_prefix}-xgboost-*-barplot.pdf"
        ]
    }
    
    # Check for exact files first
    for file_type, patterns in file_patterns.items():
        if file_type not in file_types:
            continue
            
        found = False
        for pattern in patterns:
            # Handle glob patterns
            if "*" in pattern:
                import glob
                # Search in both dirs
                base_matches = glob.glob(os.path.join(base_dir, pattern))
                xgb_matches = glob.glob(os.path.join(xgboost_dir, pattern))
                matches = base_matches + xgb_matches
                
                if matches:
                    file_paths[file_type] = matches[0]  # Take the first match
                    found = True
                    break
            else:
                # Check exact file names in both dirs
                base_path = os.path.join(base_dir, pattern)
                xgb_path = os.path.join(xgboost_dir, pattern)
                
                if os.path.exists(base_path):
                    file_paths[file_type] = base_path
                    found = True
                    break
                elif os.path.exists(xgb_path):
                    file_paths[file_type] = xgb_path
                    found = True
                    break
    
    return file_paths


def safely_save_figure(fig, output_path, dpi=300, bbox_inches='tight', pad_inches=0.1):
    """
    Safely save a matplotlib figure and properly close it to prevent memory leaks.
    
    This function addresses the common issue where matplotlib figures aren't properly closed
    after saving, leading to the "More than 20 figures have been opened" warning and potential
    memory leaks.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_path : str
        Path where the figure should be saved
    dpi : int, default=300
        Resolution in dots per inch
    bbox_inches : str or Bbox, default='tight'
        Bounding box in inches 
    pad_inches : float, default=0.1
        Amount of padding in inches
        
    Returns
    -------
    None
    
    Notes
    -----
    After this function is called, the figure is closed and should not be used.
    Follow the best practices documented in meta_spliceai/splice_engine/docs/matplotlib_best_practices.md
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the figure
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        
        # Explicitly close the figure to prevent memory leaks
        fig.clf()
        plt.close(fig)
    except Exception as e:
        print(f"Error saving figure to {output_path}: {str(e)}")


def select_samples_for_analysis(
    X, 
    y, 
    y_pred=None, 
    y_pred_proba=None, 
    sample_selection="misclassified", 
    n_samples=10, 
    custom_indices=None, 
    random_state=42
):
    """
    Select samples for targeted local feature importance visualization.
    
    This function implements systematic sample selection for more targeted analysis
    of model behavior, supporting various selection strategies.
    
    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : array-like
        True labels
    y_pred : array-like, optional
        Model predictions (required for "misclassified" selection)
    y_pred_proba : array-like, optional
        Prediction probabilities (required for confidence-based selections)
    sample_selection : str, default="misclassified"
        Selection strategy:
        - "random": Original method, selects random samples
        - "high_confidence": Samples where model is most confident
        - "low_confidence": Samples where model is least confident
        - "border": Samples near the decision boundary (most uncertain)
        - "misclassified": Focuses on samples the model got wrong
        - "custom": Allows providing specific sample indices
    n_samples : int, default=10
        Number of samples to select
    custom_indices : list, optional
        List of specific indices for custom selection
    random_state : int, default=42
        Random seed for reproducible selection
        
    Returns
    -------
    numpy.ndarray
        Array of selected indices
    
    Notes
    -----
    This implementation gracefully falls back to random selection when the
    requested selection method is not possible with the available data.
    """
    import numpy as np
    
    # Check inputs and initialize
    n_samples = min(n_samples, len(X))
    rng = np.random.RandomState(random_state)
    
    # Custom selection (highest priority)
    if sample_selection == "custom" and custom_indices is not None:
        return np.array(custom_indices)[:n_samples]
    
    # Misclassified samples
    elif sample_selection == "misclassified" and y_pred is not None and y is not None:
        misclassified = np.where(y != y_pred)[0]
        if len(misclassified) == 0:
            print("Warning: No misclassified samples found. Falling back to random selection.")
            return rng.choice(len(X), size=n_samples, replace=False)
        elif len(misclassified) < n_samples:
            print(f"Warning: Only {len(misclassified)} misclassified samples found, "
                 f"which is less than the requested {n_samples}.")
            return misclassified
        else:
            return rng.choice(misclassified, size=n_samples, replace=False)
    
    # Confidence-based selections (require probability estimates)
    elif sample_selection in ["high_confidence", "low_confidence", "border"] and y_pred_proba is not None:
        # Get the probability of the predicted class
        if y_pred_proba.ndim > 1:  # Multi-class
            # Get the highest probability for each sample
            confidences = np.max(y_pred_proba, axis=1)
        else:  # Binary classification with single probability
            # Convert to probability of predicted class
            confidences = np.where(y_pred_proba > 0.5, y_pred_proba, 1 - y_pred_proba)
        
        if sample_selection == "high_confidence":
            # Most confident predictions (highest probability)
            indices = np.argsort(confidences)[-n_samples:]
        elif sample_selection == "low_confidence":
            # Least confident predictions (lowest probability)
            indices = np.argsort(confidences)[:n_samples]
        else:  # "border"
            # Samples closest to the decision boundary (closest to 0.5)
            if y_pred_proba.ndim > 1:  # Multi-class
                # For multi-class, find samples with smallest margin between top classes
                sorted_probs = np.sort(y_pred_proba, axis=1)
                margins = sorted_probs[:, -1] - sorted_probs[:, -2]  # Top vs second probability
                indices = np.argsort(margins)[:n_samples]
            else:  # Binary
                # Find samples closest to 0.5 probability
                indices = np.argsort(np.abs(y_pred_proba - 0.5))[:n_samples]
        
        return indices
    
    # Default to random selection
    else:
        if sample_selection != "random":
            print(f"Warning: Cannot use '{sample_selection}' selection strategy "
                 f"with the available data. Falling back to random selection.")
        return rng.choice(len(X), size=n_samples, replace=False)


def get_sample_shap_files(
    experiment_dir: str,
    splice_type: str,
    error_model: str
) -> List[str]:
    """
    Get all sample-specific SHAP plot files for a given error model.
    
    This function searches for sample-specific SHAP visualization files
    that follow the pattern "{prefix}-sample{ID}.pdf" where ID can be any number.
    
    Parameters
    ----------
    experiment_dir : str
        Full path to the experiment directory containing outputs
    splice_type : str
        Type of splice site ("donor", "acceptor", or "any")
    error_model : str
        Error model name (e.g., "fp_vs_tp")
        
    Returns
    -------
    list
        List of paths to all sample-specific SHAP plot files, sorted by sample ID
    """
    # Error label mapping
    error_label_map = {
        "fp_vs_tp": ("FP", "TP"),
        "fn_vs_tp": ("FN", "TP"),
        "fn_vs_tn": ("FN", "TN"),
        "fp_vs_tn": ("FP", "TN")
    }
    
    # Convert model name format
    error_label, correct_label = error_label_map.get(error_model, ("", ""))
    if not error_label:
        return []
    
    # Create the base file prefix for this model
    model_prefix = f"{splice_type}_{error_label.lower()}_vs_{correct_label.lower()}"
    
    # File paths can be either in the experiment directory or in the xgboost subdirectory
    base_dir = os.path.join(experiment_dir)
    xgboost_dir = os.path.join(base_dir, "xgboost")
    
    # Use glob pattern to find all sample files
    import glob
    base_matches = glob.glob(os.path.join(base_dir, f"{model_prefix}-sample*.pdf"))
    xgb_matches = glob.glob(os.path.join(xgboost_dir, f"{model_prefix}-sample*.pdf"))
    
    # Combine and sort the results - sort numerically by the sample ID when possible
    all_matches = base_matches + xgb_matches
    
    # Try to sort by the numeric sample ID if possible
    try:
        # Extract sample numbers for sorting
        import re
        pattern = re.compile(f"{model_prefix}-sample(\\d+).pdf")
        
        def get_sample_id(file_path):
            file_name = os.path.basename(file_path)
            match = pattern.match(file_name)
            if match:
                return int(match.group(1))
            return 0  # Default if no match
        
        all_matches.sort(key=get_sample_id)
    except Exception:
        # Fall back to regular sorting if regex fails
        all_matches.sort()
    
    return all_matches


def get_analysis_summary(
    experiment_dir: str,
    splice_types: List[str] = None,
    error_models: List[str] = None
) -> Dict:
    """
    Get a comprehensive summary of all analysis outputs for the given experiment.
    
    Parameters
    ----------
    experiment_dir : str
        Full path to the experiment directory containing outputs
    splice_types : list, default=None
        List of splice types to check, if None checks all ["donor", "acceptor", "any"]
    error_models : list, default=None
        List of error models to check, if None checks all ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
        
    Returns
    -------
    dict
        Nested dictionary with structure: 
        {splice_type: {error_model: {"status": status, "files": {file_type: path}}}}
    """
    # Set default values
    if splice_types is None:
        splice_types = ["donor", "acceptor", "any"]
    if error_models is None:
        error_models = ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn", "fp_vs_tn"]
    
    # Get all possible file types
    all_file_types = [
        "feature_distributions", "feature_importance_comparison", 
        "global_importance", "shap_summary", "shap_beeswarm",
        "roc_curve", "pr_curve", "local_shap", 
        "importance_files", "importance_plots", "done"
    ]
    
    # Initialize result dictionary
    summary_dict = {}
    
    # Iterate over all combinations
    for splice_type in splice_types:
        summary_dict[splice_type] = {}
        
        for error_model in error_models:
            # Get model status
            status = "pending"
            
            # Check for status file
            file_paths = get_model_file_paths(
                experiment_dir, splice_type, error_model, ["done"]
            )
            
            if "done" in file_paths:
                status = "completed"
            
            # Get all file paths
            all_files = get_model_file_paths(
                experiment_dir, splice_type, error_model, all_file_types
            )
            
            # Get sample-specific files
            sample_files = get_sample_shap_files(experiment_dir, splice_type, error_model)
            
            # If we found any files but no done marker, mark as "in_progress"
            if all_files and "done" not in all_files:
                status = "in_progress"
            
            # Store results
            summary_dict[splice_type][error_model] = {
                "status": status,
                "files": all_files,
                "sample_files": sample_files,
                "file_count": len(all_files) + len(sample_files)
            }
    
    return summary_dict
