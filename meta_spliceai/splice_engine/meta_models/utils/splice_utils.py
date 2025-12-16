"""
Utility functions for splice site analysis, adjustment, and processing.

UPDATED 2025-10-31: Now uses the new score-based adjustment detection module
(infer_score_adjustments.py) instead of the old position-based module.
"""

import os
import json
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Any

from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, 
    print_with_indent
)


def prepare_splice_site_adjustments(
    local_dir: str,
    ss_annotations_df: pd.DataFrame,
    sample_predictions: Optional[Dict[str, Any]] = None,
    use_empirical: bool = True,
    save_adjustments: bool = True,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Determine optimal splice site SCORE adjustments using correlated probability vectors.
    
    UPDATED 2025-10-31: Now uses score-based adjustment detection (infer_score_adjustments.py)
    instead of position-based adjustment. This maintains 100% coverage and probability constraints.
    
    Parameters
    ----------
    local_dir : str
        Directory to store adjustment information
    ss_annotations_df : pd.DataFrame
        Splice site annotations DataFrame (will be converted to Polars)
    sample_predictions : Optional[Dict[str, Any]], optional
        Sample predictions to use for adjustment inference, by default None
        Format: {'gene_id': {'donor_prob': array, 'acceptor_prob': array, ...}}
    use_empirical : bool, optional
        Whether to use empirical data-driven adjustment detection, by default True
    save_adjustments : bool, optional
        Whether to save the adjustment information, by default True
    verbosity : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'success': bool
        - 'adjustment_dict': Dict[str, Dict[str, int]]
        - 'adjustment_file': str
        
    Notes
    -----
    The adjustment detection now uses the score-shifting paradigm:
    - Shifts entire probability vectors (donor, acceptor, neither) together
    - Maintains probability constraint (sum = 1.0)
    - Creates splice-type-specific views when needed
    - 100% coverage (no position collisions)
    """
    result = {
        'success': False,
        'adjustment_dict': None,
        'adjustment_file': None
    }
    
    # First check if we already have saved adjustments
    adjustment_file = os.path.join(local_dir, "splice_site_adjustments.json")
    result['adjustment_file'] = adjustment_file
    
    if os.path.exists(adjustment_file):
        if verbosity >= 1:
            print_emphasized("[info] Loading existing splice site adjustments")
        
        with open(adjustment_file, 'r') as f:
            adjustment_dict = json.load(f)
            
        result['adjustment_dict'] = adjustment_dict
        result['success'] = True
        
        if verbosity >= 1:
            print_with_indent(f"Donor sites:    +{adjustment_dict['donor']['plus']} on plus strand, +{adjustment_dict['donor']['minus']} on minus strand", indent_level=1)
            print_with_indent(f"Acceptor sites: +{adjustment_dict['acceptor']['plus']} on plus strand, {adjustment_dict['acceptor']['minus']} on minus strand", indent_level=1)
        
        return result
    
    # If sample predictions not provided, can't perform empirical inference
    if sample_predictions is None and use_empirical:
        if verbosity >= 1:
            print_emphasized("[warn] Sample predictions required for empirical adjustment inference")
            print_with_indent("Using zero adjustments (assuming base model is aligned)", indent_level=1)
        use_empirical = False
    
    # Import NEW score-based adjustment utilities
    from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
        auto_detect_score_adjustments,
        save_adjustment_dict
    )
    
    if verbosity >= 1:
        print_emphasized("[action] Detecting splice site score adjustments (correlated probability vectors)")
    
    # Convert pandas DataFrame to polars for the new module
    if isinstance(ss_annotations_df, pd.DataFrame):
        ss_annotations_pl = pl.from_pandas(ss_annotations_df)
    else:
        ss_annotations_pl = ss_annotations_df
    
    # Use auto-detection with or without empirical approach
    adjustment_dict = auto_detect_score_adjustments(
        annotations_df=ss_annotations_pl,
        pred_results=sample_predictions,
        use_empirical=use_empirical,
        search_range=(-5, 5),
        threshold=0.5,
        verbose=(verbosity >= 1)
    )
    
    result['adjustment_dict'] = adjustment_dict
    result['success'] = True
    
    # Save adjustments for future use
    if save_adjustments:
        save_adjustment_dict(
            adjustment_dict=adjustment_dict,
            output_path=adjustment_file,
            verbose=(verbosity >= 1)
        )
    
    return result