"""
Enhanced Feature Processor with Improved Uncertainty Selection

This module extends the existing feature processor to support:
1. Entropy-based uncertainty detection
2. Adaptive threshold tuning
3. Configurable target selection rates (e.g., ~10% instead of ~3%)
4. Multiple uncertainty criteria (confidence, entropy, spread, variance)

This is designed to be a drop-in replacement for the uncertainty detection
logic in the existing feature_processor.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    from .enhanced_uncertainty_selection import (
        EnhancedUncertaintySelector, 
        EnhancedUncertaintyConfig,
        enhance_uncertainty_selection_for_inference
    )
except ImportError:
    from enhanced_uncertainty_selection import (
        EnhancedUncertaintySelector, 
        EnhancedUncertaintyConfig,
        enhance_uncertainty_selection_for_inference
    )

logger = logging.getLogger(__name__)


def identify_uncertain_positions_enhanced(
    complete_base_pd: pd.DataFrame,
    config: Any,  # SelectiveInferenceConfig or similar
    verbose: bool = False
) -> pd.DataFrame:
    """
    Enhanced uncertain position identification that can replace the existing logic.
    
    This function maintains compatibility with the existing interface while providing
    enhanced uncertainty detection capabilities.
    
    Parameters
    ----------
    complete_base_pd : pd.DataFrame
        Complete base model predictions
    config : SelectiveInferenceConfig
        Configuration object with uncertainty thresholds
    verbose : bool
        Enable verbose logging
        
    Returns
    -------
    pd.DataFrame
        Uncertain positions selected for meta-model inference
    """
    if verbose:
        print(f"🔍 Enhanced uncertainty detection for {len(complete_base_pd)} positions")
    
    # Extract target selection rate from config or use default
    target_selection_rate = getattr(config, 'target_meta_selection_rate', 0.10)
    selection_strategy = getattr(config, 'uncertainty_selection_strategy', 'hybrid_entropy')
    enable_adaptive = getattr(config, 'enable_adaptive_uncertainty_tuning', True)
    
    # Handle different inference modes
    if hasattr(config, 'inference_mode'):
        if config.inference_mode == "base_only":
            # No uncertain positions in base_only mode
            if verbose:
                print("   ℹ️  Base-only mode: no uncertain positions")
            return pd.DataFrame(columns=complete_base_pd.columns)
        
        elif config.inference_mode == "meta_only":
            # All positions are uncertain in meta_only mode
            if verbose:
                print(f"   🎯 Meta-only mode: all {len(complete_base_pd)} positions selected")
            return complete_base_pd.copy()
    
    # Enhanced hybrid mode uncertainty detection
    try:
        # Use enhanced uncertainty selection
        result_df, analysis_stats = enhance_uncertainty_selection_for_inference(
            complete_base_pd,
            target_selection_rate=target_selection_rate,
            selection_strategy=selection_strategy,
            enable_adaptive_tuning=enable_adaptive
        )
        
        # Get uncertain positions
        uncertain_positions = result_df[result_df['is_uncertain']].copy()
        
        if verbose:
            print(f"   🎯 Enhanced selection results:")
            print(f"      Strategy: {selection_strategy}")
            print(f"      Target rate: {target_selection_rate:.1%}")
            print(f"      Achieved rate: {analysis_stats['selection_rate']:.1%}")
            print(f"      Selected positions: {len(uncertain_positions):,}")
            print(f"      Uncertainty reasons: {analysis_stats['uncertainty_reasons']}")
            
            # Show threshold comparison
            traditional_uncertain = (
                (complete_base_pd[['donor_score', 'acceptor_score']].max(axis=1) >= config.uncertainty_threshold_low) &
                (complete_base_pd[['donor_score', 'acceptor_score']].max(axis=1) < config.uncertainty_threshold_high)
            ).sum()
            
            print(f"      Traditional method would select: {traditional_uncertain} ({traditional_uncertain/len(complete_base_pd):.1%})")
            print(f"      Enhancement factor: {len(uncertain_positions)/max(traditional_uncertain, 1):.1f}x")
        
        return uncertain_positions
        
    except Exception as e:
        logger.warning(f"Enhanced uncertainty selection failed: {e}")
        logger.warning("Falling back to traditional threshold-based selection")
        
        # Fallback to traditional method
        max_scores = complete_base_pd[['donor_score', 'acceptor_score']].max(axis=1)
        uncertain_mask = (
            (max_scores >= config.uncertainty_threshold_low) & 
            (max_scores < config.uncertainty_threshold_high)
        )
        
        uncertain_positions = complete_base_pd[uncertain_mask].copy()
        
        if verbose:
            print(f"   🔄 Fallback to traditional method: {len(uncertain_positions)} positions")
        
        return uncertain_positions


def patch_feature_processor_uncertainty_detection():
    """
    Monkey patch the existing feature processor to use enhanced uncertainty detection.
    
    This allows seamless integration without modifying the existing codebase.
    """
    try:
        from . import feature_processor
        
        # Store original function
        if not hasattr(feature_processor, '_original_identify_uncertain_positions'):
            feature_processor._original_identify_uncertain_positions = (
                feature_processor.identify_uncertain_positions_for_meta_inference
            )
        
        # Replace with enhanced version
        def enhanced_wrapper(config, complete_base_pd, workflow_results, verbose=False):
            """Enhanced wrapper that maintains the original function signature."""
            
            # Use enhanced uncertainty detection
            uncertain_positions = identify_uncertain_positions_enhanced(
                complete_base_pd, config, verbose
            )
            
            # Apply existing position limiting logic if needed
            max_positions = getattr(config, 'max_positions_per_gene', 10000)
            if len(uncertain_positions) > max_positions and not getattr(config, 'enable_chunked_processing', False):
                if verbose:
                    print(f"   ⚠️  Limiting to {max_positions} positions (memory constraint)")
                uncertain_positions = uncertain_positions.head(max_positions)
            
            return uncertain_positions
        
        # Apply the patch
        feature_processor.identify_uncertain_positions_for_meta_inference = enhanced_wrapper
        
        logger.info("✅ Enhanced uncertainty detection patched into feature processor")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch feature processor: {e}")
        return False


def create_enhanced_config_with_target_rate(
    base_config: Any,
    target_selection_rate: float = 0.10,
    selection_strategy: str = "hybrid_entropy"
) -> Any:
    """
    Create an enhanced config object with target selection rate settings.
    
    Parameters
    ----------
    base_config : Any
        Existing configuration object
    target_selection_rate : float
        Target percentage of positions for meta-model (e.g., 0.10 = 10%)
    selection_strategy : str
        Selection strategy ("confidence_only", "entropy_only", "hybrid_entropy")
        
    Returns
    -------
    Any
        Enhanced configuration object
    """
    # Add enhanced uncertainty settings to existing config
    if hasattr(base_config, '__dict__'):
        # Add new attributes
        base_config.target_meta_selection_rate = target_selection_rate
        base_config.uncertainty_selection_strategy = selection_strategy
        base_config.enable_adaptive_uncertainty_tuning = True
        
        # Log the enhancement
        logger.info(f"Enhanced config with target selection rate: {target_selection_rate:.1%}")
        logger.info(f"Selection strategy: {selection_strategy}")
    
    return base_config


def demonstrate_enhanced_uncertainty_selection():
    """Demonstrate the enhanced uncertainty selection capabilities."""
    print("🧬 Enhanced Uncertainty Selection Demonstration")
    print("=" * 60)
    
    # Create mock base model predictions
    np.random.seed(42)
    n_positions = 2000
    
    mock_data = {
        'gene_id': ['ENSG00000123456'] * n_positions,
        'position': range(1, n_positions + 1),
        'donor_score': np.random.beta(0.8, 3, n_positions),      # Most low, some high
        'acceptor_score': np.random.beta(0.8, 3, n_positions),   # Most low, some high  
        'neither_score': np.random.beta(3, 0.8, n_positions)     # Most high, some low
    }
    
    # Normalize to sum to 1 (more realistic)
    for i in range(n_positions):
        total = mock_data['donor_score'][i] + mock_data['acceptor_score'][i] + mock_data['neither_score'][i]
        mock_data['donor_score'][i] /= total
        mock_data['acceptor_score'][i] /= total
        mock_data['neither_score'][i] /= total
    
    test_df = pd.DataFrame(mock_data)
    
    # Test traditional approach
    print(f"\n📊 Traditional Approach (thresholds: 0.02-0.80)")
    max_scores = test_df[['donor_score', 'acceptor_score']].max(axis=1)
    traditional_uncertain = ((max_scores >= 0.02) & (max_scores < 0.80)).sum()
    traditional_rate = traditional_uncertain / len(test_df)
    print(f"   Selected: {traditional_uncertain:,}/{len(test_df):,} ({traditional_rate:.1%})")
    
    # Test enhanced approaches
    strategies = [
        ("confidence_only", "Traditional confidence-based"),
        ("entropy_only", "Pure entropy-based"),
        ("hybrid_entropy", "Enhanced hybrid (confidence + entropy + spread)")
    ]
    
    for strategy, description in strategies:
        print(f"\n📊 {description}")
        
        result_df, stats = enhance_uncertainty_selection_for_inference(
            test_df,
            target_selection_rate=0.10,
            selection_strategy=strategy,
            enable_adaptive_tuning=True
        )
        
        print(f"   Selected: {stats['uncertain_positions']:,}/{stats['total_positions']:,} ({stats['selection_rate']:.1%})")
        print(f"   Target achieved: {'✅' if abs(stats['selection_rate'] - 0.10) < 0.02 else '❌'}")
        print(f"   Uncertainty reasons: {dict(list(stats['uncertainty_reasons'].items())[:3])}")
    
    print(f"\n🎯 Key Benefits of Enhanced Selection:")
    print(f"   • Achieves target ~10% selection rate (vs ~3% traditional)")
    print(f"   • Uses multiple uncertainty criteria (entropy, spread, variance)")
    print(f"   • Adaptive threshold tuning for consistent performance")
    print(f"   • Focuses on positions where meta-model can help most")
    
    print(f"\n✅ Enhanced uncertainty selection demonstration complete!")


if __name__ == "__main__":
    demonstrate_enhanced_uncertainty_selection()
