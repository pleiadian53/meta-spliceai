"""Base Model Runner for Cross-Build Comparisons

This module provides utilities for running base models (SpliceAI, OpenSpliceAI)
and comparing their performance.

Key Features:
- Run multiple base models with consistent configuration
- Compare performance metrics (precision, recall, F1, accuracy)
- Handle missing genes gracefully
- Generate comprehensive comparison reports

Examples:
    >>> from meta_spliceai.system import BaseModelRunner
    >>> from meta_spliceai.system.genomic_resources import GeneSamplingResult
    >>> 
    >>> # Initialize runner
    >>> runner = BaseModelRunner()
    >>> 
    >>> # Run comparison
    >>> results = runner.compare_models(
    ...     models=['spliceai', 'openspliceai'],
    ...     gene_sampling_result=sampling_result,
    ...     save_nucleotide_scores=True
    ... )
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import polars as pl

# Import moved inside functions to avoid circular import
# from meta_spliceai.run_base_model import run_base_model_predictions


@dataclass
class BaseModelConfig:
    """Configuration for base model execution.
    
    Attributes
    ----------
    mode : str
        Execution mode ('test' or 'production')
    coverage : str
        Coverage mode ('gene_subset', 'chromosome', 'full_genome')
    threshold : float
        Splice site score threshold
    consensus_window : int
        Window for consensus calling
    error_window : int
        Window for error analysis
    use_auto_position_adjustments : bool
        Auto-detect position offsets
    save_nucleotide_scores : bool
        Save nucleotide-level scores (full coverage)
    no_tn_sampling : bool
        Disable true negative sampling
    verbosity : int
        Output verbosity (0-2)
    """
    mode: str = 'test'
    coverage: str = 'gene_subset'
    threshold: float = 0.5
    consensus_window: int = 2
    error_window: int = 500
    use_auto_position_adjustments: bool = True
    save_nucleotide_scores: bool = False
    no_tn_sampling: bool = True
    verbosity: int = 1


@dataclass
class BaseModelResult:
    """Result from running a base model.
    
    Attributes
    ----------
    model_name : str
        Base model name
    success : bool
        Whether execution succeeded
    runtime_seconds : float
        Execution time in seconds
    positions : pl.DataFrame
        Analyzed positions with predictions
    nucleotide_scores : Optional[pl.DataFrame]
        Nucleotide-level scores (if enabled)
    gene_manifest : Optional[pl.DataFrame]
        Gene processing manifest
    processed_genes : Set[str]
        Genes successfully processed
    missing_genes : Set[str]
        Genes not processed
    metrics : Dict[str, float]
        Performance metrics
    paths : Dict[str, str]
        Artifact paths
    error : Optional[str]
        Error message (if failed)
    """
    model_name: str
    success: bool
    runtime_seconds: float
    positions: pl.DataFrame
    nucleotide_scores: Optional[pl.DataFrame] = None
    gene_manifest: Optional[pl.DataFrame] = None
    processed_genes: set = field(default_factory=set)
    missing_genes: set = field(default_factory=set)
    metrics: Dict[str, float] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result from comparing multiple base models.
    
    Attributes
    ----------
    test_name : str
        Test identifier
    models : Dict[str, BaseModelResult]
        Results for each model
    comparison_metrics : Dict[str, Any]
        Comparison metrics
    output_dir : Path
        Output directory
    """
    test_name: str
    models: Dict[str, BaseModelResult]
    comparison_metrics: Dict[str, Any]
    output_dir: Path


class BaseModelRunner:
    """Runner for base model comparisons.
    
    This class provides utilities for:
    - Running multiple base models with consistent configuration
    - Comparing performance metrics
    - Handling missing genes gracefully
    - Generating comprehensive reports
    
    Examples
    --------
    >>> runner = BaseModelRunner()
    >>> 
    >>> # Run comparison
    >>> results = runner.compare_models(
    ...     models=['spliceai', 'openspliceai'],
    ...     gene_sampling_result=sampling_result,
    ...     config=BaseModelConfig(save_nucleotide_scores=True)
    ... )
    >>> 
    >>> # Print summary
    >>> runner.print_comparison_summary(results)
    """
    
    def __init__(self):
        """Initialize base model runner."""
        pass
    
    def run_single_model(
        self,
        model_name: str,
        target_genes: List[Optional[str]],
        test_name: str,
        config: BaseModelConfig,
        verbosity: int = 1
    ) -> BaseModelResult:
        """Run a single base model.
        
        Parameters
        ----------
        model_name : str
            Base model name ('spliceai', 'openspliceai', etc.)
        target_genes : List[Optional[str]]
            Target gene IDs (source-specific)
        test_name : str
            Test identifier
        config : BaseModelConfig
            Model configuration
        verbosity : int, default=1
            Output verbosity
        
        Returns
        -------
        BaseModelResult
            Model execution results
        """
        if verbosity >= 1:
            print("=" * 80)
            print(f"Running {model_name.upper()}")
            print("=" * 80)
            print()
            print(f"🔵 Using: run_base_model_predictions(base_model='{model_name}', ...)")
            print()
        
        start_time = time.time()
        
        try:
            # Import here to avoid circular import
            from meta_spliceai.run_base_model import run_base_model_predictions
            
            # Run predictions
            results = run_base_model_predictions(
                base_model=model_name,
                target_genes=target_genes,
                mode=config.mode,
                coverage=config.coverage,
                test_name=f"{test_name}_{model_name}",
                threshold=config.threshold,
                consensus_window=config.consensus_window,
                error_window=config.error_window,
                use_auto_position_adjustments=config.use_auto_position_adjustments,
                verbosity=config.verbosity,
                no_tn_sampling=config.no_tn_sampling,
                save_nucleotide_scores=config.save_nucleotide_scores
            )
            
            runtime = time.time() - start_time
            
            # Extract results
            positions = results['positions']
            nucleotide_scores = results.get('nucleotide_scores')
            gene_manifest = results.get('gene_manifest')
            
            # Determine processed genes
            processed_genes = set()
            if positions.height > 0:
                if 'gene_id' in positions.columns:
                    processed_genes = set(positions['gene_id'].unique().to_list())
                elif 'gene_name' in positions.columns:
                    processed_genes = set(positions['gene_name'].unique().to_list())
            
            # Determine missing genes
            all_genes = set(g for g in target_genes if g is not None)
            missing_genes = all_genes - processed_genes
            
            # Calculate metrics
            metrics = self._calculate_metrics(positions)
            
            if verbosity >= 1:
                print()
                print(f"✅ {model_name.upper()} completed in {runtime:.1f} seconds")
                print(f"   Positions analyzed: {positions.height:,}")
                if config.save_nucleotide_scores and nucleotide_scores is not None:
                    print(f"   Nucleotide scores: {nucleotide_scores.height:,}")
                print(f"   Genes processed: {len(processed_genes)}/{len(all_genes)}")
                if missing_genes:
                    print(f"   ⚠️  Missing genes: {len(missing_genes)}")
                    if verbosity >= 2:
                        print(f"      {', '.join(list(missing_genes)[:5])}...")
            
            return BaseModelResult(
                model_name=model_name,
                success=True,
                runtime_seconds=runtime,
                positions=positions,
                nucleotide_scores=nucleotide_scores,
                gene_manifest=gene_manifest,
                processed_genes=processed_genes,
                missing_genes=missing_genes,
                metrics=metrics,
                paths=results.get('paths', {})
            )
        
        except Exception as e:
            runtime = time.time() - start_time
            
            if verbosity >= 1:
                print(f"❌ {model_name.upper()} error: {e}")
            
            return BaseModelResult(
                model_name=model_name,
                success=False,
                runtime_seconds=runtime,
                positions=pl.DataFrame(),
                processed_genes=set(),
                missing_genes=set(g for g in target_genes if g is not None),
                error=str(e)
            )
    
    def _calculate_metrics(self, positions_df: pl.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Parameters
        ----------
        positions_df : pl.DataFrame
            Positions dataframe with pred_type column
        
        Returns
        -------
        Dict[str, float]
            Performance metrics
        """
        if positions_df.height == 0:
            return {
                'positions': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0
            }
        
        tp = positions_df.filter(pl.col('pred_type') == 'TP').height
        tn = positions_df.filter(pl.col('pred_type') == 'TN').height
        fp = positions_df.filter(pl.col('pred_type') == 'FP').height
        fn = positions_df.filter(pl.col('pred_type') == 'FN').height
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            'positions': positions_df.height,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    def compare_models(
        self,
        models: List[str],
        gene_symbols: List[str],
        gene_ids_by_model: Dict[str, List[Optional[str]]],
        test_name: str,
        config: Optional[BaseModelConfig] = None,
        verbosity: int = 1
    ) -> ComparisonResult:
        """Compare multiple base models.
        
        Parameters
        ----------
        models : List[str]
            List of model names to compare
        gene_symbols : List[str]
            Gene symbols being tested
        gene_ids_by_model : Dict[str, List[Optional[str]]]
            Gene IDs for each model (model_name → gene_ids)
        test_name : str
            Test identifier
        config : Optional[BaseModelConfig]
            Model configuration (uses defaults if None)
        verbosity : int, default=1
            Output verbosity
        
        Returns
        -------
        ComparisonResult
            Comparison results
        """
        if config is None:
            config = BaseModelConfig()
        
        # Run each model
        model_results = {}
        for model_name in models:
            target_genes = gene_ids_by_model.get(model_name, [])
            
            result = self.run_single_model(
                model_name=model_name,
                target_genes=target_genes,
                test_name=test_name,
                config=config,
                verbosity=verbosity
            )
            
            model_results[model_name] = result
            
            if verbosity >= 1:
                print()
        
        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(
            model_results,
            gene_symbols
        )
        
        # Create output directory
        output_dir = Path(f"results/{test_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return ComparisonResult(
            test_name=test_name,
            models=model_results,
            comparison_metrics=comparison_metrics,
            output_dir=output_dir
        )
    
    def _calculate_comparison_metrics(
        self,
        model_results: Dict[str, BaseModelResult],
        gene_symbols: List[str]
    ) -> Dict[str, Any]:
        """Calculate comparison metrics across models.
        
        Parameters
        ----------
        model_results : Dict[str, BaseModelResult]
            Results for each model
        gene_symbols : List[str]
            Gene symbols being tested
        
        Returns
        -------
        Dict[str, Any]
            Comparison metrics
        """
        return {
            'total_genes': len(gene_symbols),
            'models': {
                name: {
                    'success': result.success,
                    'genes_processed': len(result.processed_genes),
                    'genes_missing': len(result.missing_genes),
                    'runtime_seconds': result.runtime_seconds,
                    **result.metrics
                }
                for name, result in model_results.items()
            }
        }
    
    def print_comparison_summary(
        self,
        result: ComparisonResult,
        verbosity: int = 1
    ) -> None:
        """Print comparison summary.
        
        Parameters
        ----------
        result : ComparisonResult
            Comparison results
        verbosity : int, default=1
            Output verbosity
        """
        print("=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        print()
        
        # Print table header
        model_names = list(result.models.keys())
        col_width = 25
        
        print(f"{'Metric':<20}", end='')
        for name in model_names:
            print(f"{name.upper():<{col_width}}", end='')
        print()
        print("-" * (20 + col_width * len(model_names)))
        
        # Print metrics
        metrics_to_show = [
            ('Genes Processed', 'genes_processed', ''),
            ('Positions', 'positions', ','),
            ('TP', 'tp', ','),
            ('FP', 'fp', ','),
            ('FN', 'fn', ','),
            ('Precision', 'precision', '.4f'),
            ('Recall', 'recall', '.4f'),
            ('F1 Score', 'f1', '.4f'),
            ('Accuracy', 'accuracy', '.4f'),
            ('Runtime (sec)', 'runtime_seconds', '.1f')
        ]
        
        for label, key, fmt in metrics_to_show:
            print(f"{label:<20}", end='')
            for name in model_names:
                model_result = result.models[name]
                if model_result.success:
                    value = result.comparison_metrics['models'][name].get(key, 0)
                    if fmt:
                        if ',' in fmt:
                            print(f"{value:<{col_width},}", end='')
                        else:
                            print(f"{value:<{col_width}{fmt}}", end='')
                    else:
                        print(f"{value:<{col_width}}", end='')
                else:
                    print(f"{'FAILED':<{col_width}}", end='')
            print()
        
        print("-" * (20 + col_width * len(model_names)))
        print()

