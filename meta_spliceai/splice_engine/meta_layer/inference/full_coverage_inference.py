"""
Full coverage inference pipeline for meta-layer.

This module addresses the "gap problem" in meta-layer inference:
- Training artifacts contain SUBSAMPLED positions (~10% of gene)
- Full coverage inference needs ALL positions in a gene

Pipeline:
1. Run base model on ALL positions (full coverage mode)
2. Generate features for all positions  
3. Apply meta-model to recalibrate all positions
4. Output: [gene_length, 3] predictions for each gene

Usage:
    >>> from meta_spliceai.splice_engine.meta_layer.inference import FullCoveragePredictor
    >>> 
    >>> predictor = FullCoveragePredictor(
    ...     meta_model_path='models/best_model.pt',
    ...     base_model='openspliceai'
    ... )
    >>> 
    >>> # Predict for specific genes
    >>> results = predictor.predict_genes(['BRCA1', 'TP53'])
    >>> 
    >>> # Results contain full coverage predictions
    >>> brca1_scores = results['BRCA1']
    >>> print(brca1_scores.shape)  # [gene_length, 3]
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch

from ..core.config import MetaLayerConfig
from ..models import MetaSpliceModel
from .predictor import MetaLayerPredictor

logger = logging.getLogger(__name__)


@dataclass
class FullCoverageResult:
    """Result of full coverage prediction for a gene."""
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    gene_start: int
    gene_end: int
    gene_length: int
    
    # Full coverage scores: [gene_length, 3]
    donor_scores: np.ndarray
    acceptor_scores: np.ndarray
    neither_scores: np.ndarray
    
    # Delta from base model (optional)
    donor_delta: Optional[np.ndarray] = None
    acceptor_delta: Optional[np.ndarray] = None
    
    # Base model scores for reference
    base_donor_scores: Optional[np.ndarray] = None
    base_acceptor_scores: Optional[np.ndarray] = None
    
    def validate_length(self) -> bool:
        """Validate that output length matches gene length."""
        expected = self.gene_length
        actual_donor = len(self.donor_scores)
        actual_acceptor = len(self.acceptor_scores)
        actual_neither = len(self.neither_scores)
        
        if actual_donor != expected:
            logger.error(f"Donor scores length mismatch: {actual_donor} != {expected}")
            return False
        if actual_acceptor != expected:
            logger.error(f"Acceptor scores length mismatch: {actual_acceptor} != {expected}")
            return False
        if actual_neither != expected:
            logger.error(f"Neither scores length mismatch: {actual_neither} != {expected}")
            return False
        
        return True
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert to DataFrame with per-position predictions."""
        positions = np.arange(self.gene_start, self.gene_end + 1)
        
        data = {
            'gene_id': [self.gene_id] * self.gene_length,
            'gene_name': [self.gene_name] * self.gene_length,
            'chrom': [self.chrom] * self.gene_length,
            'strand': [self.strand] * self.gene_length,
            'position': positions,
            'relative_position': np.arange(self.gene_length),
            'meta_donor_score': self.donor_scores,
            'meta_acceptor_score': self.acceptor_scores,
            'meta_neither_score': self.neither_scores,
        }
        
        if self.base_donor_scores is not None:
            data['base_donor_score'] = self.base_donor_scores
        if self.base_acceptor_scores is not None:
            data['base_acceptor_score'] = self.base_acceptor_scores
        if self.donor_delta is not None:
            data['donor_delta'] = self.donor_delta
        if self.acceptor_delta is not None:
            data['acceptor_delta'] = self.acceptor_delta
        
        return pl.DataFrame(data)


class FullCoveragePredictor:
    """
    Full coverage inference for meta-layer model.
    
    Addresses the gap between:
    - Training artifacts (subsampled positions)
    - Inference requirement (ALL positions in a gene)
    
    Pipeline:
    1. Run base model with save_nucleotide_scores=True
    2. Generate features for all positions
    3. Apply meta-model to recalibrate all positions
    4. Output: [gene_length, 3] for each gene
    
    Parameters
    ----------
    meta_model_path : str or Path
        Path to trained meta-model checkpoint.
    base_model : str
        Base model to use ('spliceai', 'openspliceai').
    config : MetaLayerConfig, optional
        Configuration object.
    device : str, optional
        Device for inference ('auto', 'cpu', 'cuda', 'mps').
    """
    
    def __init__(
        self,
        meta_model_path: Union[str, Path],
        base_model: str = 'openspliceai',
        config: Optional[MetaLayerConfig] = None,
        device: Optional[str] = None
    ):
        self.meta_model_path = Path(meta_model_path)
        self.base_model = base_model
        self.config = config or MetaLayerConfig(base_model=base_model)
        
        # Load meta-model
        self.predictor = MetaLayerPredictor.from_checkpoint(
            self.meta_model_path,
            config=self.config
        )
        
        if device:
            self.predictor.device = torch.device(device)
        
        logger.info(f"FullCoveragePredictor initialized")
        logger.info(f"  Base model: {self.base_model}")
        logger.info(f"  Meta model: {self.meta_model_path}")
    
    def predict_genes(
        self,
        target_genes: List[str],
        batch_size: int = 256,
        verbosity: int = 1,
        save_intermediate: bool = False,
        output_dir: Optional[Path] = None
    ) -> Dict[str, FullCoverageResult]:
        """
        Predict full coverage scores for specified genes.
        
        Parameters
        ----------
        target_genes : list of str
            Gene names or IDs to predict.
        batch_size : int
            Batch size for meta-model inference.
        verbosity : int
            Verbosity level (0=silent, 1=normal, 2=detailed).
        save_intermediate : bool
            Whether to save intermediate base model results.
        output_dir : Path, optional
            Directory for intermediate outputs.
        
        Returns
        -------
        dict
            Dictionary mapping gene_name -> FullCoverageResult.
        """
        logger.info(f"Running full coverage prediction for {len(target_genes)} genes")
        
        # Step 1: Run base model with full coverage
        if verbosity >= 1:
            print(f"[1/4] Running base model ({self.base_model}) with full coverage...")
        
        base_results = self._run_base_model_full_coverage(
            target_genes, 
            verbosity=verbosity,
            save_intermediate=save_intermediate,
            output_dir=output_dir
        )
        
        if not base_results['success']:
            raise RuntimeError(f"Base model failed: {base_results.get('error')}")
        
        nucleotide_scores = base_results['nucleotide_scores']
        gene_manifest = base_results['gene_manifest']
        
        if verbosity >= 1:
            print(f"    Base model produced {nucleotide_scores.height:,} nucleotide scores")
        
        # Step 2: Generate features for all positions
        if verbosity >= 1:
            print(f"[2/4] Generating features for all positions...")
        
        features_df = self._generate_features(nucleotide_scores, verbosity)
        
        if verbosity >= 1:
            print(f"    Generated {features_df.height:,} feature vectors")
        
        # Step 3: Apply meta-model
        if verbosity >= 1:
            print(f"[3/4] Applying meta-model...")
        
        meta_predictions = self._apply_meta_model(features_df, batch_size, verbosity)
        
        # Step 4: Assemble full coverage results
        if verbosity >= 1:
            print(f"[4/4] Assembling full coverage results...")
        
        results = self._assemble_results(
            meta_predictions, 
            nucleotide_scores,
            gene_manifest,
            verbosity
        )
        
        # Validate output lengths
        for gene_name, result in results.items():
            if not result.validate_length():
                logger.warning(f"Length validation failed for {gene_name}")
        
        if verbosity >= 1:
            print(f"\n✅ Full coverage prediction complete!")
            print(f"   Genes processed: {len(results)}")
            for gene_name, result in results.items():
                print(f"   - {gene_name}: {result.gene_length:,} positions")
        
        return results
    
    def _run_base_model_full_coverage(
        self,
        target_genes: List[str],
        verbosity: int = 1,
        save_intermediate: bool = False,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """Run base model with full coverage mode."""
        try:
            from meta_spliceai import run_base_model_predictions
        except ImportError:
            raise ImportError(
                "Could not import run_base_model_predictions. "
                "Make sure meta_spliceai is properly installed."
            )
        
        # Determine test name
        test_name = f"meta_layer_full_coverage_{self.base_model}"
        
        # Run base model with full coverage
        results = run_base_model_predictions(
            base_model=self.base_model,
            target_genes=target_genes,
            save_nucleotide_scores=True,  # KEY: Enable full coverage
            mode='test',
            coverage='gene_subset',
            test_name=test_name,
            verbosity=verbosity - 1 if verbosity > 0 else 0,
            no_tn_sampling=True  # We want all positions
        )
        
        return results
    
    def _generate_features(
        self,
        nucleotide_scores: pl.DataFrame,
        verbosity: int = 1
    ) -> pl.DataFrame:
        """
        Generate features for all nucleotide positions.
        
        This applies the same feature engineering as the training artifacts,
        but to ALL positions (not just subsampled ones).
        """
        # The nucleotide_scores DataFrame should have:
        # - position, chrom, strand
        # - donor_score, acceptor_score, neither_score
        # - sequence (501nt context)
        
        # Apply feature derivation
        # This mirrors what's done in the base layer artifact generation
        
        df = nucleotide_scores
        
        # Check required columns
        required = ['donor_score', 'acceptor_score', 'neither_score']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Add context features (if neighboring positions available)
        df = self._add_context_features(df)
        
        return df
    
    def _add_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived probability features."""
        # Calculate derived features matching the schema
        df = df.with_columns([
            # Relative donor probability
            (pl.col('donor_score') / 
             (pl.col('donor_score') + pl.col('acceptor_score') + 1e-10)
            ).alias('relative_donor_probability'),
            
            # Splice probability (donor + acceptor)
            (pl.col('donor_score') + pl.col('acceptor_score')
            ).alias('splice_probability'),
            
            # Donor-acceptor difference
            (pl.col('donor_score') - pl.col('acceptor_score')
            ).alias('donor_acceptor_diff'),
            
            # Splice-neither difference
            (pl.col('donor_score') + pl.col('acceptor_score') - pl.col('neither_score')
            ).alias('splice_neither_diff'),
            
            # Probability entropy
            (-pl.col('donor_score') * (pl.col('donor_score') + 1e-10).log() -
             pl.col('acceptor_score') * (pl.col('acceptor_score') + 1e-10).log() -
             pl.col('neither_score') * (pl.col('neither_score') + 1e-10).log()
            ).alias('probability_entropy'),
        ])
        
        return df
    
    def _add_context_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add context features from neighboring positions."""
        # For full coverage, we can compute context from neighbors
        # This is a simplified version - the full implementation would
        # use sliding windows
        
        # For now, add placeholder context features
        # In production, this should use proper window-based computation
        
        df = df.with_columns([
            pl.lit(0.0).alias('context_score_m2'),
            pl.lit(0.0).alias('context_score_m1'),
            pl.lit(0.0).alias('context_score_p1'),
            pl.lit(0.0).alias('context_score_p2'),
        ])
        
        # TODO: Implement proper context computation using gene-wise grouping
        # df = df.sort(['gene_id', 'position'])
        # df = df.with_columns([
        #     pl.col('donor_score').shift(-2).over('gene_id').alias('context_score_m2'),
        #     ...
        # ])
        
        return df
    
    def _apply_meta_model(
        self,
        features_df: pl.DataFrame,
        batch_size: int = 256,
        verbosity: int = 1
    ) -> pl.DataFrame:
        """Apply meta-model to all positions."""
        from ..data import MetaLayerDataset
        
        # Create dataset (without leakage check for inference)
        dataset = MetaLayerDataset(
            features_df,
            max_seq_length=self.config.max_seq_length,
            check_leakage=False  # Skip leakage check for inference
        )
        
        # Get predictions
        result = self.predictor.predict_dataframe(
            features_df,
            batch_size=batch_size
        )
        
        return result
    
    def _assemble_results(
        self,
        meta_predictions: pl.DataFrame,
        nucleotide_scores: pl.DataFrame,
        gene_manifest: pl.DataFrame,
        verbosity: int = 1
    ) -> Dict[str, FullCoverageResult]:
        """Assemble results per gene."""
        results = {}
        
        # Get unique genes
        if 'gene_id' in meta_predictions.columns:
            gene_col = 'gene_id'
        elif 'gene_name' in meta_predictions.columns:
            gene_col = 'gene_name'
        else:
            raise ValueError("No gene identifier column found")
        
        unique_genes = meta_predictions[gene_col].unique().to_list()
        
        for gene in unique_genes:
            gene_df = meta_predictions.filter(pl.col(gene_col) == gene).sort('position')
            
            # Get gene info from manifest
            gene_info = gene_manifest.filter(pl.col(gene_col) == gene)
            if gene_info.height == 0:
                logger.warning(f"Gene {gene} not found in manifest")
                continue
            
            gene_info = gene_info.row(0, named=True)
            
            # Extract scores
            result = FullCoverageResult(
                gene_id=gene_info.get('gene_id', gene),
                gene_name=gene_info.get('gene_name', gene),
                chrom=gene_info.get('chrom', ''),
                strand=gene_info.get('strand', '+'),
                gene_start=gene_info.get('gene_start', 0),
                gene_end=gene_info.get('gene_end', 0),
                gene_length=gene_df.height,
                donor_scores=gene_df['meta_donor_score'].to_numpy(),
                acceptor_scores=gene_df['meta_acceptor_score'].to_numpy(),
                neither_scores=gene_df['meta_neither_score'].to_numpy(),
            )
            
            # Add base model scores if available
            if 'donor_score' in gene_df.columns:
                result.base_donor_scores = gene_df['donor_score'].to_numpy()
            if 'acceptor_score' in gene_df.columns:
                result.base_acceptor_scores = gene_df['acceptor_score'].to_numpy()
            
            # Compute deltas
            if result.base_donor_scores is not None:
                result.donor_delta = result.donor_scores - result.base_donor_scores
            if result.base_acceptor_scores is not None:
                result.acceptor_delta = result.acceptor_scores - result.base_acceptor_scores
            
            results[gene] = result
        
        return results


def predict_full_coverage(
    meta_model_path: Union[str, Path],
    target_genes: List[str],
    base_model: str = 'openspliceai',
    output_path: Optional[Union[str, Path]] = None,
    verbosity: int = 1
) -> Dict[str, FullCoverageResult]:
    """
    Convenience function for full coverage prediction.
    
    Parameters
    ----------
    meta_model_path : str or Path
        Path to trained meta-model.
    target_genes : list of str
        Genes to predict.
    base_model : str
        Base model to use.
    output_path : str or Path, optional
        Path to save results (as TSV).
    verbosity : int
        Verbosity level.
    
    Returns
    -------
    dict
        Gene name -> FullCoverageResult mapping.
    
    Examples
    --------
    >>> results = predict_full_coverage(
    ...     'models/best_model.pt',
    ...     ['BRCA1', 'TP53'],
    ...     base_model='openspliceai'
    ... )
    >>> 
    >>> # Check output length matches gene length
    >>> brca1 = results['BRCA1']
    >>> assert len(brca1.donor_scores) == brca1.gene_length
    """
    predictor = FullCoveragePredictor(
        meta_model_path=meta_model_path,
        base_model=base_model
    )
    
    results = predictor.predict_genes(
        target_genes=target_genes,
        verbosity=verbosity
    )
    
    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Concatenate all gene DataFrames
        all_dfs = [result.to_dataframe() for result in results.values()]
        combined = pl.concat(all_dfs)
        combined.write_csv(output_path, separator='\t')
        
        if verbosity >= 1:
            print(f"Saved full coverage results to {output_path}")
    
    return results

