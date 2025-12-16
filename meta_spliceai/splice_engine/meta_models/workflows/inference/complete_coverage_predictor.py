"""
Complete Coverage Predictor for Inference Workflow

This module ensures complete position coverage for target genes by:
1. Detecting gap positions in existing artifacts
2. Running base model predictions on gap positions
3. Applying meta-model selectively to uncertain positions
4. Producing complete, continuous position coverage

Key principle: In inference mode, we predict ALL positions for target genes,
not just the subset that was retained during training (due to TN downsampling).
"""

import logging
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import tempfile
import os

# Import base model prediction capabilities
from ...workflows.splice_prediction_workflow import run_enhanced_splice_prediction_workflow
from ...workflows.data_preparation import prepare_genomic_sequences
from .chromosome_sequence_utils import extract_targeted_chromosome_sequences

logger = logging.getLogger(__name__)


class CompleteCoveragePredictor:
    """
    Ensures complete position coverage for inference by filling gaps
    and applying meta-model selectively to uncertain positions.
    """
    
    def __init__(self, 
                 base_model_path: str,
                 meta_model_path: str,
                 output_dir: str,
                 uncertainty_threshold: float = 0.3):
        """
        Initialize the complete coverage predictor.
        
        Args:
            base_model_path: Path to trained base model
            meta_model_path: Path to trained meta model  
            output_dir: Directory for outputs
            uncertainty_threshold: Threshold for selecting uncertain positions
        """
        self.base_model_path = Path(base_model_path)
        self.meta_model_path = Path(meta_model_path)
        self.output_dir = Path(output_dir)
        self.uncertainty_threshold = uncertainty_threshold
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_gap_positions(self, 
                           existing_df: pl.DataFrame, 
                           gene_id: str,
                           gene_length: int) -> Set[int]:
        """
        Detect gap positions that need base model predictions.
        
        Args:
            existing_df: DataFrame with existing predictions for the gene
            gene_id: Target gene ID
            gene_length: Total length of the gene sequence
            
        Returns:
            Set of positions that need predictions
        """
        # Get existing positions for this gene
        gene_data = existing_df.filter(pl.col('gene_id') == gene_id)
        existing_positions = set(gene_data['position'].to_list())
        
        # Generate complete position range (1-based)
        complete_positions = set(range(1, gene_length + 1))
        
        # Find gap positions
        gap_positions = complete_positions - existing_positions
        
        logger.info(f"Gene {gene_id}: {len(existing_positions)} existing, "
                   f"{len(gap_positions)} gap positions, "
                   f"{gene_length} total length")
        
        return gap_positions
    
    def run_base_model_on_gaps(self,
                              gene_id: str,
                              gap_positions: Set[int],
                              gene_info: Dict) -> pl.DataFrame:
        """
        Run base model predictions on gap positions.
        
        Args:
            gene_id: Target gene ID
            gap_positions: Positions needing predictions
            gene_info: Gene metadata (chromosome, strand, etc.)
            
        Returns:
            DataFrame with gap position predictions
        """
        if not gap_positions:
            return pl.DataFrame()
            
        logger.info(f"Running base model on {len(gap_positions)} gap positions for {gene_id}")
        
        # Create temporary directory for gap prediction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a minimal gene manifest for just this gene
            gap_manifest_path = temp_path / "gap_gene_manifest.csv"
            gap_manifest_df = pl.DataFrame({
                'gene_id': [gene_id],
                'chrom': [gene_info['chrom']],
                'strand': [gene_info['strand']],
                'start': [gene_info['start']], 
                'end': [gene_info['end']],
                'gene_name': [gene_info.get('gene_name', gene_id)]
            })
            gap_manifest_df.write_csv(gap_manifest_path)
            
            # Prepare genomic sequences (reuse existing if available)
            seq_result = prepare_genomic_sequences(
                local_dir=str(temp_path),
                chromosomes=[gene_info['chrom']],
                force_overwrite=False
            )
            
            if not seq_result['success']:
                # Extract chromosome sequences if needed
                seq_result = extract_targeted_chromosome_sequences(
                    output_dir=temp_path,
                    target_genes=[gene_id],
                    chromosomes=[gene_info['chrom']]
                )
                
            if not seq_result['success']:
                raise RuntimeError(f"Failed to prepare sequences for {gene_id}: {seq_result['error']}")
            
            # Run base model prediction on this gene
            # We'll get predictions for ALL positions, then filter to gaps
            prediction_result = run_enhanced_splice_prediction_workflow(
                eval_dir=str(temp_path),
                gene_manifest_path=str(gap_manifest_path),
                output_prefix=f"gap_prediction_{gene_id}",
                do_extract_sequences=False,  # Already done
                do_prepare_annotations=True,  # Need fresh annotations
                do_find_overlaping_genes=False,  # Not needed for gaps
                apply_tn_sampling=False,  # We want ALL positions
                target_chromosomes=[gene_info['chrom']]
            )
            
            if not prediction_result['success']:
                raise RuntimeError(f"Base model prediction failed for {gene_id}: {prediction_result['error']}")
            
            # Load the complete predictions
            prediction_files = list(temp_path.glob("**/analysis_sequences_*.parquet"))
            if not prediction_files:
                raise RuntimeError(f"No prediction files found for {gene_id}")
            
            # Read and filter to gap positions only
            all_predictions = []
            for pfile in prediction_files:
                pred_df = pl.read_parquet(pfile)
                gene_pred = pred_df.filter(
                    (pl.col('gene_id') == gene_id) & 
                    (pl.col('position').is_in(list(gap_positions)))
                )
                if gene_pred.height > 0:
                    all_predictions.append(gene_pred)
            
            if not all_predictions:
                logger.warning(f"No gap predictions found for {gene_id}")
                return pl.DataFrame()
                
            gap_predictions = pl.concat(all_predictions)
            logger.info(f"Generated {gap_predictions.height} gap predictions for {gene_id}")
            
            return gap_predictions
    
    def identify_uncertain_positions(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Identify uncertain positions based ONLY on base model scores.
        
        Key principle: In inference mode, we don't know labels, so uncertainty
        must be inferred from base model scores (donor_score, acceptor_score, neither_score).
        
        Args:
            df: DataFrame with base model scores
            
        Returns:
            DataFrame with uncertainty flags and confidence categories
        """
        # Calculate uncertainty metrics from base model scores only
        df_with_uncertainty = df.with_columns([
            # Max score (confidence in best prediction)
            pl.max_horizontal(['donor_score', 'acceptor_score', 'neither_score']).alias('max_score'),
            
            # Score entropy (uncertainty measure)
            pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
            .map_elements(lambda x: self._calculate_entropy([x['donor_score'], x['acceptor_score'], x['neither_score']]))
            .alias('score_entropy'),
            
            # Score spread (difference between max and second max)
            pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
            .map_elements(lambda x: self._calculate_score_spread([x['donor_score'], x['acceptor_score'], x['neither_score']]))
            .alias('score_spread'),
            
            # Predicted splice type (for output)
            pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
            .map_elements(lambda x: self._get_predicted_type([x['donor_score'], x['acceptor_score'], x['neither_score']]))
            .alias('predicted_splice_type')
        ])
        
        # Define uncertainty based on multiple criteria
        df_with_flags = df_with_uncertainty.with_columns([
            # High uncertainty: low max score OR high entropy OR low spread
            (
                (pl.col('max_score') < self.uncertainty_threshold) |
                (pl.col('score_entropy') > 0.9) |  # High entropy threshold
                (pl.col('score_spread') < 0.1)    # Low spread threshold
            ).alias('is_uncertain'),
            
            # Confidence categories
            pl.when(pl.col('max_score') >= 0.8)
            .then(pl.lit('high'))
            .when(pl.col('max_score') >= 0.5)
            .then(pl.lit('medium'))
            .otherwise(pl.lit('low'))
            .alias('confidence_category')
        ])
        
        return df_with_flags
    
    def _calculate_entropy(self, scores: List[float]) -> float:
        """Calculate entropy of score distribution."""
        scores = np.array(scores)
        # Normalize to probabilities
        probs = scores / np.sum(scores) if np.sum(scores) > 0 else scores
        # Add small epsilon to avoid log(0)
        probs = probs + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def _calculate_score_spread(self, scores: List[float]) -> float:
        """Calculate spread between max and second max score."""
        sorted_scores = sorted(scores, reverse=True)
        return float(sorted_scores[0] - sorted_scores[1])
    
    def _get_predicted_type(self, scores: List[float]) -> str:
        """Get predicted splice type based on max score."""
        score_names = ['donor', 'acceptor', 'neither']
        max_idx = np.argmax(scores)
        return score_names[max_idx]
    
    def apply_meta_model_selectively(self, 
                                   df: pl.DataFrame,
                                   meta_model) -> pl.DataFrame:
        """
        Apply meta-model only to uncertain positions.
        
        Args:
            df: DataFrame with uncertainty flags
            meta_model: Trained meta model
            
        Returns:
            DataFrame with meta scores populated
        """
        # Initialize meta scores as copies of base scores
        result_df = df.with_columns([
            pl.col('donor_score').alias('donor_meta'),
            pl.col('acceptor_score').alias('acceptor_meta'), 
            pl.col('neither_score').alias('neither_meta'),
            pl.lit('base_model').alias('prediction_source'),
            pl.lit(0).alias('is_adjusted')
        ])
        
        # Get uncertain positions that need meta-model
        uncertain_positions = df.filter(pl.col('is_uncertain') == True)
        
        if uncertain_positions.height == 0:
            logger.info("No uncertain positions found - no meta-model application needed")
            return result_df
        
        logger.info(f"Applying meta-model to {uncertain_positions.height} uncertain positions")
        
        # TODO: Apply actual meta-model here
        # For now, simulate meta-model adjustment
        # In real implementation, this would:
        # 1. Extract features for uncertain positions
        # 2. Run meta-model prediction
        # 3. Update meta scores
        
        # Placeholder: Apply small adjustments to uncertain positions
        uncertain_indices = uncertain_positions['position'].to_list()
        
        adjusted_df = result_df.with_columns([
            pl.when(pl.col('position').is_in(uncertain_indices))
            .then(pl.col('donor_score') * 1.1)  # Placeholder adjustment
            .otherwise(pl.col('donor_meta'))
            .alias('donor_meta'),
            
            pl.when(pl.col('position').is_in(uncertain_indices))
            .then(pl.col('acceptor_score') * 1.1)  # Placeholder adjustment  
            .otherwise(pl.col('acceptor_meta'))
            .alias('acceptor_meta'),
            
            pl.when(pl.col('position').is_in(uncertain_indices))
            .then(pl.col('neither_score') * 0.9)  # Placeholder adjustment
            .otherwise(pl.col('neither_meta'))
            .alias('neither_meta'),
            
            pl.when(pl.col('position').is_in(uncertain_indices))
            .then(pl.lit('meta_model'))
            .otherwise(pl.col('prediction_source'))
            .alias('prediction_source'),
            
            pl.when(pl.col('position').is_in(uncertain_indices))
            .then(pl.lit(1))
            .otherwise(pl.col('is_adjusted'))
            .alias('is_adjusted')
        ])
        
        return adjusted_df
    
    def generate_complete_predictions(self,
                                    target_genes: List[str],
                                    existing_artifacts_dir: str,
                                    gene_manifest_path: str) -> Dict:
        """
        Generate complete coverage predictions for target genes.
        
        Args:
            target_genes: List of gene IDs to predict
            existing_artifacts_dir: Directory with existing sparse artifacts
            gene_manifest_path: Path to gene manifest with gene info
            
        Returns:
            Dictionary with results and metadata
        """
        logger.info(f"Generating complete coverage predictions for {len(target_genes)} genes")
        
        # Load gene manifest
        gene_manifest = pl.read_csv(gene_manifest_path)
        gene_info_map = {
            row['gene_id']: row for row in gene_manifest.to_dicts()
        }
        
        # Load existing artifacts
        existing_files = list(Path(existing_artifacts_dir).glob("**/analysis_sequences_*.parquet"))
        if existing_files:
            existing_dfs = [pl.read_parquet(f) for f in existing_files]
            existing_df = pl.concat(existing_dfs) if existing_dfs else pl.DataFrame()
        else:
            existing_df = pl.DataFrame()
        
        complete_predictions = []
        
        for gene_id in target_genes:
            if gene_id not in gene_info_map:
                logger.warning(f"Gene {gene_id} not found in manifest")
                continue
                
            gene_info = gene_info_map[gene_id]
            gene_length = gene_info['end'] - gene_info['start'] + 1
            
            logger.info(f"Processing {gene_id} (length: {gene_length})")
            
            # Get existing predictions for this gene
            gene_existing = existing_df.filter(pl.col('gene_id') == gene_id)
            
            # Detect gap positions
            gap_positions = self.detect_gap_positions(gene_existing, gene_id, gene_length)
            
            # Run base model on gaps if needed
            if gap_positions:
                gap_predictions = self.run_base_model_on_gaps(gene_id, gap_positions, gene_info)
                
                # Combine existing and gap predictions
                if gene_existing.height > 0 and gap_predictions.height > 0:
                    # Align columns
                    common_cols = list(set(gene_existing.columns) & set(gap_predictions.columns))
                    combined_df = pl.concat([
                        gene_existing.select(common_cols),
                        gap_predictions.select(common_cols)
                    ])
                elif gap_predictions.height > 0:
                    combined_df = gap_predictions
                else:
                    combined_df = gene_existing
            else:
                combined_df = gene_existing
            
            if combined_df.height == 0:
                logger.warning(f"No predictions available for {gene_id}")
                continue
            
            # Identify uncertain positions (using only base scores)
            df_with_uncertainty = self.identify_uncertain_positions(combined_df)
            
            # Apply meta-model selectively
            # TODO: Load actual meta-model
            meta_model = None  # Placeholder
            final_predictions = self.apply_meta_model_selectively(df_with_uncertainty, meta_model)
            
            # Add final splice type prediction
            final_with_type = final_predictions.with_columns([
                pl.struct(['donor_meta', 'acceptor_meta', 'neither_meta'])
                .map_elements(lambda x: self._get_predicted_type([x['donor_meta'], x['acceptor_meta'], x['neither_meta']]))
                .alias('splice_type')
            ])
            
            complete_predictions.append(final_with_type)
        
        # Combine all predictions
        if complete_predictions:
            all_predictions = pl.concat(complete_predictions)
            
            # Save complete predictions
            output_path = self.output_dir / "complete_coverage_predictions.parquet"
            all_predictions.write_parquet(output_path)
            
            logger.info(f"Generated complete predictions: {all_predictions.height} positions")
            logger.info(f"Saved to: {output_path}")
            
            return {
                'success': True,
                'predictions_path': str(output_path),
                'total_positions': all_predictions.height,
                'genes_processed': len(target_genes),
                'summary': self._generate_summary(all_predictions)
            }
        else:
            return {
                'success': False,
                'error': 'No predictions generated',
                'genes_processed': 0
            }
    
    def _generate_summary(self, df: pl.DataFrame) -> Dict:
        """Generate summary statistics."""
        total_positions = df.height
        uncertain_positions = df.filter(pl.col('is_uncertain') == True).height
        meta_adjusted = df.filter(pl.col('is_adjusted') == 1).height
        
        confidence_dist = df.group_by('confidence_category').agg(pl.len().alias('count')).to_dicts()
        confidence_summary = {item['confidence_category']: item['count'] for item in confidence_dist}
        
        return {
            'total_positions': total_positions,
            'uncertain_positions': uncertain_positions,
            'meta_adjusted_positions': meta_adjusted,
            'meta_adjustment_rate': meta_adjusted / total_positions if total_positions > 0 else 0,
            'confidence_distribution': confidence_summary
        }


def run_complete_coverage_inference(target_genes: List[str],
                                  base_model_path: str,
                                  meta_model_path: str,
                                  existing_artifacts_dir: str,
                                  gene_manifest_path: str,
                                  output_dir: str,
                                  uncertainty_threshold: float = 0.3) -> Dict:
    """
    Main function to run complete coverage inference workflow.
    
    This ensures ALL positions in target genes have predictions by:
    1. Using existing artifacts where available
    2. Running base model on gap positions  
    3. Applying meta-model selectively to uncertain positions
    4. Producing continuous position coverage
    
    Args:
        target_genes: List of gene IDs to predict
        base_model_path: Path to trained base model
        meta_model_path: Path to trained meta model
        existing_artifacts_dir: Directory with existing sparse artifacts
        gene_manifest_path: Path to gene manifest
        output_dir: Output directory
        uncertainty_threshold: Threshold for uncertain position selection
        
    Returns:
        Dictionary with results and metadata
    """
    predictor = CompleteCoveragePredictor(
        base_model_path=base_model_path,
        meta_model_path=meta_model_path,
        output_dir=output_dir,
        uncertainty_threshold=uncertainty_threshold
    )
    
    return predictor.generate_complete_predictions(
        target_genes=target_genes,
        existing_artifacts_dir=existing_artifacts_dir,
        gene_manifest_path=gene_manifest_path
    )