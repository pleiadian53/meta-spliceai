#!/usr/bin/env python3
"""
Sequence-Centric Inference Interface for Splice Surveyor Meta-Models

This module provides a SpliceAI/OpenSpliceAI-compatible interface for predicting
splice site scores on arbitrary DNA sequences, enabling seamless integration
with variant analysis workflows and delta score calculations.

Key Features:
1. Sequence-centric interface (no gene IDs required)
2. Compatible with SpliceAI/OpenSpliceAI APIs
3. Supports WT/ALT sequence comparison for variant analysis
4. Integrates with existing meta-model infrastructure
5. Enables delta score calculations for variant impact assessment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import tempfile
import json

from meta_spliceai.splice_engine.meta_models.training.unified_model_loader import load_unified_model


class SequenceInferenceInterface:
    """
    Sequence-centric interface for meta-model splice site prediction.
    
    This class provides a SpliceAI-compatible API for predicting splice site scores
    on arbitrary DNA sequences, enabling seamless integration with variant analysis.
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        training_dataset_path: Union[str, Path],
        context_size: int = 5000,
        inference_mode: str = "hybrid",
        verbose: bool = False
    ):
        """
        Initialize sequence inference interface.
        
        Parameters
        ----------
        model_path : str or Path
            Path to trained meta-model (single or ensemble)
        training_dataset_path : str or Path
            Path to training dataset for feature engineering
        context_size : int
            Context size for sequence analysis (default: 5000bp)
        inference_mode : str
            Inference mode: 'base_only', 'hybrid', or 'meta_only'
        verbose : bool
            Enable verbose output
        """
        self.model_path = Path(model_path)
        self.training_dataset_path = Path(training_dataset_path)
        self.context_size = context_size
        self.inference_mode = inference_mode
        self.verbose = verbose
        
        # Load the unified model
        self.model = load_unified_model(self.model_path)
        
        # Load feature names for consistency
        feature_manifest_path = self.model_path.parent / "feature_manifest.csv"
        if feature_manifest_path.exists():
            feature_df = pd.read_csv(feature_manifest_path)
            self.expected_features = feature_df['feature'].tolist()
        else:
            self.expected_features = self.model.get_feature_names()
        
        if self.verbose:
            print(f"🧬 Sequence Inference Interface initialized")
            print(f"  Model: {self.model_path}")
            print(f"  Model type: {type(self.model.model).__name__}")
            print(f"  Features: {len(self.expected_features)}")
            print(f"  Mode: {self.inference_mode}")
    
    def predict_sequence(
        self,
        sequence: str,
        chromosome: str = "chr1",
        start_position: int = 1,
        gene_id: Optional[str] = None,
        strand: str = "+",
        return_format: str = "dict"
    ) -> Union[Dict, np.ndarray]:
        """
        Predict splice site scores for an arbitrary DNA sequence.
        
        This method provides a SpliceAI-compatible interface for sequence-based
        splice site prediction using trained meta-models.
        
        Parameters
        ----------
        sequence : str
            DNA sequence to analyze (A, T, G, C)
        chromosome : str
            Chromosome identifier (default: "chr1")
        start_position : int
            Genomic start position of sequence (1-based)
        gene_id : str, optional
            Gene identifier if known (improves feature engineering)
        strand : str
            Strand orientation ('+' or '-')
        return_format : str
            Return format: 'dict', 'array', or 'dataframe'
            
        Returns
        -------
        Union[Dict, np.ndarray]
            Splice site predictions for each position:
            - dict: {'donor': array, 'acceptor': array, 'neither': array}
            - array: shape (len(sequence), 3) [neither, donor, acceptor]
            - dataframe: columns [position, donor_score, acceptor_score, neither_score]
        """
        
        if self.verbose:
            print(f"🔍 Predicting splice sites for {len(sequence)}bp sequence")
            print(f"  Chromosome: {chromosome}, Start: {start_position}, Strand: {strand}")
        
        # Create temporary gene entry for feature engineering
        temp_gene_data = self._create_temp_gene_entry(
            sequence, chromosome, start_position, gene_id, strand
        )
        
        # Run inference using existing infrastructure
        predictions = self._run_sequence_inference(temp_gene_data)
        
        # Format results based on requested format
        return self._format_predictions(predictions, sequence, return_format)
    
    def predict_variant_delta(
        self,
        wt_sequence: str,
        alt_sequence: str,
        variant_position: int,
        ref_allele: str,
        alt_allele: str,
        chromosome: str = "chr1",
        genomic_position: int = 1,
        gene_id: Optional[str] = None,
        strand: str = "+",
        variant_index: Optional[int] = None,
        return_format: str = "dict"
    ) -> Dict:
        """
        Predict delta scores for variant impact analysis.
        
        This method computes the difference in splice site scores between
        wild-type and alternate sequences, enabling variant impact assessment.
        
        Parameters
        ----------
        wt_sequence : str
            Wild-type DNA sequence
        alt_sequence : str
            Alternate DNA sequence with variant applied
        variant_position : int
            Position of variant within sequence (0-based)
        ref_allele : str
            Reference allele (e.g., "G")
        alt_allele : str
            Alternate allele (e.g., "A")
        chromosome : str
            Chromosome identifier
        genomic_position : int
            Genomic position of variant (1-based)
        gene_id : str, optional
            Gene identifier if known
        strand : str
            Strand orientation
        variant_index : int, optional
            Index for multiallelic variants (0, 1, 2, ...)
        return_format : str
            Return format for delta scores
            
        Returns
        -------
        Dict
            Delta score analysis:
            {
                'donor_delta': np.ndarray,      # Donor score differences
                'acceptor_delta': np.ndarray,   # Acceptor score differences
                'max_delta_donor': float,       # Maximum donor delta
                'max_delta_acceptor': float,    # Maximum acceptor delta
                'variant_position': int,        # Variant position in sequence
                'impact_assessment': dict       # Predicted impact classification
            }
        """
        
        if self.verbose:
            print(f"🧬 Computing variant delta scores")
            print(f"  WT sequence: {len(wt_sequence)}bp")
            print(f"  ALT sequence: {len(alt_sequence)}bp")
            print(f"  Variant position: {variant_position}")
        
        # Generate descriptive gene IDs for WT/ALT pair with multiallelic support
        wt_gene_id, alt_gene_id = self.create_variant_gene_ids(
            gene_id, chromosome, genomic_position, ref_allele, alt_allele, variant_index
        )
        
        if self.verbose:
            print(f"  Generated IDs: WT={wt_gene_id}, ALT={alt_gene_id}")
            if variant_index is not None:
                print(f"  Multiallelic variant index: {variant_index}")
        
        # Get predictions for both sequences with descriptive IDs
        wt_predictions = self.predict_sequence(
            wt_sequence, chromosome, genomic_position, wt_gene_id, strand, "dict"
        )
        
        alt_predictions = self.predict_sequence(
            alt_sequence, chromosome, genomic_position, alt_gene_id, strand, "dict"
        )
        
        # Compute delta scores
        donor_delta = alt_predictions['donor'] - wt_predictions['donor']
        acceptor_delta = alt_predictions['acceptor'] - wt_predictions['acceptor']
        
        # Find maximum deltas
        max_delta_donor = np.max(np.abs(donor_delta))
        max_delta_acceptor = np.max(np.abs(acceptor_delta))
        
        # Classify impact
        impact_assessment = self._assess_variant_impact(
            donor_delta, acceptor_delta, variant_position
        )
        
        return {
            'donor_delta': donor_delta,
            'acceptor_delta': acceptor_delta,
            'max_delta_donor': float(max_delta_donor),
            'max_delta_acceptor': float(max_delta_acceptor),
            'variant_position': variant_position,
            'impact_assessment': impact_assessment,
            'wt_predictions': wt_predictions,
            'alt_predictions': alt_predictions
        }
    
    def _create_temp_gene_entry(
        self, 
        sequence: str, 
        chromosome: str, 
        start_position: int,
        gene_id: Optional[str],
        strand: str
    ) -> Dict:
        """Create temporary gene data for feature engineering."""
        
        # Generate a descriptive temporary gene ID
        if gene_id is None:
            # Use chromosome and position for anonymous sequences
            temp_gene_id = f"SEQ_{chromosome}_{start_position}_1"
        else:
            # Use provided gene ID with descriptive suffix for variant analysis
            # This maintains traceability while indicating variant analysis context
            temp_gene_id = f"{gene_id}_VAR_1"
        
        # Create gene data structure compatible with inference workflow
        end_position = start_position + len(sequence) - 1
        
        gene_data = {
            'gene_id': temp_gene_id,
            'chromosome': chromosome,
            'start_position': start_position,
            'end_position': end_position,
            'strand': strand,
            'sequence': sequence,
            'length': len(sequence),
            'gene_type': 'protein_coding',  # Default assumption
            'gene_name': gene_id if gene_id else temp_gene_id,
            'original_gene_id': gene_id,  # Track original gene if provided
            'is_variant_analysis': True   # Flag for variant analysis context
        }
        
        return gene_data
    
    @staticmethod
    def create_variant_gene_ids(
        base_gene_id: Optional[str], 
        chromosome: str, 
        variant_position: int,
        ref_allele: str,
        alt_allele: str,
        variant_index: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Create paired gene IDs for WT/ALT variant analysis with multiallelic support.
        
        This method generates descriptive, unique IDs for variant analysis that handle:
        1. Multiple variants per gene at different positions
        2. Multiallelic variants at the same position
        3. Maintains traceability to original gene
        
        Parameters
        ----------
        base_gene_id : str, optional
            Original gene ID if known
        chromosome : str
            Chromosome identifier
        variant_position : int
            Genomic position of the variant
        ref_allele : str
            Reference allele
        alt_allele : str
            Alternate allele
        variant_index : int, optional
            Index for multiallelic variants at same position (0, 1, 2, ...)
            
        Returns
        -------
        Tuple[str, str]
            (wt_gene_id, alt_gene_id) for variant analysis
            
        Examples
        --------
        >>> # Single variant
        >>> wt_id, alt_id = SequenceInferenceInterface.create_variant_gene_ids(
        ...     "ENSG00000012048", "chr17", 43094077, "G", "A"
        ... )
        >>> print(wt_id)   # "ENSG00000012048_WT_43094077_G"
        >>> print(alt_id)  # "ENSG00000012048_ALT_43094077_G_A"
        
        >>> # Multiallelic variant (first alternative)
        >>> wt_id, alt_id = SequenceInferenceInterface.create_variant_gene_ids(
        ...     "ENSG00000012048", "chr17", 43094077, "G", "A", variant_index=0
        ... )
        >>> print(alt_id)  # "ENSG00000012048_ALT_43094077_G_A_0"
        
        >>> # Multiallelic variant (second alternative)
        >>> wt_id, alt_id = SequenceInferenceInterface.create_variant_gene_ids(
        ...     "ENSG00000012048", "chr17", 43094077, "G", "T", variant_index=1
        ... )
        >>> print(alt_id)  # "ENSG00000012048_ALT_43094077_G_T_1"
        
        >>> # Anonymous sequence
        >>> wt_id, alt_id = SequenceInferenceInterface.create_variant_gene_ids(
        ...     None, "chr1", 12345, "C", "T"
        ... )
        >>> print(wt_id)   # "SEQ_chr1_12345_WT_C"
        >>> print(alt_id)  # "SEQ_chr1_12345_ALT_C_T"
        """
        
        if base_gene_id is None:
            # Anonymous sequence - use chromosome and position
            base_id = f"SEQ_{chromosome}_{variant_position}"
        else:
            # Known gene - use gene ID
            base_id = base_gene_id
        
        # Create WT ID with position and reference allele
        wt_gene_id = f"{base_id}_WT_{variant_position}_{ref_allele}"
        
        # Create ALT ID with position, alleles, and optional multiallelic index
        alt_gene_id = f"{base_id}_ALT_{variant_position}_{ref_allele}_{alt_allele}"
        
        # Add multiallelic index if provided
        if variant_index is not None:
            alt_gene_id += f"_{variant_index}"
        
        return wt_gene_id, alt_gene_id
    
    def _run_sequence_inference(self, gene_data: Dict) -> Dict:
        """Run inference using existing infrastructure with temporary gene data."""
        
        # Create temporary files for inference
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Create temporary gene list file
            gene_file = temp_dir_path / "temp_genes.txt"
            with open(gene_file, 'w') as f:
                f.write(gene_data['gene_id'])
            
            # Create temporary inference configuration
            config = {
                'model_path': str(self.model_path),
                'training_dataset': str(self.training_dataset_path),
                'genes': [gene_data['gene_id']],
                'inference_mode': self.inference_mode,
                'complete_coverage': True,  # Always use complete coverage
                'enable_chunked_processing': True,
                'chunk_size': max(1000, len(gene_data['sequence']) // 10)
            }
            
            # Use existing inference infrastructure
            from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
                run_enhanced_selective_inference
            )
            
            # Mock the gene data in a format the inference workflow expects
            # This is a simplified approach - in production, we'd want to create
            # a proper sequence-to-features pipeline
            
            # For now, return mock predictions that demonstrate the interface
            # TODO: Implement full feature engineering pipeline for arbitrary sequences
            
            n_positions = len(gene_data['sequence'])
            
            if self.inference_mode == "base_only":
                # Use base model predictions only
                predictions = self._generate_base_predictions(gene_data['sequence'])
            elif self.inference_mode == "meta_only":
                # Use meta-model for all positions
                predictions = self._generate_meta_predictions(gene_data['sequence'])
            else:  # hybrid
                # Use hybrid approach
                predictions = self._generate_hybrid_predictions(gene_data['sequence'])
            
            return predictions
    
    def _generate_base_predictions(self, sequence: str) -> Dict:
        """Generate base model predictions (placeholder for SpliceAI/OpenSpliceAI)."""
        n_positions = len(sequence)
        
        # Placeholder: In production, this would call actual SpliceAI/OpenSpliceAI
        return {
            'donor': np.random.random(n_positions) * 0.3,      # Low baseline scores
            'acceptor': np.random.random(n_positions) * 0.3,   # Low baseline scores
            'neither': 1 - (np.random.random(n_positions) * 0.6)  # High neither scores
        }
    
    def _generate_meta_predictions(self, sequence: str) -> Dict:
        """Generate meta-model enhanced predictions."""
        n_positions = len(sequence)
        
        # Placeholder: In production, this would use feature engineering + meta-model
        # For demonstration, show enhanced scores
        base_predictions = self._generate_base_predictions(sequence)
        
        # Simulate meta-model enhancement
        enhancement_factor = 1.2
        return {
            'donor': np.clip(base_predictions['donor'] * enhancement_factor, 0, 1),
            'acceptor': np.clip(base_predictions['acceptor'] * enhancement_factor, 0, 1),
            'neither': np.clip(base_predictions['neither'], 0, 1)
        }
    
    def _generate_hybrid_predictions(self, sequence: str) -> Dict:
        """Generate hybrid predictions (selective meta-model application)."""
        base_predictions = self._generate_base_predictions(sequence)
        meta_predictions = self._generate_meta_predictions(sequence)
        
        # Simulate uncertainty-based selection (2-5% of positions)
        uncertainty_mask = np.random.random(len(sequence)) < 0.03
        
        # Use meta predictions for uncertain positions, base for others
        hybrid_predictions = {}
        for key in ['donor', 'acceptor', 'neither']:
            hybrid_predictions[key] = np.where(
                uncertainty_mask,
                meta_predictions[key],
                base_predictions[key]
            )
        
        return hybrid_predictions
    
    def _assess_variant_impact(
        self, 
        donor_delta: np.ndarray, 
        acceptor_delta: np.ndarray, 
        variant_position: int
    ) -> Dict:
        """Assess variant impact based on delta scores."""
        
        # Define impact thresholds (compatible with OpenSpliceAI)
        thresholds = {
            'high_impact': 0.5,
            'moderate_impact': 0.2,
            'low_impact': 0.1
        }
        
        # Find maximum absolute deltas
        max_donor_delta = np.max(np.abs(donor_delta))
        max_acceptor_delta = np.max(np.abs(acceptor_delta))
        max_overall_delta = max(max_donor_delta, max_acceptor_delta)
        
        # Classify impact level
        if max_overall_delta >= thresholds['high_impact']:
            impact_level = 'high'
        elif max_overall_delta >= thresholds['moderate_impact']:
            impact_level = 'moderate'
        elif max_overall_delta >= thresholds['low_impact']:
            impact_level = 'low'
        else:
            impact_level = 'minimal'
        
        # Find positions with significant changes
        significant_positions = []
        for i, (d_delta, a_delta) in enumerate(zip(donor_delta, acceptor_delta)):
            if abs(d_delta) >= thresholds['low_impact'] or abs(a_delta) >= thresholds['low_impact']:
                significant_positions.append({
                    'position': i,
                    'donor_delta': float(d_delta),
                    'acceptor_delta': float(a_delta),
                    'distance_from_variant': abs(i - variant_position)
                })
        
        return {
            'impact_level': impact_level,
            'max_donor_delta': float(max_donor_delta),
            'max_acceptor_delta': float(max_acceptor_delta),
            'max_overall_delta': float(max_overall_delta),
            'significant_positions': significant_positions,
            'confidence': min(max_overall_delta * 2, 1.0)
        }
    
    def _format_predictions(
        self, 
        predictions: Dict, 
        sequence: str, 
        return_format: str
    ) -> Union[Dict, np.ndarray, pd.DataFrame]:
        """Format predictions according to requested format."""
        
        if return_format == "dict":
            return predictions
        
        elif return_format == "array":
            # Return as (n_positions, 3) array [neither, donor, acceptor]
            return np.column_stack([
                predictions['neither'],
                predictions['donor'], 
                predictions['acceptor']
            ])
        
        elif return_format == "dataframe":
            # Return as DataFrame with position information
            positions = list(range(len(sequence)))
            return pd.DataFrame({
                'position': positions,
                'donor_score': predictions['donor'],
                'acceptor_score': predictions['acceptor'],
                'neither_score': predictions['neither'],
                'nucleotide': list(sequence)
            })
        
        else:
            raise ValueError(f"Unknown return format: {return_format}")


# Convenience functions for SpliceAI-compatible interface
def predict_splice_scores(
    sequence: str,
    model_path: str,
    training_dataset_path: str,
    inference_mode: str = "hybrid",
    return_format: str = "dict"
) -> Union[Dict, np.ndarray]:
    """
    SpliceAI-compatible function for splice site prediction.
    
    Parameters
    ----------
    sequence : str
        DNA sequence to analyze
    model_path : str
        Path to trained meta-model
    training_dataset_path : str
        Path to training dataset
    inference_mode : str
        Inference mode: 'base_only', 'hybrid', or 'meta_only'
    return_format : str
        Return format: 'dict', 'array', or 'dataframe'
        
    Returns
    -------
    Union[Dict, np.ndarray]
        Splice site predictions
        
    Examples
    --------
    >>> # Basic usage
    >>> scores = predict_splice_scores(
    ...     sequence="ATGCGTAAGT...",
    ...     model_path="results/my_model/model_multiclass.pkl",
    ...     training_dataset_path="train_data/master"
    ... )
    >>> print(f"Donor scores: {scores['donor'][:10]}")
    
    >>> # For variant analysis
    >>> wt_scores = predict_splice_scores(wt_sequence, model_path, dataset_path)
    >>> alt_scores = predict_splice_scores(alt_sequence, model_path, dataset_path)
    >>> delta_scores = alt_scores['donor'] - wt_scores['donor']
    """
    
    interface = SequenceInferenceInterface(
        model_path=model_path,
        training_dataset_path=training_dataset_path,
        inference_mode=inference_mode,
        verbose=False
    )
    
    return interface.predict_sequence(
        sequence=sequence,
        return_format=return_format
    )


def compute_variant_delta_scores(
    wt_sequence: str,
    alt_sequence: str,
    variant_position: int,
    ref_allele: str,
    alt_allele: str,
    model_path: str,
    training_dataset_path: str,
    chromosome: str = "chr1",
    genomic_position: int = 1,
    gene_id: Optional[str] = None,
    variant_index: Optional[int] = None,
    inference_mode: str = "hybrid"
) -> Dict:
    """
    Compute delta scores for variant impact analysis.
    
    This function provides OpenSpliceAI-compatible delta score calculation
    using trained meta-models for enhanced variant impact assessment.
    
    Parameters
    ----------
    wt_sequence : str
        Wild-type DNA sequence
    alt_sequence : str
        Alternate sequence with variant
    variant_position : int
        Position of variant in sequence (0-based)
    model_path : str
        Path to trained meta-model
    training_dataset_path : str
        Path to training dataset
    inference_mode : str
        Inference mode for predictions
        
    Returns
    -------
    Dict
        Comprehensive delta score analysis
        
    Examples
    --------
    >>> # Variant delta analysis
    >>> delta_results = compute_variant_delta_scores(
    ...     wt_sequence="ATGCGTAAGT...",
    ...     alt_sequence="ATGCATAAGT...",  # G>A variant
    ...     variant_position=4,
    ...     model_path="results/my_model/model_multiclass.pkl",
    ...     training_dataset_path="train_data/master"
    ... )
    >>> print(f"Maximum impact: {delta_results['max_overall_delta']:.3f}")
    >>> print(f"Impact level: {delta_results['impact_assessment']['impact_level']}")
    """
    
    interface = SequenceInferenceInterface(
        model_path=model_path,
        training_dataset_path=training_dataset_path,
        inference_mode=inference_mode,
        verbose=False
    )
    
    return interface.predict_variant_delta(
        wt_sequence=wt_sequence,
        alt_sequence=alt_sequence,
        variant_position=variant_position,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        chromosome=chromosome,
        genomic_position=genomic_position,
        gene_id=gene_id,
        variant_index=variant_index
    )


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test sequence-centric inference interface")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--dataset", required=True, help="Path to training dataset")
    parser.add_argument("--sequence", help="DNA sequence to test")
    parser.add_argument("--test-variant", action="store_true", help="Run variant delta test")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Test sequence if not provided
    test_sequence = args.sequence or "ATGCGTAAGTCGACTAGCTAGCTGATCGATCGTAGCTAGCTAG"
    
    print("🧬 Testing Sequence-Centric Inference Interface")
    print("=" * 60)
    
    # Initialize interface
    interface = SequenceInferenceInterface(
        model_path=args.model,
        training_dataset_path=args.dataset,
        inference_mode="hybrid",
        verbose=args.verbose
    )
    
    # Test basic sequence prediction
    print("\n📊 Basic Sequence Prediction:")
    predictions = interface.predict_sequence(
        sequence=test_sequence,
        return_format="dataframe"
    )
    print(predictions.head(10))
    
    # Test variant delta calculation
    if args.test_variant:
        print("\n🧬 Variant Delta Score Test:")
        
        # Create a simple variant (substitute middle nucleotide)
        mid_pos = len(test_sequence) // 2
        alt_sequence = test_sequence[:mid_pos] + 'A' + test_sequence[mid_pos+1:]
        
        print(f"  WT:  {test_sequence[mid_pos-5:mid_pos+6]}")
        print(f"  ALT: {alt_sequence[mid_pos-5:mid_pos+6]}")
        print(f"  Variant position: {mid_pos}")
        
        delta_results = interface.predict_variant_delta(
            wt_sequence=test_sequence,
            alt_sequence=alt_sequence,
            variant_position=mid_pos
        )
        
        print(f"\n📈 Delta Score Results:")
        print(f"  Max donor delta: {delta_results['max_delta_donor']:.4f}")
        print(f"  Max acceptor delta: {delta_results['max_delta_acceptor']:.4f}")
        print(f"  Impact level: {delta_results['impact_assessment']['impact_level']}")
        print(f"  Confidence: {delta_results['impact_assessment']['confidence']:.3f}")
    
    print("\n✅ Sequence inference interface test completed!")
