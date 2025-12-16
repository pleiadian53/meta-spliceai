#!/usr/bin/env python3
"""
Selective Feature Generation for Meta-Model Inference

This module implements selective feature generation for uncertain positions,
enabling actual meta-model predictions while maintaining efficiency.

Key Features:
- Generate features only for uncertain positions (not all positions)
- Use proper context length (500bp) and k-mer size (3-mers) matching training
- Extract sequence context and generate k-mer features
- Create feature matrices compatible with trained meta-models
- Efficient processing for selective inference workflows

Author: Splice Surveyor Team
"""

import os
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import polars as pl
import numpy as np

from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_inference_workflow import _load_training_schema
from meta_spliceai.splice_engine.sequence_featurizer import extract_analysis_sequences


def extract_sequence_context_for_positions(
    positions_df: pd.DataFrame,
    window_size: int = 500,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract sequence context around specified positions.
    
    Parameters
    ----------
    positions_df : pd.DataFrame
        DataFrame with columns: gene_id, position, chrom
    window_size : int, default=500
        Context window size (±window_size/2 around each position)
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sequence context for each position
    """
    if verbose:
        print(f"[selective-feature-gen] Extracting sequence context for {len(positions_df)} positions")
        print(f"[selective-feature-gen] Using window size: ±{window_size//2}bp")
    
    # Create sequence dataframe from gene info
    # This will need to interface with existing sequence extraction
    # For now, create a placeholder structure
    
    # Group by gene to extract sequences efficiently
    sequence_contexts = []
    
    for gene_id, gene_positions in positions_df.groupby('gene_id'):
        if verbose:
            print(f"[selective-feature-gen] Processing {len(gene_positions)} positions for gene {gene_id}")
        
        # Extract sequence context for this gene's positions
        # This would interface with the existing sequence featurizer
        try:
            # Call existing extract_analysis_sequences function
            # Need to adapt it for selective positions
            gene_contexts = _extract_gene_sequence_contexts(
                gene_id=gene_id,
                positions=gene_positions['position'].tolist(),
                chromosome=gene_positions['chrom'].iloc[0],
                window_size=window_size,
                verbose=verbose
            )
            sequence_contexts.extend(gene_contexts)
            
        except Exception as e:
            if verbose:
                print(f"[selective-feature-gen] Warning: Failed to extract context for gene {gene_id}: {e}")
            continue
    
    if not sequence_contexts:
        if verbose:
            print("[selective-feature-gen] Warning: No sequence contexts extracted")
        return pd.DataFrame()
    
    # Convert to DataFrame
    context_df = pd.DataFrame(sequence_contexts)
    
    if verbose:
        print(f"[selective-feature-gen] ✅ Extracted context for {len(context_df)} positions")
    
    return context_df


def _extract_gene_sequence_contexts(
    gene_id: str,
    positions: List[int],
    chromosome: str,
    window_size: int = 500,
    verbose: bool = True
) -> List[Dict]:
    """
    Extract sequence contexts for specific positions in a gene.
    
    Parameters
    ----------
    gene_id : str
        Gene identifier
    positions : List[int]
        List of genomic positions
    chromosome : str
        Chromosome identifier
    window_size : int, default=500
        Context window size
        
    Returns
    -------
    List[Dict]
        List of context dictionaries for each position
    """
    from Bio import SeqIO
    from meta_spliceai.system.genomic_resources import create_systematic_manager
    
    contexts = []
    
    # Get FASTA file path using systematic genomic resource manager
    try:
        # Use systematic genomic resource manager
        manager = create_systematic_manager()
        fasta_file = manager.get_fasta_path()
        
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
    except Exception as e:
        print(f"[selective-feature-gen] Error accessing genomic resources: {e}")
        return contexts
    
    # Load genome sequence
    try:
        genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
        
        if verbose:
            print(f"[selective-feature-gen] Loaded {len(genome)} chromosomes from FASTA")
            print(f"[selective-feature-gen] Available chromosomes: {list(genome.keys())[:10]}...")  # First 10
            print(f"[selective-feature-gen] Searching for chromosome: {repr(chromosome)} (type: {type(chromosome)})")
        
        # Convert chromosome to string to handle numpy types
        chromosome_str = str(chromosome)
        if verbose:
            print(f"[selective-feature-gen] Converted to string: {repr(chromosome_str)}")
            
        if chromosome_str not in genome:
            print(f"[selective-feature-gen] Warning: Chromosome {chromosome} not found in FASTA")
            print(f"[selective-feature-gen] Chromosome type: {type(chromosome)}, repr: {repr(chromosome)}")
            print(f"[selective-feature-gen] Available chromosomes: {list(genome.keys())}")
            # Try to find similar chromosomes
            similar_chroms = [k for k in genome.keys() if str(chromosome) in str(k)]
            print(f"[selective-feature-gen] Similar chromosomes: {similar_chroms}")
            return contexts
            
        chrom_seq = genome[chromosome_str].seq
        if verbose:
            print(f"[selective-feature-gen] Successfully loaded chromosome {chromosome_str} sequence (length: {len(chrom_seq):,})")
        
    except Exception as e:
        print(f"[selective-feature-gen] Error loading FASTA: {e}")
        import traceback
        traceback.print_exc()
        return contexts
    
    half_window = window_size // 2
    
    for position in positions:
        try:
            # Extract sequence context around position (1-based to 0-based conversion)
            start_pos = max(0, position - half_window - 1)  # Convert to 0-based
            end_pos = min(len(chrom_seq), position + half_window)
            
            # Get sequence from chromosome
            sequence = str(chrom_seq[start_pos:end_pos])
            
            if sequence and len(sequence) >= window_size // 2:  # Minimum sequence length
                context = {
                    'gene_id': gene_id,
                    'position': position,
                    'chrom': chromosome,
                    'window_start': start_pos + 1,  # Convert back to 1-based for reporting
                    'window_end': end_pos,
                    'sequence': sequence.upper(),  # Normalize to uppercase
                    'sequence_length': len(sequence)
                }
                contexts.append(context)
                
        except Exception as e:
            print(f"[selective-feature-gen] Warning: Failed to extract sequence for position {position}: {e}")
            continue
    
    return contexts


def generate_kmer_features_for_sequences(
    sequence_contexts_df: pd.DataFrame,
    kmer_size: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate k-mer features for sequence contexts.
    
    Parameters
    ----------
    sequence_contexts_df : pd.DataFrame
        DataFrame with sequence contexts
    kmer_size : int, default=3
        Size of k-mers to generate (should match training data)
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        DataFrame with k-mer features for each position
    """
    if verbose:
        print(f"[selective-feature-gen] Generating {kmer_size}-mer features for {len(sequence_contexts_df)} sequences")
    
    # Generate all possible k-mers
    nucleotides = ['A', 'T', 'G', 'C']
    all_kmers = _generate_all_kmers(nucleotides, kmer_size)
    
    if verbose:
        print(f"[selective-feature-gen] Total possible {kmer_size}-mers: {len(all_kmers)}")
    
    feature_rows = []
    
    for idx, row in sequence_contexts_df.iterrows():
        try:
            sequence = row['sequence'].upper()
            
            # Count k-mers in sequence
            kmer_counts = _count_kmers_in_sequence(sequence, kmer_size)
            
            # Normalize counts to frequencies
            total_kmers = sum(kmer_counts.values())
            if total_kmers > 0:
                kmer_freqs = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
            else:
                kmer_freqs = {kmer: 0.0 for kmer in all_kmers}
            
            # Create feature row
            feature_row = {
                'gene_id': row['gene_id'],
                'position': row['position'],
                'chrom': row['chrom'],
                'sequence_length': row['sequence_length']
            }
            
            # Add k-mer frequency features
            for kmer in all_kmers:
                feature_row[f'{kmer_size}mer_{kmer}'] = kmer_freqs.get(kmer, 0.0)
            
            feature_rows.append(feature_row)
            
        except Exception as e:
            if verbose:
                print(f"[selective-feature-gen] Warning: Failed to generate features for position {row['position']}: {e}")
            continue
    
    if not feature_rows:
        if verbose:
            print("[selective-feature-gen] Warning: No k-mer features generated")
        return pd.DataFrame()
    
    features_df = pd.DataFrame(feature_rows)
    
    if verbose:
        kmer_cols = [col for col in features_df.columns if f'{kmer_size}mer_' in col]
        print(f"[selective-feature-gen] ✅ Generated {len(kmer_cols)} k-mer features for {len(features_df)} positions")
    
    return features_df


def _generate_all_kmers(nucleotides: List[str], k: int) -> List[str]:
    """Generate all possible k-mers of length k."""
    if k == 1:
        return nucleotides
    
    kmers = []
    for nuc in nucleotides:
        for sub_kmer in _generate_all_kmers(nucleotides, k - 1):
            kmers.append(nuc + sub_kmer)
    
    return sorted(kmers)


def _count_kmers_in_sequence(sequence: str, k: int) -> Dict[str, int]:
    """Count k-mers in a sequence."""
    kmer_counts = {}
    
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        
        # Only count valid k-mers (no N's or other ambiguous nucleotides)
        if all(nuc in 'ATGC' for nuc in kmer):
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    
    return kmer_counts


def create_feature_matrix_for_meta_model(
    kmer_features_df: pd.DataFrame,
    base_predictions_df: pd.DataFrame,
    training_schema_path: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create feature matrix compatible with trained meta-model.
    
    Parameters
    ----------
    kmer_features_df : pd.DataFrame
        DataFrame with k-mer features
    base_predictions_df : pd.DataFrame
        DataFrame with base model predictions
    training_schema_path : str
        Path to training schema for feature alignment
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Feature matrix ready for meta-model inference
    """
    if verbose:
        print(f"[selective-feature-gen] Creating feature matrix for {len(kmer_features_df)} positions")
        print(f"[selective-feature-gen] Loading training schema from: {training_schema_path}")
    
    # Load training schema to get expected feature names and order
    try:
        schema_info = _load_training_schema(training_schema_path)
        expected_features = schema_info.get('feature_names', [])
        
        if verbose:
            print(f"[selective-feature-gen] Expected features from training: {len(expected_features)}")
            
    except Exception as e:
        if verbose:
            print(f"[selective-feature-gen] Warning: Could not load training schema: {e}")
            print("[selective-feature-gen] Using available features instead")
        expected_features = None
    
    # Merge k-mer features with base predictions
    feature_matrix = kmer_features_df.merge(
        base_predictions_df[['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score']],
        on=['gene_id', 'position'],
        how='inner'
    )
    
    if verbose:
        print(f"[selective-feature-gen] Merged features: {len(feature_matrix)} positions")
    
    # Align with training schema if available
    if expected_features:
        # Add missing features with zeros
        missing_features = []
        for feature in expected_features:
            if feature not in feature_matrix.columns:
                feature_matrix[feature] = 0.0
                missing_features.append(feature)
        
        if missing_features and verbose:
            print(f"[selective-feature-gen] Added {len(missing_features)} missing features with zeros")
        
        # Reorder columns to match training schema
        available_features = [f for f in expected_features if f in feature_matrix.columns]
        extra_cols = ['gene_id', 'position', 'chrom']
        
        # Keep essential columns + training features
        final_columns = extra_cols + available_features
        feature_matrix = feature_matrix[final_columns]
        
        if verbose:
            print(f"[selective-feature-gen] Final feature matrix: {len(feature_matrix)} × {len(available_features)} features")
    
    return feature_matrix


def run_selective_feature_generation(
    uncertain_positions_df: pd.DataFrame,
    base_predictions_df: pd.DataFrame,
    training_schema_path: str,
    kmer_size: int = 3,
    window_size: int = 500,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Complete selective feature generation pipeline.
    
    Parameters
    ----------
    uncertain_positions_df : pd.DataFrame
        DataFrame with uncertain positions that need meta-model predictions
    base_predictions_df : pd.DataFrame
        DataFrame with base model predictions
    training_schema_path : str
        Path to training schema directory
    kmer_size : int, default=3
        K-mer size (should match training data)
    window_size : int, default=500
        Sequence context window size
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Feature matrix ready for meta-model inference
    """
    if verbose:
        print(f"[selective-feature-gen] 🚀 Starting selective feature generation")
        print(f"[selective-feature-gen] Target positions: {len(uncertain_positions_df)}")
        print(f"[selective-feature-gen] K-mer size: {kmer_size}")
        print(f"[selective-feature-gen] Context window: ±{window_size//2}bp")
    
    # Step 1: Extract sequence contexts
    sequence_contexts = extract_sequence_context_for_positions(
        uncertain_positions_df,
        window_size=window_size,
        verbose=verbose
    )
    
    if sequence_contexts.empty:
        if verbose:
            print("[selective-feature-gen] ❌ No sequence contexts extracted")
        return pd.DataFrame()
    
    # Step 2: Generate k-mer features
    kmer_features = generate_kmer_features_for_sequences(
        sequence_contexts,
        kmer_size=kmer_size,
        verbose=verbose
    )
    
    if kmer_features.empty:
        if verbose:
            print("[selective-feature-gen] ❌ No k-mer features generated")
        return pd.DataFrame()
    
    # Step 3: Create final feature matrix
    feature_matrix = create_feature_matrix_for_meta_model(
        kmer_features,
        base_predictions_df,
        training_schema_path,
        verbose=verbose
    )
    
    if verbose:
        print(f"[selective-feature-gen] ✅ Feature generation complete: {len(feature_matrix)} positions ready for meta-model inference")
    
    return feature_matrix


# Global cache for feature enrichment data to avoid repeated loading
_FEATURE_ENRICHMENT_CACHE = {}

def _get_cached_feature_data(data_type: str, target_genes: List[str], verbose: bool = True):
    """
    Get cached feature data or load it if not available.
    This prevents repeated loading of large datasets during feature enrichment.
    """
    cache_key = f"{data_type}_{','.join(sorted(target_genes))}"
    
    if cache_key in _FEATURE_ENRICHMENT_CACHE:
        if verbose:
            print(f"[cache] Using cached {data_type} data for {len(target_genes)} genes")
        return _FEATURE_ENRICHMENT_CACHE[cache_key]
    
    if verbose:
        print(f"[cache] Loading {data_type} data for {len(target_genes)} genes...")
    
    # Load the data (this is the expensive operation we want to cache)
    if data_type == "gene_features":
        from meta_spliceai.splice_engine.meta_models.features.genomic_features import run_genomic_gtf_feature_extraction
        from meta_spliceai.system.genomic_resources import create_systematic_manager
        
        # Create systematic manager to get paths
        sm = create_systematic_manager()
        gtf_file = sm.get_gtf_file()
        
        # Load gene features and filter to target genes
        gene_features = run_genomic_gtf_feature_extraction(gtf_file)
        filtered_features = gene_features[gene_features['gene_id'].isin(target_genes)]
        
        _FEATURE_ENRICHMENT_CACHE[cache_key] = filtered_features
        return filtered_features
    
    return None

# Global cache for genomic features to avoid reloading
_genomic_features_cache = {}

def _run_selective_base_model_pass(
    uncertain_positions_df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run targeted SpliceAI base model pass for specific genes to get probability tensors.
    
    This implements a streamlined version of run_enhanced_splice_prediction_workflow()
    that focuses only on target genes and assumes all genomic datasets are present.
    
    Parameters
    ----------
    uncertain_positions_df : pd.DataFrame
        DataFrame with columns: gene_id, position, chrom
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Base model predictions in format: {gene_id: {donor_prob: array, acceptor_prob: array, neither_prob: array}}
    """
    from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import run_enhanced_splice_prediction_workflow
    from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
    from meta_spliceai.system.genomic_resources import create_systematic_manager
    import tempfile
    import os
    
    if verbose:
        print(f"[selective-base-pass] Running targeted SpliceAI workflow for {len(uncertain_positions_df)} uncertain positions")
    
    # Get unique genes to process
    target_genes = uncertain_positions_df['gene_id'].unique().tolist()
    
    if verbose:
        print(f"[selective-base-pass] Processing {len(target_genes)} genes: {target_genes[:3]}{'...' if len(target_genes) > 3 else ''}")
    
    try:
        # Get genomic resources
        manager = create_systematic_manager()
        
        # Create temporary directory for selective inference
        with tempfile.TemporaryDirectory(prefix="selective_inference_") as temp_dir:
            
            # Create targeted SpliceAI config
            config = SpliceAIConfig(
                gtf_file=str(manager.get_gtf_path()),
                genome_fasta=str(manager.get_fasta_path()),
                eval_dir=temp_dir,
                
                # Enable minimal data preparation for new genes
                do_extract_annotations=False,  # Annotations exist
                do_extract_splice_sites=False,  # Splice sites exist  
                do_extract_sequences=True,      # ENABLE: Need sequences for target genes
                do_find_overlaping_genes=False, # Overlapping genes exist
                
                # Efficient settings for selective inference
                test_mode=False,
                threshold=0.5,
                consensus_window=2,
                error_window=500
            )
            
            if verbose:
                print(f"[selective-base-pass] Running targeted SpliceAI workflow...")
                print(f"[selective-base-pass] Target genes: {len(target_genes)} genes")
                print(f"[selective-base-pass] Temp dir: {temp_dir}")
            
            # Run the enhanced splice prediction workflow with target genes
            workflow_results = run_enhanced_splice_prediction_workflow(
                config=config,
                target_genes=target_genes,
                target_chromosomes=None,  # Let it auto-detect from target genes
                verbosity=1 if verbose else 0,
                no_final_aggregate=False,  # We want the predictions
                action="predict"
            )
            
            if not workflow_results.get('success', False):
                if verbose:
                    print(f"[selective-base-pass] ❌ Targeted workflow failed")
                return {}
            
            # Extract positions DataFrame which contains the probability scores
            positions_df = workflow_results.get('positions')
            
            if positions_df is None or positions_df.height == 0:
                if verbose:
                    print(f"[selective-base-pass] ❌ No positions data generated")
                return {}
            
            if verbose:
                print(f"[selective-base-pass] ✅ Generated {positions_df.height} position predictions")
                
                # Check for the required probability columns
                required_cols = ['donor_score', 'acceptor_score', 'neither_score']
                available_cols = [col for col in required_cols if col in positions_df.columns]
                print(f"[selective-base-pass] Available score columns: {available_cols}")
            
            # CRITICAL: Apply feature enrichment BEFORE conversion to get gene metadata
            if verbose:
                print(f"[selective-base-pass] Applying feature enrichment to add gene metadata...")
            
            try:
                from meta_spliceai.splice_engine.meta_models.features.feature_enrichment import apply_feature_enrichers
                
                # Convert to pandas for enrichment
                positions_pd = positions_df.to_pandas()
                
                # Get unique gene IDs to optimize data loading
                unique_genes = positions_pd['gene_id'].unique().tolist()
                if verbose:
                    print(f"[selective-base-pass] Optimizing for {len(unique_genes)} unique genes: {unique_genes}")
                
                # Apply lightweight feature enrichment - only essential features
                try:
                    if verbose:
                        print(f"[selective-base-pass] Applying lightweight feature enrichment...")
                    
                    # OPTIMIZATION: Only enrich positions that actually need meta-model recalibration
                    # The base model pass already provides all the genomic features we need
                    # We only need to add the missing features for uncertain positions
                    
                    # Filter to only uncertain positions (these are the ones that need meta-model)
                    uncertain_mask = (
                        (positions_pd['donor_score'] >= 0.02) & (positions_pd['donor_score'] < 0.80) |
                        (positions_pd['acceptor_score'] >= 0.02) & (positions_pd['acceptor_score'] < 0.80)
                    )
                    uncertain_positions = positions_pd[uncertain_mask].copy()
                    
                    if verbose:
                        print(f"[selective-base-pass] Found {len(uncertain_positions)} uncertain positions out of {len(positions_pd)} total")
                        print(f"[selective-base-pass] Only enriching uncertain positions for meta-model compatibility")
                    
                    if len(uncertain_positions) > 0:
                        # Apply feature enrichment only to uncertain positions
                        uncertain_enriched = apply_feature_enrichers(
                            uncertain_positions,
                            enrichers=["gene_level", "length_features", "performance_features", "overlap_features", "distance_features"],
                            verbose=verbose if verbose >= 2 else 0
                        )
                        
                        # Merge enriched uncertain positions back with confident positions
                        confident_positions = positions_pd[~uncertain_mask].copy()
                        
                        # Handle different return types from feature enrichment
                        if hasattr(uncertain_enriched, 'to_pandas'):
                            uncertain_enriched = uncertain_enriched.to_pandas()
                        elif not hasattr(uncertain_enriched, 'to_numpy'):
                            uncertain_enriched = pd.DataFrame(uncertain_enriched)
                        
                        # Combine confident and enriched uncertain positions
                        positions_enriched = pd.concat([confident_positions, uncertain_enriched], ignore_index=True)
                    else:
                        # No uncertain positions, use original data
                        positions_enriched = positions_pd
                    
                    # Handle different return types from feature enrichment
                    if hasattr(positions_enriched, 'to_pandas'):
                        # If it's a polars DataFrame, convert to pandas first
                        positions_enriched = positions_enriched.to_pandas()
                    elif hasattr(positions_enriched, 'to_numpy'):
                        # If it's a pandas DataFrame, use as is
                        pass
                    else:
                        # Unknown type, try to convert
                        positions_enriched = pd.DataFrame(positions_enriched)
                    
                    # Convert back to polars
                    positions_df = pl.from_pandas(positions_enriched)
                    
                except Exception as e:
                    if verbose:
                        print(f"[selective-base-pass] ⚠️ Feature enrichment failed: {e}")
                        print(f"[selective-base-pass] Continuing with basic features only")
                    # Keep the original positions_df if enrichment fails
                
                if verbose:
                    print(f"[selective-base-pass] ✅ Feature enrichment completed")
                    print(f"[selective-base-pass] Available columns after enrichment: {list(positions_df.columns)}")
                    
            except Exception as e:
                if verbose:
                    print(f"[selective-base-pass] ⚠️ Feature enrichment failed: {e}")
                    print(f"[selective-base-pass] Continuing with basic features only")
            
            # Convert Polars DataFrame to the expected prediction format
            predictions = _convert_positions_to_predictions(positions_df, target_genes, verbose=verbose)
            
            if verbose:
                print(f"[selective-base-pass] ✅ Converted to prediction format for {len(predictions)} genes")
            
            return predictions
        
    except Exception as e:
        if verbose:
            print(f"[selective-base-pass] ❌ Error in targeted base model pass: {e}")
            import traceback
            traceback.print_exc()
        return {}


def _generate_complete_features_from_predictions(
    base_predictions: Dict[str, Dict[str, np.ndarray]],
    uncertain_positions_df: pd.DataFrame,
    training_schema_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate complete feature matrix from base model predictions using enhanced_process_predictions_with_all_scores.
    
    Parameters
    ----------
    base_predictions : Dict[str, Dict[str, np.ndarray]]
        Raw base model predictions from SpliceAI
    uncertain_positions_df : pd.DataFrame
        Original uncertain positions data
    training_schema_path : str, optional
        Path to training schema for feature alignment
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Complete feature matrix with all derived features (should be 124 features)
    """
    from meta_spliceai.splice_engine.meta_models.core.enhanced_workflow import enhanced_process_predictions_with_all_scores
    from meta_spliceai.system.genomic_resources import create_systematic_manager
    import polars as pl
    
    if verbose:
        print(f"[feature-generation] Generating complete features from base predictions")
    
    try:
        # Load splice site annotations
        manager = create_systematic_manager()
        splice_sites_path = Path(manager.cfg.data_root) / "splice_sites.tsv"
        
        if not splice_sites_path.exists():
            if verbose:
                print(f"[feature-generation] Warning: Splice sites file not found: {splice_sites_path}")
            return pd.DataFrame()
        
        ss_annotations_df = pl.read_csv(
            str(splice_sites_path), 
            separator="\t",
            schema_overrides={"chrom": pl.Utf8}  # Handle chromosome X, Y as strings
        )
        
        # Generate complete features using enhanced processing
        if verbose:
            print(f"[feature-generation] Calling enhanced_process_predictions_with_all_scores...")
        
        # Convert absolute genomic coordinates to relative gene coordinates for each gene
        ss_annotations_relative = ss_annotations_df.clone()
        
        for gene_id in base_predictions.keys():
            if gene_id not in base_predictions:
                continue
                
            gene_data = base_predictions[gene_id]
            gene_start = gene_data.get('gene_start')
            gene_end = gene_data.get('gene_end')
            strand = gene_data.get('strand', '+')
            
            if gene_start is None or gene_end is None:
                if verbose:
                    print(f"[feature-generation] Warning: Missing gene coordinates for {gene_id}, skipping coordinate conversion")
                continue
            
            # Filter annotations for this gene
            gene_mask = ss_annotations_relative['gene_id'] == gene_id
            gene_annotations = ss_annotations_relative.filter(gene_mask)
            
            if gene_annotations.height == 0:
                continue
            
            # Convert absolute coordinates to relative coordinates
            if strand == '+':
                relative_starts = gene_annotations['start'] - gene_start
                relative_ends = gene_annotations['end'] - gene_start
            else:
                relative_starts = gene_end - gene_annotations['end']
                relative_ends = gene_end - gene_annotations['start']
            
            # Update the annotations with relative coordinates
            # Convert to pandas for easier manipulation
            ss_annotations_pd = ss_annotations_relative.to_pandas()
            
            # Update coordinates for this gene
            gene_mask = ss_annotations_pd['gene_id'] == gene_id
            # Ensure proper dtype conversion - convert to numpy array first
            ss_annotations_pd.loc[gene_mask, 'start'] = np.array(relative_starts, dtype=int)
            ss_annotations_pd.loc[gene_mask, 'end'] = np.array(relative_ends, dtype=int)
            
            # Convert back to polars
            ss_annotations_relative = pl.from_pandas(ss_annotations_pd)
        
        error_df, positions_df = enhanced_process_predictions_with_all_scores(
            predictions=base_predictions,
            ss_annotations_df=ss_annotations_relative,
            add_derived_features=True,  # This is crucial - generates all derived features
            verbose=verbose if verbose >= 2 else 0
        )
        
        if positions_df.is_empty():
            if verbose:
                print(f"[feature-generation] ❌ No positions returned from enhanced processing")
            return pd.DataFrame()
        
        # Convert to pandas for consistency with existing workflow
        feature_matrix = positions_df.to_pandas()
        
        # Filter to only uncertain positions FIRST
        uncertain_positions_set = set(zip(uncertain_positions_df['gene_id'], uncertain_positions_df['position']))
        
        if 'gene_id' in feature_matrix.columns and 'position' in feature_matrix.columns:
            mask = feature_matrix.apply(
                lambda row: (row['gene_id'], row['position']) in uncertain_positions_set,
                axis=1
            )
            feature_matrix = feature_matrix[mask]
        
        # CRITICAL: Apply feature enrichment to get essential metadata columns
        if verbose:
            print(f"[feature-generation] Applying feature enrichment to add gene metadata...")
        
        try:
            from meta_spliceai.splice_engine.meta_models.features.feature_enrichment import apply_feature_enrichers
            
            # Apply feature enrichment to get gene_start, gene_end, strand, etc.
            feature_matrix = apply_feature_enrichers(
                feature_matrix,
                enrichers=["gene_level", "length_features", "performance_features", "overlap_features", "distance_features"],  # Full feature set for meta-model compatibility
                verbose=verbose if verbose >= 2 else 0
            )
            
            if verbose:
                print(f"[feature-generation] ✅ Feature enrichment completed")
                print(f"[feature-generation] Available columns after enrichment: {list(feature_matrix.columns)}")
                
        except Exception as e:
            if verbose:
                print(f"[feature-generation] ⚠️ Feature enrichment failed: {e}")
                print(f"[feature-generation] Continuing with basic features only")
        
        if verbose:
            print(f"[feature-generation] ✅ Generated {len(feature_matrix)} features with {feature_matrix.shape[1]} columns")
            if verbose >= 2:
                print(f"[feature-generation] Feature columns: {feature_matrix.columns.tolist()[:20]}...")
        
        return feature_matrix
        
    except Exception as e:
        if verbose:
            print(f"[feature-generation] ❌ Error generating features: {e}")
            import traceback
            traceback.print_exc()
        return pd.DataFrame()


def run_selective_feature_generation_v2(
    uncertain_positions_df: pd.DataFrame,
    training_schema_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Main entry point for selective feature generation using the corrected approach.
    
    This implements the correct strategy based on artifact availability:
    1. Check which uncertain positions exist in analysis_sequences_* artifacts
    2. For existing positions: Extract features directly from artifacts
    3. For missing positions (downsampled TN "holes"): Run selective base-model pass
    4. Combine both to create complete feature matrix
    
    Parameters
    ----------
    uncertain_positions_df : pd.DataFrame
        DataFrame with uncertain positions. Must have columns: gene_id, position, chrom
    training_schema_path : str, optional
        Path to training schema directory
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Complete feature matrix ready for meta-model inference
    """
    if verbose:
        print(f"[selective-feature-gen] 🚀 Starting selective feature generation v3 (corrected)")
        print(f"[selective-feature-gen] Target positions: {len(uncertain_positions_df)}")
        print(f"[selective-feature-gen] Strategy: Check artifacts + selective base-model for missing positions")
    
    try:
        # Step 1: Identify which positions exist in artifacts vs missing ("holes")
        existing_positions, missing_positions = _partition_positions_by_artifact_availability(
            uncertain_positions_df,
            verbose=verbose
        )
        
        feature_matrices = []
        
        # Step 2a: For existing positions, extract features directly from artifacts
        if not existing_positions.empty:
            if verbose:
                print(f"[selective-feature-gen] Extracting features from artifacts for {len(existing_positions)} existing positions")
            
            existing_features = _extract_features_from_artifacts(
                existing_positions,
                training_schema_path=training_schema_path,
                verbose=verbose
            )
            
            if not existing_features.empty:
                feature_matrices.append(existing_features)
        
        # Step 2b: For missing positions, run selective base-model pass
        if not missing_positions.empty:
            if verbose:
                print(f"[selective-feature-gen] Running selective base-model pass for {len(missing_positions)} missing positions (TN holes)")
            
            missing_features = _generate_features_for_missing_positions(
                missing_positions,
                training_schema_path=training_schema_path,
                verbose=verbose
            )
            
            if not missing_features.empty:
                feature_matrices.append(missing_features)
        
        # Step 3: Combine all feature matrices
        if feature_matrices:
            feature_matrix = pd.concat(feature_matrices, ignore_index=True)
            
            if verbose:
                print(f"[selective-feature-gen] ✅ Combined feature generation successful: {feature_matrix.shape}")
                print(f"[selective-feature-gen]    - From artifacts: {len(existing_positions)} positions")
                print(f"[selective-feature-gen]    - From base-model: {len(missing_positions)} positions")
        else:
            if verbose:
                print(f"[selective-feature-gen] ❌ No features generated from any source")
            feature_matrix = pd.DataFrame()
        
        return feature_matrix
        
    except Exception as e:
        if verbose:
            print(f"[selective-feature-gen] ❌ Error in selective feature generation: {e}")
            import traceback
            traceback.print_exc()
        return pd.DataFrame()


def _partition_positions_by_artifact_availability(
    uncertain_positions_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition uncertain positions into those that exist in artifacts vs missing ("holes").
    
    Parameters
    ----------
    uncertain_positions_df : pd.DataFrame
        DataFrame with columns: gene_id, position, chrom
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (existing_positions, missing_positions)
    """
    from meta_spliceai.system.genomic_resources import create_systematic_manager
    import polars as pl
    from pathlib import Path
    
    if verbose:
        print(f"[artifact-check] Checking artifact availability for {len(uncertain_positions_df)} positions")
    
    try:
        # Get analysis sequences directory
        manager = create_systematic_manager()
        eval_dir = Path(manager.cfg.data_root) / "spliceai_eval"
        analysis_sequences_pattern = Path(eval_dir) / "analysis_sequences_*.parquet"
        
        # Find all analysis sequences files
        analysis_files = list(Path(eval_dir).glob("analysis_sequences_*.parquet"))
        
        if not analysis_files:
            if verbose:
                print(f"[artifact-check] No analysis_sequences_*.parquet files found in {eval_dir}")
            # All positions are missing - need base model pass
            return pd.DataFrame(), uncertain_positions_df
        
        if verbose:
            print(f"[artifact-check] Found {len(analysis_files)} analysis_sequences files")
        
        # Load and combine all analysis sequences to check position coverage
        existing_positions_set = set()
        
        for file_path in analysis_files:
            try:
                df = pl.read_parquet(str(file_path))
                if 'gene_id' in df.columns and 'position' in df.columns:
                    for row in df.select(['gene_id', 'position']).to_pandas().itertuples():
                        existing_positions_set.add((row.gene_id, row.position))
            except Exception as e:
                if verbose:
                    print(f"[artifact-check] Warning: Could not read {file_path}: {e}")
                continue
        
        if verbose:
            print(f"[artifact-check] Found {len(existing_positions_set)} existing (gene_id, position) pairs in artifacts")
        
        # Partition uncertain positions
        uncertain_position_pairs = set(zip(uncertain_positions_df['gene_id'], uncertain_positions_df['position']))
        
        existing_pairs = uncertain_position_pairs & existing_positions_set
        missing_pairs = uncertain_position_pairs - existing_positions_set
        
        # Create DataFrames for existing and missing positions
        existing_positions = uncertain_positions_df[
            uncertain_positions_df.apply(lambda row: (row['gene_id'], row['position']) in existing_pairs, axis=1)
        ].copy()
        
        missing_positions = uncertain_positions_df[
            uncertain_positions_df.apply(lambda row: (row['gene_id'], row['position']) in missing_pairs, axis=1)
        ].copy()
        
        if verbose:
            print(f"[artifact-check] ✅ Partitioned positions:")
            print(f"[artifact-check]    - Existing in artifacts: {len(existing_positions)}")
            print(f"[artifact-check]    - Missing (TN holes): {len(missing_positions)}")
        
        return existing_positions, missing_positions
        
    except Exception as e:
        if verbose:
            print(f"[artifact-check] ❌ Error during partitioning: {e}")
        # Return all as missing if we can't check artifacts
        return pd.DataFrame(), uncertain_positions_df


def _extract_features_from_artifacts(
    existing_positions_df: pd.DataFrame,
    training_schema_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract complete features for positions that exist in analysis_sequences_* artifacts.
    
    Parameters
    ----------
    existing_positions_df : pd.DataFrame
        DataFrame with positions that exist in artifacts
    training_schema_path : str, optional
        Path to training schema for feature alignment
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Feature matrix extracted from artifacts
    """
    from meta_spliceai.system.genomic_resources import create_systematic_manager
    import polars as pl
    from pathlib import Path
    
    if verbose:
        print(f"[artifact-extract] Extracting features for {len(existing_positions_df)} positions from artifacts")
    
    try:
        # Get analysis sequences directory
        manager = create_systematic_manager()
        eval_dir = Path(manager.cfg.data_root) / "spliceai_eval"
        
        # Find relevant analysis sequences files
        analysis_files = list(Path(eval_dir).glob("analysis_sequences_*.parquet"))
        
        if not analysis_files:
            if verbose:
                print(f"[artifact-extract] No analysis_sequences files found")
            return pd.DataFrame()
        
        # Create lookup set for target positions
        target_positions = set(zip(existing_positions_df['gene_id'], existing_positions_df['position']))
        
        # Extract matching rows from all analysis files
        feature_rows = []
        
        for file_path in analysis_files:
            try:
                df = pl.read_parquet(str(file_path))
                
                # Filter to target positions
                df_pandas = df.to_pandas()
                if 'gene_id' in df_pandas.columns and 'position' in df_pandas.columns:
                    mask = df_pandas.apply(
                        lambda row: (row['gene_id'], row['position']) in target_positions,
                        axis=1
                    )
                    matching_rows = df_pandas[mask]
                    
                    if len(matching_rows) > 0:
                        feature_rows.append(matching_rows)
                        if verbose:
                            print(f"[artifact-extract] Found {len(matching_rows)} matches in {file_path.name}")
                            
            except Exception as e:
                if verbose:
                    print(f"[artifact-extract] Warning: Could not process {file_path}: {e}")
                continue
        
        if not feature_rows:
            if verbose:
                print(f"[artifact-extract] No matching features found in artifacts")
            return pd.DataFrame()
        
        # Combine all feature rows
        feature_matrix = pd.concat(feature_rows, ignore_index=True)
        
        if verbose:
            print(f"[artifact-extract] ✅ Extracted features: {feature_matrix.shape}")
            
        return feature_matrix
        
    except Exception as e:
        if verbose:
            print(f"[artifact-extract] ❌ Error extracting from artifacts: {e}")
        return pd.DataFrame()


def _generate_features_for_missing_positions(
    missing_positions_df: pd.DataFrame,
    training_schema_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate features for missing positions using selective base-model pass.
    
    This is the original functionality for positions not found in artifacts.
    
    Parameters
    ----------
    missing_positions_df : pd.DataFrame
        DataFrame with positions missing from artifacts (the "holes")
    training_schema_path : str, optional
        Path to training schema
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Feature matrix for missing positions
    """
    if verbose:
        print(f"[missing-features] Generating features for {len(missing_positions_df)} missing positions via base-model pass")
    
    try:
        # Step 1: Run selective base-model pass for missing positions
        base_predictions = _run_selective_base_model_pass(
            missing_positions_df,
            verbose=verbose
        )
        
        if not base_predictions:
            if verbose:
                print(f"[missing-features] ❌ No base model predictions generated")
            return pd.DataFrame()
        
        # Step 2: Generate complete feature matrix
        feature_matrix = _generate_complete_features_from_predictions(
            base_predictions,
            missing_positions_df,
            training_schema_path=training_schema_path,
            verbose=verbose
        )
        
        if verbose:
            if not feature_matrix.empty:
                print(f"[missing-features] ✅ Generated features for missing positions: {feature_matrix.shape}")
            else:
                print(f"[missing-features] ❌ Feature generation failed for missing positions")
        
        return feature_matrix
        
    except Exception as e:
        if verbose:
            print(f"[missing-features] ❌ Error generating features for missing positions: {e}")
            import traceback
            traceback.print_exc()
        return pd.DataFrame()


def _load_gene_sequences_for_genes(
    target_genes: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load gene sequences for target genes from existing sequence files.
    
    Parameters
    ----------
    target_genes : List[str]
        List of gene IDs to load sequences for
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Gene sequences DataFrame compatible with predict_splice_sites_for_genes
    """
    from meta_spliceai.system.genomic_resources import create_systematic_manager
    import polars as pl
    from pathlib import Path
    
    if verbose:
        print(f"[gene-sequences] Loading sequences for {len(target_genes)} genes")
    
    try:
        # Get sequence files directory
        manager = create_systematic_manager()
        eval_dir = Path(manager.cfg.data_root) / "spliceai_eval"
        
        # Find gene sequence files
        sequence_files = list(Path(eval_dir).glob("gene_sequence_*.parquet"))
        
        if not sequence_files:
            if verbose:
                print(f"[gene-sequences] No gene_sequence_*.parquet files found in {eval_dir}")
            return pd.DataFrame()
        
        if verbose:
            print(f"[gene-sequences] Found {len(sequence_files)} sequence files")
        
        # Load and filter to target genes
        gene_sequences = []
        
        for file_path in sequence_files:
            try:
                df = pl.read_parquet(str(file_path))
                
                # Filter to target genes
                if 'gene_id' in df.columns:
                    target_df = df.filter(pl.col('gene_id').is_in(target_genes))
                    if target_df.height > 0:
                        gene_sequences.append(target_df.to_pandas())
                        if verbose:
                            print(f"[gene-sequences] Found {target_df.height} target genes in {file_path.name}")
                            
            except Exception as e:
                if verbose:
                    print(f"[gene-sequences] Warning: Could not read {file_path}: {e}")
                continue
        
        if not gene_sequences:
            if verbose:
                print(f"[gene-sequences] No sequences found for target genes")
            return pd.DataFrame()
        
        # Combine all sequences
        combined_sequences = pd.concat(gene_sequences, ignore_index=True)
        
        if verbose:
            print(f"[gene-sequences] ✅ Loaded sequences: {combined_sequences.shape}")
            
        return combined_sequences
        
    except Exception as e:
        if verbose:
            print(f"[gene-sequences] ❌ Error loading gene sequences: {e}")
        return pd.DataFrame()


def _convert_absolute_to_relative_coordinates(absolute_positions: List[int], gene_start: int, gene_end: int, strand: str) -> List[int]:
    """
    Convert absolute genomic coordinates to relative gene coordinates.
    
    Parameters:
    - absolute_positions: List of absolute genomic positions
    - gene_start: Gene start position in absolute coordinates
    - gene_end: Gene end position in absolute coordinates  
    - strand: Gene strand ('+' or '-')
    
    Returns:
    - List of relative positions (0-based indices into gene sequence)
    """
    relative_positions = []
    
    for abs_pos in absolute_positions:
        if strand == '+':
            # Positive strand: relative = absolute - gene_start
            rel_pos = abs_pos - gene_start
        else:
            # Negative strand: relative = gene_end - absolute
            rel_pos = gene_end - abs_pos
        
        # Ensure position is within gene bounds
        if 0 <= rel_pos < (gene_end - gene_start):
            relative_positions.append(rel_pos)
        else:
            print(f"Warning: Position {abs_pos} outside gene bounds [{gene_start}, {gene_end}], skipping")
    
    return relative_positions


def _convert_positions_to_predictions(
    positions_df: pl.DataFrame,
    target_genes: List[str],
    verbose: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert positions DataFrame to the expected prediction format for enhanced_process_predictions_with_all_scores.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        Positions DataFrame from run_enhanced_splice_prediction_workflow
    target_genes : List[str]
        List of target gene IDs
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Predictions in format: {gene_id: {donor_prob: array, acceptor_prob: array, neither_prob: array}}
    """
    import numpy as np
    
    if verbose:
        print(f"[convert-predictions] Converting positions DataFrame to prediction format")
        print(f"[convert-predictions] Input: {positions_df.height} positions, {len(target_genes)} target genes")
    
    predictions = {}
    
    try:
        # Convert to pandas for easier processing
        positions_pd = positions_df.to_pandas()
        
        # Group by gene_id and extract probability arrays
        for gene_id in target_genes:
            gene_positions = positions_pd[positions_pd['gene_id'] == gene_id].copy()
            
            if len(gene_positions) == 0:
                if verbose:
                    print(f"[convert-predictions] Warning: No positions found for gene {gene_id}")
                continue
            
            # Sort by position to ensure consistent ordering
            gene_positions = gene_positions.sort_values('position')
            
            # Validate required metadata columns
            required_columns = ['gene_start', 'gene_end', 'strand', 'chrom']
            missing_columns = [col for col in required_columns if col not in gene_positions.columns]
            if missing_columns:
                raise ValueError(f"Missing required metadata columns for gene {gene_id}: {missing_columns}")
            
            # Extract probability scores
            predictions[gene_id] = {
                'donor_prob': gene_positions['donor_score'].values,
                'acceptor_prob': gene_positions['acceptor_score'].values,
                'neither_prob': gene_positions['neither_score'].values,
                
                # Include metadata required by enhanced_process_predictions_with_all_scores
                'positions': gene_positions['position'].values,
                'gene_id': gene_id,
                'strand': gene_positions['strand'].iloc[0],
                'chromosome': gene_positions['chrom'].iloc[0],
                'gene_start': gene_positions['gene_start'].iloc[0],
                'gene_end': gene_positions['gene_end'].iloc[0]
            }
            
            if verbose and len(predictions) <= 3:  # Show details for first few genes
                print(f"[convert-predictions] Gene {gene_id}: {len(gene_positions)} positions")
                print(f"[convert-predictions]   - donor_prob range: [{gene_positions['donor_score'].min():.3f}, {gene_positions['donor_score'].max():.3f}]")
                print(f"[convert-predictions]   - acceptor_prob range: [{gene_positions['acceptor_score'].min():.3f}, {gene_positions['acceptor_score'].max():.3f}]")
                print(f"[convert-predictions]   - neither_prob range: [{gene_positions['neither_score'].min():.3f}, {gene_positions['neither_score'].max():.3f}]")
        
        if verbose:
            print(f"[convert-predictions] ✅ Successfully converted {len(predictions)} gene predictions")
        
        return predictions
        
    except Exception as e:
        if verbose:
            print(f"[convert-predictions] ❌ Error converting predictions: {e}")
            import traceback
            traceback.print_exc()
        return {}


def load_feature_manifest(model_path: str) -> List[str]:
    """
    Load the feature manifest from a trained model directory.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model directory (e.g., results/gene_cv_pc_1000_3mers_run_4)
        
    Returns
    -------
    List[str]
        List of feature names used during training
    """
    model_dir = Path(model_path)
    feature_manifest_path = model_dir / "feature_manifest.csv"
    
    if not feature_manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest not found at {feature_manifest_path}")
    
    # Load feature names from CSV
    manifest_df = pd.read_csv(feature_manifest_path)
    feature_names = manifest_df['feature'].tolist()
    
    return feature_names


def harmonize_features_with_training(
    feature_matrix: pd.DataFrame,
    training_features: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Harmonize feature matrix to match exactly the features used during training.
    
    This focuses on k-mer feature consistency while ensuring other features are present.
    K-mer features may vary due to sequence context differences, while probability
    and genomic features should be consistent by design.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Generated feature matrix
    training_features : List[str]
        List of feature names used during training
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Harmonized feature matrix with exactly the same columns as training
    """
    if verbose:
        print(f"[harmonization] Harmonizing features with training data")
        print(f"[harmonization] Training features: {len(training_features)}")
        print(f"[harmonization] Available features: {len(feature_matrix.columns)}")
    
    # Separate features by type
    # Detect k-mer features dynamically using regex (any k-mer pattern like 3mer_, 4mer_, 5mer_, etc.)
    kmer_pattern = re.compile(r'^\d+mer_')  # Matches any k-mer feature (1mer_, 2mer_, 3mer_, etc.)
    kmer_features = [f for f in training_features if kmer_pattern.match(f)]
    probability_features = [f for f in training_features if any(x in f for x in ['score', 'prob', 'ratio', 'diff', 'entropy', 'signal', 'peak', 'surge', 'context', 'weighted'])]
    genomic_features = [f for f in training_features if f not in kmer_features + probability_features]
    
    if verbose:
        print(f"[harmonization] Feature breakdown:")
        print(f"[harmonization]   - K-mer features: {len(kmer_features)}")
        print(f"[harmonization]   - Probability features: {len(probability_features)}")
        print(f"[harmonization]   - Genomic features: {len(genomic_features)}")
    
    # Check for missing features by type
    missing_kmer = set(kmer_features) - set(feature_matrix.columns)
    missing_prob = set(probability_features) - set(feature_matrix.columns)
    missing_genomic = set(genomic_features) - set(feature_matrix.columns)
    
    # Handle missing k-mer features (expected - fill with zeros)
    if missing_kmer:
        if verbose:
            print(f"[harmonization] ⚠️ Missing k-mer features: {len(missing_kmer)} (filling with zeros)")
            print(f"[harmonization] Missing k-mers: {list(missing_kmer)[:10]}...")
        for feature in missing_kmer:
            feature_matrix[feature] = 0.0
    
    # Handle missing probability features (error - should not happen)
    if missing_prob:
        if verbose:
            print(f"[harmonization] ❌ ERROR: Missing probability features: {len(missing_prob)}")
            print(f"[harmonization] Missing: {list(missing_prob)[:10]}...")
        raise ValueError(f"Missing {len(missing_prob)} probability features - this indicates a feature generation error")
    
    # Handle missing genomic features (error - should not happen)
    if missing_genomic:
        if verbose:
            print(f"[harmonization] ❌ ERROR: Missing genomic features: {len(missing_genomic)}")
            print(f"[harmonization] Missing: {list(missing_genomic)[:10]}...")
        raise ValueError(f"Missing {len(missing_genomic)} genomic features - this indicates a feature enrichment error")
    
    # Check for extra features
    extra_features = set(feature_matrix.columns) - set(training_features)
    if extra_features:
        if verbose:
            print(f"[harmonization] ℹ️ Extra features: {len(extra_features)} (will be dropped)")
    
    # Reorder columns to match training exactly
    harmonized_matrix = feature_matrix[training_features]
    
    if verbose:
        print(f"[harmonization] ✅ Harmonized matrix shape: {harmonized_matrix.shape}")
        print(f"[harmonization] ✅ All required features present")
        print(f"[harmonization] ✅ Columns match training: {list(harmonized_matrix.columns) == training_features}")
    
    return harmonized_matrix