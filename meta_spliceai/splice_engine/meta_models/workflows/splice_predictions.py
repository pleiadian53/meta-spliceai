"""
Enhanced SpliceAI prediction workflow for meta models.

This module extends the original SpliceAI prediction workflow to capture
all three probability scores (donor, acceptor, neither) for each position,
enabling more comprehensive analysis for meta models.
"""

import os
from tabnanny import verbose
import time
import random
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import asdict
import pandas as pd
import polars as pl
from tqdm import tqdm
import re

# Import original functionality from appropriate modules
from meta_spliceai.splice_engine.run_spliceai_workflow import (
    predict_splice_sites_for_genes,
    load_chromosome_sequence_streaming,
    adjust_chunk_size,
    subsample_dataframe,
    print_emphasized,
    print_with_indent,
    print_section_separator
)
from meta_spliceai.splice_engine.evaluate_models import evaluate_splice_site_errors
from meta_spliceai.splice_engine.analysis_utils import check_and_subset_invalid_transcript_ids
from meta_spliceai.splice_engine.workflow_utils import align_and_append
from meta_spliceai.splice_engine.utils_df import is_dataframe_empty
from meta_spliceai.splice_engine.sequence_featurizer import extract_analysis_sequences
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.utils_fs import read_splice_sites

# Import meta model utilities
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
# from meta_spliceai.splice_engine.utils_bio import normalize_strand

def process_predictions_with_all_scores(
    predictions: Dict[str, Any],
    ss_annotations_df: pl.DataFrame,
    threshold: float = 0.5,
    consensus_window: int = 2,
    error_window: int = 500,
    return_positions_df: bool = True,
    analyze_position_offsets: bool = False, 
    verbose: int = 0
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Process SpliceAI predictions to include all three probability scores.
    
    This is an enhanced version of evaluate_splice_site_errors that captures
    all three probability scores (donor, acceptor, neither) for each position.
    
    Parameters
    ----------
    predictions : Dict[str, Any]
        Output from predict_splice_sites_for_genes, containing per-nucleotide probabilities
    ss_annotations_df : pl.DataFrame
        Splice site annotations with columns: chrom, start, end, strand, site_type, gene_id, transcript_id
    threshold : float, optional
        Threshold for calling splice sites, by default 0.5
    consensus_window : int, optional
        Window size for consensus calling, by default 2
    error_window : int, optional
        Window size for error analysis, by default 500
    return_positions_df : bool, optional
        Whether to return positions DataFrame, by default True
    analyze_position_offsets : bool, optional
        Whether to analyze position offsets between predictions and annotations, by default False
        
    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        error_df: DataFrame with error analysis
        positions_df: DataFrame with all positions and their three probabilities
    """
    # Preload overlapping gene metadata to avoid loading it twice
    from meta_spliceai.splice_engine.splice_error_analyzer import SpliceAnalyzer
    sa = SpliceAnalyzer()
    overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()
    
    # Call the original function first to get the error analysis
    # Pass all required parameters to evaluate_splice_site_errors
    # The function might need additional parameters that we don't provide by default
    error_df, positions_df = evaluate_splice_site_errors(
        ss_annotations_df,  # First parameter is annotations_df
        predictions,     # Second parameter is pred_results
        threshold=threshold,
        consensus_window=consensus_window,
        error_window=error_window,
        return_positions_df=True,  # Explicitly request positions_df
        collect_tn=True,           # Collect true negatives
        predicted_delta_correction=True,  # Apply delta correction for better accuracy
        verbose=verbose,                 # Reduce verbosity in function call by default
        overlapping_genes_metadata=overlapping_genes_metadata  # Pass preloaded metadata
    )

    # NOTE: positions_df is a dataframe with the following columns: 
    #       - pred_type: 'TP', 'FP', 'FN', 'TN' and optionally 'PRED'
    #       - position: a strand-dependent relative position of the predicted splice site
    #       - score: the probability of the predicted splice site

    if not return_positions_df:
        return error_df, None
    
    # Enhance the positions DataFrame by adding all three probabilities
    enhanced_positions = []
    
    for gene_id, gene_preds in predictions.items():
        if 'positions' not in gene_preds or len(gene_preds['positions']) == 0:
            continue
            
        # Get strand for this gene
        strand = gene_preds.get('strand', '+')
            
        for pos_idx, position in enumerate(gene_preds['positions']):
            pos_int = int(position)  # predicted splice site position
            
            # Get probability scores for this position
            donor_score = gene_preds['donor_prob'][pos_idx]
            acceptor_score = gene_preds['acceptor_prob'][pos_idx]
            neither_score = gene_preds['neither_prob'][pos_idx]
            
            # Find the matching row in positions_df using a flexible window approach
            # This accounts for small systematic differences between annotations
            match_filter = (
                (positions_df['gene_id'] == gene_id) & 
                (positions_df['position'] >= pos_int - consensus_window) &
                (positions_df['position'] <= pos_int + consensus_window)
            )
                
            matching_rows = positions_df.filter(match_filter)
            
            # If we don't find a match, let's add this entry anyway with basic info
            # This ensures we capture predictions even if they don't match annotations
            if matching_rows.height == 0:
                # Get all three probability scores
                donor_score = gene_preds['donor_prob'][pos_idx]
                acceptor_score = gene_preds['acceptor_prob'][pos_idx]
                neither_score = gene_preds['neither_prob'][pos_idx]
                
                # Only include entries with significant scores
                max_score = max(donor_score, acceptor_score)
                if max_score >= threshold:
                    # Determine splice type based on highest score
                    splice_type = 'donor' if donor_score > acceptor_score else 'acceptor'
                    
                    # Find all transcripts for this gene to preserve transcript diversity
                    gene_transcripts = ss_annotations_df.filter(pl.col('gene_id') == gene_id).select('transcript_id').unique()
                    
                    # If no transcripts found, use a placeholder
                    if gene_transcripts.height == 0:
                        # Create a proper transcript ID placeholder
                        placeholder_transcript = f"ENST_{gene_id.replace('ENSG', '')}_PRED"
                        
                        # Add an entry for this position even without annotation match
                        enhanced_positions.append({
                            'gene_id': gene_id,
                            'transcript_id': placeholder_transcript,
                            'chrom': gene_preds.get('seqname', ''),
                            'position': int(position),
                            'pred_type': 'PRED',  # Mark as prediction-only (no annotation match)
                            'strand': strand,
                            'donor_score': donor_score,
                            'acceptor_score': acceptor_score,
                            'neither_score': neither_score,
                            'splice_type': splice_type
                        })
                    else:
                        # Add an entry for each transcript of this gene to preserve transcript diversity
                        for transcript_id in gene_transcripts.select('transcript_id').to_series():
                            enhanced_positions.append({
                                'gene_id': gene_id,
                                'transcript_id': transcript_id,
                                'chrom': gene_preds.get('seqname', ''),
                                'position': int(position),
                                'pred_type': 'PRED',  # Mark as prediction-only (no annotation match)
                                'strand': strand,
                                'donor_score': donor_score,
                                'acceptor_score': acceptor_score,
                                'neither_score': neither_score,
                                'splice_type': splice_type
                            })
            else:
                # Process matching rows as before
                for row_idx in range(matching_rows.height):
                    row = matching_rows.row(row_idx)
                    pred_type = row[positions_df.columns.index('pred_type')]
                    splice_type = row[positions_df.columns.index('splice_type')]
                    transcript_id = row[positions_df.columns.index('transcript_id')]
                    
                    # Get all three probability scores
                    donor_score = gene_preds['donor_prob'][pos_idx]
                    acceptor_score = gene_preds['acceptor_prob'][pos_idx]
                    neither_score = gene_preds['neither_prob'][pos_idx]
                    
                    # Create an enhanced position entry
                    enhanced_positions.append({
                        'gene_id': gene_id,
                        'transcript_id': transcript_id,
                        'chrom': gene_preds.get('seqname', ''),
                        'position': int(position),
                        'pred_type': pred_type,
                        'strand': strand,
                        'donor_score': donor_score,
                        'acceptor_score': acceptor_score,
                        'neither_score': neither_score,
                        'splice_type': splice_type
                    })
    
    # Create enhanced positions DataFrame
    if enhanced_positions:
        enhanced_positions_df = pl.DataFrame(enhanced_positions)
    else:
        # Create empty DataFrame with the correct schema when no matches are found
        enhanced_positions_df = pl.DataFrame(schema={
            'gene_id': pl.Utf8,
            'transcript_id': pl.Utf8,
            'chrom': pl.Utf8,
            'position': pl.Int64,
            'pred_type': pl.Utf8,
            'strand': pl.Utf8,
            'donor_score': pl.Float64,
            'acceptor_score': pl.Float64,
            'neither_score': pl.Float64,
            'splice_type': pl.Utf8
        })
        print(f"[warning] No matching positions found between predictions and splice site annotations.")
        print(f"[debug] Number of gene IDs in predictions: {len(predictions)}")
        print(f"[debug] Number of positions in positions_df: {positions_df.height}")
        # Print sample of gene IDs for debugging
        if len(predictions) > 0:
            print(f"[debug] Sample gene IDs in predictions: {list(predictions.keys())[:5]}")
        if positions_df.height > 0:
            print(f"[debug] Sample gene IDs in positions_df: {positions_df.filter(pl.col('gene_id').is_not_null()).select('gene_id').unique().head(5)}")
    
    if analyze_position_offsets:
        print_emphasized("\nAnalyzing position offsets between predictions and annotations...")
        
        # Analyze position offsets between predictions and annotations
        offset_df = pl.DataFrame({
            'gene_id': [],
            'transcript_id': [],
            'chrom': [],
            'position': [],
            'pred_type': [],
            'strand': [],
            'splice_type': [],
            'offset': []
        })
        
        # Track how many positions were analyzed and matched
        total_positions = 0
        matched_positions = 0
        
        for gene_id, gene_preds in predictions.items():
            if 'positions' not in gene_preds or len(gene_preds['positions']) == 0:
                continue
                
            # Get strand for this gene
            strand = gene_preds.get('strand', '+')
            seqname = gene_preds.get('seqname', '')
            
            # Track positions for this gene
            gene_positions = len(gene_preds['positions'])
            gene_matches = 0
                
            for pos_idx, position in enumerate(gene_preds['positions']):
                pos_int = int(position)
                total_positions += 1
                
                # Get probability scores for this position
                donor_score = gene_preds['donor_prob'][pos_idx]
                acceptor_score = gene_preds['acceptor_prob'][pos_idx]
                
                # Only analyze positions with significant probability
                max_score = max(donor_score, acceptor_score)
                if max_score < threshold:
                    continue
                    
                # Determine splice type based on highest score
                splice_type = 'donor' if donor_score > acceptor_score else 'acceptor'
                
                # Find the matching row in positions_df using a flexible window approach
                # This accounts for small systematic differences between annotations
                # First try exact matching on gene_id, then find positions within window
                annotation_matches = ss_annotations_df.filter(
                    (pl.col('gene_id') == gene_id) &
                    (pl.col('site_type') == splice_type) &
                    (pl.col('strand') == strand) &
                    (pl.col('start') >= pos_int - consensus_window * 2) &  # Use wider window for analysis
                    (pl.col('start') <= pos_int + consensus_window * 2)
                )
                
                if annotation_matches.height > 0:
                    gene_matches += 1
                    matched_positions += 1
                    
                    for row_idx in range(annotation_matches.height):
                        row = annotation_matches.row(row_idx)
                        # Adjust column indices for ss_annotations_df schema
                        annot_cols = annotation_matches.columns
                        transcript_id = row[annot_cols.index('transcript_id')] if 'transcript_id' in annot_cols else 'unknown'
                        site_pos = row[annot_cols.index('start')]
                        
                        # Calculate position offset - positive means prediction is downstream of annotation
                        offset = pos_int - site_pos
                        
                        # Add an entry for this position offset
                        offset_df = offset_df.vstack({
                            'gene_id': [gene_id],
                            'transcript_id': [transcript_id],
                            'chrom': [seqname],
                            'position': [pos_int],
                            'pred_type': ['MATCH'],
                            'strand': [strand],
                            'splice_type': [splice_type],
                            'offset': [offset]
                        })
            
            # Print per-gene matching statistics if verbosity is high enough
            if gene_positions > 0:
                match_rate = (gene_matches / gene_positions) * 100
                if gene_matches == 0:
                    print(f"[warn] Gene {gene_id} ({seqname}): No position matches found out of {gene_positions} positions")
        
        # Print summary statistics for position offsets
        print(f"\n[info] Position offset analysis summary:")
        print(f"  Total positions analyzed: {total_positions}")
        match_percentage = (matched_positions/total_positions*100) if total_positions > 0 else 0
        print(f"  Positions with matches: {matched_positions} ({match_percentage:.1f}%)")
        
        if offset_df.height > 0:
            print(f"[info] Position offset statistics:")
            print(f"  Mean offset: {offset_df['offset'].mean()}")
            print(f"  Median offset: {offset_df['offset'].median()}")
            print(f"  Standard deviation: {offset_df['offset'].std()}")
            
            # Print offset distributions for different splice types and strands
            for splice_type in ['donor', 'acceptor']:
                for strand in ['+', '-']:
                    filter_df = offset_df.filter((offset_df['splice_type'] == splice_type) & (offset_df['strand'] == strand))
                    if filter_df.height > 0:
                        print(f"[info] Offset distribution for {splice_type} sites on {strand} strand:")
                        print(f"  Mean offset: {filter_df['offset'].mean()}")
                        print(f"  Median offset: {filter_df['offset'].median()}")
                        print(f"  Standard deviation: {filter_df['offset'].std()}")
                        
                        # Count offsets within specific ranges
                        exact_match = filter_df.filter(pl.col('offset') == 0).height
                        within_1nt = filter_df.filter((pl.col('offset') >= -1) & (pl.col('offset') <= 1)).height
                        within_2nt = filter_df.filter((pl.col('offset') >= -2) & (pl.col('offset') <= 2)).height
                        
                        total = filter_df.height
                        print(f"  Exact matches: {exact_match} ({exact_match/total*100:.1f}%)")
                        print(f"  Within ±1nt: {within_1nt} ({within_1nt/total*100:.1f}%)")
                        print(f"  Within ±2nt: {within_2nt} ({within_2nt/total*100:.1f}%)")
        else:
            print(f"[warn] No position offsets found - couldn't match predicted and annotated positions")
            print(f"[debug] This could indicate a problem with strand information or position coordinates")
            # Provide sample data for debugging
            if total_positions > 0:
                print(f"\n[debug] Sample predicted positions:")
                for gene_id, gene_preds in list(predictions.items())[:2]:  # First two genes
                    if 'positions' in gene_preds and len(gene_preds['positions']) > 0:
                        print(f"  Gene {gene_id}: {gene_preds['positions'][:5]} (strand: {gene_preds.get('strand', 'unknown')})")
                
                print(f"\n[debug] Sample annotation positions:")
                sample_annot = ss_annotations_df.head(5) if ss_annotations_df.height > 0 else "No annotations"
                print(f"  {sample_annot}")
    
    return error_df, enhanced_positions_df


def run_enhanced_splice_prediction_workflow(
    config: Optional[SpliceAIConfig] = None, 
    target_genes: Optional[List[str]] = None,
    verbosity: int = 1,
    **kwargs
) -> Dict[str, pl.DataFrame]:
    """
    Run the enhanced SpliceAI prediction workflow, with improvements:
    - Supports multi-class probability scores (donor, acceptor, neither)
    - Preserves transcript ID mapping when multiple transcripts share splice sites
    - Allows filtering to specific target genes for focused analysis
    
    Parameters:
    ----------
    config : SpliceAIConfig, optional
        Configuration for SpliceAI workflow. If None, uses default config.
    target_genes : List[str], optional
        List of gene symbols or IDs to focus on. If provided, will only process these genes.
    verbosity : int, default=1
        Controls output verbosity:
        0 = minimal (errors and final summary only)
        1 = normal (chromosome progress and important warnings)
        2 = detailed (all chunk processing, memory usage, etc.)
    **kwargs : dict
        Additional parameters for SpliceAI workflow.
        
    Returns
    -------
    Dict[str, pl.DataFrame]
        Dictionary containing various DataFrames produced by the workflow:
        - 'error_analysis': Error analysis DataFrame with TP/FP/FN classifications
           Columns ['error_type', 'gene_id', 'position', 'splice_type', 'strand', 'transcript_id', 'window_end', 'window_start']

        - 'positions': Enhanced positions DataFrame with all three probabilities
           Columns: ['gene_id', 'transcript_id', 'error_typ', 'position', 'splice_type', 'strand', 'window_start', 'window_end', 'chrom']
        
        - 'analysis_sequences': Analysis sequences DataFrame for downstream analysis
           Columns: ['gene_id', 'transcript_id', 'chrom', 'position', 'strand', 'sequence', 'splice_type', 'error_type', 'window_start', 'window_end']
        
    Examples
    --------
    # Process all genes (slow)
    results = run_enhanced_splice_prediction_workflow()
    
    # Process only specific genes for testing (fast)
    results = run_enhanced_splice_prediction_workflow(
        target_genes=['STMN2', 'UNC13A', 'ENSG00000104435']
    )
    
    # Process in test mode with minimal genes
    results = run_enhanced_splice_prediction_workflow(test_mode=True)
    """
    # Create a configuration object if not provided
    if config is None:
        config = SpliceAIConfig(**kwargs)
        
    # Mix in any overrides from kwargs (using the pattern from the memory)
    config_dict = asdict(config)
    for param_name, param_value in kwargs.items():
        if param_name in config_dict:
            config_dict[param_name] = param_value
    config = SpliceAIConfig(**config_dict)
    
    # Extract configuration parameters
    gtf_file = config.gtf_file
    genome_fasta = config.genome_fasta
    eval_dir = config.eval_dir
    output_subdir = config.output_subdir
    format = config.format
    separator = config.separator
    threshold = config.threshold
    consensus_window = config.consensus_window
    error_window = config.error_window
    test_mode = config.test_mode
    chromosomes = config.chromosomes
    
    # Data preparation switches
    do_extract_annotations = config.do_extract_annotations
    do_extract_splice_sites = config.do_extract_splice_sites
    do_extract_sequences = config.do_extract_sequences
    do_find_overlaping_genes = config.do_find_overlaping_genes
    use_precomputed_overlapping_genes = config.use_precomputed_overlapping_genes
    
    if verbosity >= 1:
        print_emphasized("[workflow] Starting enhanced SpliceAI prediction workflow...")
        print_with_indent(f"[params] threshold: {threshold}, consensus_window: {consensus_window}, error_window: {error_window}", indent_level=1)
        print_with_indent(f"[params] output_directory: {os.path.join(eval_dir, output_subdir)}", indent_level=1)
        print_with_indent(f"[params] data preparation: extract_annotations={do_extract_annotations}, extract_splice_sites={do_extract_splice_sites}, extract_sequences={do_extract_sequences}", indent_level=1)
    
    # Convert path strings to absolute paths if they're not already
    if not os.path.isabs(eval_dir):
        eval_dir = os.path.abspath(eval_dir)
    
    # Set up handlers
    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)
    data_handler = MetaModelDataHandler(eval_dir, separator=separator)
    
    # Create output directory if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    
    # Define local directory for intermediates (similar to original workflow)
    local_dir = os.path.dirname(eval_dir)
    
    # 1. Extract gene annotations (optional)
    if do_extract_annotations:
        from meta_spliceai.splice_engine.extract_genomic_features import extract_annotations
        if verbosity >= 1:
            print_emphasized("[action] Extract gene annotations...")
        
        db_file = os.path.join(local_dir, 'annotations.db')
        output_file = os.path.join(local_dir, 'annotations_all_transcripts.tsv')
        extract_annotations(gtf_file, db_file=db_file, output_file=output_file, sep=separator)
        
        # Test loading annotations
        annotation_file_path = os.path.join(local_dir, "annotations_all_transcripts.csv")
        if os.path.exists(annotation_file_path):
            if verbosity >= 1:
                print("[info] Loading transcript annotations (exon, CDS, 5'UTR, and 3'UTR annotations):")
            annot_df = pd.read_csv(annotation_file_path, sep=',')
            if verbosity >= 1:
                print(annot_df.head())
    
    # 2. Extract splice site annotations (optional)
    if verbosity >= 1:
        print_emphasized("[data] Loading splice site annotations...")
    
    # Default file extension format for splice site annotations
    ss_format = 'tsv'  # Maintain TSV format for splice site annotations
    
    # Splice site annotations should be in TSV format by default
    splice_sites_file_path = os.path.join(local_dir, f"splice_sites.{ss_format}")
    
    if do_extract_splice_sites:
        from meta_spliceai.splice_engine.extract_genomic_features import extract_splice_sites_workflow
        if verbosity >= 1:
            print_emphasized("[action] Extract splice sites for all transcripts...")
        
        splice_sites_file_path = extract_splice_sites_workflow(
            data_prefix=local_dir, 
            gtf_file=gtf_file, 
            consensus_window=consensus_window
        )
    
    # Load the splice sites for this genome
    if verbosity >= 1:
        print_emphasized("[data] Loading splice site annotations file")
        
    if not os.path.exists(splice_sites_file_path):
        print(f"Error: Splice sites file not found at {splice_sites_file_path}")
        print("Please set do_extract_splice_sites=True or provide a valid splice sites file.")
        return {}
    
    ss_annotations_df = read_splice_sites(splice_sites_file_path, separator=separator, dtypes=None)
    
    # Use the existing utility function from utils_df.py
    if ss_annotations_df is None or is_dataframe_empty(ss_annotations_df):
        print("Error: Failed to load splice site annotations.")
        return {}
    
    if verbosity >= 1:
        print(f"[info] Splice-site dataframe: shape={ss_annotations_df.shape}")
        print(ss_annotations_df.head())
    
    # 3. Extract DNA sequences (optional) - for gene mode
    if verbosity >= 1:
        print_emphasized("[action] Extract DNA sequences")
    mode = 'gene'  # Use gene mode for SpliceAI predictions
    seq_format = 'parquet'  # Set separate format for sequence files (parquet is more efficient)
    seq_type = 'full'  # Use full gene sequences
    
    if mode == 'gene':
        from meta_spliceai.splice_engine.extract_genomic_features import gene_sequence_retrieval_workflow
        
        # Use consistent file naming with original workflow (no chromosome in filename)
        if seq_type == 'minmax':
            output_file = f"gene_sequence_minmax.{seq_format}"
        else:
            output_file = f"gene_sequence.{seq_format}" 
        
        seq_df_path = os.path.join(local_dir, output_file)
        
        if do_extract_sequences:
            if verbosity >= 1:
                print_with_indent(f"Extracting {seq_type} gene sequences...", indent_level=1)
            
            # Check if we should extract for all chromosomes or just specific ones
            if chromosomes and not test_mode:
                # Extract sequences for each chromosome to match load_chromosome_sequence_streaming expectations
                for chrom in chromosomes:
                    chrom_output = os.path.join(local_dir, f"gene_sequence_{chrom}.{seq_format}")
                    if not os.path.exists(chrom_output):
                        if verbosity >= 1:
                            print_with_indent(f"Extracting {seq_type} gene sequences for chromosome {chrom}...", indent_level=2)
                        gene_sequence_retrieval_workflow(
                            gtf_file, genome_fasta, gene_tx_map=None, 
                            output_file=chrom_output, mode=seq_type,
                            chromosomes=[chrom]
                        )
                    else:
                        if verbosity >= 1:
                            print_with_indent(f"Using existing sequences for chromosome {chrom}", indent_level=2)
            else:
                # Default case: extract all sequences
                gene_sequence_retrieval_workflow(
                    gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path, mode=seq_type
                )
    else:
        from meta_spliceai.splice_engine.extract_genomic_features import transcript_sequence_retrieval_workflow
        
        # No seq_type for transcript mode
        seq_df_path = os.path.join(local_dir, f"tx_sequence.{seq_format}")
        
        if do_extract_sequences:
            if verbosity >= 1:
                print_with_indent("Extracting transcript sequences...", indent_level=1)
            transcript_sequence_retrieval_workflow(
                gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path
            )
    
    # 4. Handle overlapping genes (optional)
    # First check if a global overlapping genes file exists
    global_overlapping_file = "data/ensembl/overlapping_gene_counts.tsv"
    local_overlapping_file = os.path.join(local_dir, "overlapping_gene_counts.tsv")
    
    if use_precomputed_overlapping_genes:
        # Check for existing overlapping genes file in order of preference
        if os.path.exists(global_overlapping_file):
            # Use the global precomputed file
            overlapping_gene_file_path = global_overlapping_file
            if verbosity >= 1:
                print(f"[info] Using existing overlapping genes file: {overlapping_gene_file_path}")
        elif os.path.exists(local_overlapping_file):
            # Use local file if it exists
            overlapping_gene_file_path = local_overlapping_file
            if verbosity >= 1:
                print(f"[info] Using local overlapping genes file: {overlapping_gene_file_path}")
        elif do_find_overlaping_genes:
            # Only compute if explicitly requested and no existing file found
            from meta_spliceai.splice_engine.overlapping_gene_mapper import precompute_overlapping_genes
            if verbosity >= 1:
                print_emphasized("[action] Precomputing overlapping genes...")
            
            filter_valid_splice_sites = kwargs.get('filter_valid_splice_sites', True)
            min_exons = kwargs.get('min_exons', 2)
            
            overlapping_gene_file_path = local_overlapping_file
            precompute_overlapping_genes(
                gtf_file=gtf_file,
                output_file=overlapping_gene_file_path,
                filter_valid_splice_sites=filter_valid_splice_sites,
                min_exons=min_exons
            )
        else:
            overlapping_gene_file_path = None
            if verbosity >= 1:
                print(f"[info] No overlapping genes file found and do_find_overlaping_genes=False, skipping")
            
        # Load overlapping gene information if file exists
        if overlapping_gene_file_path and os.path.exists(overlapping_gene_file_path):
            overlapping_df = pd.read_csv(overlapping_gene_file_path, sep=separator)
            if verbosity >= 1:
                print(f"[info] Overlapping genes dataframe: shape={overlapping_df.shape}")
                print(overlapping_df.head())

    ###########################################

    # 5. Set up chromosomes to process
    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y'] 
        # or add 'MT' for mitochondrial DNA
    
    # If targeting specific genes, try to optimize which chromosomes to process
    if target_genes is not None and not test_mode and (chromosomes is None or len(chromosomes) > 5):
        # Only attempt optimization if we're looking at many chromosomes
        if verbosity >= 1:
            print_emphasized("[optimize] Determining which chromosomes contain target genes...")
        
        # First check if we already have gene-to-chromosome mapping info
        gene_chrom_file = os.path.join(local_dir, "gene_chromosome_map.tsv")
        if os.path.exists(gene_chrom_file):
            # Load existing mapping
            if verbosity >= 1:
                print_with_indent("Loading existing gene-chromosome mapping", indent_level=1)
            gene_chrom_df = pd.read_csv(gene_chrom_file, sep='\t')
            gene_chrom_map = dict(zip(gene_chrom_df['gene_id'], gene_chrom_df['chromosome']))
            gene_name_map = dict(zip(gene_chrom_df['gene_name'], gene_chrom_df['chromosome']))
        else:
            # Create mapping from annotations
            if verbosity >= 1:
                print_with_indent("Creating gene-chromosome mapping from annotations", indent_level=1)
            
            # Quick scan of gene annotations to build gene-to-chromosome map
            gene_chrom_map = {}
            gene_name_map = {}
            
            # Use GTF file to determine chromosomes for target genes
            with open(gtf_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 9:
                        continue
                    
                    if parts[2] == "gene":
                        chrom = parts[0]
                        attributes = parts[8]
                        
                        # Extract gene_id
                        gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
                        if gene_id_match:
                            gene_id = gene_id_match.group(1)
                            gene_chrom_map[gene_id] = chrom
                        
                        # Extract gene_name if available
                        gene_name_match = re.search(r'gene_name "([^"]+)"', attributes)
                        if gene_name_match:
                            gene_name = gene_name_match.group(1)
                            gene_name_map[gene_name] = chrom
            
            # Save mapping for future use
            gene_chrom_df = pd.DataFrame({
                'gene_id': list(gene_chrom_map.keys()),
                'gene_name': [gene_name_map.get(gene_id, "") for gene_id in gene_chrom_map.keys()],
                'chromosome': list(gene_chrom_map.values())
            })
            gene_chrom_df.to_csv(gene_chrom_file, sep='\t', index=False)
        
        # Find which chromosomes contain our target genes
        target_chromosomes = set()
        for gene in target_genes:
            if gene in gene_chrom_map:
                target_chromosomes.add(gene_chrom_map[gene])
            elif gene in gene_name_map:
                target_chromosomes.add(gene_name_map[gene])
        
        if target_chromosomes:
            # Filter to only process chromosomes containing target genes
            chromosomes = sorted(list(target_chromosomes))
            if verbosity >= 1:
                print_emphasized(f"[optimize] Filtered to {len(chromosomes)} chromosome(s) containing target genes: {', '.join(chromosomes)}")
        else:
            if verbosity >= 1:
                print_emphasized("[warn] Could not determine which chromosomes contain target genes")
                print_with_indent("Will process all specified chromosomes", indent_level=1)
    
    if test_mode:
        if verbosity >= 1:
            print("[mode] Test mode enabled - using subset of chromosomes")
        # If test_mode, use either the explicitly specified chromosomes or just chromosome 21
        if not chromosomes:
            chromosomes = ['21']
        if verbosity >= 1:
            print_with_indent(f"Processing chromosomes: {', '.join(chromosomes)}", indent_level=1)
    
    # 6. Load SpliceAI models
    if verbosity >= 1:
        print_emphasized("[action] Loading SpliceAI models...")
    try:
        from keras.models import load_model
        from meta_spliceai.splice_engine.run_spliceai_workflow import load_spliceai_models
        
        models = load_spliceai_models()
        if verbosity >= 1:
            print_with_indent(f"[info] SpliceAI models loaded successfully (n={len(models)})", indent_level=1)
    except Exception as e:
        if verbosity >= 1:
            print_emphasized(f"[error] Failed to load SpliceAI models: {str(e)}")
        raise
    
    # 7. Initialize result containers
    performance_files = []
    error_analysis_files = []
    splice_positions_files = []
    analysis_sequence_files = []
    
    full_performance_df = None
    full_delta_df = None
    full_error_df = None
    full_positions_df = None
    
    # Initialize DataFrames for collecting results
    full_error_df = pl.DataFrame()
    full_positions_df = pl.DataFrame()
    full_analysis_sequence_df = pl.DataFrame()
    
    # Statistics tracking
    total_genes_processed = 0
    total_chunk_time = 0
    start_time = time.time()
    
    # Process each chromosome
    for chr in tqdm(chromosomes, desc="Processing chromosomes"):
        chr = str(chr)
        
        # Load sequence data for this chromosome using the original helper function
        # This correctly handles chromosome-specific files (e.g., gene_sequence_19.tsv)
        if verbosity >= 1:
            print_emphasized(f"[info] Loading sequences for chromosome {chr}")
        try:
            lazy_seq_df = load_chromosome_sequence_streaming(seq_df_path, chr, format=seq_format)
        except FileNotFoundError as e:
            if do_extract_sequences:
                print(f"[error] Sequence file for chromosome {chr} not found: {e}")
                print("[info] You may need to run the extraction step separately for each chromosome.")
                print("       Check that the gene sequence files follow the pattern: gene_sequence_CHROM.tsv")
            else:
                print(f"[error] Sequence file for chromosome {chr} not found and extraction is disabled.")
                print("       Enable sequence extraction with do_extract_sequences=True or")
                print("       provide pre-extracted sequence files.")
            continue
        
        # Initialize chunk processing
        default_chunk_size = 500 if not test_mode else 50
        chunk_size = default_chunk_size
        seq_len_avg = 50000  # Assume an average sequence length
        num_genes = lazy_seq_df.select(pl.col('gene_id').n_unique()).collect().item()
        
        if verbosity >= 1:
            print_emphasized(f"[info] Processing {num_genes} genes from chromosome {chr}")
        
        # Process genes in chunks
        for chunk_start in range(0, num_genes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_genes)
            chunk_start_time = time.time()
            
            # Adjust chunk size based on memory
            previous_chunk_size = chunk_size
            chunk_size = adjust_chunk_size(chunk_size, seq_len_avg, 
                            verbose=1 if chunk_start > 0 and chunk_size != default_chunk_size else 0)
            
            # Print message only if chunk size changed
            if verbosity >= 1 and chunk_size != previous_chunk_size:
                print(f"[info] Chunk size adjusted from {previous_chunk_size} to {chunk_size}")
            
            # Load current chunk
            seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()
            if verbosity >= 2:
                print_emphasized(f"[info] Processing genes {chunk_start + 1} to {chunk_end} out of {num_genes}")
            
            # Filter genes if target_genes is provided
            if target_genes is not None:
                original_count = seq_chunk.height
                seq_chunk = seq_chunk.filter(
                    pl.col('gene_id').is_in(target_genes) | 
                    pl.col('gene_name').is_in(target_genes)
                )
                filtered_count = seq_chunk.height
                
                if filtered_count == 0:
                    if verbosity >= 2:
                        print_emphasized(f"[warn] No genes matched the target_genes filter. Skipping chunk.")
                    continue
                    
                if verbosity >= 1:
                    print_emphasized(f"[info] Filtered from {original_count} to {filtered_count} genes based on target_genes")
            
            # Predict splice sites
            if verbosity >= 1:
                print_emphasized("[pred] Running SpliceAI predictions...")
            predictions = predict_splice_sites_for_genes(
                seq_chunk, models=models, context=10000, efficient_output=True
            )
            
            # Enhanced evaluation with all probabilities
            if verbosity >= 1:
                print_emphasized("[eval] Evaluating predictions with enhanced scoring...")
            error_df_chunk, positions_df_chunk = process_predictions_with_all_scores(
                predictions,
                ss_annotations_df,
                threshold=threshold,
                consensus_window=consensus_window,
                error_window=error_window,
                return_positions_df=True,
                analyze_position_offsets=True, 
                verbose = verbosity >= 1
            )
            
            # Check if positions DataFrame has the expected columns before proceeding
            required_columns = ['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score']
            missing_columns = [col for col in required_columns if col not in positions_df_chunk.columns]
            
            if missing_columns:
                if verbosity >= 1:
                    print_emphasized(f"[warning] Positions DataFrame missing required columns: {missing_columns}")
                    print(f"[debug] Available columns: {positions_df_chunk.columns}")
                    if positions_df_chunk.height == 0:
                        print(f"[debug] DataFrame is empty (height=0), likely no predictions were found")
                    else:
                        print(f"[debug] First row: {positions_df_chunk.head(1)}")
                continue  # Skip processing this chunk
            
            # Verify that the DataFrame isn't empty
            if positions_df_chunk.height == 0:
                if verbosity >= 1:
                    print_emphasized(f"[warning] No splice site predictions found for this chunk")
                continue  # Skip processing this chunk
            
            # Add chromosome column
            positions_df_chunk = positions_df_chunk.with_columns(pl.lit(chr).alias('chrom'))
            
            # If this is the first chunk for our target genes, display validation info
            if target_genes is not None and (full_positions_df is None or full_positions_df.height == 0) and positions_df_chunk.height > 0:
                # Compare a subset of columns between original positions_df and enhanced
                if verbosity >= 1:
                    print_emphasized("\n===== VALIDATION: Original vs Enhanced DataFrame =====")
                    print(f"Original positions_df columns: {', '.join(sorted(positions_df_chunk.columns))}")
                # Show sample rows for comparison
                sample_size = min(5, positions_df_chunk.height)
                sample_rows = positions_df_chunk.sample(sample_size, seed=42)
                
                if verbosity >= 1:
                    print(f"\nShowing {sample_size} sample rows to verify enhancement:")
                
                # Display a subset of columns for comparison
                display_cols = ['gene_id', 'transcript_id', 'position', 'pred_type', 'splice_type']
                
                # Original columns (from evaluate_splice_site_errors)
                if 'score' in positions_df_chunk.columns:
                    if verbosity >= 1:
                        print("\nOriginal 'score' column in positions_df:")
                        print(sample_rows.select(display_cols + ['score']).head(sample_size))
                
                # New enhanced columns (3 probability scores)
                if verbosity >= 1:
                    print("\nNew enhanced columns in positions_df:")
                    prob_cols = ['donor_score', 'acceptor_score', 'neither_score']
                    print(sample_rows.select(display_cols + prob_cols).head(sample_size))
                
                if verbosity >= 1:
                    print("===== END VALIDATION =====\n")
            
            # Save and process results if we have data
            if positions_df_chunk.height > 0:
                # Safely extract analysis sequences if required columns exist
                required_columns = ['gene_id', 'position']
                can_extract_sequences = all(col in positions_df_chunk.columns for col in required_columns)
                
                if can_extract_sequences:
                    try:
                        if verbosity >= 2:
                            print("[extract] Extracting analysis sequences...")
                        
                        # IMPORTANT: The correct order is positions_df first, then seq_df
                        df_seq = extract_analysis_sequences(
                            positions_df_chunk, 
                            seq_chunk,
                            include_empty_entries=True, 
                            verbose=(verbosity >= 2)
                        )
                    except Exception as e:
                        if verbosity >= 1:
                            print(f"[warning] Could not extract analysis sequences: {str(e)}")
                            print(f"[debug] Positions DataFrame columns: {positions_df_chunk.columns}")
                        df_seq = None
                else:
                    if verbosity >= 1:
                        print(f"[warning] Cannot extract analysis sequences - missing required columns")
                        print(f"[debug] Required: {required_columns}, Available: {positions_df_chunk.columns}")
                    df_seq = None
                
                # Save intermediate files using the existing handlers
                # For positions
                splice_position_path = data_handler.save_splice_positions(
                    positions_df_chunk, 
                    chr=chr,
                    chunk_start=chunk_start+1,
                    chunk_end=chunk_end,
                    enhanced=True,
                    output_subdir=output_subdir
                )
                splice_positions_files.append(splice_position_path)
                
                # For error analysis
                error_analysis_path = data_handler.save_error_analysis(
                    error_df_chunk,
                    chr=chr,
                    chunk_start=chunk_start+1,
                    chunk_end=chunk_end,
                    aggregated=False,
                    output_subdir=output_subdir
                )
                if verbosity >= 2:
                    print(f"[i/o] Saved chunk error analysis to {error_analysis_path}")
                error_analysis_files.append(error_analysis_path)
                
                # Save analysis sequences if we have them
                if df_seq is not None and hasattr(df_seq, 'height') and df_seq.height > 0:
                    # For analysis sequences
                    analysis_seq_path = data_handler.save_analysis_sequences(
                        df_seq,
                        chr=chr,
                        chunk_start=chunk_start+1,
                        chunk_end=chunk_end,
                        aggregated=False,
                        output_subdir=output_subdir
                    )
                    if verbosity >= 2:
                        print(f"[i/o] Saved chunk analysis sequences to {analysis_seq_path}")
                    analysis_sequence_files.append(analysis_seq_path)
                
                # Combine with full results
                full_positions_df = align_and_append(
                    full_positions_df,
                    positions_df_chunk,
                    strict=False
                )
                
                full_error_df = align_and_append(
                    full_error_df,
                    error_df_chunk,
                    strict=False
                )
                
                if df_seq is not None and hasattr(df_seq, 'height') and df_seq.height > 0:
                    full_analysis_sequence_df = align_and_append(
                        full_analysis_sequence_df,
                        df_seq,
                        strict=False
                    )
            
            # Update statistics
            chunk_time = time.time() - chunk_start_time
            total_chunk_time += chunk_time
            total_genes_processed += seq_chunk.height
            
            # Estimate time remaining
            avg_time_per_gene = total_chunk_time / total_genes_processed if total_genes_processed > 0 else 0
            genes_remaining = sum(
                lazy_seq_df.select(pl.col('gene_id').n_unique()).collect().item() 
                for chr in chromosomes if chr not in [str(chr)]
            )
            estimated_time_remaining = avg_time_per_gene * genes_remaining
            
            if verbosity >= 2:
                print_emphasized(f"[time] Chunk processed in {chunk_time:.2f}s")
                print_with_indent(f"[time] Estimated time remaining: {estimated_time_remaining/60:.2f} minutes", indent_level=1)
    
    # Check for invalid transcript IDs
    if full_positions_df.height > 0:
        abnormal_positions_df = check_and_subset_invalid_transcript_ids(
            full_positions_df
        )
        
        if not is_dataframe_empty(abnormal_positions_df):
            if verbosity >= 1:
                print_emphasized("[warning] Found abnormal transcript IDs in final dataset")
                print_with_indent("Sample of abnormal data:", indent_level=1)
                display_df = subsample_dataframe(abnormal_positions_df, columns=None, num_rows=10, random=True)
                print(display_df)
    
    # Check if we have any valid results before saving
    if full_positions_df is None or full_positions_df.height == 0:
        print_emphasized("[warning] No valid splice site predictions were found for any chromosome")
        print_with_indent("This may be due to:", indent_level=1)
        print_with_indent("1. Coordinate mismatches between predictions and annotations", indent_level=1)
        print_with_indent("2. Different reference genome versions", indent_level=1)
        print_with_indent("3. Target genes having no valid splice sites in annotations", indent_level=1)
        
        # Return empty results
        return {
            'error_analysis': full_error_df if full_error_df is not None else pl.DataFrame(),
            'positions': pl.DataFrame(),
            'analysis_sequences': pl.DataFrame()
        }

    # Save final aggregated files using the handler
    positions_path = data_handler.save_splice_positions(
        full_positions_df, 
        enhanced=True,
        aggregated=True, 
        output_subdir=output_subdir
    )
    
    error_path = data_handler.save_error_analysis(
        full_error_df,
        aggregated=True,
        output_subdir=output_subdir
    )
    
    if full_analysis_sequence_df.height > 0:
        sequences_path = data_handler.save_analysis_sequences(
            full_analysis_sequence_df,
            aggregated=True,
            output_subdir=output_subdir
        )
        if verbosity >= 1:
            print(f"[i/o] Saved analysis sequences to {sequences_path}")
    else:
        sequences_path = None
        if verbosity >= 1:
            print("[i/o] No analysis sequences to save (skipped)")
    
    total_time = time.time() - start_time
    if verbosity >= 1:
        print_emphasized(f"[done] Enhanced SpliceAI workflow completed in {total_time/60:.2f} minutes")
        print_with_indent(f"[stats] Processed {total_genes_processed} genes across {len(chromosomes)} chromosomes", indent_level=1)
        print_with_indent(f"[i/o] Enhanced positions file saved to: {positions_path}", indent_level=1)
        if full_positions_df is not None and full_positions_df.height > 0:
            print_with_indent(f"[columns] Positions DataFrame: {', '.join(full_positions_df.columns)}", indent_level=2)
        
        print_with_indent(f"[i/o] Enhanced error analysis saved to: {error_path}", indent_level=1)
        if full_error_df is not None and full_error_df.height > 0:
            print_with_indent(f"[columns] Error analysis DataFrame: {', '.join(full_error_df.columns)}", indent_level=2)
        
        if sequences_path:
            print_with_indent(f"[i/o] Enhanced analysis sequences saved to: {sequences_path}", indent_level=1)
            if full_analysis_sequence_df is not None and full_analysis_sequence_df.height > 0:
                print_with_indent(f"[columns] Analysis sequences DataFrame: {', '.join(full_analysis_sequence_df.columns)}", indent_level=2)
    
    return {
        'error_analysis': full_error_df,
        'positions': full_positions_df,
        'analysis_sequences': full_analysis_sequence_df
    }
