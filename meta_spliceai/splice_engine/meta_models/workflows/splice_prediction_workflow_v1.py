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
    # load_chromosome_sequence_streaming,
    adjust_chunk_size,
    subsample_dataframe,
    print_emphasized,
    print_with_indent,
    print_section_separator
)

from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import (
    normalize_chromosome_names
)

from meta_spliceai.splice_engine.evaluate_models import evaluate_splice_site_errors
from meta_spliceai.splice_engine.analysis_utils import check_and_subset_invalid_transcript_ids
from meta_spliceai.splice_engine.workflow_utils import align_and_append
from meta_spliceai.splice_engine.utils_df import is_dataframe_empty
# Use the meta_models version of extract_analysis_sequences to keep the workflow self-contained
from meta_spliceai.splice_engine.meta_models.workflows.sequence_data_utils import extract_analysis_sequences
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.utils_fs import read_splice_sites

# Import meta model utilities
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
# from meta_spliceai.splice_engine.utils_bio import normalize_strand
from meta_spliceai.splice_engine.meta_models.utils.infer_splice_site_adjustments import (
    auto_detect_splice_site_adjustments,
    apply_auto_detected_adjustments,
    calculate_prediction_statistics
)
from meta_spliceai.splice_engine.meta_models.core.enhanced_workflow import enhanced_process_predictions_with_all_scores

# Import the data preparation module
from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import (
    prepare_gene_annotations,
    prepare_splice_site_annotations,
    prepare_genomic_sequences,
    handle_overlapping_genes,
    determine_target_chromosomes,
    prepare_splice_site_adjustments,
    load_spliceai_models
)

# Import new function
from meta_spliceai.splice_engine.meta_models.utils.sequence_utils import (
    load_chromosome_sequence_streaming, 
    scan_chromosome_sequence,
    # filter_genes_by_chromosome,
    # get_sequence_context
)

def run_enhanced_splice_prediction_workflow(
    config: Optional[SpliceAIConfig] = None, 
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None, 
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
    target_chromosomes : List[str], optional
        List of chromosomes to focus on. If provided, will only process these chromosomes.
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
    seq_format = config.seq_format
    seq_mode = config.seq_mode
    seq_type = config.seq_type
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

    # Normalize target chromosomes if specified
    normalized_chromosomes = None
    if target_chromosomes:
        normalized_chromosomes = normalize_chromosome_names(target_chromosomes)
        if verbosity >= 1:
            print(f"[info] Normalized target chromosomes: {normalized_chromosomes}")
    
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
    # NOTE: Global genomic datasets (annotations, splice sites, overlapping genes)
    #       should be stored in Analyzer.shared_dir (typically the parent directory
    #       of analysis_dir and eval_dir). This allows different analysis runs to 
    #       share the same reference data, avoiding redundant storage and computation.
    
    # 1. Prepare gene annotations
    annot_result = prepare_gene_annotations(
        local_dir=local_dir,
        gtf_file=gtf_file,
        do_extract=do_extract_annotations,
        target_chromosomes=target_chromosomes,  # Pass chromosomes directly
        use_shared_db=True,  # Use the shared database to save space and time
        separator=separator,
        verbosity=verbosity
    )
    # NOTE: Process entire set of annotations for all chromosomes
    #       If target_chromosomes is specified, filter after extraction

    # 2. Prepare splice site annotations
    ss_result = prepare_splice_site_annotations(
        local_dir=local_dir,
        gtf_file=gtf_file,
        do_extract=do_extract_splice_sites,
        # gene_annotations_df=None,  # None by default, will use annotations from step 1
        target_chromosomes=target_chromosomes,  # Filter by chromosomes if specified
        consensus_window=consensus_window,
        separator=separator,
        verbosity=verbosity
    )
    # NOTE: When providing "gene_annotations_df", the function will filter the full splice site
    #       annotations to only include genes in the provided dataframe.

    if not ss_result['success']:
        print("Error: Failed to prepare splice site annotations.")
        return {}
    
    ss_annotations_df = ss_result['splice_sites_df']
    
    # 3. Prepare genomic sequences
    seq_result = prepare_genomic_sequences(
        local_dir=local_dir,
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        mode=seq_mode,
        seq_type=seq_type,
        do_extract=do_extract_sequences,
        chromosomes=target_chromosomes,  # Filter by chromosomes if specified
        test_mode=test_mode,
        seq_format=seq_format,  # parquet by default
        verbosity=verbosity
    )
    
    # 4. Handle overlapping genes
    overlap_result = handle_overlapping_genes(
        local_dir=local_dir,
        gtf_file=gtf_file,
        do_find=do_find_overlaping_genes,
        filter_valid_splice_sites=kwargs.get('filter_valid_splice_sites', True),
        min_exons=kwargs.get('min_exons', 2),
        separator=separator,
        verbosity=verbosity
    )
    
    # 5. Determine which chromosomes to process
    chrom_result = determine_target_chromosomes(
        local_dir=local_dir,
        gtf_file=gtf_file,
        target_genes=target_genes,
        chromosomes=chromosomes,
        test_mode=test_mode,
        separator=separator,
        verbosity=verbosity
    )
    
    chromosomes = chrom_result['chromosomes']
    
    # 6. Load SpliceAI models
    model_result = load_spliceai_models(verbosity=verbosity)
    
    if not model_result['success']:
        print(f"Error: Failed to load SpliceAI models: {model_result['error']}")
        return {}
    
    models = model_result['models']
    
    # 7. Prepare splice site position adjustments
    # For empirical inference, we need a sample of predictions first
    sample_predictions = None
    if config.use_auto_position_adjustments:
        # If targeting specific genes, we can use them as the sample for adjustment detection
        if verbosity >= 1:
            print_emphasized("[action] Preparing splice site position adjustments")
        
        # If we have target genes, we can get a sample of predictions for them first
        # to use in empirical adjustment detection
        if target_genes is not None and len(target_genes) > 0:
            # Find which chromosome has the target genes
            sample_chromosome = next(iter(chrom_result.get('chromosomes', ['21'])))
            
            if verbosity >= 1:
                print_with_indent(f"Using target genes from chromosome {sample_chromosome} for adjustment detection", indent_level=1)
            
            # Load sequences for this chromosome
            try:
                sample_seq_df = None
                chrom_seq_file = os.path.join(local_dir, f"gene_sequence_{sample_chromosome}.{format}")
                
                if os.path.exists(chrom_seq_file):
                    sample_seq_df = load_chromosome_sequence_streaming(chrom_seq_file, sample_chromosome, format=format).collect()
                else:
                    # Try the main sequence file
                    main_seq_file = seq_result.get('main_sequence_file')
                    if main_seq_file and os.path.exists(main_seq_file):
                        sample_seq_df = load_chromosome_sequence_streaming(main_seq_file, sample_chromosome, format=format).collect()
                
                if sample_seq_df is not None:
                    # Filter to target genes if specified
                    if target_genes:
                        sample_seq_df = sample_seq_df.filter(
                            pl.col('gene_id').is_in(target_genes) | 
                            pl.col('gene_name').is_in(target_genes)
                        )
                    
                    # Limit to a small number of genes for sample
                    if sample_seq_df.height > 20:
                        sample_seq_df = sample_seq_df.sample(20, seed=42)
                    
                    # Run predictions on sample
                    if verbosity >= 1:
                        print_with_indent(f"Running sample predictions on {sample_seq_df.height} genes for adjustment detection", indent_level=1)
                    
                    sample_predictions = predict_splice_sites_for_genes(
                        sample_seq_df, models=models, context=10000, efficient_output=True
                    )
            except Exception as e:
                if verbosity >= 1:
                    print_with_indent(f"Could not generate sample predictions for adjustment detection: {e}", indent_level=1)
                    print_with_indent("Will use default adjustments instead", indent_level=1)
                sample_predictions = None
        
        # Get optimal position adjustments
        adjustment_result = prepare_splice_site_adjustments(
            local_dir=local_dir,
            ss_annotations_df=ss_annotations_df,
            sample_predictions=sample_predictions,
            use_empirical=config.use_auto_position_adjustments,
            save_adjustments=True,
            verbosity=verbosity
        )
        
        adjustment_dict = adjustment_result.get('adjustment_dict')
        
        if verbosity >= 1 and adjustment_dict:
            print_emphasized("[info] Position adjustments that will be applied:")
            print_with_indent(f"Donor sites:    +{adjustment_dict['donor']['plus']} on plus strand, +{adjustment_dict['donor']['minus']} on minus strand", indent_level=1)
            print_with_indent(f"Acceptor sites: +{adjustment_dict['acceptor']['plus']} on plus strand, +{adjustment_dict['acceptor']['minus']} on minus strand", indent_level=1)
    else:
        adjustment_dict = {
            'donor': {'plus': 0, 'minus': 0},
            'acceptor': {'plus': 0, 'minus': 0}
        }
        if verbosity >= 1:
            print_emphasized("[info] Automatic position adjustments disabled")
    
    # ──────────────────────────────────────────────────────────────────────────────
    # Main Workflow: Per‑chromosome / per‑chunk   Prediction → Evaluation → Save artefacts
    # ──────────────────────────────────────────────────────────────────────────────

    ################################################################################
    # Initialise run-wide containers & timers
    ################################################################################
    probability_floor = kwargs.get("probability_floor", 0.005)

    full_error_df           = None
    full_positions_df       = None
    full_analysis_seq_df    = None      # optional – only if you collect ±500 nt windows

    total_genes_processed   = 0
    total_chunk_time        = 0.0
    run_start_time          = time.time()

    # ── load SpliceAI ensemble once (GPU friendly) ────────────────────────────────
    action  = kwargs.get("action", "predict")
    if action == "predict":
        from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_spliceai_ensemble   # <- thin wrapper
        models = load_spliceai_ensemble(context=10_000)
    else:
        models = None

    ################################################################################
    # iterate chromosomes
    ################################################################################
    processed_chroms = []
    for chr_ in tqdm(chromosomes, desc="Processing chromosomes"):
        chr_ = str(chr_)

        # • Lazy‑load sequences for this chromosome
        if verbosity >= 1:
            print_emphasized(f"[info] Loading sequences for chromosome {chr_}")
        try:
            # Use the new utility function for lazily loading chromosome sequences
            lazy_seq_df = scan_chromosome_sequence(
                seq_result=seq_result,
                chromosome=chr_,
                format=seq_format,
                separator=separator,
                verbosity=verbosity
            )
        except FileNotFoundError as e:
            print(f"[warning] Sequence file for chr{chr_} not found – skipping. ({e})")
            continue

        default_chunk_size = 500 if not test_mode else 50
        chunk_size         = default_chunk_size
        seq_len_avg        = 50_000   # used by adjust_chunk_size()
        n_genes            = lazy_seq_df.select(pl.col("gene_id").n_unique()).collect().item()

        if verbosity >= 1:
            print_emphasized(f"[info] chr{chr_}: {n_genes} genes to process")

        ############################################################################
        # iterate gene chunks inside this chromosome
        ############################################################################
        for chunk_start in range(0, n_genes, chunk_size):
            chunk_end   = min(chunk_start + chunk_size, n_genes)
            chunk_start_time = time.time()

            # adapt chunk_size if memory pressure changed
            chunk_size  = adjust_chunk_size(chunk_size, seq_len_avg)

            # materialise this chunk
            seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()

            # optional gene filter
            if target_genes:
                before = seq_chunk.height
                seq_chunk = seq_chunk.filter(
                    pl.col("gene_id").is_in(target_genes) |
                    pl.col("gene_name").is_in(target_genes)
                )
                after  = seq_chunk.height
                if after == 0:
                    continue
                if verbosity >= 1:
                    print_emphasized(f"[info] target_genes filter: {before} → {after}")

            if verbosity >= 1:
                print_emphasized(
                    f"[pred] chr{chr_} genes {chunk_start+1}-{chunk_end}  "
                    f"(size={seq_chunk.height})")

            ########################################################################
            # 8.1  SPLICE‑SITE PREDICTION
            ########################################################################
            if action == "predict":
                predictions = predict_splice_sites_for_genes(
                    seq_chunk, models=models, context=10_000, efficient_output=True)
            else:   # loading from disk not shown – add if desired
                predictions = {}

            ########################################################################
            # 8.2  ENHANCED EVALUATION (new meta_models core)
            ########################################################################
            error_df_chunk, positions_df_chunk = enhanced_process_predictions_with_all_scores(
                    predictions=predictions,
                    ss_annotations_df=ss_annotations_df,
                    threshold=config.threshold,
                    consensus_window=config.consensus_window,
                    error_window=config.error_window,
                    analyze_position_offsets=True,
                    collect_tn=True,
                    predicted_delta_correction=True,  # Use adjustments
                    splice_site_adjustments=adjustment_dict,  # Pass the detected adjustments
                    add_derived_features=True,
                    verbose=2
            )
   
            # quick sanity: skip empty chunks
            if positions_df_chunk.height == 0:
                continue

            # add chromosome column if absent
            if "chrom" not in positions_df_chunk.columns:
                positions_df_chunk = positions_df_chunk.with_columns(
                    pl.lit(chr_).alias("chrom"))

            ########################################################################
            # 8.3  OPTIONAL  merge extra contextual features
            ########################################################################
            df_seq = extract_analysis_sequences(seq_chunk, positions_df_chunk, include_empty_entries=True, verbose=1)
            print_with_indent(f"[info] shape(df_seq): {df_seq.shape}, n(genes): {df_seq['gene_id'].n_unique()}", indent_level=2)
            print_with_indent(f"[info] Columns: {df_seq.columns}", indent_level=2)
            
            if full_analysis_seq_df is None:
                full_analysis_seq_df = df_seq
            else: 
                full_analysis_seq_df = align_and_append(full_analysis_seq_df, df_seq, strict=False)

            if kwargs.get("add_kmer_features", False):
                positions_df_chunk = merge_contextual_features(
                    positions_df_chunk,
                    gene_meta_df = kwargs.get("gene_meta_df"),
                    kmer_df      = kwargs.get("kmer_df"))

            ########################################################################
            # 8.4  SAVE chunk artefacts
            ########################################################################
            data_handler.save_error_analysis(
                error_df_chunk, chr=chr_, chunk_start=chunk_start+1, chunk_end=chunk_end)
            data_handler.save_splice_positions(
                positions_df_chunk, chr=chr_, chunk_start=chunk_start+1, chunk_end=chunk_end,
                enhanced=True)

            ########################################################################
            # 8.5  ACCUMULATE globals
            ########################################################################
            if full_error_df is None:
                full_error_df = error_df_chunk
            else:
                full_error_df = align_and_append(full_error_df, error_df_chunk, strict=False)

            if full_positions_df is None:
                full_positions_df = positions_df_chunk
            else:
                full_positions_df = align_and_append(full_positions_df,
                                                    positions_df_chunk, strict=False)

            ########################################################################
            # 8.6  PROGRESS  (time per gene, ETA)
            ########################################################################
            chunk_time = time.time() - chunk_start_time
            total_chunk_time      += chunk_time
            total_genes_processed += positions_df_chunk.select(
                                        pl.col("gene_id").n_unique()).item()

            if verbosity >= 2:
                tqdm.write(f"  └─ chunk {chunk_start+1}-{chunk_end} "
                        f"done in {chunk_time:,.1f}s")

        processed_chroms.append(chr_)
    # end‑for chromosomes
    
    # Summary after completing all chromosomes and gene chunks
    print_emphasized(f"[success] Completed chromosome-level prediction and evaluation")
    if processed_chroms:
        processed_chrom_str = ", ".join(processed_chroms) if len(processed_chroms) <= 10 else f"{len(processed_chroms)} chromosomes"
        print_with_indent(f"Processed chromosomes: {processed_chrom_str}", indent_level=1)
    print_with_indent(f"Total genes processed: {total_genes_processed:,}", indent_level=1)
    print_with_indent(f"Total processing time: {total_chunk_time:.1f} seconds", indent_level=1)
    if full_positions_df is not None:
        print_with_indent(f"Total positions analyzed: {full_positions_df.height:,}", indent_level=1)
    print_section_separator()

    ################################################################################
    # 9. FINAL AGGREGATION & SAVE
    ################################################################################
    if full_positions_df is not None and full_positions_df.height:

        if verbosity >= 1:
            print_emphasized("[debug] Columns in positions DataFrame before save:")
            print(f"Number of columns: {len(full_positions_df.columns)}")
            print(f"Column names: {full_positions_df.columns}")

            # Check specifically for context columns
            context_cols = [col for col in full_positions_df.columns if col.startswith('context_')]
            if context_cols:
                print(f"Found {len(context_cols)} context columns: {context_cols}")
            else:
                print("No context columns found in DataFrame!")

            # Check specifically for derived feature columns
            derived_prefixes = ['donor_diff', 'donor_surge', 'donor_peak']
            derived_cols = [col for col in full_positions_df.columns 
                        if any(col.startswith(prefix) for prefix in derived_prefixes)]
            if derived_cols:
                print(f"Found {len(derived_cols)} derived feature columns, sample: {derived_cols[:5]}...")
            else:
                print("No derived feature columns found in DataFrame!")

        pos_path = data_handler.save_splice_positions(
            full_positions_df, aggregated=True, enhanced=True)
        err_path = data_handler.save_error_analysis(
            full_error_df,      aggregated=True)
        if verbosity >= 1:
            print_emphasized(f"[done] saved aggregated positions → {pos_path}")
            print_emphasized(f"[done] saved aggregated errors    → {err_path}")
    else:
        print_emphasized("[warning] No valid predictions produced by workflow")

    runtime_min = (time.time() - run_start_time) / 60.0
    print_emphasized(f"[time] Total runtime: {runtime_min:,.1f} min")

    # Create result dictionary with success status and available dataframes
    result = {
        "success": True,  # Indicate workflow completed successfully
        "error_analysis": full_error_df if 'full_error_df' in locals() and full_error_df is not None else pl.DataFrame(),
        "positions": full_positions_df if 'full_positions_df' in locals() and full_positions_df is not None else pl.DataFrame(),
        "analysis_sequences": full_analysis_seq_df if 'full_analysis_seq_df' in locals() and full_analysis_seq_df is not None else pl.DataFrame()
    }
    
    # Include overlapping genes data if it was processed
    if do_find_overlaping_genes and 'overlap_result' in locals() and overlap_result is not None:
        result["overlapping_genes"] = overlap_result
    
    return result
