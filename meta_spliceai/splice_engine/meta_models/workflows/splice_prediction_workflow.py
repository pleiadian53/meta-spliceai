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
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig, GeneManifest
from meta_spliceai.splice_engine.meta_models.core.position_types import (
    PositionType, absolute_to_relative
)
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
    no_final_aggregate: bool = False,
    no_tn_sampling: bool = False,
    position_id_mode: str = 'genomic',  # NEW: Position identification strategy
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
    no_tn_sampling : bool, default=False
        If True, preserve all TN positions without sampling. If False, apply TN sampling based on tn_sample_factor.
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

    # Control memory-heavy final aggregation
    do_aggregate = not no_final_aggregate
    if no_final_aggregate and verbosity >= 1:
        print_emphasized("[info] Final aggregation disabled (--no-final-aggregate)")
    
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
    # CRITICAL: Override config.chromosomes with target_chromosomes if provided
    normalized_chromosomes = None
    if target_chromosomes:
        normalized_chromosomes = normalize_chromosome_names(target_chromosomes)
        chromosomes = normalized_chromosomes  # FIX: Use normalized target chromosomes for processing
        if verbosity >= 1:
            print(f"[info] Normalized target chromosomes: {normalized_chromosomes}")
    
    # ============================================================================
    # ARTIFACT MANAGER SETUP
    # ============================================================================
    # Initialize artifact manager for systematic artifact management
    artifact_manager = config.get_artifact_manager()
    
    if verbosity >= 1:
        print_emphasized("[workflow] Starting enhanced SpliceAI prediction workflow...")
        print_with_indent(f"[params] threshold: {threshold}, consensus_window: {consensus_window}, error_window: {error_window}", indent_level=1)
        print_with_indent(f"[params] output_directory: {os.path.join(eval_dir, output_subdir)}", indent_level=1)
        print_with_indent(f"[params] data preparation: extract_annotations={do_extract_annotations}, extract_splice_sites={do_extract_splice_sites}, extract_sequences={do_extract_sequences}", indent_level=1)
        print()
        print_emphasized("[artifact_manager] Artifact Management Configuration")
        artifact_manager.print_summary()
    
    # Convert path strings to absolute paths if they're not already
    if not os.path.isabs(eval_dir):
        eval_dir = os.path.abspath(eval_dir)

    # Set up handlers
    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)
    data_handler = MetaModelDataHandler(eval_dir, separator=separator)
    
    # Create output directory if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    
    # Define local directory for intermediates (similar to original workflow)
    # Use config.local_dir if provided, otherwise default to parent of eval_dir
    local_dir = config.local_dir if config.local_dir is not None else os.path.dirname(eval_dir)
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

    # 1.5. Derive gene features (gene_type, gene_length, etc.)
    # This is needed for biotype-specific gene sampling and analysis
    try:
        from meta_spliceai.system.genomic_resources.derive import GenomicDataDeriver
        from meta_spliceai.system.genomic_resources import Registry
        
        # Create deriver using the local_dir (build-specific directory)
        # Infer build and release from the GTF file path if possible
        registry = None
        if 'GRCh37' in gtf_file:
            registry = Registry(build='GRCh37', release='87')
        elif 'GRCh38' in gtf_file:
            # Try to infer release from filename
            import re
            match = re.search(r'\.(\d+)\.gtf', gtf_file)
            release = match.group(1) if match else '112'
            registry = Registry(build='GRCh38', release=release)
        
        deriver = GenomicDataDeriver(
            data_dir=local_dir,
            registry=registry,
            verbosity=verbosity
        )
        
        # Derive gene features if not already present
        gene_features_result = deriver.derive_gene_features(
            output_filename='gene_features.tsv',
            target_chromosomes=target_chromosomes,
            force_overwrite=False  # Use existing if available
        )
        
        if gene_features_result['success'] and verbosity >= 1:
            print_with_indent(
                f"[info] Gene features available: {gene_features_result['gene_features_file']}",
                indent_level=1
            )
    except Exception as e:
        if verbosity >= 1:
            print_with_indent(
                f"[warning] Could not derive gene features: {e}",
                indent_level=1
            )
            print_with_indent(
                "[warning] Gene biotype information may not be available for downstream analysis",
                indent_level=1
            )

    # 2. Prepare splice site annotations
    ss_result = prepare_splice_site_annotations(
        local_dir=local_dir,
        gtf_file=gtf_file,
        do_extract=do_extract_splice_sites,
        # gene_annotations_df=None,  # None by default, will use annotations from step 1
        target_chromosomes=target_chromosomes,  # Filter by chromosomes if specified
        consensus_window=consensus_window,
        fasta_file=genome_fasta,  # Enable OpenSpliceAI fallback when splice_sites.tsv is missing
        separator=separator,
        verbosity=verbosity
    )
    # NOTE: When providing "gene_annotations_df", the function will filter the full splice site
    #       annotations to only include genes in the provided dataframe.

    if not ss_result['success']:
        print("Error: Failed to prepare splice site annotations.")
        return {}
    
    ss_annotations_df = ss_result['splice_sites_df']
    
    # CRITICAL: Standardize splice site schema for consistency across the system
    # This handles synonymous column names (e.g., site_type → splice_type)
    from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema
    if verbosity >= 1:
        print_with_indent("[schema] Standardizing splice site annotations before workflow...", indent_level=1)
        print_with_indent(f"[schema] Columns before: {ss_annotations_df.columns}", indent_level=2)
    ss_annotations_df = standardize_splice_sites_schema(
        ss_annotations_df, 
        verbose=True  # Always verbose for debugging
    )
    if verbosity >= 1:
        print_with_indent(f"[schema] Columns after: {ss_annotations_df.columns}", indent_level=2)
    
    # DEBUG: Log target genes before sequence preparation
    if verbosity >= 1:
        if target_genes:
            print(f"[debug] Preparing sequences for {len(target_genes)} target genes: {target_genes[:5]}{'...' if len(target_genes) > 5 else ''}")
        else:
            print(f"[debug] No target genes specified - will process all genes")
    
    # 3. Prepare genomic sequences
    # IMPORTANT: When target genes are specified, we need to extract sequences if they don't exist
    # This is because the per-chromosome sequence files may not be pre-extracted
    force_extract_for_target_genes = target_genes and do_extract_sequences == False
    if force_extract_for_target_genes and verbosity >= 1:
        print("[info] Target genes specified - forcing sequence extraction from FASTA")
    
    seq_result = prepare_genomic_sequences(
        local_dir=local_dir,
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        mode=seq_mode,
        seq_type=seq_type,
        do_extract=do_extract_sequences or force_extract_for_target_genes,  # Force extraction for target genes
        chromosomes=target_chromosomes,  # Filter by chromosomes if specified
        genes=target_genes,  # FIX: Pass target genes for filtering
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
        verbosity=verbosity,
        target_chromosomes=target_chromosomes  # Add chromosome filtering
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
                        # Build filter condition based on available columns
                        filter_condition = pl.col('gene_id').is_in(target_genes)
                        if 'gene_name' in sample_seq_df.columns:
                            filter_condition = filter_condition | pl.col('gene_name').is_in(target_genes)
                        sample_seq_df = sample_seq_df.filter(filter_condition)
                    
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
    full_analysis_seq_df    = None      # optional – only if you collect ±500 nt windows
    full_nucleotide_scores_df = None    # NEW: Full nucleotide-level scores

    total_genes_processed   = 0
    total_chunk_time        = 0.0
    run_start_time          = time.time()
    
    # ── load base model ensemble once (GPU friendly) ──────────────────────────────
    action  = kwargs.get("action", "predict")
    model_metadata = None  # Initialize to None
    
    if action == "predict":
        from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_base_model_ensemble
        
        # Load the appropriate base model based on configuration
        models, model_metadata = load_base_model_ensemble(
            base_model=config.base_model,
            context=10_000,
            verbosity=verbosity
        )
        
        if verbosity >= 1:
            print_emphasized(f"[model] Using {model_metadata['base_model']} "
                           f"({model_metadata['genome_build']}, {model_metadata['framework']})")
    
    # ── Initialize gene manifest for tracking processing status ──────────────────
    gene_manifest = GeneManifest(
        base_model=config.base_model,
        genomic_build=model_metadata.get('genome_build', 'unknown') if model_metadata else 'unknown'
    )
    
    # Track requested genes if specified
    if target_genes:
        gene_manifest.add_requested_genes(target_genes)
        if verbosity >= 1:
            print_with_indent(f"[manifest] Tracking {len(target_genes)} requested genes", indent_level=1)

    ################################################################################
    # iterate chromosomes
    ################################################################################
    processed_chroms = []
    
    # Check if sequences were already loaded and filtered (for target genes)
    # Only use pre-loaded sequences for SMALL gene sets to avoid memory issues
    # For large gene sets or full genome, use per-chromosome streaming
    MAX_GENES_FOR_PRELOAD = 1000  # Threshold for memory-efficient processing
    
    use_preloaded_sequences = (target_genes and 
                               len(target_genes) <= MAX_GENES_FOR_PRELOAD and
                               seq_result.get('sequences_df') is not None and 
                               seq_result['sequences_df'] is not None)
    
    if target_genes and len(target_genes) > MAX_GENES_FOR_PRELOAD:
        if verbosity >= 1:
            print_emphasized(f"[info] Large gene set ({len(target_genes)} genes) - using per-chromosome streaming for memory efficiency")
    
    if use_preloaded_sequences:
        # Get pre-loaded sequences
        preloaded_df = seq_result['sequences_df']
        
        # Convert to polars if needed
        if not isinstance(preloaded_df, pl.DataFrame):
            import pandas as pd
            if isinstance(preloaded_df, pd.DataFrame):
                preloaded_df = pl.from_pandas(preloaded_df)
        
        # Check if DataFrame is actually empty
        if preloaded_df.height == 0:
            if verbosity >= 1:
                print_emphasized(f"[warning] Pre-loaded sequences DataFrame is empty - falling back to per-chromosome loading")
            use_preloaded_sequences = False
        else:
            if verbosity >= 1:
                print_emphasized(f"[info] Using pre-loaded sequences for {len(target_genes)} target genes")
            
            # Standardize schema to ensure consistent column names (seqname → chrom)
            from meta_spliceai.system.genomic_resources import standardize_gene_features_schema
            preloaded_df = standardize_gene_features_schema(preloaded_df, verbose=False)
            
            # Get unique chromosomes from the filtered data
            actual_chromosomes = preloaded_df.select(pl.col('chrom').unique()).to_series().to_list()
            if verbosity >= 1:
                print_with_indent(f"[info] Target genes found on {len(actual_chromosomes)} chromosomes: {actual_chromosomes}", indent_level=1)
            
            # Override chromosomes list to only process chromosomes with target genes
            chromosomes = actual_chromosomes
    
    for chr_ in tqdm(chromosomes, desc="Processing chromosomes"):
        chr_ = str(chr_)

        # • Load sequences for this chromosome
        if verbosity >= 1:
            print_emphasized(f"[info] Loading sequences for chromosome {chr_}")
        
        if use_preloaded_sequences:
            # Use pre-loaded filtered sequences
            try:
                # Filter pre-loaded data to this chromosome (schema is already standardized)
                chrom_df = preloaded_df.filter(pl.col('chrom') == chr_)
                
                if chrom_df.height == 0:
                    if verbosity >= 1:
                        print(f"[info] No target genes on chromosome {chr_} - skipping")
                    continue
                
                # Convert to LazyFrame for compatibility with downstream processing
                lazy_seq_df = chrom_df.lazy()
                
                if verbosity >= 1:
                    print_with_indent(f"[info] Using {chrom_df.height} pre-loaded sequences for chr{chr_}", indent_level=1)
            except Exception as e:
                print(f"[warning] Failed to filter pre-loaded sequences for chr{chr_}: {e}")
                continue
        else:
            # Original path: load per-chromosome from files
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

            ########################################################################
            # CHUNK-LEVEL CHECKPOINT: Skip if already processed
            ########################################################################
            chunk_artifact_file = os.path.join(
                data_handler.meta_dir,  # Use meta_dir, not eval_dir!
                f"analysis_sequences_{chr_}_chunk_{chunk_start+1}_{chunk_end}.tsv"
            )
            if os.path.exists(chunk_artifact_file):
                file_size = os.path.getsize(chunk_artifact_file) / (1024 * 1024)  # MB
                if verbosity >= 1:
                    print_emphasized(
                        f"[checkpoint] chr{chr_} chunk {chunk_start+1}-{chunk_end} "
                        f"already exists ({file_size:.1f} MB) - SKIPPING"
                    )
                continue  # Skip this chunk

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
            # 8.1  SPLICE‑SITE PREDICTION (with mini-batch memory optimization)
            ########################################################################
            # Mini-batch processing: Process genes in smaller batches to reduce peak memory
            # Note: This does NOT change chunk boundaries or file names - just internal processing
            MINI_BATCH_SIZE = config.mini_batch_size if hasattr(config, 'mini_batch_size') else 50
            
            # Containers for accumulating results from mini-batches
            all_mini_batch_predictions = {}  # Will hold all predictions for this chunk
            all_mini_batch_errors = []       # Will hold error_df from all mini-batches
            all_mini_batch_positions = []    # Will hold positions_df from all mini-batches
            all_mini_batch_sequences = []    # Will hold df_seq from all mini-batches
            
            n_genes_in_chunk = seq_chunk.height
            n_mini_batches = (n_genes_in_chunk + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE
            
            # Memory monitoring (optional)
            try:
                import psutil
                process = psutil.Process()
                mem_before_mb = process.memory_info().rss / (1024 * 1024)
                memory_monitoring = True
            except (ImportError, Exception):
                mem_before_mb = 0
                memory_monitoring = False
            
            if verbosity >= 1 and n_mini_batches > 1:
                mem_str = f", current memory: {mem_before_mb:.0f} MB" if memory_monitoring else ""
                print_with_indent(
                    f"[memory] Processing {n_genes_in_chunk} genes in {n_mini_batches} mini-batches "
                    f"of {MINI_BATCH_SIZE} genes each{mem_str}", 
                    indent_level=1
                )
            
            for mini_batch_idx in range(n_mini_batches):
                mini_batch_start = mini_batch_idx * MINI_BATCH_SIZE
                mini_batch_end = min(mini_batch_start + MINI_BATCH_SIZE, n_genes_in_chunk)
                mini_batch_size = mini_batch_end - mini_batch_start
                
                # Extract mini-batch from seq_chunk
                seq_mini_batch = seq_chunk[mini_batch_start:mini_batch_end]
                
                if verbosity >= 2 and n_mini_batches > 1:
                    print_with_indent(
                        f"[memory] Mini-batch {mini_batch_idx+1}/{n_mini_batches}: "
                        f"genes {mini_batch_start+1}-{mini_batch_end}",
                        indent_level=2
                    )
                
                # ═══════════════════════════════════════════════════════════════
                # 8.1.1  PREDICT (mini-batch only)
                # ═══════════════════════════════════════════════════════════════
                if action == "predict":
                    mini_batch_pred_start_time = time.time()
                    predictions_mini = predict_splice_sites_for_genes(
                        seq_mini_batch, 
                        models=models, 
                        context=10_000, 
                        efficient_output=True,
                        show_gene_progress=False,  # Disable inner progress bar
                        verbosity=verbosity
                    )
                    mini_batch_pred_time = time.time() - mini_batch_pred_start_time
                    
                    # Accumulate predictions
                    all_mini_batch_predictions.update(predictions_mini)
                    
                    # Track genes in manifest
                    for row in seq_mini_batch.iter_rows(named=True):
                        gene_id = row['gene_id']
                        gene_name = row.get('gene_name', gene_id)
                        
                        if gene_id in predictions_mini:
                            pred_info = predictions_mini[gene_id]
                            num_nucleotides = len(pred_info.get('donor_prob', []))
                            
                            gene_ss = ss_annotations_df.filter(pl.col('gene_id') == gene_id)
                            num_splice_sites = gene_ss.height if hasattr(gene_ss, 'height') else len(gene_ss)
                            
                            gene_manifest.mark_processed(
                                gene_id=gene_id,
                                gene_name=gene_name,
                                num_nucleotides=num_nucleotides,
                                num_splice_sites=num_splice_sites,
                                processing_time=mini_batch_pred_time / mini_batch_size
                            )
                        else:
                            gene_manifest.mark_failed(
                                gene_id=gene_id,
                                gene_name=gene_name,
                                status='prediction_failed',
                                reason='No predictions generated'
                            )
                else:
                    predictions_mini = {}
                
                # ═══════════════════════════════════════════════════════════════
                # 8.1.2  EVALUATE (mini-batch only)
                # ═══════════════════════════════════════════════════════════════
                if predictions_mini:
                    error_df_mini, positions_df_mini = enhanced_process_predictions_with_all_scores(
                        predictions=predictions_mini,
                        ss_annotations_df=ss_annotations_df,
                        threshold=config.threshold,
                        consensus_window=config.consensus_window,
                        error_window=config.error_window,
                        analyze_position_offsets=True,
                        collect_tn=True,
                        no_tn_sampling=no_tn_sampling,
                        predicted_delta_correction=True,
                        splice_site_adjustments=adjustment_dict,
                        add_derived_features=True,
                        verbose=0 if n_mini_batches > 1 else 2  # Reduce verbosity for mini-batches
                    )
                    
                    # Add chromosome column if absent
                    if positions_df_mini.height > 0:
                        if "chrom" not in positions_df_mini.columns:
                            positions_df_mini = positions_df_mini.with_columns(
                                pl.lit(chr_).alias("chrom"))
                        
                        # ═══════════════════════════════════════════════════════════
                        # 8.1.3  EXTRACT SEQUENCES (mini-batch only)
                        # ═══════════════════════════════════════════════════════════
                        df_seq_mini = extract_analysis_sequences(
                            seq_mini_batch,
                            positions_df_mini,
                            include_empty_entries=True,
                            essential_columns_only=False,
                            drop_transcript_id=False,
                            window_size=250,
                            position_id_mode=position_id_mode,
                            preserve_transcript_list=True,
                            verbose=0 if n_mini_batches > 1 else 1  # Reduce verbosity
                        )
                        
                        # Accumulate for later concatenation
                        all_mini_batch_errors.append(error_df_mini)
                        all_mini_batch_positions.append(positions_df_mini)
                        all_mini_batch_sequences.append(df_seq_mini)
                
                # Free mini-batch memory immediately
                del seq_mini_batch, predictions_mini
                if 'error_df_mini' in locals():
                    del error_df_mini
                if 'positions_df_mini' in locals():
                    del positions_df_mini
                if 'df_seq_mini' in locals():
                    del df_seq_mini
                
                # Force garbage collection after each mini-batch
                import gc
                gc.collect()
            
            # ═══════════════════════════════════════════════════════════════════
            # 8.1.4  CONSOLIDATE mini-batch results into chunk-level data
            # ═══════════════════════════════════════════════════════════════════
            # Now we have all predictions and sequences - consolidate them
            predictions = all_mini_batch_predictions  # This is the full chunk predictions
            
            # Concatenate all mini-batch results for errors, positions, and sequences
            if all_mini_batch_errors:
                error_df_chunk = pl.concat(all_mini_batch_errors)
            else:
                error_df_chunk = pl.DataFrame()  # Empty if no results
            
            if all_mini_batch_positions:
                positions_df_chunk = pl.concat(all_mini_batch_positions)
            else:
                positions_df_chunk = pl.DataFrame()  # Empty if no results
            
            if all_mini_batch_sequences:
                df_seq = pl.concat(all_mini_batch_sequences)
            else:
                df_seq = pl.DataFrame()  # Empty if no results
            
            # Clean up accumulation containers
            del all_mini_batch_predictions, all_mini_batch_errors, all_mini_batch_positions, all_mini_batch_sequences
            
            if verbosity >= 1 and n_mini_batches > 1:
                # Memory monitoring after consolidation
                if memory_monitoring:
                    mem_after_mb = process.memory_info().rss / (1024 * 1024)
                    mem_delta = mem_after_mb - mem_before_mb
                    mem_str = f", memory: {mem_after_mb:.0f} MB (Δ{mem_delta:+.0f} MB)"
                else:
                    mem_str = ""
                
                print_with_indent(
                    f"[memory] Consolidated {n_mini_batches} mini-batches: "
                    f"{error_df_chunk.height} errors, {positions_df_chunk.height} positions, "
                    f"{df_seq.height} sequences{mem_str}",
                    indent_level=1
                )

            ########################################################################
            # 8.1.5  CAPTURE NUCLEOTIDE-LEVEL SCORES (NEW - Optional)
            ########################################################################
            # Extract full nucleotide-level scores from predictions
            # NOTE: Disabled by default to avoid massive data volumes
            # Enable with config.save_nucleotide_scores = True
            if predictions and config.save_nucleotide_scores:
                nucleotide_scores_chunk = []
                for gene_id, pred_data in predictions.items():
                    # Get gene metadata
                    gene_row = seq_chunk.filter(pl.col('gene_id') == gene_id)
                    if gene_row.height == 0:
                        continue
                    
                    gene_info = gene_row.to_dicts()[0]
                    gene_name = gene_info.get('gene_name', gene_id)
                    gene_start = gene_info.get('start', 0)
                    gene_end = gene_info.get('end', 0)
                    strand = gene_info.get('strand', '+')
                    chrom = gene_info.get('seqname', chr_)
                    
                    # Extract probability arrays
                    donor_probs = pred_data.get('donor_prob', [])
                    acceptor_probs = pred_data.get('acceptor_prob', [])
                    neither_probs = pred_data.get('neither_prob', [])
                    
                    # IMPORTANT: Position coordinate semantics
                    # - 'positions' from run_spliceai_workflow.py contains ABSOLUTE genomic coordinates
                    # - We convert to RELATIVE positions for the 'position' column in nucleotide_scores
                    # - See meta_models/core/position_types.py for coordinate system documentation
                    absolute_positions = pred_data.get('positions', list(range(1, len(donor_probs) + 1)))
                    
                    # Create nucleotide-level DataFrame for this gene
                    n_positions = len(donor_probs)
                    if n_positions > 0:
                        # Convert absolute → relative using the position_types module
                        # This ensures consistent handling across the codebase
                        relative_positions = absolute_to_relative(
                            absolute_positions, 
                            gene_start=gene_start, 
                            gene_end=gene_end, 
                            strand=strand
                        )
                        
                        gene_nucleotide_df = pl.DataFrame({
                            'gene_id': [gene_id] * n_positions,
                            'gene_name': [gene_name] * n_positions,
                            'chrom': [chrom] * n_positions,
                            'strand': [strand] * n_positions,
                            'position': relative_positions,  # RELATIVE position (1-indexed, 5' to 3' in transcription space)
                            'genomic_position': absolute_positions,  # ABSOLUTE genomic coordinate
                            'donor_score': donor_probs,
                            'acceptor_score': acceptor_probs,
                            'neither_score': neither_probs,
                        })
                        nucleotide_scores_chunk.append(gene_nucleotide_df)
                
                # Concatenate all gene nucleotide scores for this chunk
                if nucleotide_scores_chunk:
                    chunk_nucleotide_df = pl.concat(nucleotide_scores_chunk)
                    
                    # Append to full nucleotide scores
                    if full_nucleotide_scores_df is None:
                        full_nucleotide_scores_df = chunk_nucleotide_df
                    else:
                        full_nucleotide_scores_df = pl.concat([full_nucleotide_scores_df, chunk_nucleotide_df])

            ########################################################################
            # 8.2  POST-PROCESSING (deduplication and validation)
            ########################################################################
            # NOTE: Evaluation and sequence extraction already done in mini-batch loop above
            # Here we just do final checks and save results
            
            # Quick sanity: skip empty chunks
            if positions_df_chunk.height == 0:
                if verbosity >= 1:
                    print_with_indent("[warning] Empty positions_df_chunk - skipping", indent_level=1)
                continue
            
            # Check for duplicates (should be rare after deduplication in extract_analysis_sequences)
            dupes = (
                df_seq.group_by(["gene_id", "position", "strand", "splice_type"])
                .agg(pl.col("sequence").count())
                .filter(pl.col("sequence") > 1)
            )
            if dupes.height > 0:
                # Log warning but don't fail - deduplication already handled in extract_analysis_sequences
                print_with_indent(
                    f"[warning] Found {dupes.height} duplicate sequence groups after extraction. "
                    f"This is expected when multiple transcripts share splice sites.",
                    indent_level=2
                )
                if verbosity >= 2:
                    print_with_indent(f"[debug] Duplicate groups:\n{dupes}", indent_level=2)
                
                # Perform additional deduplication if needed
                n_before = df_seq.height
                df_seq = df_seq.unique(subset=["gene_id", "position", "strand", "splice_type"])
                n_after = df_seq.height
                if n_before != n_after:
                    print_with_indent(
                        f"[info] Deduplicated {n_before - n_after} rows ({n_before} → {n_after})",
                        indent_level=2
                    )

            print_with_indent(
                f"[info] shape(df_seq): {df_seq.shape}, n(genes): {df_seq['gene_id'].n_unique()}",
                indent_level=2,
            )
            print_with_indent(f"[info] Columns: {df_seq.columns}", indent_level=2)

            # Write this chunk's sequences straight to disk; don't keep in RAM
            analysis_seq_path = data_handler.save_analysis_sequences(
                df_seq,
                chr=chr_,
                chunk_start=chunk_start + 1,
                chunk_end=chunk_end,
                aggregated=False,  # chunk-level file to minimise memory footprint
                output_subdir=output_subdir,
            )
            print_with_indent(
                f"[info] saved analysis sequences → {analysis_seq_path}",
                indent_level=2,
            )

            # Explicitly free memory for this chunk
            del df_seq

            if kwargs.get("add_kmer_features", False):
                positions_df_chunk = merge_contextual_features(
                    positions_df_chunk,
                    gene_meta_df = kwargs.get("gene_meta_df"),
                    kmer_df      = kwargs.get("kmer_df"))

            ########################################################################
            # 8.4  SAVE chunk artefacts
            ########################################################################
            data_handler.save_error_analysis(
                 error_df_chunk, chr=chr_, chunk_start=chunk_start+1, chunk_end=chunk_end,
                 output_subdir=output_subdir,
            )
            data_handler.save_splice_positions(
                 positions_df_chunk, chr=chr_, chunk_start=chunk_start+1, chunk_end=chunk_end,
                 enhanced=True,
                 output_subdir=output_subdir,
            )

            ########################################################################
            # 8.5  ACCUMULATE globals
            ########################################################################
            if do_aggregate:
                if full_error_df is None:
                    full_error_df = error_df_chunk
                else:
                    full_error_df = align_and_append(full_error_df, error_df_chunk, strict=False)

                if full_positions_df is None:
                    full_positions_df = positions_df_chunk
                else:
                    full_positions_df = align_and_append(full_positions_df,
                                                         positions_df_chunk, strict=False)

                # ── Progress: aggregated rows so far ───────────────────────────
                if verbosity >= 2:
                    tqdm.write(f"[aggregate] total positions aggregated so far: {full_positions_df.height:,}")

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
    if do_aggregate and full_positions_df is not None:
        print_with_indent(f"Total positions analyzed: {full_positions_df.height:,}", indent_level=1)
    print_section_separator()

    ################################################################################
    # 9. FINAL AGGREGATION & SAVE
    ################################################################################
    if do_aggregate and full_positions_df is not None and full_positions_df.height:

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

        # ====================================================================
        # ARTIFACT MANAGER: Check overwrite policy before saving
        # ====================================================================
        positions_artifact = artifact_manager.get_artifact_path('full_splice_positions_enhanced.tsv')
        errors_artifact = artifact_manager.get_artifact_path('full_splice_errors.tsv')
        
        should_save_positions = artifact_manager.should_overwrite(positions_artifact)
        should_save_errors = artifact_manager.should_overwrite(errors_artifact)
        
        if verbosity >= 1:
            if not should_save_positions:
                print_with_indent(
                    f"[artifact_manager] Skipping positions save (production mode, file exists): {positions_artifact}",
                    indent_level=1
                )
            if not should_save_errors:
                print_with_indent(
                    f"[artifact_manager] Skipping errors save (production mode, file exists): {errors_artifact}",
                    indent_level=1
                )
        
        # Route aggregated outputs to a sub-directory only when explicitly requested.
        _subdir_kw = {"output_subdir": output_subdir} if output_subdir and output_subdir != "meta_models" else {}

        # Save positions if allowed by overwrite policy
        if should_save_positions:
            pos_path = data_handler.save_splice_positions(
                full_positions_df,
                aggregated=True,
                enhanced=True,
                **_subdir_kw,
            )
            if verbosity >= 1:
                print_emphasized(f"[done] saved aggregated positions → {pos_path}")
        else:
            pos_path = str(positions_artifact)
            if verbosity >= 1:
                print_with_indent(f"[info] Using existing positions: {pos_path}", indent_level=1)
        
        # Save errors if allowed by overwrite policy
        if should_save_errors:
            err_path = data_handler.save_error_analysis(
                full_error_df,
                aggregated=True,
                **_subdir_kw,
            )
            if verbosity >= 1:
                print_emphasized(f"[done] saved aggregated errors    → {err_path}")
        else:
            err_path = str(errors_artifact)
            if verbosity >= 1:
                print_with_indent(f"[info] Using existing errors: {err_path}", indent_level=1)
        
        # ====================================================================
        # SAVE NUCLEOTIDE-LEVEL SCORES (Optional - only if enabled)
        # ====================================================================
        if config.save_nucleotide_scores and full_nucleotide_scores_df is not None and full_nucleotide_scores_df.height > 0:
            nucleotide_scores_artifact = artifact_manager.get_artifact_path('nucleotide_scores.tsv', create_dir=True)
            should_save_nucleotide_scores = artifact_manager.should_overwrite(nucleotide_scores_artifact)
            
            if should_save_nucleotide_scores:
                full_nucleotide_scores_df.write_csv(str(nucleotide_scores_artifact), separator='\t')
                if verbosity >= 1:
                    print_emphasized(f"[done] saved nucleotide scores → {nucleotide_scores_artifact}")
                    print_with_indent(
                        f"  Total nucleotides: {full_nucleotide_scores_df.height:,}",
                        indent_level=1
                    )
                    print_with_indent(
                        f"  Genes: {full_nucleotide_scores_df['gene_id'].n_unique()}",
                        indent_level=1
                    )
            else:
                if verbosity >= 1:
                    print_with_indent(
                        f"[artifact_manager] Skipping nucleotide scores save (production mode, file exists): {nucleotide_scores_artifact}",
                        indent_level=1
                    )
        
        # ====================================================================
        # SAVE GENE MANIFEST
        # ====================================================================
        manifest_artifact = artifact_manager.get_artifact_path('gene_manifest.tsv', create_dir=True)
        should_save_manifest = artifact_manager.should_overwrite(manifest_artifact)
        
        if should_save_manifest:
            # Update num_positions for processed genes from final positions DataFrame
            if full_positions_df is not None and full_positions_df.height > 0:
                position_counts = full_positions_df.group_by('gene_id').agg(
                    pl.count().alias('num_positions')
                ).to_dict(as_series=False)
                
                for gene_id, count in zip(position_counts['gene_id'], position_counts['num_positions']):
                    if gene_id in gene_manifest.entries:
                        gene_manifest.entries[gene_id].num_positions = count
            
            # Save manifest
            gene_manifest.save(str(manifest_artifact), use_polars=True)
            if verbosity >= 1:
                manifest_summary = gene_manifest.get_summary()
                print_emphasized(f"[done] saved gene manifest → {manifest_artifact}")
                print_with_indent(
                    f"  Processed: {manifest_summary['processed_genes']}/{manifest_summary['total_genes']} genes",
                    indent_level=1
                )
                if manifest_summary['failed_genes'] > 0:
                    print_with_indent(
                        f"  Failed: {manifest_summary['failed_genes']} genes",
                        indent_level=1
                    )
        else:
            if verbosity >= 1:
                print_with_indent(
                    f"[artifact_manager] Skipping manifest save (production mode, file exists): {manifest_artifact}",
                    indent_level=1
                )

    else:
        if do_aggregate:
            print_emphasized("[warning] No valid predictions produced by workflow")
        else:
            print_emphasized("[info] Final aggregation skipped (--no-final-aggregate)")

    runtime_min = (time.time() - run_start_time) / 60.0
    print_emphasized(f"[time] Total runtime: {runtime_min:,.1f} min")

    # Create result dictionary with success status and available dataframes
    result = {
        "success": True,  # Indicate workflow completed successfully
        "error_analysis": full_error_df if do_aggregate and full_error_df is not None else pl.DataFrame(),
        "positions": full_positions_df if do_aggregate and full_positions_df is not None else pl.DataFrame(),
        "analysis_sequences": full_analysis_seq_df if 'full_analysis_seq_df' in locals() and full_analysis_seq_df is not None else pl.DataFrame(),
        "gene_manifest": gene_manifest.to_dataframe(use_polars=True),  # NEW: Gene processing manifest
        "nucleotide_scores": full_nucleotide_scores_df if full_nucleotide_scores_df is not None else pl.DataFrame(),  # NEW: Nucleotide-level scores
        "paths": {
            "eval_dir": eval_dir,
            # Ensure we point to the exact sub-directory that holds this run's artefacts
            "output_dir": data_handler._get_output_dir(output_subdir=output_subdir) if hasattr(data_handler, '_get_output_dir') else None,
            # Artifact manager paths
            "artifacts_dir": str(artifact_manager.get_artifacts_dir()),
            "positions_artifact": str(artifact_manager.get_artifact_path('full_splice_positions_enhanced.tsv')),
            "errors_artifact": str(artifact_manager.get_artifact_path('full_splice_errors.tsv')),
            "manifest_artifact": str(artifact_manager.get_artifact_path('gene_manifest.tsv')),  # NEW
            "nucleotide_scores_artifact": str(artifact_manager.get_artifact_path('nucleotide_scores.tsv')),  # NEW
        },
        "artifact_manager": {
            "mode": artifact_manager.config.mode,
            "coverage": artifact_manager.config.coverage,
            "test_name": artifact_manager.config.test_name,
            "summary": artifact_manager.get_summary()
        },
        "manifest_summary": gene_manifest.get_summary()  # NEW: Manifest summary statistics
    }
    
    # Include overlapping genes data if it was processed
    if do_find_overlaping_genes and 'overlap_result' in locals() and overlap_result is not None:
        result["overlapping_genes"] = overlap_result
    
    return result
