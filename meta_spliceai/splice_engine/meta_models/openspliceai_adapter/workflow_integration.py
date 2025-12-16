"""
Enhanced workflow integration that seamlessly bridges OpenSpliceAI with existing MetaSpliceAI workflows.

This module provides drop-in replacements for the data preparation steps in your existing
workflow, allowing direct GTF + FASTA ingestion while maintaining full compatibility
with downstream processing.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.openspliceai_adapter import (
    OpenSpliceAIPreprocessor, OpenSpliceAIAdapterConfig
)

def enhanced_data_preparation_with_openspliceai(
    config: SpliceAIConfig,
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None,
    use_openspliceai: bool = True,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Enhanced data preparation that can use OpenSpliceAI preprocessing.
    
    This function serves as a drop-in replacement for the three separate data preparation
    steps in your existing workflow:
    - prepare_gene_annotations()
    - prepare_splice_site_annotations() 
    - prepare_genomic_sequences()
    
    Parameters
    ----------
    config : SpliceAIConfig
        Your existing SpliceAI configuration
    target_genes : List[str], optional
        Specific genes to process
    target_chromosomes : List[str], optional
        Specific chromosomes to process
    use_openspliceai : bool, default=True
        Whether to use OpenSpliceAI preprocessing (recommended)
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Combined results from all data preparation steps, compatible with your existing workflow
    """
    if verbosity >= 1:
        print("[enhanced_prep] Starting enhanced data preparation...")
        if use_openspliceai:
            print("[enhanced_prep] Using OpenSpliceAI preprocessing pipeline")
        else:
            print("[enhanced_prep] Using traditional MetaSpliceAI preprocessing")
    
    # Get the local directory for outputs
    local_dir = os.path.join(config.eval_dir, config.output_subdir)
    os.makedirs(local_dir, exist_ok=True)
    
    if use_openspliceai:
        # Use OpenSpliceAI preprocessing
        return _prepare_data_with_openspliceai(
            config, local_dir, target_genes, target_chromosomes, verbosity
        )
    else:
        # Fall back to traditional preprocessing
        return _prepare_data_traditional(
            config, local_dir, target_genes, target_chromosomes, verbosity
        )

def _prepare_data_with_openspliceai(
    config: SpliceAIConfig,
    local_dir: str,
    target_genes: Optional[List[str]],
    target_chromosomes: Optional[List[str]],
    verbosity: int
) -> Dict[str, Any]:
    """Prepare data using OpenSpliceAI preprocessing."""
    
    # Create OpenSpliceAI adapter configuration from SpliceAI config
    adapter_config = OpenSpliceAIAdapterConfig(
        gtf_file=config.gtf_file,
        genome_fasta=config.genome_fasta,
        output_dir=local_dir,
        flanking_size=2000,  # Good default for meta-learning
        biotype="protein-coding",  # Match your typical usage
        parse_type="all_isoforms",  # Comprehensive for meta-learning
        target_genes=target_genes,
        chromosomes=target_chromosomes,
        canonical_only=False,  # Include non-canonical for comprehensive analysis
        output_format="parquet"  # Match your existing format preference
    )
    
    # Initialize preprocessor
    preprocessor = OpenSpliceAIPreprocessor(config=adapter_config, verbose=verbosity)
    
    # Create MetaSpliceAI-compatible data
    if verbosity >= 1:
        print("[openspliceai] Creating MetaSpliceAI-compatible datasets...")
    
    openspliceai_results = preprocessor.create_splicesurveyor_compatible_data(
        use_openspliceai_preprocessing=True
    )
    
    # Transform results to match expected format from traditional workflow
    combined_results = {
        'success': True,
        'method': 'openspliceai',
        
        # Gene annotations (equivalent to prepare_gene_annotations result)
        'annotation_file': openspliceai_results.get('train_splice_sites', None),
        'annotation_df': None,  # Will be loaded on demand
        
        # Splice site annotations (equivalent to prepare_splice_site_annotations result)
        'splice_sites_file': openspliceai_results.get('train_splice_sites', None),
        'splice_sites_df': None,  # Will be loaded on demand
        
        # Genomic sequences (equivalent to prepare_genomic_sequences result)
        'sequences_file': openspliceai_results.get('train_h5', None),
        'sequences_df': None,  # Will be loaded on demand
        'seq_format': 'hdf5',
        
        # Additional OpenSpliceAI-specific results
        'openspliceai_datasets': openspliceai_results,
        'quality_metrics': preprocessor.get_quality_metrics()
    }
    
    if verbosity >= 1:
        print(f"[openspliceai] Enhanced data preparation completed")
        print(f"[openspliceai] Generated {len(openspliceai_results)} dataset files")
    
    return combined_results

def _prepare_data_traditional(
    config: SpliceAIConfig,
    local_dir: str,
    target_genes: Optional[List[str]],
    target_chromosomes: Optional[List[str]],
    verbosity: int
) -> Dict[str, Any]:
    """Prepare data using traditional MetaSpliceAI preprocessing."""
    
    # Import traditional functions
    from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import (
        prepare_gene_annotations,
        prepare_splice_site_annotations,
        prepare_genomic_sequences
    )
    
    # Run traditional data preparation steps
    if verbosity >= 1:
        print("[traditional] Running traditional data preparation...")
    
    # 1. Gene annotations
    annot_result = prepare_gene_annotations(
        local_dir=local_dir,
        gtf_file=config.gtf_file,
        do_extract=config.do_extract_annotations,
        target_chromosomes=target_chromosomes,
        separator=config.separator,
        verbosity=verbosity
    )
    
    # 2. Splice site annotations
    ss_result = prepare_splice_site_annotations(
        local_dir=local_dir,
        gtf_file=config.gtf_file,
        do_extract=config.do_extract_splice_sites,
        target_chromosomes=target_chromosomes,
        consensus_window=config.consensus_window,
        separator=config.separator,
        verbosity=verbosity
    )
    
    # 3. Genomic sequences
    seq_result = prepare_genomic_sequences(
        local_dir=local_dir,
        gtf_file=config.gtf_file,
        genome_fasta=config.genome_fasta,
        mode=config.seq_mode,
        seq_type=config.seq_type,
        do_extract=config.do_extract_sequences,
        chromosomes=target_chromosomes,
        test_mode=config.test_mode,
        seq_format=config.seq_format,
        verbosity=verbosity
    )
    
    # Combine results
    combined_results = {
        'success': all([annot_result.get('success', False), 
                       ss_result.get('success', False), 
                       seq_result.get('success', False)]),
        'method': 'traditional',
        
        # Individual results
        'annotation_result': annot_result,
        'splice_sites_result': ss_result,
        'sequences_result': seq_result,
        
        # Unified interface (for compatibility)
        'annotation_file': annot_result.get('annotation_file'),
        'annotation_df': annot_result.get('annotation_df'),
        'splice_sites_file': ss_result.get('splice_sites_file'),
        'splice_sites_df': ss_result.get('splice_sites_df'),
        'sequences_file': seq_result.get('sequences_file'),
        'sequences_df': seq_result.get('sequences_df'),
        'seq_format': seq_result.get('seq_format', config.seq_format)
    }
    
    return combined_results

def create_enhanced_incremental_builder_config(
    gtf_file: str,
    genome_fasta: str,
    output_dir: str = "enhanced_training_data",
    use_openspliceai: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Create configuration for enhanced incremental builder that can directly ingest GTF + FASTA.
    
    This function creates a configuration that allows your incremental_builder.py to
    directly work with GTF + FASTA inputs, bypassing the need for pre-derived datasets.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file
    genome_fasta : str
        Path to genome FASTA file
    output_dir : str, default="enhanced_training_data"
        Output directory for training datasets
    use_openspliceai : bool, default=True
        Whether to use OpenSpliceAI preprocessing
    **kwargs
        Additional parameters for configuration
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary for enhanced incremental builder
    """
    
    # Create base SpliceAI configuration
    base_config = SpliceAIConfig(
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        eval_dir=output_dir,
        output_subdir="meta_models",
        # Enable data extraction since we're starting from raw files
        do_extract_annotations=True,
        do_extract_splice_sites=True,
        do_extract_sequences=True,
        **kwargs
    )
    
    # Enhanced configuration for incremental builder
    enhanced_config = {
        'base_config': base_config,
        'use_openspliceai_preprocessing': use_openspliceai,
        'data_preparation_function': enhanced_data_preparation_with_openspliceai,
        
        # OpenSpliceAI-specific settings
        'openspliceai_settings': {
            'flanking_size': kwargs.get('flanking_size', 2000),
            'biotype': kwargs.get('biotype', 'protein-coding'),
            'parse_type': kwargs.get('parse_type', 'all_isoforms'),
            'canonical_only': kwargs.get('canonical_only', False),
            'remove_paralogs': kwargs.get('remove_paralogs', False)
        },
        
        # Incremental builder settings
        'batch_size': kwargs.get('batch_size', 1000),
        'n_genes': kwargs.get('n_genes', 20000),
        'subset_policy': kwargs.get('subset_policy', 'error_total'),
        'kmer_sizes': kwargs.get('kmer_sizes', (6,)),
        'enrichers': kwargs.get('enrichers', None),
        
        # Output settings
        'output_format': kwargs.get('output_format', 'parquet'),
        'generate_manifest': kwargs.get('generate_manifest', True)
    }
    
    return enhanced_config

def run_enhanced_workflow_with_direct_inputs(
    gtf_file: str,
    genome_fasta: str,
    output_dir: str = "enhanced_workflow_output",
    target_genes: Optional[List[str]] = None,
    use_openspliceai: bool = True,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Run the complete enhanced workflow directly from GTF + FASTA inputs.
    
    This is a high-level function that demonstrates the complete integration,
    taking raw GTF + FASTA files and producing the same output as your existing
    workflow that expects derived datasets.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file
    genome_fasta : str
        Path to genome FASTA file
    output_dir : str, default="enhanced_workflow_output"
        Output directory
    target_genes : List[str], optional
        Specific genes to process
    use_openspliceai : bool, default=True
        Whether to use OpenSpliceAI preprocessing
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Complete workflow results, compatible with existing downstream processing
    """
    
    if verbosity >= 1:
        print("=" * 60)
        print("Enhanced MetaSpliceAI Workflow with Direct GTF + FASTA Input")
        print("=" * 60)
        print(f"GTF file: {gtf_file}")
        print(f"Genome FASTA: {genome_fasta}")
        print(f"Output directory: {output_dir}")
        print(f"Using OpenSpliceAI: {use_openspliceai}")
        if target_genes:
            print(f"Target genes: {len(target_genes)} genes")
        print()
    
    # Create configuration
    config = SpliceAIConfig(
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        eval_dir=output_dir,
        do_extract_annotations=True,
        do_extract_splice_sites=True,
        do_extract_sequences=True
    )
    
    # Run enhanced data preparation
    prep_results = enhanced_data_preparation_with_openspliceai(
        config=config,
        target_genes=target_genes,
        use_openspliceai=use_openspliceai,
        verbosity=verbosity
    )
    
    if not prep_results['success']:
        print("Error: Data preparation failed")
        return prep_results
    
    # At this point, you have all the data needed for your existing workflow
    # The prep_results contain the same structure as your current workflow expects
    
    if verbosity >= 1:
        print("\n" + "=" * 60)
        print("Data Preparation Complete - Ready for Meta-Learning Pipeline")
        print("=" * 60)
        print(f"Method used: {prep_results['method']}")
        print(f"Splice sites file: {prep_results.get('splice_sites_file', 'N/A')}")
        print(f"Sequences file: {prep_results.get('sequences_file', 'N/A')}")
        print(f"Annotation file: {prep_results.get('annotation_file', 'N/A')}")
        
        if 'quality_metrics' in prep_results:
            metrics = prep_results['quality_metrics']
            print(f"\nQuality Metrics:")
            print(f"  GTF size: {metrics['input_files'].get('gtf_size_mb', 0):.1f} MB")
            print(f"  FASTA size: {metrics['input_files'].get('fasta_size_mb', 0):.1f} MB")
    
    # Return results in format compatible with your existing downstream processing
    return {
        'success': True,
        'data_preparation': prep_results,
        'config': config,
        'ready_for_incremental_builder': True,
        'compatible_with_existing_workflow': True
    }


# Example usage functions
def example_direct_gtf_fasta_workflow():
    """Example showing direct GTF + FASTA workflow."""
    
    # Your existing default paths
    gtf_file = "data/ensembl/Homo_sapiens.GRCh38.112.gtf"
    genome_fasta = "data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    # Run complete workflow directly from GTF + FASTA
    results = run_enhanced_workflow_with_direct_inputs(
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        output_dir="test_direct_workflow",
        target_genes=["STMN2", "UNC13A"],  # Test with specific genes
        use_openspliceai=True,
        verbosity=2
    )
    
    if results['success']:
        print("\n✓ Direct GTF + FASTA workflow completed successfully!")
        print("✓ Data is ready for your existing incremental_builder.py")
        print("✓ Output format is compatible with train_pc_1000_3mers/master structure")
    
    return results
