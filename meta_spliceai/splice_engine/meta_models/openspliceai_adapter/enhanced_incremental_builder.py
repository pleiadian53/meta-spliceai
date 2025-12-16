"""
Enhanced Incremental Builder with Direct GTF + FASTA Support

This module extends your existing incremental_builder.py to support direct GTF + FASTA
input using OpenSpliceAI preprocessing, while maintaining full compatibility with your
existing workflow and output format (train_pc_1000_3mers/master structure).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.builder.incremental_builder import (
    incremental_build_training_dataset as original_incremental_build
)
from meta_spliceai.splice_engine.meta_models.openspliceai_adapter.workflow_integration import (
    enhanced_data_preparation_with_openspliceai,
    create_enhanced_incremental_builder_config
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

def enhanced_incremental_build_training_dataset(
    # Direct GTF + FASTA inputs (new capability)
    gtf_file: Optional[str] = None,
    genome_fasta: Optional[str] = None,
    use_openspliceai_preprocessing: bool = True,
    
    # Existing parameters (maintained for compatibility)
    eval_dir: Optional[str] = None,
    output_dir: str = "enhanced_training_dataset",
    n_genes: int = 20_000,
    subset_policy: str = "error_total",
    batch_size: int = 1_000,
    kmer_sizes: tuple = (6,),
    enrichers: Optional[List[str]] = None,
    downsample_kwargs: Optional[Dict] = None,
    batch_rows: int = 500_000,
    gene_types: Optional[List[str]] = None,
    additional_gene_ids: Optional[List[str]] = None,
    
    # OpenSpliceAI-specific parameters
    flanking_size: int = 2000,
    biotype: str = "protein-coding",
    parse_type: str = "all_isoforms",
    canonical_only: bool = False,
    remove_paralogs: bool = False,
    
    # Control parameters
    overwrite: bool = False,
    generate_manifest: bool = True,
    verbose: int = 1,
    **kwargs
) -> Path:
    """
    Enhanced incremental training dataset builder with direct GTF + FASTA support.
    
    This function extends your existing incremental builder to directly accept GTF + FASTA
    inputs using OpenSpliceAI preprocessing, while producing the same output format as
    your current workflow (compatible with train_pc_1000_3mers/master structure).
    
    Parameters
    ----------
    gtf_file : str, optional
        Path to GTF annotation file. If provided, enables direct GTF + FASTA workflow.
    genome_fasta : str, optional
        Path to genome FASTA file. Required if gtf_file is provided.
    use_openspliceai_preprocessing : bool, default=True
        Whether to use OpenSpliceAI preprocessing (recommended for direct GTF + FASTA input)
    
    # ... (all other parameters same as original incremental_build_training_dataset)
    
    Returns
    -------
    Path
        Path to the master partitioned dataset directory (same as original)
    """
    
    if verbose >= 1:
        print("=" * 70)
        print("Enhanced Incremental Training Dataset Builder")
        print("=" * 70)
        
        if gtf_file and genome_fasta:
            print("🆕 Direct GTF + FASTA Input Mode")
            print(f"   GTF file: {gtf_file}")
            print(f"   Genome FASTA: {genome_fasta}")
            print(f"   Using OpenSpliceAI: {use_openspliceai_preprocessing}")
        else:
            print("📁 Traditional Derived Dataset Mode")
            print(f"   Eval directory: {eval_dir}")
        
        print(f"   Output directory: {output_dir}")
        print(f"   Genes to process: {n_genes}")
        print(f"   Batch size: {batch_size}")
        print()
    
    # Determine workflow mode
    direct_input_mode = gtf_file is not None and genome_fasta is not None
    
    if direct_input_mode:
        # NEW: Direct GTF + FASTA workflow
        return _build_from_direct_inputs(
            gtf_file=gtf_file,
            genome_fasta=genome_fasta,
            use_openspliceai_preprocessing=use_openspliceai_preprocessing,
            output_dir=output_dir,
            n_genes=n_genes,
            subset_policy=subset_policy,
            batch_size=batch_size,
            kmer_sizes=kmer_sizes,
            enrichers=enrichers,
            downsample_kwargs=downsample_kwargs,
            batch_rows=batch_rows,
            gene_types=gene_types,
            additional_gene_ids=additional_gene_ids,
            flanking_size=flanking_size,
            biotype=biotype,
            parse_type=parse_type,
            canonical_only=canonical_only,
            remove_paralogs=remove_paralogs,
            overwrite=overwrite,
            generate_manifest=generate_manifest,
            verbose=verbose,
            **kwargs
        )
    else:
        # EXISTING: Traditional workflow with derived datasets
        if verbose >= 1:
            print("🔄 Using existing incremental builder workflow...")
        
        return original_incremental_build(
            eval_dir=eval_dir,
            output_dir=output_dir,
            n_genes=n_genes,
            subset_policy=subset_policy,
            batch_size=batch_size,
            kmer_sizes=kmer_sizes,
            enrichers=enrichers,
            downsample_kwargs=downsample_kwargs,
            batch_rows=batch_rows,
            gene_types=gene_types,
            additional_gene_ids=additional_gene_ids,
            overwrite=overwrite,
            generate_manifest=generate_manifest,
            verbose=verbose,
            **kwargs
        )

def _build_from_direct_inputs(
    gtf_file: str,
    genome_fasta: str,
    use_openspliceai_preprocessing: bool,
    output_dir: str,
    n_genes: int,
    subset_policy: str,
    batch_size: int,
    kmer_sizes: tuple,
    enrichers: Optional[List[str]],
    downsample_kwargs: Optional[Dict],
    batch_rows: int,
    gene_types: Optional[List[str]],
    additional_gene_ids: Optional[List[str]],
    flanking_size: int,
    biotype: str,
    parse_type: str,
    canonical_only: bool,
    remove_paralogs: bool,
    overwrite: bool,
    generate_manifest: bool,
    verbose: int,
    **kwargs
) -> Path:
    """Build training dataset from direct GTF + FASTA inputs."""
    
    if verbose >= 1:
        print("🔧 Step 1: Enhanced Data Preparation with OpenSpliceAI")
    
    # Create temporary evaluation directory for data preparation
    temp_eval_dir = os.path.join(output_dir, "temp_data_preparation")
    os.makedirs(temp_eval_dir, exist_ok=True)
    
    # Create SpliceAI configuration for data preparation
    config = SpliceAIConfig(
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        eval_dir=temp_eval_dir,
        output_subdir="openspliceai_processed",
        do_extract_annotations=True,
        do_extract_splice_sites=True,
        do_extract_sequences=True,
        seq_format="parquet"  # Match your existing preference
    )
    
    # Run enhanced data preparation
    prep_results = enhanced_data_preparation_with_openspliceai(
        config=config,
        target_genes=additional_gene_ids,  # Use additional_gene_ids if provided
        use_openspliceai=use_openspliceai_preprocessing,
        verbosity=verbose
    )
    
    if not prep_results['success']:
        raise RuntimeError("Enhanced data preparation failed")
    
    if verbose >= 1:
        print("✅ Data preparation completed successfully")
        print(f"   Method: {prep_results['method']}")
        print(f"   Splice sites: {prep_results.get('splice_sites_file', 'N/A')}")
        print(f"   Sequences: {prep_results.get('sequences_file', 'N/A')}")
        print()
        print("🔧 Step 2: Running Incremental Builder with Prepared Data")
    
    # Now run the original incremental builder with the prepared data
    # Point it to the temporary evaluation directory we just created
    final_output = original_incremental_build(
        eval_dir=temp_eval_dir,
        output_dir=output_dir,
        n_genes=n_genes,
        subset_policy=subset_policy,
        batch_size=batch_size,
        kmer_sizes=kmer_sizes,
        enrichers=enrichers,
        downsample_kwargs=downsample_kwargs,
        batch_rows=batch_rows,
        gene_types=gene_types,
        additional_gene_ids=additional_gene_ids,
        overwrite=overwrite,
        generate_manifest=generate_manifest,
        verbose=verbose,
        **kwargs
    )
    
    if verbose >= 1:
        print("✅ Enhanced incremental builder completed successfully!")
        print(f"   Final output: {final_output}")
        print(f"   Compatible with train_pc_1000_3mers/master structure: ✓")
        print()
        print("🧹 Cleaning up temporary files...")
    
    # Clean up temporary directory
    import shutil
    try:
        shutil.rmtree(temp_eval_dir)
        if verbose >= 2:
            print(f"   Removed temporary directory: {temp_eval_dir}")
    except Exception as e:
        if verbose >= 1:
            print(f"   Warning: Could not remove temporary directory: {e}")
    
    return final_output

def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface for enhanced incremental builder."""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Incremental Training Dataset Builder with Direct GTF + FASTA Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct GTF + FASTA workflow (NEW)
  python enhanced_incremental_builder.py \\
    --gtf-file data/ensembl/Homo_sapiens.GRCh38.112.gtf \\
    --genome-fasta data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
    --output-dir enhanced_training_data \\
    --n-genes 1000 \\
    --use-openspliceai
  
  # Traditional workflow (EXISTING)
  python enhanced_incremental_builder.py \\
    --eval-dir data/ensembl/spliceai_eval \\
    --output-dir traditional_training_data \\
    --n-genes 1000
  
  # Disease-focused analysis
  python enhanced_incremental_builder.py \\
    --gtf-file data/ensembl/Homo_sapiens.GRCh38.112.gtf \\
    --genome-fasta data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
    --output-dir als_analysis \\
    --flanking-size 10000 \\
    --biotype protein-coding \\
    --additional-gene-ids STMN2,UNC13A,TARDBP \\
    --use-openspliceai
        """
    )
    
    # Input source options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--gtf-file', type=str,
        help='Path to GTF annotation file (enables direct GTF + FASTA workflow)'
    )
    input_group.add_argument(
        '--eval-dir', type=str,
        help='Evaluation directory with derived datasets (traditional workflow)'
    )
    
    # Required for direct GTF + FASTA workflow
    parser.add_argument(
        '--genome-fasta', type=str,
        help='Path to genome FASTA file (required with --gtf-file)'
    )
    
    # OpenSpliceAI preprocessing options
    parser.add_argument(
        '--use-openspliceai', action='store_true', default=True,
        help='Use OpenSpliceAI preprocessing (default: True)'
    )
    parser.add_argument(
        '--no-openspliceai', dest='use_openspliceai', action='store_false',
        help='Disable OpenSpliceAI preprocessing'
    )
    parser.add_argument(
        '--flanking-size', type=int, default=2000, choices=[80, 400, 2000, 10000],
        help='Context window size for OpenSpliceAI (default: 2000)'
    )
    parser.add_argument(
        '--biotype', type=str, default='protein-coding',
        choices=['protein-coding', 'non-coding', 'all'],
        help='Gene biotype filter (default: protein-coding)'
    )
    parser.add_argument(
        '--parse-type', type=str, default='all_isoforms',
        choices=['canonical', 'all_isoforms'],
        help='Transcript parsing type (default: all_isoforms)'
    )
    parser.add_argument(
        '--canonical-only', action='store_true', default=False,
        help='Include only canonical splice sites'
    )
    parser.add_argument(
        '--remove-paralogs', action='store_true', default=False,
        help='Remove paralogous sequences (recommended for disease studies)'
    )
    
    # Standard incremental builder options
    parser.add_argument(
        '--output-dir', type=str, default='enhanced_training_dataset',
        help='Output directory for training dataset (default: enhanced_training_dataset)'
    )
    parser.add_argument(
        '--n-genes', type=int, default=20000,
        help='Number of genes to process (default: 20000)'
    )
    parser.add_argument(
        '--subset-policy', type=str, default='error_total',
        choices=['error_total', 'error_fp', 'error_fn', 'random', 'custom'],
        help='Gene selection policy (default: error_total)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1000,
        help='Number of genes per batch (default: 1000)'
    )
    parser.add_argument(
        '--kmer-sizes', type=int, nargs='+', default=[6],
        help='K-mer sizes for feature extraction (default: [6])'
    )
    parser.add_argument(
        '--gene-types', type=str, nargs='+',
        help='Gene types to include (e.g., protein_coding lncRNA)'
    )
    parser.add_argument(
        '--additional-gene-ids', type=str,
        help='Comma-separated list of specific gene IDs to include'
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--no-manifest', dest='generate_manifest', action='store_false', default=True,
        help='Skip generating gene manifest file'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1, choices=[0, 1, 2],
        help='Verbosity level (0=quiet, 1=normal, 2=detailed)'
    )
    
    return parser

def main():
    """Main entry point for enhanced incremental builder."""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.gtf_file and not args.genome_fasta:
        parser.error("--genome-fasta is required when using --gtf-file")
    
    # Parse additional gene IDs
    additional_gene_ids = None
    if args.additional_gene_ids:
        additional_gene_ids = [g.strip() for g in args.additional_gene_ids.split(',')]
    
    # Convert kmer_sizes to tuple
    kmer_sizes = tuple(args.kmer_sizes)
    
    try:
        # Run enhanced incremental builder
        output_path = enhanced_incremental_build_training_dataset(
            gtf_file=args.gtf_file,
            genome_fasta=args.genome_fasta,
            use_openspliceai_preprocessing=args.use_openspliceai,
            eval_dir=args.eval_dir,
            output_dir=args.output_dir,
            n_genes=args.n_genes,
            subset_policy=args.subset_policy,
            batch_size=args.batch_size,
            kmer_sizes=kmer_sizes,
            gene_types=args.gene_types,
            additional_gene_ids=additional_gene_ids,
            flanking_size=args.flanking_size,
            biotype=args.biotype,
            parse_type=args.parse_type,
            canonical_only=args.canonical_only,
            remove_paralogs=args.remove_paralogs,
            overwrite=args.overwrite,
            generate_manifest=args.generate_manifest,
            verbose=args.verbose
        )
        
        print("\n" + "=" * 70)
        print("🎉 Enhanced Incremental Builder Completed Successfully!")
        print("=" * 70)
        print(f"📁 Output directory: {output_path}")
        print(f"📊 Dataset structure: Compatible with train_pc_1000_3mers/master")
        print(f"🔗 Ready for meta-learning pipeline: ✓")
        
        if args.gtf_file:
            print(f"🆕 Direct GTF + FASTA workflow: ✓")
            print(f"🧬 OpenSpliceAI preprocessing: {'✓' if args.use_openspliceai else '✗'}")
        
        print("\nNext steps:")
        print("1. Examine the generated dataset in the output directory")
        print("2. Use the dataset with your existing meta-learning pipeline")
        print("3. Compare results with traditional preprocessing if desired")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
