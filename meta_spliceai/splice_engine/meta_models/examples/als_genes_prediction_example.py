#!/usr/bin/env python3
"""
Example script for running enhanced SpliceAI predictions on ALS-related genes.

This script demonstrates how to:
1. Run the enhanced splice prediction workflow on specific genes
2. Access the enhanced prediction results with donor, acceptor, and neither probabilities
3. Analyze splice junctions using the new junction analysis utilities

References:
- Brown et al. (2022). TDP-43 loss and ALS-risk SNPs drive mis-splicing and depletion 
  of UNC13A. Nature, 603(7899), 131-137.
- Melamed et al. (2019). Premature polyadenylation-mediated loss of stathmin-2 is a 
  hallmark of TDP-43-dependent neurodegeneration. Nature Neuroscience, 22(2), 180-190.
"""

import time
import pandas as pd
import polars as pl

# Import the enhanced splice prediction workflow
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_predictions import run_enhanced_splice_prediction_workflow
from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer

# Import the new junction analysis utilities
from meta_spliceai.splice_engine.meta_models.utils import (
    identify_splice_junctions,
    report_junction_statistics
)


def main():
    """Run enhanced SpliceAI predictions on ALS-related genes."""
    # Start timer
    start_time = time.time()
    
    # Get the basic paths from ErrorAnalyzer
    gtf_file = ErrorAnalyzer.gtf_file
    eval_dir = ErrorAnalyzer.eval_dir
    
    # Genome FASTA path (typically near the GTF file)
    genome_fasta = ErrorAnalyzer.genome_fasta
    # "/mnt/nfs1/splice-surveyor/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    # Print information about the example
    print("=" * 80)
    print(" Enhanced SpliceAI Predictions for ALS-related Genes ")
    print("=" * 80)
    print(f"GTF file: {gtf_file}")
    print(f"Genome FASTA: {genome_fasta}")
    print(f"Evaluation directory: {eval_dir}")
    
    # Define ALS-related genes of interest
    als_genes = [
        'STMN2',     # Stathmin-2 - affected by TDP-43 loss in ALS
        'UNC13A',    # UNC13A - contains ALS-risk SNPs that affect splicing
        'ENSG00000104435'  # Using Ensembl ID for STMN2 as an example of mixed ID/name usage

        # 'SOD1',    # Superoxide dismutase 1
        # 'FUS',     # Fused in sarcoma
        # 'TARDBP',  # TAR DNA binding protein (TDP-43)
        # 'C9orf72', # Chromosome 9 open reading frame 72
        # 'NEK1',    # NIMA-related kinase 1
        # 'TBK1',    # TANK-binding kinase 1
        # 'VCP',     # Valosin-containing protein
        # 'OPTN',    # Optineurin
        # 'UBQLN2',  # Ubiquilin 2
        # 'CHCHD10', # Coiled-coil-helix-coiled-coil-helix domain-containing protein 10
        # 'MATR3',   # Matrin 3
        # 'TUBA4A',  # Tubulin alpha 4a
        # 'PFN1'     # Profilin 1
    ]
    
    print(f"\nTargeting ALS-related genes: {', '.join(als_genes)}")
    
    # Option 1: Specify chromosomes (faster when you know them)
    # STMN2 is on chr8, UNC13A is on chr19
    specified_chromosomes = ["8", "19"]
    
    # Option 2: Process all chromosomes (slower but more flexible)
    # all_chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]
    
    # Option 3: Dynamically determine chromosomes (requires additional dependency)
    # This is just an example of how you might look up chromosomes
    # In practice, you could use a mapping file or database query
    try:
        from meta_spliceai.utils.gtf_utils import get_gene_chromosomes
        gene_to_chrom = get_gene_chromosomes(gtf_file, gene_ids=als_genes)
        if gene_to_chrom:
            dynamic_chromosomes = sorted(list(set(gene_to_chrom.values())))
            print(f"Dynamically determined chromosomes: {', '.join(dynamic_chromosomes)}")
            chromosomes_to_use = dynamic_chromosomes
        else:
            print("Couldn't determine chromosomes dynamically, using specified ones.")
            chromosomes_to_use = specified_chromosomes
    except (ImportError, AttributeError):
        print("get_gene_chromosomes function not available, using specified chromosomes.")
        chromosomes_to_use = specified_chromosomes
    
    # Create a configuration for enhanced SpliceAI predictions
    config = SpliceAIConfig(
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        eval_dir=eval_dir,
        output_subdir="als_predictions",  # Custom subdirectory specific to ALS genes
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        test_mode=False,  # Use test mode (processes fewer chromosomes)
        chromosomes=chromosomes_to_use,  # Use determined chromosomes

        # Data processing are disabled by default
        # do_extract_sequences=False,       # Disable sequence extraction
        # do_extract_splice_sites=False,    # Disable splice site extraction 
        # do_extract_annotations=False      # Disable annotation extraction
    )
    
    # Run the enhanced SpliceAI prediction workflow with target genes
    print("\nRunning enhanced SpliceAI prediction workflow for ALS genes...")
    workflow_results = run_enhanced_splice_prediction_workflow(
        config=config,
        target_genes=als_genes,  # Only process these specific genes
        verbosity=1,  # Control output detail: 0=minimal, 1=normal progress, 2=all details
        test_mode=False  # Set to False to process all genes
    )
    
    # Display information about the results
    print("\nEnhanced SpliceAI prediction workflow completed!")
    
    # Access the positions DataFrame with all three probabilities
    positions_df = workflow_results.get('positions')
    if positions_df is not None and positions_df.height > 0:
        print(f"\nEnhanced positions DataFrame shape: {positions_df.shape}")
        
        # Show gene counts
        gene_counts = positions_df.group_by('gene_id').agg(
            pl.len().alias('position_count'),  # Use len() instead of count() for Polars
            pl.n_unique('transcript_id').alias('transcript_count')
        ).sort('position_count', descending=True)
        
        print("\nGene summary:")
        print(gene_counts)
        
        # Display example of alternative splicing by counting positions used in multiple transcripts
        print("\nAlternative splicing analysis (positions shared across transcripts):")
        transcript_sharing = positions_df.group_by(['gene_id', 'position'])\
            .agg(pl.n_unique('transcript_id').alias('transcripts_using_site'))\
            .filter(pl.col('transcripts_using_site') > 1)\
            .sort('transcripts_using_site', descending=True)\
            .head(10)
        
        print(transcript_sharing)
        
        # Display a sample with the three probability scores
        print("\nSample of positions with all three probability scores:")
        sample_cols = [
            'gene_id', 'transcript_id', 'position', 'pred_type', 
            'splice_type', 'donor_score', 'acceptor_score', 'neither_score'
        ]
        
        # Ensure all expected columns are present
        sample_cols = [col for col in sample_cols if col in positions_df.columns]
        
        # Show a sample sorted by position
        sample_df = positions_df.select(sample_cols).sort('position').head(10)
        print(sample_df)
        
        # Identify junctions using the utility function
        donor_threshold = 0.9
        acceptor_threshold = 0.9
        
        print(f"\nIdentifying potential splice site junctions (donor sites with score > {donor_threshold} and acceptor sites with score > {acceptor_threshold})")
        
        # Call the utility function to identify junctions
        junctions_df, donor_sites, acceptor_sites = identify_splice_junctions(
            positions_df, 
            donor_threshold=donor_threshold, 
            acceptor_threshold=acceptor_threshold
        )
        
        # Generate reports on the identified junctions
        junction_counts, junction_samples = report_junction_statistics(
            junctions_df, donor_sites, acceptor_sites
        )
        
        # Show high-confidence splice sites
        print("\nHigh-confidence splice sites (donor score > 0.9 or acceptor score > 0.9):")
        high_conf_sites = positions_df.filter(
            (pl.col('donor_score') > 0.9) | (pl.col('acceptor_score') > 0.9)
        ).sort(
            ['gene_id', 'position']
        ).head(10)
        
        print(high_conf_sites.select([
            'gene_id', 'transcript_id', 'position', 'splice_type', 'donor_score', 'acceptor_score'
        ]))
    else:
        print("\nNo enhanced positions data found.")
    
    # Calculate and display execution time
    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
    
    # Show the path to access the full dataset
    print("\nTo load the enhanced positions data in your own code:")
    print("```python")
    print("import polars as pl")
    print("from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler")
    print("from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer")
    print("from meta_spliceai.splice_engine.meta_models.utils import identify_splice_junctions, report_junction_statistics")
    print("")
    print("# Load positions dataframe")
    print("data_handler = MetaModelDataHandler('/home/bchiu/work/splice-surveyor/outputs/eval')")
    print("positions_df = data_handler.load_splice_positions(")
    print("    subject=\"splice_positions_enhanced\", # Important to specify enhanced version")
    print("    output_subdir=\"meta_models\"")
    print(")")
    print("")
    print("# Analyze junctions")
    print("junctions_df, donor_sites, acceptor_sites = identify_splice_junctions(positions_df)")
    print("junction_counts, junction_samples = report_junction_statistics(junctions_df, donor_sites, acceptor_sites)")
    print("```")


if __name__ == "__main__":
    main()
