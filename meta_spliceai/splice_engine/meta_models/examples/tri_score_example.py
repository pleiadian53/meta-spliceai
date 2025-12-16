"""
Example: Three Probability Score Enhanced Workflow

This script demonstrates how to use the enhanced workflow with all three probability scores
(donor, acceptor, neither) for each position, for more comprehensive splice site analysis.
"""

import os
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Import the enhanced modules
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.core.enhanced_workflow import enhanced_process_predictions_with_all_scores
from meta_spliceai.splice_engine.meta_models.workflows.splice_predictions import predict_splice_sites_for_genes
from meta_spliceai.splice_engine.run_spliceai_workflow import load_spliceai_models, load_chromosome_sequence_streaming
from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.utils_fs import read_splice_sites


def main():
    """Run an example of the tri-score enhanced splice prediction workflow."""
    
    # Get the basic paths from ErrorAnalyzer
    gtf_file = ErrorAnalyzer.gtf_file
    eval_dir = ErrorAnalyzer.eval_dir
    
    # Print information about the example
    print("=" * 80)
    print(" Three Probability Score Enhanced Workflow Example ")
    print("=" * 80)
    print(f"GTF file: {gtf_file}")
    print(f"Evaluation directory: {eval_dir}")
    
    # List of specific genes to analyze
    target_genes = {'STMN2': '8', 'UNC13A': '19', 'ENSG00000104435': '1'}
    # ["TARDBP", "SOD1", "FUS"]  # ALS-related genes
    
    # Create a configuration
    config = SpliceAIConfig(
        gtf_file=gtf_file,
        eval_dir=eval_dir,
        output_subdir="tri_score_analysis",
        threshold=0.3,  # Lower threshold to capture more potential splice sites
        consensus_window=2,
        error_window=500,
        test_mode=True
    )
    
    # Load annotations using the new MetaModelDataHandler.load_splice_annotations method
    data_handler = MetaModelDataHandler(eval_dir=eval_dir)
    print(f"\nLoading splice site annotations...")
    annotations_df = data_handler.load_splice_annotations()
    
    print(f"Loaded annotations: {annotations_df.shape}")
    
    # Process target genes with SpliceAI to get predictions
    print(f"\nProcessing target genes: {target_genes}")
    
    # First load the gene sequences
    print("Loading gene sequences...")
    gene_df = load_gene_sequences_for_targets(target_genes, config, seq_type="standard", prioritized_chromosomes=['8', '19'])
    if gene_df is None or gene_df.height == 0:
        print("Error: Could not load gene sequences for the target genes.")
        # Try minmax sequences if standard sequences are not found
        print("Attempting to load minmax sequences instead...")
        gene_df = load_gene_sequences_for_targets(target_genes, config, seq_type="minmax", prioritized_chromosomes=['8', '19'])
        if gene_df is None or gene_df.height == 0:
            print("Error: Could not load any gene sequences for the target genes.")
            return
    
    print(f"Loaded sequences for {gene_df.height} genes")
        
    # Load SpliceAI models
    print("Loading SpliceAI models...")
    models = load_spliceai_models()
    
    # Get the gene predictions with all three scores
    print("Running SpliceAI predictions...")
    predictions = predict_splice_sites_for_genes(
        gene_df,
        models=models,
        context=10000,
        efficient_output=True  # This already returns all three scores
    )
    
    # Process the predictions with enhanced workflow
    print("\nProcessing predictions with enhanced workflow...")
    error_df, positions_df = enhanced_process_predictions_with_all_scores(
        predictions=predictions,
        ss_annotations_df=annotations_df,
        threshold=config.threshold,
        consensus_window=config.consensus_window,
        error_window=config.error_window,
        analyze_position_offsets=True,
        collect_tn=True,
        predicted_delta_correction=True,
        verbose=2
    )
    
    # Display results
    print("\nEnhanced analysis complete!")
    
    if positions_df.height > 0:
        print(f"\nPositions DataFrame shape: {positions_df.shape}")
        print(f"Columns: {positions_df.columns}")
        
        # Show distribution of prediction types
        pred_types = positions_df.group_by("pred_type").agg(
            pl.count("position").alias("count")
        ).sort("count", descending=True)
        
        print("\nPrediction type distribution:")
        print(pred_types)
        
        # Display sample of positions with all three scores
        print("\nSample positions with all three probability scores:")
        sample_cols = [
            'gene_id', 'transcript_id', 'position', 'pred_type', 
            'splice_type', 'donor_score', 'acceptor_score', 'neither_score',
            'donor_acceptor_ratio', 'splice_neither_ratio'
        ]
        sample_df = positions_df.select(sample_cols).sample(n=5)
        print(sample_df)
        
        # Save to disk
        output_path = os.path.join(eval_dir, config.output_subdir, "tri_score_positions.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        positions_df.write_parquet(output_path)
        print(f"\nSaved positions DataFrame to: {output_path}")
        
        # Simple visualization of the probability scores
        try:
            # Convert to pandas for easier plotting
            pd_df = positions_df.filter(
                (pl.col("pred_type").is_in(["TP", "FP", "FN"])) & 
                (pl.col("splice_type") == "donor")
            ).to_pandas()
            
            if len(pd_df) > 0:
                # Create visualization
                plt.figure(figsize=(12, 6))
                
                # Plot donor vs acceptor probability scores
                plt.subplot(1, 2, 1)
                for pred_type in ["TP", "FP", "FN"]:
                    subset = pd_df[pd_df["pred_type"] == pred_type]
                    if len(subset) > 0:
                        plt.scatter(
                            subset["donor_score"], 
                            subset["acceptor_score"], 
                            alpha=0.5, 
                            label=pred_type
                        )
                
                plt.xlabel("Donor Probability")
                plt.ylabel("Acceptor Probability")
                plt.title("Donor vs Acceptor Probabilities")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Plot splice vs neither probabilities
                plt.subplot(1, 2, 2)
                for pred_type in ["TP", "FP", "FN"]:
                    subset = pd_df[pd_df["pred_type"] == pred_type]
                    if len(subset) > 0:
                        plt.scatter(
                            subset["donor_score"] + subset["acceptor_score"], 
                            subset["neither_score"], 
                            alpha=0.5, 
                            label=pred_type
                        )
                
                plt.xlabel("Splice Probability (Donor + Acceptor)")
                plt.ylabel("Neither Probability")
                plt.title("Splice vs Neither Probabilities")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(eval_dir, config.output_subdir, "tri_score_visualization.png")
                plt.savefig(plot_path)
                print(f"Visualization saved to: {plot_path}")
                plt.close()
            else:
                print("Not enough data for visualization")
        except Exception as e:
            print(f"Visualization error: {e}")
    
    if error_df.height > 0:
        print(f"\nError DataFrame shape: {error_df.shape}")
        print(f"Columns: {error_df.columns}")
        
        # Display error types
        error_types = error_df.group_by("error_type").agg(
            pl.count("position").alias("count")
        ).sort("count", descending=True)
        
        print("\nError type distribution:")
        print(error_types)
        
        # Save to disk
        output_path = os.path.join(eval_dir, config.output_subdir, "tri_score_errors.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        error_df.write_parquet(output_path)
        print(f"Saved error DataFrame to: {output_path}")
    
    print("\nDone!")


def load_gene_sequences_for_targets(target_genes, config, seq_type="standard", prioritized_chromosomes=None):
    """
    Load gene sequences for target genes from sequence files.
    
    Parameters
    ----------
    target_genes : List[str] or Dict[str, Union[str, int]]
        Either a list of gene IDs or names to load, or a dictionary mapping gene IDs/names to 
        chromosome numbers (e.g., {'STMN2': '8', 'UNC13A': '19'})
    config : SpliceAIConfig
        Configuration object with paths and settings
    seq_type : str, optional
        Type of gene sequence files to load:
        - "standard": Regular gene start to gene end (default)
        - "minmax": Min transcript start to max transcript end
    prioritized_chromosomes : List[Union[str, int]], optional
        List of chromosome numbers to check first, e.g. ['8', '19']
        
    Returns
    -------
    pl.DataFrame
        DataFrame containing the gene sequences for target genes
    """
    from meta_spliceai.splice_engine.run_spliceai_workflow import load_chromosome_sequence_streaming
    import glob
    
    # Handle the case where target_genes is a dictionary that maps genes to chromosomes
    if isinstance(target_genes, dict):
        gene_to_chrom = target_genes.copy()
        target_genes = list(gene_to_chrom.keys())
        # Extract prioritized chromosomes from the dictionary values if not explicitly provided
        if prioritized_chromosomes is None:
            prioritized_chromosomes = list(set(str(chrom) for chrom in gene_to_chrom.values()))
    else:
        gene_to_chrom = None
        
    # Convert prioritized_chromosomes to strings if provided
    if prioritized_chromosomes:
        prioritized_chromosomes = [str(chrom) for chrom in prioritized_chromosomes]
        print(f"Prioritizing chromosomes: {', '.join(prioritized_chromosomes)}")
    
    # Sequence file is typically stored as gene_sequence.parquet or gene_sequence_[chr].parquet
    seq_format = 'parquet'  # Efficient format for sequence data
    local_dir = os.path.dirname(config.eval_dir)
    
    # Determine file pattern based on seq_type
    if seq_type.lower() == "minmax":
        main_seq_file = os.path.join(local_dir, f"gene_sequence_minmax.{seq_format}")
        chr_pattern = os.path.join(local_dir, f"gene_sequence_minmax_*.{seq_format}")
        file_prefix = "gene_sequence_minmax_"
    else:  # standard
        main_seq_file = os.path.join(local_dir, f"gene_sequence.{seq_format}")
        chr_pattern = os.path.join(local_dir, f"gene_sequence_*.{seq_format}")
        file_prefix = "gene_sequence_"
    
    # Track which genes we've found
    found_genes = set()
    
    # If main sequence file exists, use it
    if os.path.exists(main_seq_file):
        print(f"Found main sequence file: {main_seq_file}")
        try:
            # Load from main sequence file
            seq_df = pl.read_parquet(main_seq_file)
            filtered_df = seq_df.filter(
                pl.col('gene_id').is_in(target_genes) | 
                pl.col('gene_name').is_in(target_genes)
            )
            if filtered_df.height > 0:
                print(f"Found {filtered_df.height} target genes in main sequence file")
                # Track which genes we found
                found_genes.update(filtered_df.get_column('gene_id').to_list())
                found_genes.update(filtered_df.get_column('gene_name').to_list())
                
                # If we found all target genes, return immediately
                if all(gene in found_genes for gene in target_genes):
                    print("All target genes found in main sequence file")
                    return filtered_df
                else:
                    remaining_genes = [gene for gene in target_genes if gene not in found_genes]
                    print(f"Still need to find genes: {', '.join(remaining_genes)}")
            else:
                print(f"No target genes found in main sequence file")
        except Exception as e:
            print(f"Error loading sequences from main file: {e}")
    
    # If we get here, either main file doesn't exist or target genes weren't found
    print(f"Looking for chromosome-specific sequence files for {seq_type} sequences...")
    
    # Find all available chromosome-specific files
    chr_files = glob.glob(chr_pattern)
    
    if not chr_files:
        print(f"No chromosome-specific sequence files found matching pattern: {chr_pattern}")
        return None
        
    print(f"Found {len(chr_files)} chromosome-specific sequence files")
    
    # Extract chromosome numbers and map to file paths
    chromosome_files = {}
    for file_path in chr_files:
        filename = os.path.basename(file_path)
        try:
            # Extract chromosome number from filename
            chr_num = filename.replace(file_prefix, "").split(".")[0]
            
            # Skip if we detect a "minmax" prefix in a standard search or vice versa
            if seq_type.lower() == "standard" and "minmax" in filename:
                continue
            
            chromosome_files[chr_num] = file_path
            print(f"  Found sequence file for chromosome {chr_num}: {filename}")
        except Exception as e:
            print(f"Could not extract chromosome number from filename: {filename}, error: {str(e)}")
    
    if not chromosome_files:
        print("Could not determine chromosome numbers from available files")
        return None
        
    print(f"Available chromosomes: {', '.join(sorted(chromosome_files.keys()))}")
    
    # List of genes we still need to find
    remaining_genes = [gene for gene in target_genes if gene not in found_genes]
    
    # Collect sequences from each chromosome file
    all_seqs = []
    
    # Process chromosomes in prioritized order if specified
    if prioritized_chromosomes:
        # First check prioritized chromosomes
        for chr_num in prioritized_chromosomes:
            if chr_num in chromosome_files:
                found_in_chrom, seq_df = _process_chromosome(
                    chr_num, chromosome_files[chr_num], remaining_genes, seq_type
                )
                if seq_df is not None and seq_df.height > 0:
                    all_seqs.append(seq_df)
                    found_genes.update(found_in_chrom)
                    remaining_genes = [gene for gene in target_genes if gene not in found_genes]
                    
                    # If all genes are found, we can stop
                    if not remaining_genes:
                        print(f"All target genes found in prioritized chromosomes")
                        break
    
    # If we still have genes to find, check the remaining chromosomes
    if remaining_genes:
        # Sort chromosomes by number to process them in a consistent order
        for chr_num in sorted(chromosome_files.keys()):
            # Skip chromosomes we've already checked
            if prioritized_chromosomes and chr_num in prioritized_chromosomes:
                continue
                
            found_in_chrom, seq_df = _process_chromosome(
                chr_num, chromosome_files[chr_num], remaining_genes, seq_type
            )
            if seq_df is not None and seq_df.height > 0:
                all_seqs.append(seq_df)
                found_genes.update(found_in_chrom)
                remaining_genes = [gene for gene in target_genes if gene not in found_genes]
                
                # If all genes are found, we can stop
                if not remaining_genes:
                    print(f"All target genes found, stopping search")
                    break
    
    if all_seqs:
        # Combine all sequences found
        gene_df = pl.concat(all_seqs)
        print(f"Combined sequences for {gene_df.height} genes")
        return gene_df
    else:
        print(f"No sequences found for target genes in any {seq_type} chromosome file")
        return None


def _process_chromosome(chr_num, file_path, target_genes, seq_type):
    """
    Process a chromosome file to find target genes.
    
    Parameters
    ----------
    chr_num : str
        Chromosome number
    file_path : str
        Path to the chromosome file
    target_genes : List[str]
        List of genes to search for
    seq_type : str
        Type of sequence file (standard or minmax)
        
    Returns
    -------
    Tuple[List[str], pl.DataFrame]
        List of genes found in this chromosome and the DataFrame of sequences
    """
    print(f"  - Loading sequences from chromosome {chr_num}")
    try:
        # Use direct file path to load sequences
        seq_df = pl.read_parquet(file_path)
        
        if seq_df is not None:
            # Filter to only keep target genes
            filtered_df = seq_df.filter(
                pl.col('gene_id').is_in(target_genes) | 
                pl.col('gene_name').is_in(target_genes)
            )
            if filtered_df.height > 0:
                # Get list of genes found in this chromosome
                found_genes = set()
                found_genes.update(filtered_df.get_column('gene_id').to_list())
                found_genes.update(filtered_df.get_column('gene_name').to_list())
                found_genes = [g for g in found_genes if g in target_genes]
                
                print(f"    Found {filtered_df.height} target genes in chromosome {chr_num}: {', '.join(found_genes)}")
                return found_genes, filtered_df
            else:
                print(f"    No target genes found in chromosome {chr_num}")
                return [], None
    except Exception as e:
        print(f"Error loading chromosome {chr_num} sequences: {e}")
        return [], None


if __name__ == "__main__":
    main()
