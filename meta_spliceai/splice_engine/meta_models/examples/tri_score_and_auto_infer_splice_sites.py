"""
Example: Three Probability Score Enhanced Workflow with Automatic Splice Site Adjustment Detection

This script demonstrates how to use the enhanced workflow with automatic detection
of splice site annotation adjustments for more robust splice site prediction analysis.
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
from meta_spliceai.splice_engine.utils import print_emphasized
from meta_spliceai.splice_engine.utils_fs import read_splice_sites

# Test and debugging utilities
from meta_spliceai.splice_engine.meta_models.utils.verify_splice_adjustment import (
    verify_adjustment_effect,
    verify_probability_position_shifts
)

# Import sequence utilities
from meta_spliceai.splice_engine.meta_models.utils.sequence_utils import (
    load_gene_sequences_for_targets
)

# Import the automatic adjustment detection module
from meta_spliceai.splice_engine.meta_models.utils.infer_splice_site_adjustments import (
    auto_detect_splice_site_adjustments,
    apply_auto_detected_adjustments,
    calculate_prediction_statistics
)

# Import the comparison plot function
from meta_spliceai.splice_engine.meta_models.utils.analyze_splice_adjustment import create_adjustment_comparison_plot


def main():
    """Run an example of the tri-score enhanced splice prediction workflow with automatic adjustment detection."""
    
    # Get the basic paths from ErrorAnalyzer
    gtf_file = ErrorAnalyzer.gtf_file
    eval_dir = ErrorAnalyzer.eval_dir
    
    # Print information about the example
    print("=" * 80)
    print(" Tri-Score Enhanced Workflow with Auto Adjustment Detection ")
    print("=" * 80)
    print(f"GTF file: {gtf_file}")
    print(f"Evaluation directory: {eval_dir}")
    
    # List of specific genes to analyze
    target_genes = {'STMN2': '8', 'UNC13A': '19', 'ENSG00000104435': '8'}
    
    # Create a configuration
    config = SpliceAIConfig(
        gtf_file=gtf_file,
        eval_dir=eval_dir,
        output_subdir="tri_score_auto_adjustment",
        threshold=0.3,  # Lower threshold to capture more potential splice sites
        consensus_window=2,
        error_window=500,
        test_mode=True
    )
    
    # Load annotations using the MetaModelDataHandler.load_splice_annotations method
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
    
    # Extract gene start positions to convert absolute to relative coordinates
    gene_start_positions = {}
    for row in gene_df.iter_rows(named=True):
        gene_id = row['gene_id']
        gene_start = row['start']
        gene_end = row['end']
        strand = row['strand']
        
        gene_start_positions[gene_id] = {
            'start': gene_start,
            'end': gene_end,
            'strand': strand,
            'length': gene_end - gene_start + 1
        }
    
    # Convert absolute positions in annotations to gene-relative positions
    print("Converting absolute positions to gene-relative positions...")
    annotations_list = []
    
    for row in annotations_df.iter_rows(named=True):
        gene_id = row['gene_id']
        
        # Skip if gene not in our loaded genes
        if gene_id not in gene_start_positions:
            continue
            
        gene_info = gene_start_positions[gene_id]
        absolute_pos = row['position']
        
        # Calculate relative position based on strand
        if gene_info['strand'] == '+':
            # For plus strand, relative position is (absolute position - gene start)
            rel_position = absolute_pos - gene_info['start']
        else:
            # For minus strand, relative position is (gene end - absolute position)
            rel_position = gene_info['end'] - absolute_pos
        
        # Check if the relative position is within gene bounds
        if 0 <= rel_position < gene_info['length']:
            # Create a new row with relative position added
            new_row = dict(row)
            new_row['rel_position'] = rel_position
            annotations_list.append(new_row)
    
    # Convert back to DataFrame
    relative_annotations_df = pl.DataFrame(annotations_list)
    
    print(f"Converted annotations: {relative_annotations_df.shape} (after filtering for loaded genes)")
        
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
    
    # Auto-detect splice site adjustments
    print("\nAuto-detecting splice site adjustments...")
    adjustment_dict = auto_detect_splice_site_adjustments(
        annotations_df=relative_annotations_df,
        pred_results=predictions,
        consensus_window=config.consensus_window,
        verbose=True
    )
    
    print("\nDetected adjustment offsets:")
    for site_type in ['donor', 'acceptor']:
        for strand_key in ['plus', 'minus']:
            print(f"  {site_type.capitalize()} sites on '{strand_key}' strand: offset = {adjustment_dict[site_type][strand_key]}")
    
    # Calculate prediction statistics
    print("\nCalculating prediction statistics...")
    
    # Without adjustments
    print("\nStatistics WITHOUT adjustments:")
    no_adj_stats = calculate_prediction_statistics(
        annotations_df=relative_annotations_df,
        pred_results=predictions,
        threshold=0.01,  # Much lower threshold to catch more sites
        adjustment_dict=None,
        verbose=0
    )
    
    # With auto-detected adjustments
    print("\nStatistics WITH auto-detected adjustments:")
    adj_stats = calculate_prediction_statistics(
        annotations_df=relative_annotations_df,
        pred_results=predictions,
        threshold=0.01,  # Much lower threshold to catch more sites
        adjustment_dict=adjustment_dict,
        verbose=0
    )
    
    print("\nPrediction statistics without adjustments:")
    print(f"  Donor sites: {no_adj_stats['donor_correct']}/{no_adj_stats['donor_total']} = {no_adj_stats['donor_accuracy']*100:.2f}%")
    print(f"  Acceptor sites: {no_adj_stats['acceptor_correct']}/{no_adj_stats['acceptor_total']} = {no_adj_stats['acceptor_accuracy']*100:.2f}%")
    
    print("\nPrediction statistics with auto-detected adjustments:")
    print(f"  Donor sites: {adj_stats['donor_correct']}/{adj_stats['donor_total']} = {adj_stats['donor_accuracy']*100:.2f}%")
    print(f"  Acceptor sites: {adj_stats['acceptor_correct']}/{adj_stats['acceptor_total']} = {adj_stats['acceptor_accuracy']*100:.2f}%")
    

    # Process the predictions with enhanced workflow - without adjustments
    print_emphasized("\nProcessing predictions without splice site adjustments...")
    error_df_without_adj, positions_df_without_adj = enhanced_process_predictions_with_all_scores(
        predictions=predictions,
        ss_annotations_df=relative_annotations_df,
        threshold=config.threshold,
        consensus_window=config.consensus_window,
        error_window=config.error_window,
        analyze_position_offsets=True,
        collect_tn=True,
        predicted_delta_correction=False,  # No adjustments
        verbose=2
    )
    
    # Process the predictions with enhanced workflow - with auto-detected adjustments
    print_emphasized("\nProcessing predictions with auto-detected splice site adjustments...")
    error_df_with_adj, positions_df_with_adj = enhanced_process_predictions_with_all_scores(
        predictions=predictions,
        ss_annotations_df=relative_annotations_df,
        threshold=config.threshold,
        consensus_window=config.consensus_window,
        error_window=config.error_window,
        analyze_position_offsets=True,
        collect_tn=True,
        predicted_delta_correction=True,  # Use adjustments
        splice_site_adjustments=adjustment_dict,  # Pass the detected adjustments
        verbose=2
    )

    # Verify probability position shifts align with expected adjustments
    shift_results = verify_probability_position_shifts(
        positions_df_without_adj, 
        positions_df_with_adj,
        adjustment_dict=adjustment_dict
    )
    
    # Compare results
    print("\nComparison of results with and without auto-detected adjustments:")
    
    if positions_df_without_adj.height > 0 and positions_df_with_adj.height > 0:
        # Distribution of prediction types without adjustments
        pred_types_without_adj = positions_df_without_adj.group_by("pred_type").agg(
            pl.count("position").alias("count")
        ).sort("count", descending=True)
        
        # Distribution of prediction types with adjustments
        pred_types_with_adj = positions_df_with_adj.group_by("pred_type").agg(
            pl.count("position").alias("count")
        ).sort("count", descending=True)
        
        print("\nPrediction type distribution without adjustments:")
        print(pred_types_without_adj)
        
        print("\nPrediction type distribution with auto-detected adjustments:")
        print(pred_types_with_adj)
        
        # Create comparison visualizations
        try:
            # Generate comparison plots for both donor and acceptor sites
            plot_path = os.path.join(eval_dir, config.output_subdir, "adjustment_comparison.pdf")
            create_adjustment_comparison_plot(
                positions_df_without_adj=positions_df_without_adj,
                positions_df_with_adj=positions_df_with_adj,
                output_path=plot_path,
                site_types=["donor", "acceptor"],
                pred_types=["TP", "FP", "FN"]
            )
            print(f"\nComparison visualization saved to: {plot_path}")
            
            # Optional: Generate a version with TNs included
            if "TN" in positions_df_without_adj["pred_type"].unique() and "TN" in positions_df_with_adj["pred_type"].unique():
                tn_plot_path = os.path.join(eval_dir, config.output_subdir, "adjustment_comparison_with_tn.pdf")
                create_adjustment_comparison_plot(
                    positions_df_without_adj=positions_df_without_adj,
                    positions_df_with_adj=positions_df_with_adj,
                    output_path=tn_plot_path,
                    site_types=["donor", "acceptor"],
                    pred_types=["TP", "FP", "FN", "TN"],
                    figsize=(16, 12)
                )
                print(f"Comparison with TNs saved to: {tn_plot_path}")
                
        except Exception as e:
            print(f"Visualization error: {e}")
    
    # Verify if adjustments are actually having an effect
    adjustment_effective = verify_adjustment_effect(positions_df_without_adj, positions_df_with_adj)
      
    if not adjustment_effective:
        print("\nWARNING: Adjustments don't appear to be affecting the probability distributions.")
        print("Check the following potential issues:")
        print("  1. Are splice_site_adjustments being correctly passed to the processing function?")
        print("  2. Is predicted_delta_correction being correctly set to True?")
        print("  3. Is the enhanced_workflow.py implementing the adjustments correctly?")
        print("  4. Are the adjustment values non-zero? Current values:")
        for site_type in adjustment_dict:
            for strand in adjustment_dict[site_type]:
                print(f"    {site_type} on {strand} strand: {adjustment_dict[site_type][strand]}")
    
    # Save results
    output_dir = os.path.join(eval_dir, config.output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save adjustments
    adjustment_df = pl.DataFrame({
        "site_type": ["donor", "donor", "acceptor", "acceptor"],
        "strand": ["+", "-", "+", "-"],
        "offset": [
            adjustment_dict["donor"]["plus"], 
            adjustment_dict["donor"]["minus"],
            adjustment_dict["acceptor"]["plus"],
            adjustment_dict["acceptor"]["minus"]
        ]
    })
    
    adjustment_path = os.path.join(output_dir, "auto_detected_adjustments.parquet")
    adjustment_df.write_parquet(adjustment_path)
    print(f"\nSaved adjustment data to: {adjustment_path}")
    
    # Save position data
    if positions_df_without_adj.height > 0:
        positions_df_without_adj.write_parquet(os.path.join(output_dir, "positions_without_adj.parquet"))
    
    if positions_df_with_adj.height > 0:
        positions_df_with_adj.write_parquet(os.path.join(output_dir, "positions_with_adj.parquet"))
    
    print("\nDone!")
    return adjustment_effective



if __name__ == "__main__":
    main()
