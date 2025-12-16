#!/usr/bin/env python
"""
FN Rescue Pipeline

This script automates the full pipeline for FN rescue analysis:
1. Identify genes with the highest FN rates from the shared dataset
2. Run the enhanced splice prediction workflow on those specific genes
3. Analyze the FN rescue potential using the context features

Usage:
    python run_fn_rescue_pipeline.py --top-genes 5 --output-dir path/to/output
"""

import os
import argparse
import pandas as pd
import polars as pl
import time
from pathlib import Path

# Import analysis modules
from meta_spliceai.splice_engine.meta_models.analysis.analyze_fn_rescue_potential import (
    load_full_positions_data,
    analyze_genes_with_most_fns,
    analyze_genes_with_most_fns_by_type,
    analyze_fn_rescue_potential_by_gene,
    evaluate_fn_rescue_potential
)

# Import workflow
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)

# Utilities
from meta_spliceai.splice_engine.meta_models.utils.workflow_utils import (
    print_emphasized, 
    print_with_indent
)

# System config for file paths
from meta_spliceai.system.config import Config

# Import shared analysis utilities
from meta_spliceai.splice_engine.meta_models.analysis.shared_analysis_utils import check_genomic_files_exist

def main():
    """Run the complete FN rescue analysis pipeline."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run the complete FN rescue analysis pipeline")
    parser.add_argument("--top-genes", type=int, default=5, 
                        help="Number of top genes with highest FN counts to analyze")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis results")
    parser.add_argument("--skip-workflow", action="store_true", 
                        help="Skip running the enhanced workflow (use if already run)")
    parser.add_argument("--force-extraction", action="store_true", 
                        help="Force extraction of genomic data even if files exist")
    parser.add_argument("--mode", type=str, choices=["gene", "transcript"], default="gene",
                        help="Mode for sequence extraction: 'gene' or 'transcript'")
    parser.add_argument("--seq-type", type=str, choices=["full", "minmax"], default="full",
                        help="Type of gene sequences to extract: 'full' or 'minmax'")
    parser.add_argument("--verbose", "-v", action="count", default=1,
                        help="Verbosity level (increase with -vv, -vvv)")
    parser.add_argument("--gene-types", type=str, nargs="+", help="Filter to specific gene types (e.g. protein_coding lncRNA)")
    parser.add_argument("--detailed-counts", action="store_true", help="Use detailed splice site counting metrics")

    args = parser.parse_args()
    
    # Create timestamp for output directory if not specified
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d")
        # Don't hardcode the full path, use Config.PROJ_DIR instead
        args.output_dir = os.path.join(Config.PROJ_DIR, "data", "ensembl", "spliceai_eval", 
                                      "meta_models", f"fn_rescue_analysis_{timestamp}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print_emphasized(f"[i/o] Output directory: {output_dir}")
    
    # Create subdirectory for each step
    identification_dir = output_dir / "1_gene_identification"
    workflow_dir = output_dir / "2_enhanced_workflow"
    analysis_dir = output_dir / "3_rescue_analysis"
    # NOTE: The special syntax `output_dir / "1_gene_identification"` is a feature of `pathlib.Path` that
    #       allows for intuitive path construction. It's equivalent to `os.path.join(output_dir, "1_gene_identification")`.
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist"

    for directory in [identification_dir, workflow_dir, analysis_dir]:
        directory.mkdir(exist_ok=True)
    
    # STEP 1: Identify genes with highest FN counts
    print_emphasized("STEP 1: Identifying genes with highest FN counts")
    
    print_with_indent("Loading full positions dataset from shared directory...", indent_level=1)
    positions_df = load_full_positions_data(verbose=args.verbose)
    
    print_with_indent(f"Finding top {args.top_genes} genes with most FNs...", indent_level=1)
    
    if args.gene_types:
        print_with_indent(f"Filtering to gene types: {', '.join(args.gene_types)}", indent_level=1)
        top_genes_df = analyze_genes_with_most_fns_by_type(
            positions_df, 
            top_n=args.top_genes, 
            gene_types=args.gene_types,
            use_detailed_counts=args.detailed_counts,
            verbose=2 if args.verbose else 1
        )
    else:
        top_genes_df = analyze_genes_with_most_fns(
            positions_df, 
            top_n=args.top_genes, 
            use_detailed_counts=args.detailed_counts
        )
    
    # Save the results
    top_genes_path = os.path.join(identification_dir, "top_fn_genes.csv")
    top_genes_df.to_csv(top_genes_path, index=False)
    print_with_indent(f"Saved top gene information to: {top_genes_path}", indent_level=1)
    
    # Get the list of gene IDs
    top_gene_ids = top_genes_df["gene_id"].tolist()
    
    # STEP 2: Run enhanced splice prediction workflow on these genes
    if not args.skip_workflow:
        print_emphasized("STEP 2: Running enhanced splice prediction workflow")
        
        # Setup the workflow directory
        test_dir = str(workflow_dir)
        print_with_indent(f"Workflow output directory: {test_dir}", indent_level=1)
        
        # Check if genomic files already exist
        if not args.force_extraction:
            existing_files = check_genomic_files_exist(mode=args.mode, seq_type=args.seq_type)
            print_with_indent("Checking for existing genomic files:", indent_level=1)
            
            # Print status for basic files
            for file_type, exists in existing_files.items():
                if file_type == "genomic_sequences_info":
                    continue
                status = "✓ Found" if exists else "✗ Missing"
                print_with_indent(f"{status}: {file_type}", indent_level=2)
            
            # Print more detailed information about genomic sequences
            seq_info = existing_files.get("genomic_sequences_info", {})
            if existing_files.get("genomic_sequences", False):
                print_with_indent("✓ Found: genomic_sequences", indent_level=2)
                for pattern, fmts in seq_info.get("complete_patterns", {}).items():
                    print_with_indent(f"  - Complete set of {pattern} files exists ({', '.join(fmts)})", indent_level=3)
            else:
                print_with_indent("✗ Missing: genomic_sequences", indent_level=2)
                for pattern, fmts in seq_info.get("pattern_details", {}).items():
                    for fmt, data in fmts.items():
                        if data["complete"]:
                            print_with_indent(f"  - Complete set of {pattern} files exists ({fmt}) but is not sufficient", indent_level=3)
                        else:
                            print_with_indent(f"  - Complete set of {pattern} files not found ({fmt})", indent_level=3)
        else:
            print_with_indent("Force extraction enabled - will regenerate all genomic data files", indent_level=1)
            existing_files = {k: False for k in ["annotations", "genomic_sequences", "splice_sites"]}
        
        # Run the enhanced workflow on the top genes
        print_with_indent(f"Running enhanced workflow for {len(top_gene_ids)} genes...", indent_level=1)
        print_with_indent(f"Target genes: {top_gene_ids}", indent_level=1)
        
        # Determine if we need to extract files based on existence checks
        # Default to NOT extracting if files exist, only extract if missing or force_extraction is set
        need_extract_sequences = not existing_files.get("genomic_sequences", False)
        need_extract_annotations = not existing_files.get("annotations", False)
        need_extract_splice_sites = not existing_files.get("splice_sites", False)
        
        # Print extraction status
        print_with_indent(f"Extraction status:", indent_level=1)
        print_with_indent(f"- Sequences: {'NEEDED' if need_extract_sequences else 'SKIPPING (files exist)'}", indent_level=2)
        print_with_indent(f"- Annotations: {'NEEDED' if need_extract_annotations else 'SKIPPING (files exist)'}", indent_level=2)
        print_with_indent(f"- Splice sites: {'NEEDED' if need_extract_splice_sites else 'SKIPPING (files exist)'}", indent_level=2)
        
        # If force extraction is enabled, override the defaults
        if args.force_extraction:
            print_with_indent("Force extraction enabled - will extract all files regardless of existence", indent_level=1)
            need_extract_sequences = True
            need_extract_annotations = True
            need_extract_splice_sites = True
        
        result = run_enhanced_splice_prediction_workflow(
            local_dir=test_dir,
            do_extract_sequences=need_extract_sequences,
            do_extract_annotations=need_extract_annotations,
            do_extract_splice_sites=need_extract_splice_sites,
            do_find_overlaping_genes=True,  # Always do this as it's specific to these genes
            target_genes=top_gene_ids,      # Process only the top FN genes
            seq_mode=args.mode,             # Pass the sequence mode (gene or transcript)
            seq_type=args.seq_type,         # Pass the sequence type (full or minmax)
            verbosity=args.verbose
        )
        
        # Check if workflow completed successfully
        if result['success']:
            print_with_indent("Enhanced workflow completed successfully!", indent_level=1)
            
            # Save workflow results summary with detailed information
            with open(os.path.join(workflow_dir, "workflow_summary.txt"), "w") as f:
                f.write("Enhanced Workflow Results\n")
                f.write("=======================\n\n")
                
                # Basic workflow status
                f.write(f"Status: {'Success' if result['success'] else 'Failed'}\n")
                
                # Dataset statistics
                f.write("\nDataset Statistics:\n")
                f.write("-----------------\n")
                
                # Positions DataFrame
                if 'positions' in result and not result['positions'].is_empty():
                    pos_df = result['positions']
                    f.write(f"Positions dataset: {pos_df.height:,} rows × {len(pos_df.columns)} columns\n")
                    
                    # Get prediction type distribution
                    try:
                        pred_counts = pos_df.group_by('pred_type').agg(pl.count()).sort("pred_type")
                        f.write("\nPrediction type counts:\n")
                        for row in pred_counts.iter_rows(named=True):
                            f.write(f"  {row['pred_type']}: {row['count']:,}\n")
                    except Exception as e:
                        f.write(f"  Could not calculate prediction type counts: {e}\n")
                    
                    # Get splice type distribution
                    try:
                        splice_counts = pos_df.group_by('splice_type').agg(pl.count()).sort("splice_type")
                        f.write("\nSplice type counts:\n")
                        for row in splice_counts.iter_rows(named=True):
                            f.write(f"  {row['splice_type']}: {row['count']:,}\n")
                    except Exception as e:
                        f.write(f"  Could not calculate splice type counts: {e}\n")
                    
                    # Count unique genes
                    try:
                        unique_genes = pos_df.select(pl.col("gene_id")).unique().height
                        f.write(f"\nUnique genes: {unique_genes:,}\n")
                    except Exception as e:
                        f.write(f"  Could not count unique genes: {e}\n")
                else:
                    f.write("Positions dataset: Not available or empty\n")
                
                # Error analysis DataFrame
                if 'error_analysis' in result and not result['error_analysis'].is_empty():
                    err_df = result['error_analysis']
                    f.write(f"\nError analysis dataset: {err_df.height:,} rows × {len(err_df.columns)} columns\n")
                else:
                    f.write("\nError analysis dataset: Not available or empty\n")
                    
                # Analysis sequences DataFrame
                if 'analysis_sequences' in result and not result['analysis_sequences'].is_empty():
                    seq_df = result['analysis_sequences']
                    f.write(f"\nAnalysis sequences dataset: {seq_df.height:,} rows × {len(seq_df.columns)} columns\n")
                else:
                    f.write("\nAnalysis sequences dataset: Not available or empty\n")
                
                # Overlapping genes
                if 'overlapping_genes' in result and result['overlapping_genes'].get('success', False):
                    ovr_df = result['overlapping_genes'].get('overlapping_df')
                    if ovr_df is not None and hasattr(ovr_df, 'height'):
                        f.write(f"\nOverlapping genes dataset: {ovr_df.height:,} rows × {len(ovr_df.columns)} columns\n")
                    elif ovr_df is not None and hasattr(ovr_df, 'shape'):
                        # Handle pandas DataFrame
                        f.write(f"\nOverlapping genes dataset: {ovr_df.shape[0]:,} rows × {ovr_df.shape[1]} columns\n")
                    else:
                        f.write("\nOverlapping genes dataset: Available but format unknown\n")
                else:
                    f.write("\nOverlapping genes dataset: Not available or empty\n")
                
                # Output files
                f.write("\nGenerated Files:\n")
                f.write("--------------\n")
                
                # Get data handler paths from the result dictionary
                try:
                    # Access paths from the result dictionary
                    if 'paths' in result and result['paths'].get('eval_dir'):
                        eval_dir = result['paths']['eval_dir']
                        meta_data_dir = os.path.join(eval_dir, "meta_models")
                        f.write(f"Data directory: {meta_data_dir}\n")
                        
                        # List common file patterns
                        for file_type in ["positions_enhanced_aggregated", "error_analysis_aggregated", "sequence_analysis"]:
                            pattern = f"*{file_type}*.{{'parquet', 'csv', 'tsv'}}"  # Using string representation for glob
                            f.write(f"  {file_type}: {pattern}\n")
                            
                        # Include output directory if available
                        if result['paths'].get('output_dir'):
                            f.write(f"\nOutput directory: {result['paths']['output_dir']}\n")
                    else:
                        f.write(f"No path information available in workflow result\n")
                except Exception as e:
                    f.write(f"Could not determine output file paths: {e}\n")
                
                # Include any other useful information from the result dictionary
                for key, value in result.items():
                    # Skip dataframes and already processed keys
                    if key in ["error_analysis", "positions", "analysis_sequences", "overlapping_genes", "success"]:
                        continue
                        
                    if isinstance(value, dict):
                        f.write(f"\n{key}:\n")
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    elif isinstance(value, (str, int, float, bool)):
                        f.write(f"\n{key}: {value}\n")
            
            # Get the enhanced positions data directly from the result
            enhanced_positions_df = result.get('positions', None)
            
            if enhanced_positions_df is None or enhanced_positions_df.height == 0:
                print_with_indent("ERROR: Enhanced positions dataframe not found in workflow results!", indent_level=1)
                print_with_indent("Will attempt to load from the expected file location...", indent_level=1)
                
                # Try to load from the expected file location
                positions_file = os.path.join(test_dir, "full_splice_positions_enhanced.tsv")
                if os.path.exists(positions_file):
                    try:
                        print_with_indent(f"Loading enhanced positions from: {positions_file}", indent_level=1)
                        # import polars as pl
                        enhanced_positions_df = pl.read_csv(positions_file, separator='\t')
                        print_with_indent(f"Successfully loaded enhanced positions with {enhanced_positions_df.height} records", indent_level=1)
                    except Exception as e:
                        print_with_indent(f"Error loading enhanced positions: {e}", indent_level=1)
                        return
                else:
                    print_with_indent("ERROR: Enhanced positions file not found!", indent_level=1)
                    return
            else:
                print_with_indent(f"Found enhanced positions dataframe with {enhanced_positions_df.height} records", indent_level=1)
        else:
            print_with_indent("ERROR: Enhanced workflow failed!", indent_level=1)
            print_with_indent(f"Error details: {result.get('error', 'Unknown error')}", indent_level=1)
            return
    else:
        print_emphasized("STEP 2: Skipping enhanced workflow (--skip-workflow flag set)")
        
        # If skipping the workflow, try to load the enhanced positions data from the expected location
        print_with_indent("Attempting to load previously generated enhanced positions data...", indent_level=1)
        
        # Look for enhanced positions in the workflow directory
        positions_file = os.path.join(workflow_dir, "full_splice_positions_enhanced.tsv")
        enhanced_positions_df = None
        
        if os.path.exists(positions_file):
            try:
                print_with_indent(f"Loading enhanced positions from: {positions_file}", indent_level=1)
                # import polars as pl
                enhanced_positions_df = pl.read_csv(positions_file, separator='\t')
                print_with_indent(f"Successfully loaded enhanced positions with {enhanced_positions_df.height} records", indent_level=1)
            except Exception as e:
                print_with_indent(f"Error loading enhanced positions: {e}", indent_level=1)
        
        if enhanced_positions_df is None or enhanced_positions_df.height == 0:
            print_with_indent("WARNING: Could not load enhanced positions data", indent_level=1)
            print_with_indent("Will use the original positions data, but rescue analysis may be limited.", indent_level=1)
            enhanced_positions_df = positions_df  # Fallback to original data
    
    # STEP 3: Run FN rescue analysis on the enhanced data
    print_emphasized("STEP 3: Analyzing FN rescue potential with enhanced features")
    
    # Reload the enhanced positions data
    print_with_indent("Running FN rescue analysis...", indent_level=1)
    
    # Create a per-gene visualization directory
    per_gene_dir = analysis_dir / "per_gene_analysis"
    os.makedirs(per_gene_dir, exist_ok=True)
    
    # Analyze each gene individually first to get visualizations
    print_with_indent(f"Generating per-gene visualizations for {len(top_gene_ids)} genes...", indent_level=1)
    
    # Track per-gene stats for the summary
    per_gene_stats = []
    
    # Process each gene individually for visualizations
    for gene_id in top_gene_ids:
        gene_dir = os.path.join(per_gene_dir, gene_id)
        os.makedirs(gene_dir, exist_ok=True)
        
        # Filter positions for just this gene
        gene_positions = enhanced_positions_df.filter(pl.col("gene_id") == gene_id)
        
        if gene_positions.height == 0:
            print_with_indent(f"Warning: No data found for gene {gene_id}", indent_level=2)
            continue
            
        print_with_indent(f"Analyzing gene {gene_id} ({gene_positions.height} positions)...", indent_level=2)
        
        # Run the individual gene analysis with visualizations
        evaluate_fn_rescue_potential(gene_positions, gene_dir)
        
        # Extract summary stats
        summary_path = os.path.join(gene_dir, "fn_rescue_summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                lines = f.readlines()
                
            # Extract FN counts from summary
            total_fns = 0
            rescued_fns = 0
            
            for line in lines:
                if "Total FNs in dataset:" in line:
                    total_fns = int(line.split(":")[1].strip().split()[0])
                elif "Total potential FN reduction:" in line:
                    rescued_fns = int(line.split(":")[1].strip().split()[0])
            
            rescue_percent = (rescued_fns / total_fns * 100) if total_fns > 0 else 0
            
            per_gene_stats.append({
                "gene_id": gene_id,
                "total_fns": total_fns,
                "rescued_fns": rescued_fns,
                "rescue_percent": rescue_percent
            })
    
    # Create a per-gene summary table
    if per_gene_stats:
        import pandas as pd
        gene_stats_df = pd.DataFrame(per_gene_stats)
        gene_stats_path = os.path.join(analysis_dir, "per_gene_rescue_summary.csv")
        gene_stats_df.to_csv(gene_stats_path, index=False)
        print_with_indent(f"Saved per-gene rescue summary to: {gene_stats_path}", indent_level=1)
        
        # Print the per-gene summary
        print_with_indent("Per-gene FN rescue summary:", indent_level=1)
        for stats in per_gene_stats:
            print_with_indent(f"Gene {stats['gene_id']}: {stats['rescued_fns']}/{stats['total_fns']} FNs rescued ({stats['rescue_percent']:.1f}%)", indent_level=2)
    
    # Run the consolidated multi-gene analysis
    print_with_indent("Running consolidated FN rescue analysis for all genes...", indent_level=1)
    result_df = analyze_fn_rescue_potential_by_gene(
        positions_df=enhanced_positions_df,  # Use the enhanced dataset from the workflow
        gene_ids=top_gene_ids,              # Focus on just these genes
        output_dir=str(analysis_dir)
    )
    
    # Print summary
    print_emphasized("Analysis Pipeline Complete!")
    print_with_indent(f"Results stored in: {output_dir}", indent_level=1)
    print_with_indent("Summary of directories:", indent_level=1)
    print_with_indent(f"  - Gene identification: {identification_dir}", indent_level=2)
    print_with_indent(f"  - Enhanced workflow: {workflow_dir}", indent_level=2)
    print_with_indent(f"  - Rescue analysis: {analysis_dir}", indent_level=2)
    
    
if __name__ == "__main__":
    main()
