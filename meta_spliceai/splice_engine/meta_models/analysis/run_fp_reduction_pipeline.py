"""
FP Reduction Pipeline

This script automates the full pipeline for FP reduction analysis:
1. Identify genes with the highest FP rates from the shared dataset
2. Run the enhanced splice prediction workflow on those specific genes
3. Analyze the FP reduction potential using the context features

Usage:
    python run_fp_reduction_pipeline.py --top-genes 5 --output-dir path/to/output
"""

import os
import argparse
import pandas as pd
import polars as pl
import time
from pathlib import Path

# Import analysis modules
from meta_spliceai.splice_engine.meta_models.analysis.analyze_fp_reduction_potential import (
    load_full_positions_data,
    analyze_genes_with_most_fps,
    analyze_genes_with_most_fps_by_type,
    analyze_fp_reduction_potential_by_gene,
    evaluate_fp_reduction_potential
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
from meta_spliceai.splice_engine.meta_models.analysis.shared_analysis_utils import check_genomic_files_exist


def main():
    """Run the complete FP reduction analysis pipeline."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run the complete FP reduction analysis pipeline")
    parser.add_argument("--top-genes", type=int, default=5, 
                        help="Number of top genes with highest FP counts to analyze")
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
        try:
            # Try to use the imported Config module
            args.output_dir = os.path.join(Config.PROJ_DIR, "data", "ensembl", "spliceai_eval", 
                                        "meta_models", f"fp_reduction_analysis_{timestamp}")
        except Exception as e:
            # Fallback to using relative path if Config doesn't work
            args.output_dir = f"spliceai_eval/meta_models/fp_reduction_analysis_{timestamp}"
            print_emphasized(f"Warning: Could not use Config.PROJ_DIR: {e}")
            print_emphasized(f"Using fallback output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print_emphasized(f"[i/o] Output directory: {output_dir}")
    
    # Create subdirectory for each step
    identification_dir = output_dir / "1_gene_identification"
    workflow_dir = output_dir / "2_enhanced_workflow"
    analysis_dir = output_dir / "3_reduction_analysis"
    # NOTE: The special syntax `output_dir / "1_gene_identification"` is a feature of `pathlib.Path` that
    #       allows for intuitive path construction. It's equivalent to `os.path.join(output_dir, "1_gene_identification")`.
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist"

    for directory in [identification_dir, workflow_dir, analysis_dir]:
        directory.mkdir(exist_ok=True)
    
    # STEP 1: Identify genes with highest FP counts
    print_emphasized("STEP 1: Identifying genes with highest FP counts")
    
    # Load the enhanced positions data
    print_with_indent("Loading enhanced positions data...", indent_level=1)
    positions_df = load_full_positions_data()
    
    print_with_indent(f"Loaded {positions_df.shape[0]} positions", indent_level=1)
    
    # Find the top N genes with the most FPs
    print_with_indent(f"Finding top {args.top_genes} genes with the most FPs...", indent_level=1)
    
    # Filter by gene type if requested
    if args.gene_types:
        print_with_indent(f"Filtering to gene types: {', '.join(args.gene_types)}", indent_level=1)
    
    # Log whether we're using detailed counting
    if args.detailed_counts:
        print_with_indent("Using detailed site counting", indent_level=1)
    else:
        print_with_indent("Using simple FP counting", indent_level=1)
    
    # Primary analysis: Find genes with most FPs by gene type (all splice types combined)
    print_with_indent(f"Finding genes with most FPs filtered by gene types: {args.gene_types if args.gene_types else 'all'}", indent_level=1)
    top_genes_all = analyze_genes_with_most_fps_by_type(
        positions_df=positions_df, 
        output_dir=os.path.join(identification_dir, "by_gene_type"),
        splice_type=None,  # Don't filter by splice type - analyze all sites
        top_n=args.top_genes,
        gene_types=args.gene_types,
        use_detailed_counts=args.detailed_counts,
        verbose=args.verbose
    )
    
    # Secondary analysis: Split by donor/acceptor for additional insight
    print_with_indent("Additionally analyzing splice site types separately...", indent_level=1)
    
    print_with_indent("Finding genes with most donor FPs...", indent_level=2)
    top_genes_donor = analyze_genes_with_most_fps_by_type(
        positions_df=positions_df, 
        output_dir=os.path.join(identification_dir, "donor_fps"),
        splice_type='donor',
        top_n=args.top_genes,
        gene_types=args.gene_types,
        use_detailed_counts=args.detailed_counts,
        verbose=args.verbose
    )
    
    print_with_indent("Finding genes with most acceptor FPs...", indent_level=2)
    top_genes_acceptor = analyze_genes_with_most_fps_by_type(
        positions_df=positions_df, 
        output_dir=os.path.join(identification_dir, "acceptor_fps"),
        splice_type='acceptor',
        top_n=args.top_genes,
        gene_types=args.gene_types,
        use_detailed_counts=args.detailed_counts,
        verbose=args.verbose
    )
    
    # For further analysis, use the genes with most FPs by gene type
    top_genes_df = top_genes_all
    
    # Save the results
    top_genes_path = os.path.join(identification_dir, "top_fp_genes.csv")
    top_genes_df.to_csv(top_genes_path, index=False)
    print_with_indent(f"Saved top gene information to: {top_genes_path}", indent_level=1)
    
    # Get the list of gene IDs (should already be strings from the analysis function)
    top_gene_ids = top_genes_df["gene_id"].tolist()
    print_with_indent(f"Using {len(top_gene_ids)} gene IDs from analysis", indent_level=1)
    
    # STEP 2: Run the enhanced splice prediction workflow on selected genes
    print_emphasized("STEP 2: Running enhanced splice prediction workflow on selected genes")
    
    if args.skip_workflow:
        print_with_indent("Skipping workflow run (--skip-workflow specified)", indent_level=1)
    else:
        # Check if genomic files already exist to avoid unnecessary extraction
        print_with_indent("Checking if genomic data files already exist...", indent_level=1)
        existing_files = check_genomic_files_exist(mode=args.mode, seq_type=args.seq_type)
        
        # First check if we have any genomic files by checking if the main keys exist
        if not ("annotations" in existing_files and "genomic_sequences" in existing_files and "splice_sites" in existing_files):
            print_with_indent("Required genomic file info not found - extraction required", indent_level=1)
            existing_files = {k: False for k in ["annotations", "genomic_sequences", "splice_sites"]}
        # Then check if force extraction is enabled
        elif args.force_extraction:
            print_with_indent("Force extraction enabled - will regenerate all genomic data files", indent_level=1)
            existing_files = {k: False for k in ["annotations", "genomic_sequences", "splice_sites"]}
        # Otherwise, examine which files exist
        else:
            # Determine base dir from Config
            base_dir = os.path.join(Config.PROJ_DIR, "data", "ensembl")
            print_with_indent(f"Found existing genomic files in data directory: {base_dir}", indent_level=1)
            print_with_indent("Status of required files:", indent_level=1)
            
            # Report on both types of annotations
            if existing_files['annotations']:
                print_with_indent(f"✓ Annotations: Available", indent_level=2)
                if 'shared_annotation' in existing_files and existing_files['shared_annotation']['exists']:
                    print_with_indent(f"  ✓ Shared annotation file exists: {existing_files['shared_annotation']['path']}", indent_level=3)
            else:
                print_with_indent(f"✗ Annotations: Missing", indent_level=2)
                if 'shared_annotation' in existing_files:
                    if existing_files['shared_annotation']['exists']:
                        print_with_indent(f"  ✓ Shared annotation file exists: {existing_files['shared_annotation']['path']}", indent_level=3)
                    else:
                        print_with_indent(f"  ✗ Shared annotation file missing: {existing_files['shared_annotation']['path']}", indent_level=3)
                        print_with_indent(f"    (This file takes a long time to extract)", indent_level=3)
            print_with_indent(f"✓ Splice Sites: {existing_files['splice_sites']}", indent_level=2)
            
            if existing_files["genomic_sequences"]:
                print_with_indent("✓ Genomic Sequences", indent_level=2)
                seq_info = existing_files.get("seq_info", {})
                for pattern, fmts in seq_info.get("complete_patterns", {}).items():
                    print_with_indent(f"  - Complete set of {pattern} files exists ({', '.join(fmts)})", indent_level=3)
            else:
                print_with_indent("✗ Missing: genomic_sequences", indent_level=2)
                seq_info = existing_files.get("seq_info", {})
                for pattern, fmts in seq_info.get("pattern_details", {}).items():
                    for fmt, data in fmts.items():
                        if data["complete"]:
                            print_with_indent(f"  - Complete set of {pattern} files exists ({fmt}) but is not sufficient", indent_level=3)
                        else:
                            print_with_indent(f"  - Complete set of {pattern} files not found ({fmt})", indent_level=3)
        
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
            local_dir=workflow_dir,
            do_extract_sequences=need_extract_sequences,
            do_extract_annotations=need_extract_annotations,
            do_extract_splice_sites=need_extract_splice_sites,
            do_find_overlaping_genes=True,  # Always do this as it's specific to these genes
            target_genes=top_gene_ids,      # Process only the top FP genes
            seq_mode=args.mode,             # Pass the sequence mode (gene or transcript)
            seq_type=args.seq_type,         # Pass the sequence type (full or minmax)
            verbosity=args.verbose
        )
        
        # Check if workflow completed successfully
        if result['success']:
            print_with_indent("Enhanced workflow completed successfully", indent_level=1)
            
            # Save the enhanced positions DataFrame for later use
            if 'positions' in result and not result['positions'].is_empty():
                enhanced_positions_df = result['positions']
                print_with_indent(f"Extracted enhanced positions dataset with {enhanced_positions_df.height:,} rows", indent_level=1)
            else:
                print_with_indent("Warning: Enhanced positions dataset not found in workflow result", indent_level=1)
                print_with_indent("Using original positions dataset for analysis", indent_level=1)
                enhanced_positions_df = positions_df
                
            # Print summary statistics
            print_with_indent("Summary statistics:", indent_level=1)
            if 'gene_stats' in result:
                for gene_id, stats in result['gene_stats'].items():
                    print_with_indent(f"Gene {gene_id}:", indent_level=2)
                    print_with_indent(f"- Transcripts: {stats.get('transcript_count', 'N/A')}", indent_level=3)
                    print_with_indent(f"- Splice sites: {stats.get('splice_site_count', 'N/A')}", indent_level=3)
                    print_with_indent(f"- Sequence length: {stats.get('sequence_length', 'N/A')} bp", indent_level=3)
            
            # Save detailed workflow results summary to a file
            with open(os.path.join(workflow_dir, "workflow_summary.txt"), "w") as f:
                f.write("Enhanced Workflow Results for FP Reduction\n")
                f.write("=======================================\n\n")
                
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
                    
                    # Calculate FP statistics
                    try:
                        fp_count = pos_df.filter(pl.col("pred_type") == "FP").height
                        tp_count = pos_df.filter(pl.col("pred_type") == "TP").height
                        fp_percentage = (fp_count / (fp_count + tp_count) * 100) if fp_count + tp_count > 0 else 0
                        f.write(f"\nFalse Positives: {fp_count:,} ({fp_percentage:.1f}% of positive predictions)\n")
                        
                        # FP by splice type
                        fp_by_type = pos_df.filter(pl.col("pred_type") == "FP").group_by("splice_type").agg(pl.count())
                        f.write("\nFPs by splice type:\n")
                        for row in fp_by_type.iter_rows(named=True):
                            f.write(f"  {row['splice_type']}: {row['count']:,}\n")
                    except Exception as e:
                        f.write(f"  Could not calculate FP statistics: {e}\n")
                else:
                    f.write("Positions dataset: Not available or empty\n")
                
                # Error analysis DataFrame
                if 'error_analysis' in result and not result['error_analysis'].is_empty():
                    err_df = result['error_analysis']
                    f.write(f"\nError analysis dataset: {err_df.height:,} rows × {len(err_df.columns)} columns\n")
                else:
                    f.write("\nError analysis dataset: Not available or empty\n")
                    
                # Analysis sequences DataFrame
                if 'analysis_sequences' in result and hasattr(result['analysis_sequences'], 'height') and not result['analysis_sequences'].is_empty():
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
                
                # Gene stats
                if 'gene_stats' in result:
                    f.write("\nGene Statistics:\n")
                    f.write("---------------\n")
                    for gene_id, stats in result['gene_stats'].items():
                        f.write(f"Gene {gene_id}:\n")
                        for stat_name, stat_value in stats.items():
                            f.write(f"  - {stat_name}: {stat_value}\n")
                
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
                    if key in ["error_analysis", "positions", "analysis_sequences", "overlapping_genes", "success", "gene_stats"]:
                        continue
                        
                    if isinstance(value, dict):
                        f.write(f"\n{key}:\n")
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    elif isinstance(value, (str, int, float, bool)):
                        f.write(f"\n{key}: {value}\n")
            
            print_with_indent(f"Detailed workflow summary saved to: {os.path.join(workflow_dir, 'workflow_summary.txt')}", indent_level=1)
        else:
            print_with_indent("Enhanced workflow encountered errors", indent_level=1)
            print_with_indent(f"Error: {result.get('error', 'Unknown error')}", indent_level=2)
            print_with_indent("Continuing with analysis using available data...", indent_level=1)
    
    # STEP 3: Analyze FP reduction potential with enhanced features
    print_emphasized("STEP 3: Analyzing FP reduction potential with enhanced features")
    
    # Make sure enhanced_positions_df is defined even if workflow was skipped
    if 'enhanced_positions_df' not in locals():
        print_with_indent("Enhanced positions dataset not available, using original dataset", indent_level=1)
        enhanced_positions_df = positions_df
    
    # Create a per-gene visualization directory
    per_gene_dir = analysis_dir / "per_gene_analysis"
    os.makedirs(per_gene_dir, exist_ok=True)
    
    # Track per-gene stats for the summary
    per_gene_stats = []
    
    # Check if gene-specific analysis is needed or if we should analyze global dataset
    if len(top_gene_ids) > 0:
        print_with_indent("Running FP reduction analysis...", indent_level=1)
        print_with_indent(f"Generating per-gene visualizations for {len(top_gene_ids)} genes...", indent_level=1)
        
        # Process each gene individually
        for gene_id in top_gene_ids:
            gene_dir = os.path.join(per_gene_dir, gene_id)
            os.makedirs(gene_dir, exist_ok=True)
            
            # Filter positions to just this gene
            gene_positions = positions_df.filter(pl.col('gene_id') == gene_id)
            
            if gene_positions.height == 0:
                print_with_indent(f"Warning: No data found for gene {gene_id}", indent_level=2)
                continue
            
            print_with_indent(f"Analyzing gene {gene_id} ({gene_positions.height} positions)...", indent_level=2)
            
            # Run the FP reduction analysis on this gene
            evaluate_fp_reduction_potential(gene_positions, gene_dir)
            
            # Extract summary stats
            summary_path = os.path.join(gene_dir, "fp_reduction_summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    lines = f.readlines()
                
                # Extract FP counts from summary
                total_fps = 0
                filtered_fps = 0
                
                for line in lines:
                    if "Total FPs in dataset:" in line:
                        total_fps = int(line.split(":")[1].strip().split()[0])
                    elif "Total potential FP reduction:" in line:
                        parts = line.split(":")[1].strip().split()
                        filtered_fps = int(parts[0])
                
                reduction_percent = (filtered_fps / total_fps * 100) if total_fps > 0 else 0
                
                per_gene_stats.append({
                    "gene_id": gene_id,
                    "total_fps": total_fps,
                    "filtered_fps": filtered_fps,
                    "reduction_percent": reduction_percent
                })
    else:
        print_with_indent("No genes found with sufficient FPs for analysis.", indent_level=1)
        print_with_indent("Running global FP analysis on all available data...", indent_level=1)
        evaluate_fp_reduction_potential(positions_df, analysis_dir)
    
    # Create a per-gene summary table
    if per_gene_stats:
        import pandas as pd
        gene_stats_df = pd.DataFrame(per_gene_stats)
        gene_stats_path = os.path.join(analysis_dir, "per_gene_reduction_summary.csv")
        gene_stats_df.to_csv(gene_stats_path, index=False)
        print_with_indent(f"Saved per-gene reduction summary to: {gene_stats_path}", indent_level=1)
        
        # Print the per-gene summary
        print_with_indent("Per-gene FP reduction summary:", indent_level=1)
        for stats in per_gene_stats:
            print_with_indent(f"Gene {stats['gene_id']}: {stats['filtered_fps']}/{stats['total_fps']} FPs filtered ({stats['reduction_percent']:.1f}%)", indent_level=2)
        
        # Generate an overall summary with aggregated stats
        total_fps_all = sum(stat["total_fps"] for stat in per_gene_stats)
        total_filtered_all = sum(stat["filtered_fps"] for stat in per_gene_stats)
        overall_reduction_pct = (total_filtered_all / total_fps_all * 100) if total_fps_all > 0 else 0
        
        # Print overall summary
        print_with_indent("\n=== Overall FP Reduction Potential Summary ===", indent_level=1)
        print_with_indent(f"Total FPs across analyzed genes: {total_fps_all}", indent_level=1)
        print_with_indent(f"Total potential FP reductions: {total_filtered_all}", indent_level=1)
        print_with_indent(f"Overall reduction percentage: {overall_reduction_pct:.1f}%", indent_level=1)
        
        # Save aggregate summary to file
        with open(os.path.join(analysis_dir, "multi_gene_reduction_summary.txt"), "w") as f:
            f.write("=== Overall FP Reduction Potential Summary ===\n")
            f.write(f"Total FPs across analyzed genes: {total_fps_all}\n")
            f.write(f"Total potential FP reductions: {total_filtered_all}\n")
            f.write(f"Overall reduction percentage: {overall_reduction_pct:.1f}%\n\n")
            
            f.write("Per-gene breakdown:\n")
            for stat in per_gene_stats:
                f.write(f"Gene {stat['gene_id']}: {stat['filtered_fps']} out of {stat['total_fps']} FPs ({stat['reduction_percent']:.1f}%)\n")
        
        print_with_indent(f"Saved multi-gene analysis summary to: {os.path.join(analysis_dir, 'multi_gene_reduction_summary.txt')}", indent_level=1)
    
    # Run the consolidated multi-gene analysis
    print_with_indent("Running consolidated FP reduction analysis for all genes...", indent_level=1)
    result_df = analyze_fp_reduction_potential_by_gene(
        positions_df=enhanced_positions_df,  # Use the enhanced dataset from the workflow
        gene_ids=top_gene_ids,              # Focus on just these genes
        output_dir=str(analysis_dir)
    )
    # NOTE: The consolidated analysis functions should run regardless of whether there 
    #       are per-gene stats, as they provide valuable aggregated information 
    #       across all genes. 
    
    # Print final summary
    print_emphasized("FP Reduction Pipeline Complete")
    print_with_indent(f"Analysis results saved to: {output_dir}", indent_level=1)
    print_with_indent("Review the following directories:", indent_level=1)
    print_with_indent(f"1. Gene Identification: {identification_dir}", indent_level=2)
    print_with_indent(f"2. Enhanced Workflow: {workflow_dir}", indent_level=2)
    print_with_indent(f"3. FP Reduction Analysis: {analysis_dir}", indent_level=2)


if __name__ == "__main__":
    main()
