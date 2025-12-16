#!/usr/bin/env python3
"""
Main driver script for the complete error modeling workflow.

This script orchestrates the full pipeline:
1. Data preparation from meta-model artifacts
2. Model training 
3. IG analysis and visualization
4. Report generation

Usage:
    python run_error_model_workflow.py --data_dir data/ensembl/spliceai_eval/meta_models \
                                       --output_dir output/error_analysis \
                                       --error_type FP_vs_TP
"""

import argparse
import logging
from pathlib import Path
import sys
from pathlib import Path
from typing import Dict, Optional
import logging
import argparse

import torch
import pandas as pd
import polars as pl
from transformers import set_seed, AutoTokenizer

from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.utils import print_emphasized
from .config import ErrorModelConfig
from .dataset.dataset_preparer import ErrorDatasetPreparer
from .modeling.transformer_trainer import TransformerTrainer
from .modeling.ig_analyzer import IGAnalyzer, IGAnalysisConfig

def setup_logging(output_dir: Path, level: str = "INFO"):
    """Setup logging to both console and file."""
    log_file = output_dir / "logs" / "workflow.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def load_data(data_dir, facet, error_label, correct_label, splice_type, input_file=None, sample_size=None, context_length=None, chromosomes=None, genes=None): 
    """Load data using existing ModelEvaluationFileHandler with automatic consolidation fallback.
    
    Args:
        data_dir: Directory containing analysis sequence files
        facet: Data facet name
        error_label: Label for error samples (e.g., 'FP')
        correct_label: Label for correct samples (e.g., 'TP')
        splice_type: Type of splice sites to include
        input_file: Optional direct path to input file (bypasses auto-loading)
        sample_size: Optional maximum number of samples to use
        context_length: Optional context length to trim sequences to
    """
    print_emphasized("[DATA] Loading meta-model artifacts...")
    
    # If direct input file is provided, load it directly
    if input_file and input_file.exists():
        logging.info(f"Loading from specified input file: {input_file}")
        analysis_df = pd.read_csv(input_file, sep='\t')
        
        # Apply sampling if requested
        if sample_size and len(analysis_df) > sample_size:
            logging.info(f"Sampling {sample_size} rows from {len(analysis_df)} total rows")
            # Try to balance by prediction type if available
            if 'pred_type' in analysis_df.columns:
                unique_types = analysis_df['pred_type'].unique()
                samples_per_type = sample_size // len(unique_types)
                sampled_dfs = []
                for pred_type in unique_types:
                    type_df = analysis_df[analysis_df['pred_type'] == pred_type]
                    sampled = type_df.sample(n=min(samples_per_type, len(type_df)), random_state=42)
                    sampled_dfs.append(sampled)
                analysis_df = pd.concat(sampled_dfs, ignore_index=True)
            else:
                analysis_df = analysis_df.sample(n=sample_size, random_state=42)
        
        # Trim sequences if requested
        if context_length:
            logging.info(f"Trimming sequences to {context_length}nt")
            def trim_sequence(seq):
                if len(seq) <= context_length:
                    return seq
                center = len(seq) // 2
                half_ctx = context_length // 2
                start = max(0, center - half_ctx)
                end = min(len(seq), center + half_ctx)
                return seq[start:end]
            
            analysis_df['sequence'] = analysis_df['sequence'].apply(trim_sequence)
        
        logging.info(f"Loaded {len(analysis_df)} sequences")
        return analysis_df
    
    # Skip the legacy loading logic and go directly to chunked file consolidation
    # This ensures we always use the standard consolidated file name
    consolidated_file = data_dir / "full_analysis_sequences_error_model.tsv"
    
    if consolidated_file.exists():
        logging.info(f"Loading existing consolidated file: {consolidated_file}")
        analysis_df = pd.read_csv(consolidated_file, sep='\t')
        
        # Apply context length trimming if specified
        if context_length and 'sequence' in analysis_df.columns:
            logging.info(f"Trimming sequences to {context_length}nt")
            def trim_sequence(seq):
                if len(seq) <= context_length:
                    return seq
                center = len(seq) // 2
                half_ctx = context_length // 2
                start = max(0, center - half_ctx)
                end = min(len(seq), center + half_ctx)
                return seq[start:end]
            
            analysis_df['sequence'] = analysis_df['sequence'].apply(trim_sequence)
        
        logging.info(f"Loaded {len(analysis_df):,} sequences from existing consolidated file")
        return analysis_df
    
    # If consolidated file doesn't exist, create it from chunked files
    logging.warning(f"Consolidated file not found. Auto-consolidating chunked analysis_sequences_* files...")
    
    # Import consolidation utilities
    from .dataset.consolidate_analysis_data import AnalysisDataConsolidator
    
    # Find all chunked analysis sequence files in multiple formats
    chunk_files = []
    for pattern in ["analysis_sequences_*_chunk_*.tsv", 
                   "analysis_sequences_*_chunk_*.csv",
                   "analysis_sequences_*_chunk_*.parquet"]:
        chunk_files.extend(data_dir.glob(pattern))
    
    if not chunk_files:
        raise FileNotFoundError(f"No analysis_sequences_* chunk files (.tsv/.csv/.parquet) found in {data_dir}")
    
    logging.info(f"Found {len(chunk_files)} chunk files to consolidate")
    
    # Log filtering if specified
    if chromosomes:
        logging.info(f"Filtering for chromosomes: {', '.join(str(c) for c in chromosomes)}")
    if genes:
        logging.info(f"Filtering for {len(genes)} genes: {', '.join(genes[:5])}{'...' if len(genes) > 5 else ''}")
    
    # Set default sample size limit to prevent huge consolidated files
    if not sample_size:
        sample_size = 50000  # Default limit to prevent memory issues
        logging.info(f"No sample size specified, using default limit: {sample_size:,}")
    
    # Load and consolidate chunks with early stopping for sample size
    consolidated_dfs = []
    total_rows = 0
    
    # Keep track of schema for debugging
    reference_schema = None
    
    for chunk_file in chunk_files:
        try:
            logging.info(f"Loading chunk: {chunk_file.name}")
            
            # Load based on file extension
            if chunk_file.suffix == '.parquet':
                chunk_df = pl.read_parquet(chunk_file)
            elif chunk_file.suffix == '.csv':
                chunk_df = pl.read_csv(chunk_file)
            else:  # .tsv or default
                chunk_df = pl.read_csv(chunk_file, separator='\t')
            
            # Apply chromosome filtering if specified
            if chromosomes and 'chr' in chunk_df.columns:
                # Convert chromosomes to string for comparison
                chr_filter = [str(c) for c in chromosomes]
                chunk_df = chunk_df.filter(pl.col('chr').cast(pl.Utf8).is_in(chr_filter))
                if chunk_df.height == 0:
                    logging.debug(f"No matching chromosomes in {chunk_file.name}")
                    continue
            
            # Apply gene filtering if specified (supports both gene names and IDs)
            if genes:
                gene_filter_applied = False
                
                # Try filtering by gene_name
                if 'gene_name' in chunk_df.columns:
                    name_matches = chunk_df.filter(pl.col('gene_name').is_in(genes))
                    gene_filter_applied = True
                
                # Try filtering by gene_id
                if 'gene_id' in chunk_df.columns:
                    id_matches = chunk_df.filter(pl.col('gene_id').is_in(genes))
                    gene_filter_applied = True
                    
                    # Combine both filters if both columns exist
                    if 'gene_name' in chunk_df.columns:
                        chunk_df = pl.concat([name_matches, id_matches]).unique()
                    else:
                        chunk_df = id_matches
                elif gene_filter_applied:
                    chunk_df = name_matches
                
                if gene_filter_applied and chunk_df.height == 0:
                    logging.debug(f"No matching genes in {chunk_file.name}")
                    continue
                elif not gene_filter_applied:
                    logging.debug(f"No gene_name or gene_id column in {chunk_file.name}, skipping gene filter")
            
            # Cast numeric columns to ensure consistent types
            for col in chunk_df.columns:
                if col in ['chr', 'position', 'start', 'end', 'idx']:
                    # These should be integers
                    if chunk_df[col].dtype == pl.Utf8:
                        try:
                            chunk_df = chunk_df.with_columns(pl.col(col).cast(pl.Int64))
                        except:
                            # If can't cast, keep as string
                            pass
                elif col in ['score', 'spliceai_score', 'probability']:
                    # These should be floats
                    if chunk_df[col].dtype == pl.Utf8:
                        try:
                            chunk_df = chunk_df.with_columns(pl.col(col).cast(pl.Float64))
                        except:
                            pass
            
            # Check for schema consistency
            if reference_schema is None:
                reference_schema = chunk_df.schema
                logging.debug(f"Reference schema: {reference_schema}")
            else:
                if chunk_df.schema != reference_schema:
                    logging.debug(f"Schema mismatch in {chunk_file.name}")
                    logging.debug(f"Expected: {reference_schema}")
                    logging.debug(f"Got: {chunk_df.schema}")
                
            consolidated_dfs.append(chunk_df)
            total_rows += chunk_df.height
            logging.info(f"  Loaded {chunk_df.height:,} rows (total: {total_rows:,})")
            
            # Early stopping if we have enough data
            if total_rows >= sample_size * 2:  # Load 2x sample size for better sampling
                logging.info(f"Early stopping: loaded {total_rows:,} rows (target: {sample_size:,})")
                break
                
        except Exception as chunk_error:
            logging.warning(f"Failed to load {chunk_file}: {chunk_error}")
            continue
    
    if not consolidated_dfs:
        raise FileNotFoundError(f"No valid chunk files could be loaded from {data_dir}")
    
    # Combine all chunks with schema alignment
    logging.info(f"Consolidating {len(consolidated_dfs)} chunks with {total_rows:,} total rows...")
    
    # Handle schema mismatches by using relaxed concatenation
    try:
        full_df = pl.concat(consolidated_dfs, how="vertical_relaxed")
    except Exception as e:
        logging.warning(f"Schema mismatch during concatenation: {e}")
        # Fall back to diagonal concatenation which handles mismatched schemas
        full_df = pl.concat(consolidated_dfs, how="diagonal_relaxed")
    
    # Apply gene-level subsampling to maintain genomic structure
    # Skip subsampling if specific genes were requested
    if sample_size and full_df.height > sample_size and not genes:
        if 'gene_name' in full_df.columns:
            logging.info(f"Subsampling at gene level: {sample_size:,} row limit from {full_df.height:,} total rows")
        
            # Get gene statistics
            gene_counts = full_df.group_by('gene_name').agg(
                pl.count().alias('row_count')
            ).sort('row_count', descending=True)
            
            total_genes = gene_counts.height
            logging.info(f"Found {total_genes:,} unique genes")
            
            # Try balanced sampling by prediction_type if available
            if 'prediction_type' in full_df.columns:
                unique_types = full_df['prediction_type'].unique().to_list()
                target_per_type = sample_size // len(unique_types)
                
                selected_genes = []
                for pred_type in unique_types:
                    # Get genes for this prediction type
                    type_genes = full_df.filter(
                        pl.col('prediction_type') == pred_type
                    ).select('gene_name').unique()['gene_name'].to_list()
                    
                    # Randomly sample genes until we approach the target
                    import random
                    random.seed(42)
                    random.shuffle(type_genes)
                    
                    rows_collected = 0
                    for gene in type_genes:
                        gene_rows = full_df.filter(
                            (pl.col('gene_name') == gene) & 
                            (pl.col('prediction_type') == pred_type)
                        ).height
                        
                        if rows_collected + gene_rows <= target_per_type * 1.2:  # Allow 20% overage
                            selected_genes.append(gene)
                            rows_collected += gene_rows
                        
                        if rows_collected >= target_per_type:
                            break
                
                # Filter to keep only selected genes
                full_df = full_df.filter(pl.col('gene_name').is_in(selected_genes))
                logging.info(f"Selected {len(selected_genes)} genes with {full_df.height:,} total rows")
                
            else:
                # Sample complete genes randomly
                gene_list = gene_counts['gene_name'].to_list()
                import random
                random.seed(42)
                random.shuffle(gene_list)
                
                selected_genes = []
                rows_collected = 0
                
                for gene in gene_list:
                    gene_row_count = gene_counts.filter(
                        pl.col('gene_name') == gene
                    )['row_count'][0]
                    
                    if rows_collected + gene_row_count <= sample_size * 1.2:  # Allow 20% overage
                        selected_genes.append(gene)
                        rows_collected += gene_row_count
                    
                    if rows_collected >= sample_size:
                        break
                
                # Filter to keep only selected genes
                full_df = full_df.filter(pl.col('gene_name').is_in(selected_genes))
                logging.info(f"Selected {len(selected_genes)} complete genes with {full_df.height:,} total rows")
                
        else:
            # Fallback to row-level sampling if no gene_name column
            logging.warning("No 'gene_name' column found. Using row-level sampling.")
            full_df = full_df.sample(n=sample_size, seed=42)
    
    # Save consolidated file
    logging.info(f"Saving consolidated file: {consolidated_file}")
    full_df.write_csv(consolidated_file, separator='\t')
    logging.info(f"✅ Created consolidated file with {full_df.height:,} rows")
    
    # Collect data statistics before converting to pandas
    data_stats = {
        'requested_sample_size': sample_size,
        'final_sample_size': full_df.height,
        'num_genes': full_df['gene_name'].n_unique() if 'gene_name' in full_df.columns else None,
        'num_chromosomes': full_df['chr'].n_unique() if 'chr' in full_df.columns else None,
        'chromosomes': sorted(full_df['chr'].unique().to_list()) if 'chr' in full_df.columns else None,
    }
    
    # Load the consolidated file directly with pandas
    logging.info(f"Loading consolidated file: {consolidated_file}")
    analysis_df = pd.read_csv(consolidated_file, sep='\t')
    
    # Attach statistics to dataframe for later use
    analysis_df.attrs['data_stats'] = data_stats
    
    # Apply context length trimming if specified
    if context_length and 'sequence' in analysis_df.columns:
        logging.info(f"Trimming sequences to {context_length}nt")
        def trim_sequence(seq):
            if len(seq) <= context_length:
                return seq
            center = len(seq) // 2
            half_ctx = context_length // 2
            start = max(0, center - half_ctx)
            end = min(len(seq), center + half_ctx)
            return seq[start:end]
        
        analysis_df['sequence'] = analysis_df['sequence'].apply(trim_sequence)
    
    logging.info(f"Loaded {len(analysis_df):,} sequences from consolidated file")
    return analysis_df


def prepare_datasets(analysis_df, config: ErrorModelConfig, error_label: str, correct_label: str, 
                    feature_columns: list = None):
    """Prepare datasets using ErrorDatasetPreparer.
    
    Args:
        analysis_df: DataFrame with analysis sequences
        config: Model configuration
        error_label: Label for error samples
        correct_label: Label for correct samples
        feature_columns: Optional list of specific features to use
    """
    print_emphasized("[DATA] Preparing position-centric datasets...")
    
    # If specific features are requested, configure the model to use only those
    if feature_columns:
        logging.info(f"Using specified features: {feature_columns}")
        # Update config based on available features
        config.include_base_scores = any(f in feature_columns for f in ['score', 'donor_score', 'acceptor_score', 'neither_score'])
        config.include_context_features = any('context_' in f for f in feature_columns)
        config.include_donor_features = any('donor_' in f and f != 'donor_score' for f in feature_columns)
        config.include_acceptor_features = any('acceptor_' in f and f != 'acceptor_score' for f in feature_columns)
        config.include_derived_features = any(f in feature_columns for f in ['entropy', 'max_score', 'score_std'])
    
    preparer = ErrorDatasetPreparer(config)
    
    # Convert to pandas if needed
    if hasattr(analysis_df, 'to_pandas'):
        analysis_df = analysis_df.to_pandas()
    
    dataset_info = preparer.prepare_dataset_from_dataframe(
        df=analysis_df,
        error_label=error_label,
        correct_label=correct_label
    )
    
    # Log dataset statistics
    for split, dataset in dataset_info['datasets'].items():
        labels = [sample['labels'].item() for sample in dataset]
        error_count = sum(labels)
        correct_count = len(labels) - error_count
        logging.info(f"{split.capitalize()} split: {len(dataset)} samples, labels: {{0: {correct_count}, 1: {error_count}}}")
    
    return dataset_info


def train_model(dataset_info, config: ErrorModelConfig, output_dir: Path, skip_training: bool = False):
    """Train or load the error model."""
    model_dir = output_dir / "models"
    model_path = model_dir / "best_model.pt"
    
    if skip_training and model_path.exists():
        print_emphasized("[MODEL] Loading existing model...")
        trainer = TransformerTrainer(config)
        trainer.load_model(model_path)
    else:
        print_emphasized("[MODEL] Training transformer model...")
        trainer = TransformerTrainer(config)
        
        # Train the model
        trainer.train(
            train_dataset=dataset_info['datasets']['train'],
            val_dataset=dataset_info['datasets']['val'],
            output_dir=model_dir
        )
        
        # Save model (trainer.trainer is the HuggingFace Trainer instance)
        if hasattr(trainer, 'trainer') and trainer.trainer:
            trainer.trainer.save_model(model_path)
        
        # Evaluate on test set
        test_results = trainer.evaluate(dataset_info['datasets']['test'])
        logging.info(f"Test results: {test_results}")
    
    return trainer, model_path


def run_ig_analysis(trainer, dataset_info, config: ErrorModelConfig, ig_config: IGAnalysisConfig, 
                   output_dir: Path, max_samples: int = 500):
    """Run IG analysis using IGAnalyzer."""
    print_emphasized("[IG] Running Integrated Gradients analysis...")
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ig_analyzer = IGAnalyzer(trainer.model, tokenizer, config, ig_config)
    
    # Prepare test data
    test_dataset = dataset_info['datasets']['test'][:max_samples]
    sequences = [sample['sequence'] for sample in test_dataset]
    labels = [sample['label'] for sample in test_dataset]
    
    # Get additional features if available
    additional_features = None
    if 'additional_features' in test_dataset[0]:
        import numpy as np
        additional_features = np.array([sample['additional_features'] for sample in test_dataset])
    
    # Compute attributions
    attributions = ig_analyzer.compute_attributions(
        sequences=sequences,
        labels=labels,
        additional_features=additional_features,
        target_class=1
    )
    
    # Analyze patterns
    analysis_results = ig_analyzer.analyze_error_patterns(attributions)
    
    # Save results
    ig_output_dir = output_dir / "ig_analysis"
    saved_files = ig_analyzer.save_results(attributions, analysis_results, ig_output_dir)
    
    return attributions, analysis_results, saved_files


def create_visualizations(attributions, analysis_results, output_dir: Path):
    """Create visualizations using the visualization package."""
    print_emphasized("[VIZ] Creating visualizations...")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Initialize plotters
    freq_plotter = visualization.FrequencyPlotter()
    align_plotter = visualization.AlignmentPlotter()
    
    saved_plots = {}
    
    # Token frequency comparison
    fig1 = freq_plotter.plot_token_frequency_comparison(analysis_results)
    path1 = viz_dir / "token_frequency_comparison.png"
    fig1.savefig(path1, dpi=300, bbox_inches='tight')
    saved_plots['frequency'] = path1
    
    # Attribution distribution
    fig2 = freq_plotter.plot_attribution_distribution(analysis_results)
    path2 = viz_dir / "attribution_distribution.png"
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    saved_plots['distribution'] = path2
    
    # Top tokens analysis
    fig3 = freq_plotter.plot_top_tokens_analysis(analysis_results)
    path3 = viz_dir / "top_tokens_analysis.png"
    fig3.savefig(path3, dpi=300, bbox_inches='tight')
    saved_plots['top_tokens'] = path3
    
    # Positional analysis
    fig4 = align_plotter.plot_positional_analysis(analysis_results)
    path4 = viz_dir / "positional_analysis.png"
    fig4.savefig(path4, dpi=300, bbox_inches='tight')
    saved_plots['positional'] = path4
    
    # Close figures
    import matplotlib.pyplot as plt
    plt.close('all')
    
    logging.info(f"Created {len(saved_plots)} visualizations")
    return saved_plots


def generate_report(analysis_results, saved_plots, config: ErrorModelConfig, output_dir: Path):
    """Generate summary report."""
    report_path = output_dir / "analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Error Model Analysis Report\n\n")
        
        # Configuration
        f.write("## Configuration\n")
        f.write(f"- Model: {config.model_name}\n")
        f.write(f"- Context Length: {config.context_length}\n")
        f.write(f"- Error Types: {', '.join(config.labels)}\n\n")
        
        # Key findings
        summary = analysis_results['summary']
        f.write("## Key Findings\n")
        f.write(f"- Analyzed {summary['n_error_samples']} error samples\n")
        f.write(f"- Analyzed {summary['n_correct_samples']} correct samples\n\n")
        
        # Top tokens
        f.write("## Top Important Tokens\n")
        token_ratios = analysis_results['token_analysis']['token_ratios']
        f.write("| Token | Error/Correct Ratio |\n|-------|---------------------|\n")
        for token, ratio in list(token_ratios.items())[:10]:
            f.write(f"| {token} | {ratio:.3f} |\n")
        f.write("\n")
        
        # Visualizations
        f.write("## Generated Files\n")
        for name, path in saved_plots.items():
            f.write(f"- {name}: `{path.name}`\n")
    
    logging.info(f"Generated report: {report_path}")
    return report_path


def main():
    """Main workflow orchestration."""
    parser = argparse.ArgumentParser(description="Error model workflow")
    parser.add_argument("--data-dir", type=Path, required=True, help="Meta-model artifacts directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--error-type", choices=["FP_vs_TP", "FN_vs_TP", "fp_vs_tp", "fn_vs_tp", "fp", "fn"], 
                       default="fp", help="Error type: fp, fn, fp_vs_tp, fn_vs_tp, FP_vs_TP, FN_vs_TP")
    parser.add_argument("--splice-type", choices=["donor", "acceptor", "any"], default="any")
    parser.add_argument("--facet", default="simple", help="Data facet")
    parser.add_argument("--model-name", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--context-length", type=int, help="Trim sequences to this length (default: use full sequences)")
    parser.add_argument("--batch-size", type=int, default=16)
    
    # New options for faster testing
    parser.add_argument("--input-file", type=Path, help="Direct path to input TSV file (bypasses auto-loading)")
    parser.add_argument("--sample-size", type=int, help="Maximum number of samples to use")
    parser.add_argument("--chromosomes", nargs='+', help="Specific chromosomes to include (e.g., 1 2 X)")
    parser.add_argument("--genes", nargs='+', help="Specific genes to include by name or ID (e.g., BRCA1 TP53 ENSG00000012048)")
    parser.add_argument("--genes-file", type=Path, help="File containing gene names or IDs (one per line)")
    parser.add_argument("--features", nargs='+', help="Specific feature columns to use")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-ig-samples", type=int, default=500)
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--use-mixed-precision", action="store_true", default=True, help="Enable FP16 mixed precision")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-ig", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    
    # Experiment tracking arguments
    parser.add_argument("--enable-mlflow", action="store_true", help="Enable MLflow experiment tracking")
    parser.add_argument("--mlflow-tracking-uri", type=str, help="MLflow tracking server URI")
    parser.add_argument("--mlflow-experiment-name", type=str, help="MLflow experiment name")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb-project", type=str, help="W&B project name")
    parser.add_argument("--enable-tensorboard", action="store_true", help="Enable TensorBoard logging")
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir, args.log_level)
    
    # Parse error type with support for simplified notation
    def parse_error_type(error_type_str):
        """Parse error type string into error_label and correct_label."""
        error_type_str = error_type_str.lower()
        
        # Handle simplified notation
        if error_type_str == "fp":
            return "FP", "TP", "FP_vs_TP"
        elif error_type_str == "fn":
            return "FN", "TP", "FN_vs_TP"
        
        # Handle full notation (case insensitive)
        if "_vs_" in error_type_str:
            parts = error_type_str.split("_vs_")
            error_label = parts[0].upper()
            correct_label = parts[1].upper()
            full_type = f"{error_label}_vs_{correct_label}"
            return error_label, correct_label, full_type
        
        # Fallback
        raise ValueError(f"Invalid error type: {error_type_str}")
    
    error_label, correct_label, full_error_type = parse_error_type(args.error_type)
    
    # Parse gene list from file if provided
    genes_list = args.genes if args.genes else []
    if args.genes_file:
        if args.genes_file.exists():
            with open(args.genes_file, 'r') as f:
                file_genes = [line.strip() for line in f if line.strip()]
                genes_list.extend(file_genes)
                logging.info(f"Loaded {len(file_genes)} genes from {args.genes_file}")
        else:
            logging.warning(f"Genes file not found: {args.genes_file}")
    
    # Remove duplicates if any
    if genes_list:
        genes_list = list(set(genes_list))
        logging.info(f"Total unique genes to filter: {len(genes_list)}")
    
    # Create configurations
    config = ErrorModelConfig(
        context_length=args.context_length if args.context_length else 200,
        error_label=error_label,
        correct_label=correct_label,
        splice_type=args.splice_type,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        use_mixed_precision=args.use_mixed_precision,
        output_dir=args.output_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        enable_tensorboard=args.enable_tensorboard
    )
    
    ig_config = IGAnalysisConfig()
    
    try:
        print_emphasized("🧬 STARTING ERROR MODEL WORKFLOW")
        print_emphasized(f"📊 Error Analysis: {full_error_type} ({args.splice_type} sites)")
        
        # 1. Load data
        analysis_df = load_data(
            args.data_dir, args.facet, error_label, correct_label, args.splice_type,
            input_file=args.input_file, sample_size=args.sample_size, context_length=args.context_length,
            chromosomes=args.chromosomes, genes=genes_list if genes_list else None
        )
        
        # Display data statistics
        if hasattr(analysis_df, 'attrs') and 'data_stats' in analysis_df.attrs:
            stats = analysis_df.attrs['data_stats']
            print_emphasized("📈 DATA STATISTICS")
            print(f"  • Requested sample size: {stats['requested_sample_size']:,}" if stats['requested_sample_size'] else "  • Requested sample size: No limit")
            print(f"  • Final sample size: {stats['final_sample_size']:,} rows")
            if stats['num_genes']:
                print(f"  • Number of genes: {stats['num_genes']:,}")
            if stats['num_chromosomes']:
                print(f"  • Number of chromosomes: {stats['num_chromosomes']}")
                if stats['chromosomes']:
                    chr_list = ', '.join(str(c) for c in stats['chromosomes'][:10])
                    if len(stats['chromosomes']) > 10:
                        chr_list += f", ... ({len(stats['chromosomes'])-10} more)"
                    print(f"  • Chromosomes: {chr_list}")
        
        # Display model configuration
        print_emphasized("🔧 MODEL CONFIGURATION")
        print(f"  • Model: {config.model_name}")
        print(f"  • Context length: {config.context_length}nt")
        print(f"  • Batch size: {config.batch_size}")
        print(f"  • Epochs: {config.num_epochs}")
        print(f"  • Device: {config.device}")
        print(f"  • Mixed precision: {config.use_mixed_precision}")
        
        # 2. Prepare datasets
        dataset_info = prepare_datasets(analysis_df, config, error_label, correct_label, 
                                       feature_columns=args.features)
        
        # 3. Train model
        trainer, model_path = train_model(dataset_info, config, args.output_dir, args.skip_training)
        
        # 4. IG analysis
        if not args.skip_ig:
            attributions, analysis_results, saved_files = run_ig_analysis(
                trainer, dataset_info, config, ig_config, args.output_dir, args.max_ig_samples
            )
            
            # 5. Create visualizations
            saved_plots = create_visualizations(attributions, analysis_results, args.output_dir)
            
            # 6. Generate report
            report_path = generate_report(analysis_results, saved_plots, config, args.output_dir)
        
        print_emphasized("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()
