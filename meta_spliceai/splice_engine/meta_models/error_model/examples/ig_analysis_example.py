#!/usr/bin/env python3
"""
Example script demonstrating Integrated Gradients analysis workflow.

This script shows how to:
1. Load a trained error model
2. Prepare data for IG analysis
3. Compute IG attributions
4. Analyze error patterns
5. Generate visualizations
"""

import os
import sys
from pathlib import Path
import logging
import argparse
from typing import Dict, Any, Optional

import torch
import pandas as pd
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.error_model import (
    ErrorModelConfig, 
    ErrorDatasetPreparer,
    TransformerTrainer,
    IGAnalyzer,
    visualization
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ig_analysis.log')
        ]
    )


def load_trained_model(
    model_path: Path,
    config: ErrorModelConfig
) -> tuple[TransformerTrainer, AutoTokenizer]:
    """Load a trained model and tokenizer."""
    logger = logging.getLogger(__name__)
    
    # Initialize trainer
    trainer = TransformerTrainer(config)
    
    # Load model checkpoint
    if model_path.exists():
        logger.info(f"Loading trained model from {model_path}")
        trainer.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    return trainer, tokenizer


def prepare_ig_data(
    data_dir: Path,
    config: ErrorModelConfig,
    max_samples: Optional[int] = None
) -> tuple[list, list, Optional[Any]]:
    """Prepare data for IG analysis."""
    logger = logging.getLogger(__name__)
    
    # Initialize dataset preparer
    preparer = ErrorDatasetPreparer(config)
    
    # Load and prepare dataset
    logger.info(f"Loading data from {data_dir}")
    dataset_info = preparer.prepare_dataset(data_dir)
    
    # Get test data for IG analysis
    test_data = dataset_info['datasets']['test']
    
    # Extract sequences, labels, and features
    sequences = []
    labels = []
    additional_features = []
    
    for i, sample in enumerate(test_data):
        if max_samples and i >= max_samples:
            break
            
        sequences.append(sample['sequence'])
        labels.append(sample['label'])
        
        if 'additional_features' in sample:
            additional_features.append(sample['additional_features'])
    
    # Convert features to numpy array if available
    features_array = None
    if additional_features:
        import numpy as np
        features_array = np.array(additional_features)
    
    logger.info(f"Prepared {len(sequences)} sequences for IG analysis")
    return sequences, labels, features_array


def run_ig_analysis(
    model: TransformerTrainer,
    tokenizer: AutoTokenizer,
    sequences: list,
    labels: list,
    additional_features: Optional[Any],
    config: ErrorModelConfig,
    output_dir: Path
) -> Dict[str, Any]:
    """Run complete IG analysis workflow."""
    logger = logging.getLogger(__name__)
    
    # Initialize IG analyzer
    ig_analyzer = IGAnalyzer(
        model=model.model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Compute attributions
    logger.info("Computing IG attributions...")
    attributions = ig_analyzer.compute_attributions(
        sequences=sequences,
        labels=labels,
        additional_features=additional_features,
        target_class=1  # Focus on error class
    )
    
    # Analyze error patterns
    logger.info("Analyzing error patterns...")
    analysis_results = ig_analyzer.analyze_error_patterns(
        attributions=attributions,
        error_label=1,
        correct_label=0
    )
    
    # Save results
    logger.info("Saving IG analysis results...")
    saved_files = ig_analyzer.save_results(
        attributions=attributions,
        analysis_results=analysis_results,
        output_dir=output_dir
    )
    
    return {
        'attributions': attributions,
        'analysis_results': analysis_results,
        'saved_files': saved_files
    }


def create_visualizations(
    analysis_results: Dict[str, Any],
    attributions: list,
    output_dir: Path
) -> Dict[str, Path]:
    """Create comprehensive visualizations."""
    logger = logging.getLogger(__name__)
    
    # Create visualization directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Initialize plotters
    alignment_plotter = visualization.AlignmentPlotter()
    frequency_plotter = visualization.FrequencyPlotter()
    
    saved_plots = {}
    
    # 1. Token frequency comparison
    logger.info("Creating frequency comparison plot...")
    fig1 = frequency_plotter.plot_token_frequency_comparison(
        analysis_results=analysis_results,
        title="Error vs Correct Token Frequencies"
    )
    freq_path = viz_dir / "token_frequency_comparison.png"
    fig1.savefig(freq_path, dpi=300, bbox_inches='tight')
    saved_plots['frequency_comparison'] = freq_path
    
    # 2. Attribution distribution
    logger.info("Creating attribution distribution plot...")
    fig2 = frequency_plotter.plot_attribution_distribution(
        analysis_results=analysis_results,
        title="Attribution Distribution Analysis"
    )
    dist_path = viz_dir / "attribution_distribution.png"
    fig2.savefig(dist_path, dpi=300, bbox_inches='tight')
    saved_plots['attribution_distribution'] = dist_path
    
    # 3. Top tokens analysis
    logger.info("Creating top tokens analysis...")
    fig3 = frequency_plotter.plot_top_tokens_analysis(
        analysis_results=analysis_results,
        title="Top Important Tokens Analysis"
    )
    tokens_path = viz_dir / "top_tokens_analysis.png"
    fig3.savefig(tokens_path, dpi=300, bbox_inches='tight')
    saved_plots['top_tokens'] = tokens_path
    
    # 4. Sample alignment plots
    logger.info("Creating sample alignment plots...")
    
    # Separate error and correct samples
    error_samples = [attr for attr in attributions if attr['label'] == 1]
    correct_samples = [attr for attr in attributions if attr['label'] == 0]
    
    # Create alignment plots for a few examples
    n_examples = min(3, len(error_samples), len(correct_samples))
    
    for i in range(n_examples):
        # Error example
        if i < len(error_samples):
            error_sample = error_samples[i]
            fig_error = alignment_plotter.plot_sequence_attribution(
                sequence=error_sample['sequence'],
                tokens=error_sample['tokens'],
                attributions=error_sample['attributions'],
                title=f"Error Example {i+1} - Token Attributions"
            )
            error_align_path = viz_dir / f"error_alignment_example_{i+1}.png"
            fig_error.savefig(error_align_path, dpi=300, bbox_inches='tight')
            saved_plots[f'error_alignment_{i+1}'] = error_align_path
        
        # Correct example
        if i < len(correct_samples):
            correct_sample = correct_samples[i]
            fig_correct = alignment_plotter.plot_sequence_attribution(
                sequence=correct_sample['sequence'],
                tokens=correct_sample['tokens'],
                attributions=correct_sample['attributions'],
                title=f"Correct Example {i+1} - Token Attributions"
            )
            correct_align_path = viz_dir / f"correct_alignment_example_{i+1}.png"
            fig_correct.savefig(correct_align_path, dpi=300, bbox_inches='tight')
            saved_plots[f'correct_alignment_{i+1}'] = correct_align_path
    
    # 5. Comparative attribution plot
    if error_samples and correct_samples:
        logger.info("Creating comparative attribution plot...")
        fig5 = alignment_plotter.plot_comparative_attribution(
            error_attributions=error_samples[:5],
            correct_attributions=correct_samples[:5],
            title="Error vs Correct Attribution Patterns"
        )
        comp_path = viz_dir / "comparative_attributions.png"
        fig5.savefig(comp_path, dpi=300, bbox_inches='tight')
        saved_plots['comparative'] = comp_path
    
    # 6. Positional analysis
    logger.info("Creating positional analysis plot...")
    fig6 = alignment_plotter.plot_positional_analysis(
        analysis_results=analysis_results,
        title="Positional Attribution Patterns"
    )
    pos_path = viz_dir / "positional_analysis.png"
    fig6.savefig(pos_path, dpi=300, bbox_inches='tight')
    saved_plots['positional'] = pos_path
    
    # Close all figures to save memory
    import matplotlib.pyplot as plt
    plt.close('all')
    
    logger.info(f"Created {len(saved_plots)} visualization plots in {viz_dir}")
    return saved_plots


def generate_summary_report(
    analysis_results: Dict[str, Any],
    saved_files: Dict[str, Path],
    saved_plots: Dict[str, Path],
    output_dir: Path
) -> Path:
    """Generate a summary report of the IG analysis."""
    logger = logging.getLogger(__name__)
    
    report_path = output_dir / "ig_analysis_summary.md"
    
    with open(report_path, 'w') as f:
        f.write("# Integrated Gradients Analysis Summary\n\n")
        
        # Overview
        summary = analysis_results['summary']
        f.write("## Overview\n\n")
        f.write(f"- **Error Samples**: {summary['n_error_samples']}\n")
        f.write(f"- **Correct Samples**: {summary['n_correct_samples']}\n")
        f.write(f"- **Total Samples**: {summary['n_error_samples'] + summary['n_correct_samples']}\n\n")
        
        # Key Statistics
        f.write("## Key Statistics\n\n")
        error_stats = summary['error_stats']
        correct_stats = summary['correct_stats']
        
        f.write("### Error Predictions\n")
        f.write(f"- Mean Total Attribution: {error_stats.get('mean_total_attribution', 0):.4f}\n")
        f.write(f"- Mean Token Attribution: {error_stats.get('mean_mean_attribution', 0):.4f}\n")
        f.write(f"- Max Attribution: {error_stats.get('mean_max_attribution', 0):.4f}\n")
        f.write(f"- Min Attribution: {error_stats.get('mean_min_attribution', 0):.4f}\n\n")
        
        f.write("### Correct Predictions\n")
        f.write(f"- Mean Total Attribution: {correct_stats.get('mean_total_attribution', 0):.4f}\n")
        f.write(f"- Mean Token Attribution: {correct_stats.get('mean_mean_attribution', 0):.4f}\n")
        f.write(f"- Max Attribution: {correct_stats.get('mean_max_attribution', 0):.4f}\n")
        f.write(f"- Min Attribution: {correct_stats.get('mean_min_attribution', 0):.4f}\n\n")
        
        # Top Tokens
        f.write("## Top Important Tokens\n\n")
        token_ratios = analysis_results['token_analysis']['token_ratios']
        f.write("| Token | Frequency Ratio (Error/Correct) |\n")
        f.write("|-------|----------------------------------|\n")
        
        for i, (token, ratio) in enumerate(list(token_ratios.items())[:10]):
            f.write(f"| {token} | {ratio:.3f} |\n")
        
        f.write("\n")
        
        # Files Generated
        f.write("## Generated Files\n\n")
        f.write("### Data Files\n")
        for name, path in saved_files.items():
            f.write(f"- **{name}**: `{path.name}`\n")
        
        f.write("\n### Visualization Files\n")
        for name, path in saved_plots.items():
            f.write(f"- **{name}**: `{path.name}`\n")
        
        f.write("\n")
        
        # Interpretation
        f.write("## Interpretation Guidelines\n\n")
        f.write("1. **High Frequency Ratio (>1)**: Tokens more important for error predictions\n")
        f.write("2. **Low Frequency Ratio (<1)**: Tokens more important for correct predictions\n")
        f.write("3. **Attribution Magnitude**: Strength of token influence on prediction\n")
        f.write("4. **Positional Patterns**: Spatial distribution of important tokens\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Examine high-ratio tokens for biological significance\n")
        f.write("2. Investigate positional patterns near splice sites\n")
        f.write("3. Validate findings with domain knowledge\n")
        f.write("4. Use insights for model improvement or feature engineering\n")
    
    logger.info(f"Generated summary report: {report_path}")
    return report_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run IG analysis on trained error model")
    parser.add_argument("--model_path", type=Path, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=Path, required=True,
                       help="Directory containing training data")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--config_path", type=Path,
                       help="Path to model config file")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples for IG analysis")
    parser.add_argument("--log_level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configuration
        if args.config_path and args.config_path.exists():
            # Load from file (implementation depends on config format)
            config = ErrorModelConfig()  # Use default for now
        else:
            config = ErrorModelConfig()
        
        logger.info("Starting IG analysis workflow...")
        
        # Load trained model
        model, tokenizer = load_trained_model(args.model_path, config)
        
        # Prepare data
        sequences, labels, features = prepare_ig_data(
            args.data_dir, config, args.max_samples
        )
        
        # Run IG analysis
        results = run_ig_analysis(
            model, tokenizer, sequences, labels, features, config, args.output_dir
        )
        
        # Create visualizations
        plots = create_visualizations(
            results['analysis_results'],
            results['attributions'],
            args.output_dir
        )
        
        # Generate summary report
        report_path = generate_summary_report(
            results['analysis_results'],
            results['saved_files'],
            plots,
            args.output_dir
        )
        
        logger.info("IG analysis completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Summary report: {report_path}")
        
    except Exception as e:
        logger.error(f"IG analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
