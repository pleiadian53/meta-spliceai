#!/usr/bin/env python3
"""
Benchmark script for evaluating splice site prediction models against reference datasets.

This script demonstrates how to evaluate the foundation model against benchmark datasets
and compare performance with other splice prediction methods like SpliceAI.

Example usage:
    python benchmark.py --model model_path.h5 --benchmark_data benchmark.csv --output results.json
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, 
    average_precision_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from meta_spliceai.foundation_model.deployment import SplicePredictorModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark splice site prediction models')
    
    # Required arguments
    parser.add_argument('--model', required=True, help='Path to trained model file (.h5)')
    parser.add_argument('--benchmark_data', required=True, 
                       help='Path to benchmark data CSV with sequences and labels')
    parser.add_argument('--output', required=True, help='Path to output results file (.json)')
    
    # Model parameters
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for prediction (default: 32)')
    
    # Comparison parameters
    parser.add_argument('--spliceai_scores', help='Path to pre-computed SpliceAI scores for comparison')
    parser.add_argument('--other_scores', help='Path to pre-computed scores from other models for comparison')
    
    # Output options
    parser.add_argument('--plot_dir', default='benchmark_plots',
                       help='Directory to save benchmark plots (default: benchmark_plots)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    
    return parser.parse_args()

def load_benchmark_data(benchmark_file):
    """Load benchmark dataset."""
    logger.info(f"Loading benchmark data from {benchmark_file}")
    
    try:
        data = pd.read_csv(benchmark_file)
    except Exception as e:
        logger.error(f"Error reading benchmark file: {e}")
        sys.exit(1)
    
    # Check required columns
    required_columns = ['sequence', 'label']
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Benchmark file must contain a '{col}' column")
            sys.exit(1)
    
    return data

def evaluate_model(predictions, labels, threshold=0.5):
    """Calculate performance metrics for model predictions."""
    # Threshold predictions to get binary classifications
    y_pred = (predictions >= threshold).astype(int)
    y_true = labels
    
    # Compute standard classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Compute confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compute ROC curve and AUROC
    fpr, tpr, _ = roc_curve(y_true, predictions)
    auroc = auc(fpr, tpr)
    
    # Compute PR curve and AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, predictions)
    auprc = average_precision_score(y_true, predictions)
    
    # Compute additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Return all metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': recall,
        'specificity': specificity,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'auroc': auroc,
        'auprc': auprc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision_curve': precision_curve.tolist(),
        'recall_curve': recall_curve.tolist()
    }

def plot_benchmark_results(results, plot_dir):
    """Generate and save benchmark plots."""
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    models = list(results.keys())
    
    for model_name in models:
        model_results = results[model_name]
        plt.plot(
            model_results['fpr'], 
            model_results['tpr'], 
            label=f"{model_name} (AUROC = {model_results['auroc']:.3f})"
        )
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'roc_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    
    for model_name in models:
        model_results = results[model_name]
        plt.plot(
            model_results['recall_curve'], 
            model_results['precision_curve'], 
            label=f"{model_name} (AUPRC = {model_results['auprc']:.3f})"
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'pr_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Plot benchmark metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'auprc']
    metrics_data = {metric: [results[model][metric] for model in models] for metric in metrics}
    
    plt.figure(figsize=(12, 10))
    x = np.arange(len(models))
    width = 0.1
    offsets = np.linspace(-(len(metrics)-1)/2*width, (len(metrics)-1)/2*width, len(metrics))
    
    for i, metric in enumerate(metrics):
        plt.bar(x + offsets[i], metrics_data[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved benchmark plots to {plot_dir}")

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create plots directory
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Load model
    try:
        logger.info(f"Loading model from {args.model}")
        predictor = SplicePredictorModel(
            model_path=args.model,
            threshold=args.threshold,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(args.benchmark_data)
    sequences = benchmark_data['sequence'].tolist()
    labels = benchmark_data['label'].values
    
    # Run predictions with our model
    logger.info("Running predictions with foundation model")
    predictions = predictor.predict_batch(sequences)
    
    # Evaluate our model
    logger.info("Evaluating foundation model performance")
    foundation_results = evaluate_model(predictions, labels, args.threshold)
    
    # Store all results in a dictionary
    all_results = {
        "foundation_model": foundation_results
    }
    
    # Load SpliceAI scores if provided
    if args.spliceai_scores:
        try:
            logger.info(f"Loading SpliceAI scores from {args.spliceai_scores}")
            spliceai_data = pd.read_csv(args.spliceai_scores)
            
            # Check if the score column exists
            if 'score' not in spliceai_data.columns:
                logger.error("SpliceAI file must contain a 'score' column")
                sys.exit(1)
            
            # Evaluate SpliceAI performance
            spliceai_results = evaluate_model(
                spliceai_data['score'].values, 
                labels, 
                args.threshold
            )
            all_results["spliceai"] = spliceai_results
            
        except Exception as e:
            logger.error(f"Error loading SpliceAI scores: {e}")
            sys.exit(1)
    
    # Load other model scores if provided
    if args.other_scores:
        try:
            logger.info(f"Loading other model scores from {args.other_scores}")
            other_data = pd.read_csv(args.other_scores)
            
            # Process each model's scores in the file
            for column in other_data.columns:
                if column == 'sequence' or column == 'label':
                    continue
                
                logger.info(f"Evaluating {column} performance")
                other_results = evaluate_model(
                    other_data[column].values, 
                    labels, 
                    args.threshold
                )
                all_results[column] = other_results
                
        except Exception as e:
            logger.error(f"Error loading other model scores: {e}")
            sys.exit(1)
    
    # Plot benchmark results
    plot_benchmark_results(all_results, args.plot_dir)
    
    # Print summary of results
    if args.verbose:
        logger.info("\nBenchmark Results Summary:\n" + "-" * 30)
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'auprc']
        
        # Calculate table width
        header = "Model".ljust(20)
        for metric in metrics:
            header += metric.capitalize().ljust(12)
        
        print("\n" + header)
        print("-" * len(header))
        
        for model_name, model_results in all_results.items():
            row = model_name.ljust(20)
            for metric in metrics:
                row += f"{model_results[metric]:.4f}".ljust(12)
            print(row)
        print("\n")
    
    # Save results to JSON file
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Saved benchmark results to {args.output}")

if __name__ == "__main__":
    main()
