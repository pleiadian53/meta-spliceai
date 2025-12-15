#!/usr/bin/env python3
"""
Command-line script for running predictions with the splice site foundation model.

This script allows users to easily run predictions on genomic sequences
using trained splice site prediction models.

Example usage:
    # Predict from FASTA file
    python predict.py --model model_path.h5 --fasta input.fa --output predictions.csv
    
    # Predict with sliding window
    python predict.py --model model_path.h5 --fasta input.fa --window 10000 --stride 1000 --output predictions.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from Bio import SeqIO

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from meta_spliceai.foundation_model.deployment import SplicePredictorModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('splice_predictor')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Splice site prediction using foundation model')
    
    # Required arguments
    parser.add_argument('--model', required=True, help='Path to trained model file (.h5)')
    
    # Input options (at least one required)
    input_group = parser.add_argument_group('Input Options (required)')
    input_source = input_group.add_mutually_exclusive_group(required=True)
    input_source.add_argument('--fasta', help='Path to input FASTA file')
    input_source.add_argument('--sequence', help='Direct DNA sequence input')
    input_source.add_argument('--csv', help='Path to CSV file with sequences column')
    
    # Output options
    parser.add_argument('--output', required=True, help='Path to output file (.csv)')
    
    # Model parameters
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--window', type=int, default=10000,
                       help='Window size for sliding window prediction (default: 10000)')
    parser.add_argument('--stride', type=int, default=1000,
                       help='Stride for sliding window prediction (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for prediction (default: 32)')
                       
    # Additional options
    parser.add_argument('--detailed', action='store_true',
                       help='Include additional details in output')
    parser.add_argument('--surveyor_format', action='store_true',
                       help='Format output for MetaSpliceAI integration')
    
    return parser.parse_args()

def predict_from_fasta(predictor, fasta_file, window_size, stride, detailed):
    """Run predictions on sequences in a FASTA file."""
    logger.info(f"Processing FASTA file: {fasta_file}")
    
    predictions = predictor.predict_from_fasta(
        fasta_file=fasta_file,
        window_size=window_size,
        stride=stride
    )
    
    # Filter columns based on detail level
    if not detailed:
        keep_cols = ['sequence_id', 'start', 'end', 'probability', 'is_splice_site']
        predictions = predictions[keep_cols]
        
    return predictions

def predict_from_sequence(predictor, sequence, window_size, stride, detailed):
    """Run predictions on a single DNA sequence."""
    logger.info(f"Processing direct sequence input of length {len(sequence)}")
    
    results = []
    
    # Use sliding window if sequence is longer than window size
    if len(sequence) > window_size:
        for i in range(0, len(sequence) - window_size + 1, stride):
            window = sequence[i:i+window_size]
            prob = predictor.predict_single_sequence(window)
            
            results.append({
                'sequence_id': 'input_sequence',
                'start': i,
                'end': i + window_size,
                'probability': prob,
                'is_splice_site': prob >= predictor.threshold
            })
    else:
        # Pad sequence if shorter than window size
        if len(sequence) < window_size:
            sequence = sequence + 'N' * (window_size - len(sequence))
            
        prob = predictor.predict_single_sequence(sequence)
        results.append({
            'sequence_id': 'input_sequence',
            'start': 0,
            'end': len(sequence),
            'probability': prob,
            'is_splice_site': prob >= predictor.threshold
        })
    
    predictions = pd.DataFrame(results)
    
    # Filter columns based on detail level
    if not detailed:
        keep_cols = ['sequence_id', 'start', 'end', 'probability', 'is_splice_site']
        predictions = predictions[keep_cols]
        
    return predictions

def predict_from_csv(predictor, csv_file, detailed):
    """Run predictions on sequences in a CSV file."""
    logger.info(f"Processing CSV file: {csv_file}")
    
    # Load CSV file
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Check if 'sequence' column exists
    if 'sequence' not in data.columns:
        logger.error("CSV file must contain a 'sequence' column")
        sys.exit(1)
    
    # Run predictions
    sequences = data['sequence'].tolist()
    probas = predictor.predict_batch(sequences)
    
    # Add predictions to dataframe
    data['probability'] = probas
    data['is_splice_site'] = (probas >= predictor.threshold).astype(int)
    
    # Filter columns based on detail level
    if not detailed and len(data.columns) > 10:  # Only filter if there are many columns
        essential_cols = ['sequence', 'probability', 'is_splice_site']
        id_cols = [col for col in data.columns if 'id' in col.lower() or 'name' in col.lower()]
        keep_cols = essential_cols + id_cols
        data = data[keep_cols]
        
    return data

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Load model
    try:
        logger.info(f"Loading model from {args.model}")
        predictor = SplicePredictorModel(
            model_path=args.model,
            context_length=args.window,
            threshold=args.threshold,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run predictions
    if args.fasta:
        predictions = predict_from_fasta(
            predictor=predictor,
            fasta_file=args.fasta,
            window_size=args.window,
            stride=args.stride,
            detailed=args.detailed
        )
    elif args.sequence:
        predictions = predict_from_sequence(
            predictor=predictor,
            sequence=args.sequence,
            window_size=args.window,
            stride=args.stride,
            detailed=args.detailed
        )
    elif args.csv:
        predictions = predict_from_csv(
            predictor=predictor,
            csv_file=args.csv,
            detailed=args.detailed
        )
    
    # Export predictions
    if args.surveyor_format:
        logger.info(f"Exporting predictions in MetaSpliceAI format to {args.output}")
        predictor.export_for_surveyor(predictions, args.output)
    else:
        logger.info(f"Saving predictions to {args.output}")
        predictions.to_csv(args.output, index=False)
    
    # Print summary
    n_splice_sites = predictions['is_splice_site'].sum()
    total_sites = len(predictions)
    logger.info(f"Prediction complete: {n_splice_sites} splice sites identified out of {total_sites} positions")

if __name__ == "__main__":
    main()
