#!/usr/bin/env python3
"""
Fine-tuning script for adapting splice site prediction models to custom datasets.

This script demonstrates how to fine-tune a pre-trained foundation model on
a custom dataset to improve performance for specific organisms or splice site types.

Example usage:
    python fine_tune.py --model pretrained_model.h5 --train_data custom_data.csv --output fine_tuned_model.h5
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from meta_spliceai.foundation_model.data_processor import GenomicDataProcessor
from meta_spliceai.foundation_model.trainer import SpliceSiteTrainer
from meta_spliceai.foundation_model.deployment import SplicePredictorModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fine_tune')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune splice site prediction models')
    
    # Required arguments
    parser.add_argument('--model', required=True, help='Path to pre-trained model file (.h5)')
    parser.add_argument('--output', required=True, help='Path to output fine-tuned model (.h5)')
    
    # Input data options
    input_group = parser.add_argument_group('Input Data (one required)')
    input_source = input_group.add_mutually_exclusive_group(required=True)
    input_source.add_argument('--train_data', help='Path to CSV with training sequences and labels')
    input_source.add_argument('--gtf', help='Path to GTF file for extracting splice sites')
    input_source.add_argument('--fasta', help='Path to FASTA file (only used with --gtf)')
    
    # Fine-tuning parameters
    ft_group = parser.add_argument_group('Fine-tuning Parameters')
    ft_group.add_argument('--learning_rate', type=float, default=1e-4, 
                         help='Learning rate for fine-tuning (default: 1e-4)')
    ft_group.add_argument('--epochs', type=int, default=10,
                         help='Number of training epochs (default: 10)')
    ft_group.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training (default: 32)')
    ft_group.add_argument('--freeze_layers', type=int, default=0,
                         help='Number of layers to freeze from the base model (default: 0)')
    ft_group.add_argument('--val_split', type=float, default=0.2,
                         help='Validation split ratio (default: 0.2)')
    
    # Advanced options
    adv_group = parser.add_argument_group('Advanced Options')
    adv_group.add_argument('--use_focal_loss', action='store_true',
                          help='Use focal loss instead of binary cross-entropy')
    adv_group.add_argument('--focal_gamma', type=float, default=2.0,
                          help='Gamma parameter for focal loss (default: 2.0)')
    adv_group.add_argument('--weight_decay', type=float, default=1e-4,
                          help='Weight decay for regularization (default: 1e-4)')
    adv_group.add_argument('--checkpoint_dir', default='fine_tune_checkpoints',
                          help='Directory to save checkpoints (default: fine_tune_checkpoints)')
    adv_group.add_argument('--log_dir', default='fine_tune_logs',
                          help='Directory to save TensorBoard logs (default: fine_tune_logs)')
    
    return parser.parse_args()

def load_data_from_csv(csv_file, val_split=0.2):
    """Load training data from a CSV file."""
    logger.info(f"Loading data from {csv_file}")
    
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Error reading data file: {e}")
        sys.exit(1)
    
    # Check required columns
    required_columns = ['sequence', 'label']
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Data file must contain a '{col}' column")
            sys.exit(1)
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(
        data, test_size=val_split, random_state=42, stratify=data['label']
    )
    
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    return train_data, val_data

def prepare_gtf_fasta_data(gtf_file, fasta_file, val_split=0.2):
    """Prepare training data from GTF and FASTA files."""
    logger.info(f"Processing GTF file: {gtf_file}")
    logger.info(f"Processing FASTA file: {fasta_file}")
    
    # Initialize data processor
    data_processor = GenomicDataProcessor(context_length=10000, stride=1000)
    
    # Extract splice sites from GTF
    splice_sites = data_processor.extract_splice_sites(gtf_file)
    logger.info(f"Extracted {len(splice_sites)} splice sites")
    
    # Extract genomic windows around splice sites
    windows = data_processor.extract_genomic_windows(fasta_file, splice_sites, flank_size=5000)
    logger.info(f"Extracted {len(windows)} sequence windows")
    
    # Load genome sequences for generating negative examples
    from Bio import SeqIO
    
    # Load genome into dictionary
    genome_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        genome_dict[record.id] = str(record.seq)
    
    # Generate negative examples (non-splice sites)
    n_negatives = len(windows)  # Generate same number of negatives as positives
    negative_windows = data_processor.generate_negative_examples(genome_dict, windows, n_negatives)
    logger.info(f"Generated {len(negative_windows)} negative examples")
    
    # Prepare datasets for training and validation
    all_data = pd.concat([windows, negative_windows])
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(
        all_data, test_size=val_split, random_state=42, stratify=all_data['label']
    )
    
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    return train_data, val_data

def load_pretrained_model(model_path, freeze_layers=0):
    """Load pre-trained model and optionally freeze some layers."""
    logger.info(f"Loading pre-trained model from {model_path}")
    
    try:
        # Load model with custom objects if needed
        model = keras.models.load_model(model_path)
        
        # Freeze specified number of layers
        if freeze_layers > 0:
            trainable_count = len(model.trainable_weights)
            layers_to_freeze = min(freeze_layers, len(model.layers))
            
            logger.info(f"Freezing {layers_to_freeze} layers out of {len(model.layers)} total layers")
            
            for i in range(layers_to_freeze):
                model.layers[i].trainable = False
            
            new_trainable_count = len(model.trainable_weights)
            logger.info(f"Trainable weights reduced from {trainable_count} to {new_trainable_count}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading pre-trained model: {e}")
        sys.exit(1)

def prepare_datasets(train_data, val_data, batch_size):
    """Prepare TensorFlow datasets for training."""
    # Initialize data processor
    data_processor = GenomicDataProcessor()
    
    # Convert to TensorFlow datasets
    train_dataset = data_processor.prepare_windows_dataset(train_data, batch_size=batch_size)
    val_dataset = data_processor.prepare_windows_dataset(val_data, batch_size=batch_size)
    
    return train_dataset, val_dataset

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load pre-trained model
    model = load_pretrained_model(args.model, args.freeze_layers)
    
    # Load data
    if args.train_data:
        train_data, val_data = load_data_from_csv(args.train_data, args.val_split)
    elif args.gtf and args.fasta:
        train_data, val_data = prepare_gtf_fasta_data(args.gtf, args.fasta, args.val_split)
    else:
        logger.error("Either --train_data or both --gtf and --fasta must be provided")
        sys.exit(1)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(train_data, val_data, args.batch_size)
    
    # Setup trainer
    trainer = SpliceSiteTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        focal_loss_gamma=args.focal_gamma,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.log_dir
    )
    
    # Compile model
    trainer.compile_model(use_focal_loss=args.use_focal_loss)
    
    # Calculate class weights for imbalanced dataset (if needed)
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = None
    if len(train_data['label'].unique()) > 1:
        classes = train_data['label'].unique()
        class_counts = train_data['label'].value_counts().sort_index()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Compute weights if imbalanced
        if class_counts.min() / class_counts.max() < 0.8:
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=train_data['label']
            )
            class_weights = dict(zip(classes, weights))
            logger.info(f"Class weights: {class_weights}")
    
    # Train model
    logger.info("Starting fine-tuning")
    history = trainer.train(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.epochs // 3,  # Early stopping patience
        use_lr_scheduler=True,
        class_weights=class_weights
    )
    
    # Save fine-tuned model
    model_path = trainer.save_model(os.path.dirname(args.output), os.path.basename(args.output))
    logger.info(f"Saved fine-tuned model to {model_path}")
    
    # Create predictor to verify the model
    predictor = SplicePredictorModel(
        model=model,
        threshold=0.5,
        batch_size=args.batch_size
    )
    
    # Plot training metrics
    metrics_plot_path = os.path.join(args.log_dir, 'training_metrics.png')
    trainer.plot_metrics(save_dir=args.log_dir)
    logger.info(f"Saved training metrics plot to {metrics_plot_path}")
    
    # Evaluate model on validation set
    logger.info("Evaluating model on validation set")
    results = trainer.evaluate(val_dataset, threshold=0.5)
    
    logger.info("\nFine-tuning Results Summary:")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Sensitivity/Recall: {results['sensitivity']:.4f}")
    logger.info(f"Specificity: {results['specificity']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"F1 Score: {results['f1_score']:.4f}")
    logger.info(f"AUROC: {results['auroc']:.4f}")
    logger.info(f"AUPRC: {results['auprc']:.4f}")
    
    logger.info(f"\nFine-tuning complete! The model is ready for deployment.")

if __name__ == "__main__":
    main()
