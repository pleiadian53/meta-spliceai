#!/usr/bin/env python3
"""
Proper ML Workflow Design for Meta-Model Training

This module implements the correct machine learning workflow following best practices:
1. Cross-validation on all available training data
2. Final model training on all training data
3. Evaluation using proper strategies (external test set, repeated holdouts, etc.)

Author: System Design
Created: 2024
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum
import argparse


class WorkflowMode(Enum):
    """Different workflow modes for meta-model training."""
    
    # Development/Testing mode with small samples
    TESTING = "testing"
    
    # Production mode with proper train/test split
    PRODUCTION = "production"
    
    # Research mode with external test dataset
    RESEARCH = "research"
    
    # Benchmarking mode with repeated holdouts
    BENCHMARKING = "benchmarking"


@dataclass
class DatasetSplit:
    """Represents a train/test split of the dataset."""
    train_genes: List[str]
    test_genes: List[str]
    train_positions: int
    test_positions: int
    split_ratio: float
    random_seed: int


class ProperMLWorkflow:
    """
    Implements proper ML workflow with different evaluation strategies.
    
    Key Principles:
    1. CV always uses ALL available training data (not subsets)
    2. Final model trains on ALL training data
    3. Evaluation uses separate test data (various strategies)
    """
    
    def __init__(self, workflow_mode: WorkflowMode = WorkflowMode.PRODUCTION):
        self.workflow_mode = workflow_mode
    
    def get_workflow_parameters(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Get appropriate workflow parameters based on mode.
        
        Parameters
        ----------
        args : argparse.Namespace
            Command-line arguments
            
        Returns
        -------
        Dict[str, Any]
            Workflow configuration including:
            - cv_strategy: How to perform cross-validation
            - final_training_strategy: How to train final model
            - evaluation_strategy: How to evaluate performance
        """
        
        if self.workflow_mode == WorkflowMode.TESTING:
            # Testing mode: Use small subsets for quick iteration
            return {
                "cv_strategy": {
                    "type": "subset",
                    "sample_genes": args.sample_genes or 100,
                    "description": "Quick CV on gene subset for testing"
                },
                "final_training_strategy": {
                    "type": "subset",
                    "use_cv_genes": True,
                    "description": "Train on same subset used in CV"
                },
                "evaluation_strategy": {
                    "type": "cv_only",
                    "description": "Use CV results only (no separate test)"
                }
            }
        
        elif self.workflow_mode == WorkflowMode.PRODUCTION:
            # Production mode: Proper train/test split
            return {
                "cv_strategy": {
                    "type": "full_training",
                    "train_test_split": 0.8,
                    "description": "CV on 80% of all genes"
                },
                "final_training_strategy": {
                    "type": "full_training",
                    "use_train_split": True,
                    "description": "Train on same 80% used in CV"
                },
                "evaluation_strategy": {
                    "type": "holdout_test",
                    "test_size": 0.2,
                    "description": "Evaluate on 20% holdout test set"
                }
            }
        
        elif self.workflow_mode == WorkflowMode.RESEARCH:
            # Research mode: External test dataset
            return {
                "cv_strategy": {
                    "type": "full_dataset",
                    "description": "CV on entire training dataset"
                },
                "final_training_strategy": {
                    "type": "full_dataset",
                    "description": "Train on entire training dataset"
                },
                "evaluation_strategy": {
                    "type": "external_test",
                    "test_dataset": args.test_dataset,
                    "description": "Evaluate on separate test dataset"
                }
            }
        
        elif self.workflow_mode == WorkflowMode.BENCHMARKING:
            # Benchmarking mode: Repeated train/test splits
            return {
                "cv_strategy": {
                    "type": "full_training",
                    "train_test_split": 0.8,
                    "description": "CV on training portion"
                },
                "final_training_strategy": {
                    "type": "ensemble",
                    "n_splits": 5,
                    "description": "Train multiple models on different splits"
                },
                "evaluation_strategy": {
                    "type": "repeated_holdout",
                    "n_repeats": 5,
                    "test_size": 0.2,
                    "description": "Average performance across 5 random splits"
                }
            }
    
    def create_dataset_split(
        self,
        all_genes: List[str],
        split_ratio: float = 0.8,
        random_seed: int = 42
    ) -> DatasetSplit:
        """
        Create a train/test split at the gene level.
        
        Parameters
        ----------
        all_genes : List[str]
            All unique genes in the dataset
        split_ratio : float, default=0.8
            Fraction of genes to use for training
        random_seed : int, default=42
            Random seed for reproducibility
            
        Returns
        -------
        DatasetSplit
            Train and test gene sets
        """
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Shuffle genes
        genes_shuffled = np.array(all_genes)
        np.random.shuffle(genes_shuffled)
        
        # Split
        n_train = int(len(genes_shuffled) * split_ratio)
        train_genes = genes_shuffled[:n_train].tolist()
        test_genes = genes_shuffled[n_train:].tolist()
        
        return DatasetSplit(
            train_genes=train_genes,
            test_genes=test_genes,
            train_positions=0,  # To be filled by actual data
            test_positions=0,   # To be filled by actual data
            split_ratio=split_ratio,
            random_seed=random_seed
        )
    
    def get_post_training_evaluation_config(self) -> Dict[str, Any]:
        """
        Get configuration for post-training analyses.
        
        Returns
        -------
        Dict[str, Any]
            Configuration for different analysis types
        """
        
        return {
            "feature_importance": {
                "sample_strategy": "stratified_genes",
                "max_genes": 2000,
                "description": "Use subset for SHAP analysis"
            },
            "overfitting_analysis": {
                "sample_strategy": "random_splits",
                "n_splits": 3,
                "split_size": 0.2,
                "description": "Multiple small train/val splits"
            },
            "base_vs_meta": {
                "sample_strategy": "repeated_holdout",
                "n_repeats": 5,
                "test_size": 0.2,
                "description": "Average across multiple random splits"
            },
            "threshold_optimization": {
                "sample_strategy": "validation_set",
                "val_size": 0.1,
                "description": "Use small validation set"
            }
        }


def determine_workflow_mode(args: argparse.Namespace) -> WorkflowMode:
    """
    Determine the appropriate workflow mode based on arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    WorkflowMode
        The selected workflow mode
    """
    
    # Testing mode if sample_genes is small
    if hasattr(args, 'sample_genes') and args.sample_genes is not None:
        if args.sample_genes < 100:
            return WorkflowMode.TESTING
    
    # Research mode if external test dataset provided
    if hasattr(args, 'test_dataset') and args.test_dataset is not None:
        return WorkflowMode.RESEARCH
    
    # Benchmarking mode if requested
    if hasattr(args, 'benchmark_mode') and args.benchmark_mode:
        return WorkflowMode.BENCHMARKING
    
    # Default to production mode
    return WorkflowMode.PRODUCTION


def add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add workflow-related arguments to argument parser.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to extend
    """
    
    workflow_group = parser.add_argument_group('Workflow Configuration')
    
    workflow_group.add_argument(
        '--workflow-mode',
        type=str,
        choices=['testing', 'production', 'research', 'benchmarking'],
        default=None,
        help='ML workflow mode (auto-detected if not specified)'
    )
    
    workflow_group.add_argument(
        '--test-dataset',
        type=str,
        default=None,
        help='External test dataset for research mode'
    )
    
    workflow_group.add_argument(
        '--benchmark-mode',
        action='store_true',
        help='Enable benchmarking with repeated train/test splits'
    )
    
    workflow_group.add_argument(
        '--train-test-split',
        type=float,
        default=0.8,
        help='Train/test split ratio for production mode (default: 0.8)'
    )
    
    workflow_group.add_argument(
        '--n-benchmark-splits',
        type=int,
        default=5,
        help='Number of splits for benchmarking mode (default: 5)'
    )


if __name__ == "__main__":
    # Example usage
    workflow = ProperMLWorkflow(WorkflowMode.PRODUCTION)
    
    print("=== Proper ML Workflow Examples ===\n")
    
    for mode in WorkflowMode:
        workflow.workflow_mode = mode
        config = workflow.get_workflow_parameters(
            argparse.Namespace(sample_genes=10, test_dataset=None)
        )
        
        print(f"{mode.value.upper()} MODE:")
        print(f"  CV Strategy: {config['cv_strategy']['description']}")
        print(f"  Final Training: {config['final_training_strategy']['description']}")
        print(f"  Evaluation: {config['evaluation_strategy']['description']}")
        print()



