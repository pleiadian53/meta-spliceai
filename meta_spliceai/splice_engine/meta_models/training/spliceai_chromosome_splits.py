#!/usr/bin/env python3
"""
SpliceAI-Compatible Chromosome Splits

Implements the chromosome holdout strategy used in the SpliceAI paper (2019)
to prevent homology leakage and ensure realistic evaluation.

Reference: Jaganathan et al., 2019
Train/validation/test split based on chromosome holdout.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path


class SpliceAIChromosomeSplitter:
    """
    Implements SpliceAI's chromosome holdout strategy.
    
    Original SpliceAI split:
    - Training: chr2-4, 6-9, 11-22
    - Validation: chr5, chr10  
    - Testing: chr1, chrX
    
    This prevents homology leakage by ensuring test chromosomes
    are completely unseen during training.
    """
    
    def __init__(self, custom_split: Optional[Dict[str, List[str]]] = None):
        """
        Initialize chromosome splitter.
        
        Parameters
        ----------
        custom_split : Dict[str, List[str]], optional
            Custom chromosome split. If None, uses SpliceAI default.
            Format: {'train': [...], 'validation': [...], 'test': [...]}
        """
        
        if custom_split is not None:
            self.splits = custom_split
        else:
            # Original SpliceAI chromosome split
            self.splits = {
                'train': ['2', '3', '4', '6', '7', '8', '9', '11', '12', '13', '14', 
                         '15', '16', '17', '18', '19', '20', '21', '22'],
                'validation': ['5', '10'],
                'test': ['1', 'X']
            }
    
    def get_chromosome_masks(
        self, 
        chrom_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/validation/test masks based on chromosome.
        
        Parameters
        ----------
        chrom_array : np.ndarray
            Array of chromosome identifiers
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (train_mask, validation_mask, test_mask) boolean arrays
        """
        
        # Convert to string for consistent comparison
        chrom_str = np.array([str(c) for c in chrom_array])
        
        # Create masks
        train_mask = np.isin(chrom_str, self.splits['train'])
        validation_mask = np.isin(chrom_str, self.splits['validation'])
        test_mask = np.isin(chrom_str, self.splits['test'])
        
        return train_mask, validation_mask, test_mask
    
    def get_split_indices(
        self,
        chrom_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/validation/test indices.
        
        Parameters
        ----------
        chrom_array : np.ndarray
            Array of chromosome identifiers
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (train_indices, validation_indices, test_indices)
        """
        
        train_mask, validation_mask, test_mask = self.get_chromosome_masks(chrom_array)
        
        train_indices = np.where(train_mask)[0]
        validation_indices = np.where(validation_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        return train_indices, validation_indices, test_indices
    
    def validate_split_coverage(
        self,
        chrom_array: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Dict[str, int]]:
        """
        Validate that the chromosome split covers the available data.
        
        Parameters
        ----------
        chrom_array : np.ndarray
            Array of chromosome identifiers
        verbose : bool
            Whether to print validation results
            
        Returns
        -------
        Dict[str, Dict[str, int]]
            Split coverage statistics
        """
        
        # Get available chromosomes
        available_chroms = set(str(c) for c in np.unique(chrom_array))
        
        # Check coverage for each split
        coverage = {}
        total_positions = len(chrom_array)
        
        for split_name, split_chroms in self.splits.items():
            available_split_chroms = [c for c in split_chroms if c in available_chroms]
            missing_chroms = [c for c in split_chroms if c not in available_chroms]
            
            if split_name == 'train':
                mask = np.isin([str(c) for c in chrom_array], available_split_chroms)
            elif split_name == 'validation':
                mask = np.isin([str(c) for c in chrom_array], available_split_chroms)
            else:  # test
                mask = np.isin([str(c) for c in chrom_array], available_split_chroms)
            
            coverage[split_name] = {
                'available_chromosomes': available_split_chroms,
                'missing_chromosomes': missing_chroms,
                'positions': int(mask.sum()),
                'percentage': float(mask.sum() / total_positions * 100)
            }
        
        if verbose:
            print(f"📊 CHROMOSOME SPLIT COVERAGE:")
            print(f"  Total positions: {total_positions:,}")
            print(f"  Available chromosomes: {sorted(available_chroms)}")
            print()
            
            for split_name, stats in coverage.items():
                print(f"  {split_name.upper()}:")
                print(f"    Chromosomes: {stats['available_chromosomes']}")
                print(f"    Positions: {stats['positions']:,} ({stats['percentage']:.1f}%)")
                if stats['missing_chromosomes']:
                    print(f"    Missing: {stats['missing_chromosomes']}")
                print()
        
        return coverage
    
    def create_balanced_split(
        self,
        chrom_array: np.ndarray,
        target_test_ratio: float = 0.2,
        target_val_ratio: float = 0.1
    ) -> Dict[str, List[str]]:
        """
        Create a balanced chromosome split if the default SpliceAI split is imbalanced.
        
        Parameters
        ----------
        chrom_array : np.ndarray
            Array of chromosome identifiers
        target_test_ratio : float
            Target percentage for test set
        target_val_ratio : float
            Target percentage for validation set
            
        Returns
        -------
        Dict[str, List[str]]
            Balanced chromosome split
        """
        
        # Get chromosome position counts
        chrom_counts = pd.Series([str(c) for c in chrom_array]).value_counts()
        
        # Sort chromosomes by size (largest first)
        sorted_chroms = chrom_counts.index.tolist()
        total_positions = len(chrom_array)
        
        # Allocate chromosomes to splits
        test_chroms = []
        val_chroms = []
        train_chroms = []
        
        test_positions = 0
        val_positions = 0
        
        target_test_positions = total_positions * target_test_ratio
        target_val_positions = total_positions * target_val_ratio
        
        for chrom in sorted_chroms:
            chrom_size = chrom_counts[chrom]
            
            if test_positions < target_test_positions:
                test_chroms.append(chrom)
                test_positions += chrom_size
            elif val_positions < target_val_positions:
                val_chroms.append(chrom)
                val_positions += chrom_size
            else:
                train_chroms.append(chrom)
        
        balanced_split = {
            'train': train_chroms,
            'validation': val_chroms,
            'test': test_chroms
        }
        
        return balanced_split


def create_spliceai_compatible_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    chrom_array: np.ndarray,
    genes: np.ndarray,
    splitter: Optional[SpliceAIChromosomeSplitter] = None,
    use_balanced_split: bool = False,
    verbose: bool = True
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create SpliceAI-compatible CV folds using chromosome holdout.
    
    This replaces the 24-fold LOCO-CV with a more realistic 3-way split
    that matches published SpliceAI methodology.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    chrom_array : np.ndarray
        Chromosome identifiers
    genes : np.ndarray
        Gene identifiers
    splitter : SpliceAIChromosomeSplitter, optional
        Custom chromosome splitter
    use_balanced_split : bool
        Whether to create balanced split instead of SpliceAI default
    verbose : bool
        Whether to print split information
        
    Returns
    -------
    List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]
        List of (fold_name, train_indices, val_indices, test_indices)
    """
    
    if splitter is None:
        splitter = SpliceAIChromosomeSplitter()
    
    # Validate split coverage
    coverage = splitter.validate_split_coverage(chrom_array, verbose=verbose)
    
    # Check if we need balanced split
    test_coverage = coverage['test']['percentage']
    if use_balanced_split or test_coverage < 5 or test_coverage > 40:
        if verbose:
            print(f"⚠️  Default SpliceAI split imbalanced (test: {test_coverage:.1f}%)")
            print(f"   Creating balanced split instead...")
        
        balanced_splits = splitter.create_balanced_split(chrom_array)
        splitter.splits = balanced_splits
        coverage = splitter.validate_split_coverage(chrom_array, verbose=verbose)
    
    # Get indices for the single train/val/test split
    train_indices, val_indices, test_indices = splitter.get_split_indices(chrom_array)
    
    if verbose:
        print(f"📊 FINAL SPLIT SUMMARY:")
        print(f"  Training: {len(train_indices):,} positions ({len(train_indices)/len(y)*100:.1f}%)")
        print(f"  Validation: {len(val_indices):,} positions ({len(val_indices)/len(y)*100:.1f}%)")
        print(f"  Testing: {len(test_indices):,} positions ({len(test_indices)/len(y)*100:.1f}%)")
    
    # Return as single fold (not 24 folds!)
    folds = [("chromosome_holdout", train_indices, val_indices, test_indices)]
    
    return folds


if __name__ == "__main__":
    # Test the chromosome splitter
    print("Testing SpliceAI Chromosome Splitter")
    print("=" * 50)
    
    # Create sample data
    sample_chroms = np.array(['1', '2', '3', '4', '5', '10', '11', 'X'] * 1000)
    
    splitter = SpliceAIChromosomeSplitter()
    coverage = splitter.validate_split_coverage(sample_chroms)
    
    train_idx, val_idx, test_idx = splitter.get_split_indices(sample_chroms)
    
    print(f"Sample test results:")
    print(f"  Train indices: {len(train_idx)}")
    print(f"  Validation indices: {len(val_idx)}")
    print(f"  Test indices: {len(test_idx)}")



