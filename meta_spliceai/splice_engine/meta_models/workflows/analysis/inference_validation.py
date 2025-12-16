#!/usr/bin/env python3
"""
Inference Mode Validation and Consistency Checking

This module provides tools to validate that different inference modes (base_only, meta_only, hybrid)
produce consistent position counts and behavior, helping identify actual issues vs normal variation.
"""

import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass

# Add meta_spliceai to path
sys.path.insert(0, str(Path(__file__).parents[5]))


@dataclass
class InferenceModeResult:
    """Results from a single inference mode test."""
    mode: str
    gene_id: str
    success: bool
    total_positions: int
    processing_time: float
    error_message: Optional[str] = None


class InferenceModeValidator:
    """Validator for consistency across different inference modes."""
    
    def __init__(self, model_path: str, training_dataset: str, verbose: bool = True):
        """
        Initialize the inference mode validator.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model
        training_dataset : str
            Training dataset name
        verbose : bool
            Enable verbose output
        """
        self.model_path = model_path
        self.training_dataset = training_dataset
        self.verbose = verbose
    
    def test_single_gene_across_modes(self, gene_id: str, 
                                    modes: List[str] = None) -> Dict[str, InferenceModeResult]:
        """
        Test a single gene across multiple inference modes.
        
        Parameters
        ----------
        gene_id : str
            Gene ID to test
        modes : List[str], optional
            List of inference modes to test. Default: ['base_only', 'meta_only', 'hybrid']
            
        Returns
        -------
        Dict[str, InferenceModeResult]
            Results for each inference mode
        """
        if modes is None:
            modes = ['base_only', 'meta_only', 'hybrid']
        
        results = {}
        
        if self.verbose:
            print(f"🧪 Testing gene {gene_id} across {len(modes)} inference modes")
            print("=" * 60)
        
        for mode in modes:
            if self.verbose:
                print(f"🔬 Testing {mode} mode...")
            
            result = self._run_single_inference_test(gene_id, mode)
            results[mode] = result
            
            if self.verbose:
                if result.success:
                    print(f"   ✅ {mode}: {result.total_positions:,} positions in {result.processing_time:.1f}s")
                else:
                    print(f"   ❌ {mode}: Failed - {result.error_message}")
        
        return results
    
    def _run_single_inference_test(self, gene_id: str, mode: str) -> InferenceModeResult:
        """Run a single inference test and extract results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / f"test_{mode}_{gene_id}"
            
            # Build command
            cmd = [
                "python", "-m", 
                "meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow",
                "--model", self.model_path,
                "--training-dataset", self.training_dataset,
                "--genes", gene_id,
                "--output-dir", str(output_dir),
                "--inference-mode", mode,
                "--verbose"
            ]
            
            try:
                # Run inference
                import time
                start_time = time.time()
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout
                )
                processing_time = time.time() - start_time
                
                if result.returncode != 0:
                    return InferenceModeResult(
                        mode=mode,
                        gene_id=gene_id,
                        success=False,
                        total_positions=0,
                        processing_time=processing_time,
                        error_message=f"Command failed: {result.stderr[:200]}"
                    )
                
                # Extract position count from output
                total_positions = self._extract_position_count(result.stdout)
                
                return InferenceModeResult(
                    mode=mode,
                    gene_id=gene_id,
                    success=True,
                    total_positions=total_positions,
                    processing_time=processing_time
                )
                
            except subprocess.TimeoutExpired:
                return InferenceModeResult(
                    mode=mode,
                    gene_id=gene_id,
                    success=False,
                    total_positions=0,
                    processing_time=300,
                    error_message="Timeout after 5 minutes"
                )
            except Exception as e:
                return InferenceModeResult(
                    mode=mode,
                    gene_id=gene_id,
                    success=False,
                    total_positions=0,
                    processing_time=0,
                    error_message=str(e)
                )
    
    def _extract_position_count(self, stdout: str) -> int:
        """Extract the final position count from stdout."""
        # Look for the final position count message
        lines = stdout.split('\n')
        for line in reversed(lines):
            if '📊 Total positions:' in line:
                # Extract number from line like "   📊 Total positions: 5,716"
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        count_str = parts[1].strip().replace(',', '')
                        return int(count_str)
                    except ValueError:
                        continue
        return 0
    
    def validate_inference_consistency(self, gene_ids: List[str], 
                                     modes: List[str] = None) -> Dict[str, Dict]:
        """
        Validate that inference modes produce consistent results across multiple genes.
        
        Parameters
        ----------
        gene_ids : List[str]
            List of gene IDs to test
        modes : List[str], optional
            List of inference modes to test
            
        Returns
        -------
        Dict[str, Dict]
            Validation results for each gene
        """
        if modes is None:
            modes = ['base_only', 'meta_only', 'hybrid']
        
        if self.verbose:
            print(f"🔬 INFERENCE MODE CONSISTENCY VALIDATION")
            print("=" * 70)
            print(f"📊 Testing {len(gene_ids)} genes across {len(modes)} modes")
            print()
        
        all_results = {}
        consistency_issues = []
        
        for gene_id in gene_ids:
            gene_results = self.test_single_gene_across_modes(gene_id, modes)
            all_results[gene_id] = gene_results
            
            # Check consistency
            successful_results = {mode: result for mode, result in gene_results.items() if result.success}
            
            if len(successful_results) > 1:
                position_counts = [result.total_positions for result in successful_results.values()]
                if len(set(position_counts)) > 1:
                    consistency_issues.append({
                        'gene_id': gene_id,
                        'position_counts': {mode: result.total_positions for mode, result in successful_results.items()},
                        'issue': 'Position count inconsistency across modes'
                    })
        
        # Generate summary
        if self.verbose:
            print(f"\n📋 CONSISTENCY VALIDATION SUMMARY")
            print("=" * 50)
            
            if consistency_issues:
                print(f"❌ Found {len(consistency_issues)} consistency issues:")
                for issue in consistency_issues:
                    print(f"   • {issue['gene_id']}: {issue['position_counts']}")
            else:
                print(f"✅ All {len(gene_ids)} genes show consistent position counts across modes")
        
        return {
            'gene_results': all_results,
            'consistency_issues': consistency_issues,
            'validation_passed': len(consistency_issues) == 0
        }


def validate_inference_consistency(gene_ids: List[str], 
                                 model_path: str,
                                 training_dataset: str,
                                 modes: List[str] = None) -> Dict:
    """
    Convenience function to validate inference consistency.
    
    Parameters
    ----------
    gene_ids : List[str]
        List of gene IDs to test
    model_path : str
        Path to trained model
    training_dataset : str
        Training dataset name
    modes : List[str], optional
        Inference modes to test
        
    Returns
    -------
    Dict
        Validation results
    """
    validator = InferenceModeValidator(model_path, training_dataset, verbose=True)
    return validator.validate_inference_consistency(gene_ids, modes)


def compare_inference_modes(gene_id: str, 
                          model_path: str,
                          training_dataset: str) -> Dict[str, InferenceModeResult]:
    """
    Compare all inference modes for a single gene.
    
    Parameters
    ----------
    gene_id : str
        Gene ID to test
    model_path : str
        Path to trained model
    training_dataset : str
        Training dataset name
        
    Returns
    -------
    Dict[str, InferenceModeResult]
        Results for each inference mode
    """
    validator = InferenceModeValidator(model_path, training_dataset, verbose=True)
    return validator.test_single_gene_across_modes(gene_id)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate inference mode consistency')
    parser.add_argument('--genes', nargs='+', required=True, help='Gene IDs to test')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--training-dataset', required=True, help='Training dataset name')
    parser.add_argument('--modes', nargs='+', default=['base_only', 'meta_only', 'hybrid'],
                       help='Inference modes to test')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Run validation
    results = validate_inference_consistency(
        gene_ids=args.genes,
        model_path=args.model,
        training_dataset=args.training_dataset,
        modes=args.modes
    )
    
    # Exit with appropriate code
    exit_code = 0 if results['validation_passed'] else 1
    sys.exit(exit_code)

