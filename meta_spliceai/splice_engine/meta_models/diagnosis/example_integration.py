#!/usr/bin/env python3
"""
Example integration of memory checker into CV training scripts.

This shows how to integrate the memory checker into existing training workflows.
"""

import argparse
from pathlib import Path

from .integration import check_memory_before_training


def example_cv_training_with_memory_check():
    """Example of how to integrate memory checking into CV training."""
    
    # Simulate argument parsing (replace with actual args in real script)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--row-cap', type=int, default=0)
    parser.add_argument('--skip-memory-check', action='store_true')
    
    # For demo purposes, use train_pc_1000_3mers/master
    args = parser.parse_args(['--dataset', 'train_pc_1000_3mers/master'])
    
    print("Starting CV Training with Memory Check Integration")
    print("=" * 60)
    
    # Memory check integration
    if not args.skip_memory_check:
        assessment = check_memory_before_training(
            args.dataset,
            cv_folds=args.n_folds,
            n_estimators=args.n_estimators,
            auto_adjust_row_cap=True,
            verbose=True
        )
        
        # Handle different risk levels
        if assessment.get('risk_level') == 'HIGH':
            if args.row_cap == 0:  # No row cap set
                if 'suggested_row_cap' in assessment:
                    print(f"⚠️  Auto-adjusting row cap to {assessment['suggested_row_cap']} due to memory constraints")
                    args.row_cap = assessment['suggested_row_cap']
                else:
                    print("❌ High OOM risk and no row cap suggestion available.")
                    print("   Please manually set --row-cap or use a smaller dataset.")
                    return False
            else:
                print(f"ℹ️  Using user-specified row cap: {args.row_cap}")
        
        elif assessment.get('risk_level') == 'MODERATE':
            print("⚠️  Moderate OOM risk detected. Monitoring recommended.")
        
        else:  # SAFE
            print("✅ Memory assessment passed. Proceeding with training.")
    
    # Continue with actual training (simulated)
    print(f"\n🚀 Starting training with:")
    print(f"   Dataset: {args.dataset}")
    print(f"   CV Folds: {args.n_folds}")
    print(f"   Estimators: {args.n_estimators}")
    print(f"   Row Cap: {args.row_cap if args.row_cap > 0 else 'None (full dataset)'}")
    
    # Simulate training
    print("   Training in progress...")
    print("   ✅ Training completed successfully!")
    
    return True


def add_memory_check_to_existing_script():
    """Example of minimal integration into existing script."""
    
    # This is what you would add to the beginning of main() in run_gene_cv_sigmoid.py
    def enhanced_main(args):
        # Add this block after argument parsing
        if not getattr(args, 'skip_memory_check', False):
            from meta_spliceai.splice_engine.meta_models.diagnosis.integration import check_memory_before_training
            
            assessment = check_memory_before_training(
                args.dataset,
                cv_folds=getattr(args, 'n_folds', 5),
                n_estimators=getattr(args, 'n_estimators', 500),
                verbose=True
            )
            
            # Auto-adjust row cap if high risk and no cap set
            if (assessment.get('risk_level') == 'HIGH' and 
                getattr(args, 'row_cap', 0) == 0 and
                'suggested_row_cap' in assessment):
                
                print(f"\n⚠️  Auto-adjusting row cap to {assessment['suggested_row_cap']} due to memory constraints")
                args.row_cap = assessment['suggested_row_cap']
                
                # Update environment variable
                import os
                os.environ["SS_MAX_ROWS"] = str(args.row_cap)
        
        # Continue with existing training logic...
        print("Proceeding with existing training logic...")
    
    return enhanced_main


if __name__ == "__main__":
    example_cv_training_with_memory_check()
