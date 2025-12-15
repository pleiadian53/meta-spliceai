#!/usr/bin/env python3
"""
Resume failed incremental builder runs.

This utility helps identify and resume interrupted incremental builder runs
by analyzing existing batch files and determining the last successful batch.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

def analyze_build_progress(output_dir: Path) -> Dict:
    """Analyze the progress of an incremental build."""
    batch_dir = output_dir
    master_dir = output_dir / "master"
    
    # Find all batch files
    batch_files = list(batch_dir.glob("batch_*_trim.parquet"))
    master_files = list(master_dir.glob("*.parquet")) if master_dir.exists() else []
    
    # Extract batch numbers
    batch_numbers = []
    for batch_file in batch_files:
        try:
            # Extract number from filename like "batch_00001_trim.parquet"
            num_str = batch_file.stem.split('_')[1]
            batch_numbers.append(int(num_str))
        except (IndexError, ValueError):
            continue
    
    batch_numbers.sort()
    
    return {
        'output_dir': str(output_dir),
        'total_batches_found': len(batch_numbers),
        'batch_numbers': batch_numbers,
        'last_batch': max(batch_numbers) if batch_numbers else 0,
        'master_files': len(master_files),
        'is_complete': len(master_files) > 0 and len(master_files) >= len(batch_numbers)
    }

def estimate_remaining_genes(progress: Dict, total_genes: int, batch_size: int) -> int:
    """Estimate how many genes remain to be processed."""
    completed_batches = progress['total_batches_found']
    estimated_completed_genes = completed_batches * batch_size
    remaining_genes = max(0, total_genes - estimated_completed_genes)
    return remaining_genes

def generate_resume_command(progress: Dict, original_args: List[str]) -> str:
    """Generate the command to resume the build."""
    # Remove --overwrite from original args to enable resumption
    args = [arg for arg in original_args if arg != '--overwrite']
    
    # Add output directory if not present
    output_dir = progress['output_dir']
    if '--output-dir' not in args:
        args.extend(['--output-dir', output_dir])
    
    # Ensure no --overwrite flag for resumption
    if '--overwrite' in args:
        args.remove('--overwrite')
    
    return ' '.join(args)

def main():
    parser = argparse.ArgumentParser(
        description="Resume failed incremental builder runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a failed build
  python resume_failed_build.py --analyze train_pc_1000_3mers
  
  # Resume with original command
  python resume_failed_build.py --resume train_pc_1000_3mers \\
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \\
    --n-genes 1000 --batch-size 250 --kmer-sizes 3 --output-dir train_pc_1000_3mers
        """
    )
    
    parser.add_argument(
        'output_dir',
        help='Output directory of the failed build'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Only analyze progress, don\'t resume'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true', 
        help='Generate resume command'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute the resume command immediately'
    )
    
    parser.add_argument(
        '--total-genes',
        type=int,
        help='Total number of genes in the original build'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size used in the original build'
    )
    
    args, remaining = parser.parse_known_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)
    
    # Analyze progress
    progress = analyze_build_progress(output_dir)
    
    print("=== Build Progress Analysis ===")
    print(f"Output directory: {progress['output_dir']}")
    print(f"Batches completed: {progress['total_batches_found']}")
    print(f"Last batch number: {progress['last_batch']}")
    print(f"Master files: {progress['master_files']}")
    print(f"Build complete: {progress['is_complete']}")
    
    if args.total_genes and args.batch_size:
        remaining_genes = estimate_remaining_genes(progress, args.total_genes, args.batch_size)
        print(f"Estimated remaining genes: {remaining_genes}")
    
    if progress['is_complete']:
        print("\n✓ Build appears to be complete!")
        return
    
    if args.analyze:
        return
    
    if args.resume or args.execute:
        if not remaining:
            print("\nError: Please provide the original command to resume")
            print("Example: python resume_failed_build.py --resume train_pc_1000_3mers \\")
            print("  python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \\")
            print("  --n-genes 1000 --batch-size 250 --kmer-sizes 3 --output-dir train_pc_1000_3mers")
            sys.exit(1)
        
        resume_cmd = generate_resume_command(progress, remaining)
        
        print("\n=== Resume Command ===")
        print(resume_cmd)
        
        if args.execute:
            print("\nExecuting resume command...")
            try:
                subprocess.run(resume_cmd, shell=True, check=True)
                print("✓ Resume completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"✗ Resume failed with exit code {e.returncode}")
                sys.exit(1)

if __name__ == "__main__":
    main() 