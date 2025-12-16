#!/usr/bin/env python3
"""
Examples for using the deep learning gene-aware CV system.

This script demonstrates how to use the new run_gene_cv_deep_learning.py
module with various deep learning models including TabNet, TensorFlow MLP,
and multi-modal transformer models.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully!")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False

def main():
    """Run example commands for different deep learning models."""
    
    # Base configuration
    dataset = "data/ensembl/spliceai_analysis"  # Adjust path as needed
    base_output = "results/deep_learning_cv"
    
    examples = [
        {
            "name": "TabNet Multi-Class",
            "cmd": [
                "python", "-m", "meta_spliceai.splice_engine.meta_models.training.run_gene_cv_deep_learning",
                "--dataset", dataset,
                "--out-dir", f"{base_output}/tabnet",
                "--algorithm", "tabnet",
                "--n-folds", "5",
                "--max-variants", "10000",
                "--verbose"
            ],
            "description": "TabNet with attention-based feature selection"
        },
        
        {
            "name": "TensorFlow MLP Multi-Class",
            "cmd": [
                "python", "-m", "meta_spliceai.splice_engine.meta_models.training.run_gene_cv_deep_learning",
                "--dataset", dataset,
                "--out-dir", f"{base_output}/tf_mlp_multiclass",
                "--algorithm", "tf_mlp_multiclass",
                "--n-folds", "5",
                "--max-variants", "10000",
                "--algorithm-params", '{"hidden_units": [512, 256, 128], "dropout_rate": 0.4, "epochs": 100}',
                "--verbose"
            ],
            "description": "Multi-class TensorFlow neural network"
        },
        
        {
            "name": "Multi-Modal Transformer",
            "cmd": [
                "python", "-m", "meta_spliceai.splice_engine.meta_models.training.run_gene_cv_deep_learning",
                "--dataset", dataset,
                "--out-dir", f"{base_output}/multimodal_transformer",
                "--algorithm", "multimodal_transformer",
                "--n-folds", "5",
                "--max-variants", "5000",  # Smaller for transformer
                "--algorithm-params", '{"sequence_length": 1000, "embedding_dim": 64, "num_heads": 4, "num_layers": 2}',
                "--verbose"
            ],
            "description": "Multi-modal transformer (sequence + tabular features)"
        },
        
        {
            "name": "TabNet with Custom Parameters",
            "cmd": [
                "python", "-m", "meta_spliceai.splice_engine.meta_models.training.run_gene_cv_deep_learning",
                "--dataset", dataset,
                "--out-dir", f"{base_output}/tabnet_custom",
                "--algorithm", "tabnet",
                "--n-folds", "5",
                "--max-variants", "10000",
                "--algorithm-params", '{"n_d": 128, "n_a": 128, "n_steps": 8, "gamma": 2.0, "lambda_sparse": 1e-3}',
                "--verbose"
            ],
            "description": "TabNet with custom architecture parameters"
        }
    ]
    
    print("🧬 Deep Learning Gene-Aware CV Examples")
    print("=" * 50)
    print("This script demonstrates various deep learning models")
    print("for splice site prediction using gene-aware cross-validation.")
    print()
    
    # Check if dataset exists
    if not Path(dataset).exists():
        print(f"❌ Dataset not found: {dataset}")
        print("Please adjust the dataset path in this script.")
        return
    
    # Run examples
    successful = 0
    total = len(examples)
    
    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{total}] {example['name']}")
        print(f"Description: {example['description']}")
        
        if run_command(example['cmd'], example['name']):
            successful += 1
        else:
            print(f"⚠️  {example['name']} failed - continuing with next example...")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {total - successful}/{total}")
    
    if successful > 0:
        print(f"\n✅ Results saved to: {base_output}/")
        print("Check the individual output directories for detailed results.")
    
    if successful < total:
        print(f"\n⚠️  Some examples failed. Check the error messages above.")
        print("Common issues:")
        print("- Missing dependencies (tensorflow, pytorch-tabnet)")
        print("- Dataset path incorrect")
        print("- Insufficient memory for large models")

if __name__ == "__main__":
    main()

