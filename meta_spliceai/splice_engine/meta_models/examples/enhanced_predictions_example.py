"""
Example: Enhanced SpliceAI prediction workflow with all probability scores.

This script demonstrates how to use the enhanced splice prediction workflow
to generate predictions with all three probability scores (donor, acceptor, neither)
for each position, which is essential for meta model development.
"""

import os
import pandas as pd
import polars as pl

# Import the enhanced splice prediction workflow
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_predictions import run_enhanced_splice_prediction_workflow
from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer


def main():
    """Run an example of the enhanced SpliceAI prediction workflow."""
    
    # Get the basic paths from ErrorAnalyzer
    gtf_file = ErrorAnalyzer.gtf_file
    eval_dir = ErrorAnalyzer.eval_dir
    
    # Print information about the example
    print("=" * 80)
    print(" Enhanced SpliceAI Prediction Workflow Example ")
    print("=" * 80)
    print(f"GTF file: {gtf_file}")
    print(f"Evaluation directory: {eval_dir}")
    
    # Create a configuration for enhanced SpliceAI predictions
    config = SpliceAIConfig(
        gtf_file=gtf_file,
        eval_dir=eval_dir,
        output_subdir="meta_predictions",  # Custom subdirectory for meta model prediction outputs
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        test_mode=True,  # Use test mode for example (processes fewer chromosomes)
        chromosomes=["21"]  # Only process chromosome 21 for this example
    )
    
    # Run the enhanced SpliceAI prediction workflow
    print("\nRunning enhanced SpliceAI prediction workflow...")
    results = run_enhanced_splice_prediction_workflow(config)
    
    # Display information about the results
    print("\nEnhanced SpliceAI prediction workflow completed!")
    
    # Access the positions DataFrame with all three probabilities
    positions_df = results.get('positions')
    if positions_df is not None and positions_df.height > 0:
        print(f"\nEnhanced positions DataFrame shape: {positions_df.shape}")
        print("\nSample of positions with all three probability scores:")
        
        # Display a sample with the three probability scores
        sample_cols = ['gene_id', 'transcript_id', 'position', 'pred_type', 
                        'splice_type', 'donor_score', 'acceptor_score', 'neither_score']
        
        # Ensure all expected columns are present
        sample_cols = [col for col in sample_cols if col in positions_df.columns]
        
        # Show a sample
        sample_df = positions_df.select(sample_cols).sample(n=5)
        print(sample_df)
    else:
        print("\nNo enhanced positions data found.")
    
    # Show the path to access the full dataset
    print("\nTo load the enhanced positions data in your own code:")
    print("```python")
    print("from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler")
    print("from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer")
    print("")
    print("# Initialize the data handler")
    print("data_handler = MetaModelDataHandler(eval_dir=ErrorAnalyzer.eval_dir)")
    print("")
    print("# Load the enhanced positions DataFrame")
    print("positions_df = data_handler.load_splice_positions(")
    print("    aggregated=True,")
    print("    subject='enhanced',")
    print("    enhanced=True,")
    print("    output_subdir='meta_predictions'  # Same subdirectory specified during creation")
    print(")")
    print("```")
    
    print("\nThe enhanced positions file contains all three probability scores (donor, acceptor, neither)")
    print("and is saved in a separate subdirectory to avoid any conflicts with the original analysis files.")
    
if __name__ == "__main__":
    main()
