#!/usr/bin/env python3
"""
Example: Feature Analysis Workflow

This script demonstrates how to use the probability feature analysis tools
to understand and visualize the derived features used in splice site prediction.

Usage:
    python example_feature_analysis.py
"""

import os
import sys
from pathlib import Path
import argparse

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demonstrate_feature_analysis():
    """Demonstrate the complete feature analysis workflow."""
    
    print("=" * 80)
    print("PROBABILITY FEATURE ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Create conceptual diagrams (no data needed)
    print("\n1. CREATING CONCEPTUAL DIAGRAMS")
    print("-" * 50)
    
    try:
        from create_feature_diagrams import main as create_diagrams
        print("Running: create_feature_diagrams.py --output-dir results/probability_feature_analysis/example/diagrams/")
        
        # Simulate command line arguments
        sys.argv = ['create_feature_diagrams.py', '--output-dir', 'results/probability_feature_analysis/example/diagrams']
        create_diagrams()
        
    except Exception as e:
        print(f"Error creating diagrams: {e}")
        print("You can run manually: python create_feature_diagrams.py --output-dir results/probability_feature_analysis/example/diagrams/")
    
    # Step 2: Check for sample data
    print("\n2. CHECKING FOR SAMPLE DATA")
    print("-" * 50)
    
    # Look for sample data files
    sample_data_paths = [
        "full_splice_positions_enhanced.tsv",
        "positions_enhanced_aggregated.tsv",
        "positions_enhanced_aggregated.parquet",
        "../results/positions_enhanced_aggregated.tsv"
    ]
    
    found_data = None
    for path in sample_data_paths:
        if os.path.exists(path):
            found_data = path
            print(f"Found sample data: {path}")
            break
    
    if not found_data:
        print("No sample data found. Looking for common locations...")
        print("You can generate sample data using:")
        print("  python run_fn_rescue_pipeline.py --top-genes 3")
        print("  python run_fp_reduction_pipeline.py --top-genes 3")
        print("\nOr point to existing enhanced positions data:")
        print("  python visualize_probability_features.py --data-file your_data.tsv")
    
    # Step 3: Generate analysis reports (if data available)
    if found_data:
        print("\n3. GENERATING FEATURE ANALYSIS REPORTS")
        print("-" * 50)
        
        try:
            from generate_feature_report import main as generate_reports
            print(f"Running: generate_feature_report.py --data-file {found_data}")
            
            # Simulate command line arguments
            sys.argv = ['generate_feature_report.py', '--data-file', found_data, '--output-dir', 'results/probability_feature_analysis/example/reports']
            generate_reports()
            
        except Exception as e:
            print(f"Error generating reports: {e}")
            print(f"You can run manually: python generate_feature_report.py --data-file {found_data}")
        
        # Step 4: Create visualizations
        print("\n4. CREATING FEATURE VISUALIZATIONS")
        print("-" * 50)
        
        try:
            from visualize_probability_features import main as create_visualizations
            print(f"Running: visualize_probability_features.py --data-file {found_data}")
            
            # Simulate command line arguments
            sys.argv = ['visualize_probability_features.py', '--data-file', found_data, 
                       '--output-dir', 'results/probability_feature_analysis/example/visualizations', '--sample-size', '1000']
            create_visualizations()
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            print(f"You can run manually: python visualize_probability_features.py --data-file {found_data}")
    
    # Step 5: Summary and next steps
    print("\n5. SUMMARY AND NEXT STEPS")
    print("-" * 50)
    
    print("✅ Feature analysis workflow demonstration complete!")
    print("\nGenerated outputs:")
    
    output_dirs = [
        'results/probability_feature_analysis/example/diagrams', 
        'results/probability_feature_analysis/example/reports', 
        'results/probability_feature_analysis/example/visualizations'
    ]
    for dir_name in output_dirs:
        if os.path.exists(dir_name):
            print(f"\n📁 {dir_name}/")
            try:
                files = list(Path(dir_name).glob("*"))
                for file_path in sorted(files):
                    print(f"   - {file_path.name}")
            except Exception as e:
                print(f"   Error listing files: {e}")
    
    print("\n" + "=" * 80)
    print("UNDERSTANDING YOUR FEATURES")
    print("=" * 80)
    
    print("""
Key Features and Their Meanings:

📊 SIGNAL PROCESSING FEATURES:
• donor_peak_height_ratio: How prominent is the donor peak?
  - > 2.0: Sharp, isolated peak (likely true)
  - < 1.5: Broad, weak signal (likely false)

• second_derivative: How curved is the peak?
  - Positive: Sharp peak (true splice site)
  - Negative: Broad signal (false positive)

• signal_strength: How strong is the signal above background?
  - > 0.2: Strong signal
  - < 0.05: Weak signal

🔄 CROSS-TYPE FEATURES:
• type_signal_difference: Donor vs acceptor preference
  - > +0.1: Donor preferred
  - < -0.1: Acceptor preferred
  - ≈ 0: Ambiguous

• donor_acceptor_peak_ratio: Relative peak heights
  - > 2.0: Strong donor
  - < 0.5: Strong acceptor

📍 CONTEXT FEATURES:
• context_asymmetry: Directional bias in signal
• context_neighbor_mean: Local background level
• context_max: Nearby competing signals

🎯 PRACTICAL APPLICATIONS:
• FP Reduction: Focus on peak_height_ratio, second_derivative
• FN Rescue: Look for moderate splice_probability with good peak quality
• Type Classification: Use type_signal_difference, peak_ratio
    """)
    
    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS TOOLS REFERENCE")
    print("=" * 80)
    
    print("""
🔧 AVAILABLE TOOLS:

1. create_feature_diagrams.py
   - Creates conceptual diagrams explaining signal processing
   - No data required - pure educational visualizations
   
2. generate_feature_report.py
   - Generates comprehensive text reports about features
   - Includes statistical analysis and interpretation guides
   
3. visualize_probability_features.py
   - Creates data-driven visualizations and examples
   - Shows real feature distributions and correlations
   
4. Probability_Feature_Documentation.md
   - Comprehensive reference manual for all features
   - Mathematical formulas and biological interpretations

📋 TYPICAL WORKFLOW:
1. Read the documentation (Probability_Feature_Documentation.md)
2. Create conceptual diagrams (create_feature_diagrams.py)
3. Generate reports with your data (generate_feature_report.py)
4. Create visualizations (visualize_probability_features.py)
5. Interpret results using the documentation
    """)

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Demonstrate probability feature analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true",
                       help="Run quick demonstration (diagrams only)")
    
    args = parser.parse_args()
    
    if args.quick:
        print("Quick demonstration mode - creating diagrams only...")
        try:
            from create_feature_diagrams import main as create_diagrams
            sys.argv = ['create_feature_diagrams.py', '--output-dir', 'results/probability_feature_analysis/quick/diagrams']
            create_diagrams()
        except Exception as e:
            print(f"Error: {e}")
    else:
        demonstrate_feature_analysis()

if __name__ == "__main__":
    main() 