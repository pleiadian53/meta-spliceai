#!/usr/bin/env python3
"""
Standalone script for generating comprehensive SHAP visualizations.

This script can be used to generate enhanced SHAP visualizations from existing
SHAP analysis results, useful for rerunning visualizations or customizing plots.

Usage:
    python scripts/generate_shap_visualizations.py --run-dir models/my_model_run --top-n 25
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.evaluation.shap_viz import (
    generate_comprehensive_shap_report,
    create_feature_importance_barcharts,
    create_shap_beeswarm_plots,
    create_feature_importance_heatmap
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive SHAP visualizations from existing analysis results"
    )
    
    parser.add_argument(
        "--run-dir", 
        type=str, 
        required=True,
        help="Path to model run directory containing SHAP analysis results"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        help="Path to dataset (required for beeswarm plots). If not provided, skips beeswarm plots."
    )
    
    parser.add_argument(
        "--top-n", 
        type=int, 
        default=20,
        help="Number of top features to display (default: 20)"
    )
    
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=1000,
        help="Sample size for beeswarm plots (default: 1000)"
    )
    
    parser.add_argument(
        "--plot-format", 
        type=str, 
        default="png",
        choices=["png", "pdf", "svg"],
        help="Plot format (default: png)"
    )
    
    parser.add_argument(
        "--barcharts-only", 
        action="store_true",
        help="Generate only bar charts (fastest option)"
    )
    
    parser.add_argument(
        "--no-beeswarm", 
        action="store_true",
        help="Skip beeswarm plots (faster, requires less memory)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    importance_csv = run_dir / "shap_importance_incremental.csv"
    if not importance_csv.exists():
        print(f"Error: SHAP importance file not found: {importance_csv}")
        print("Make sure you've run SHAP analysis first using the main CV script.")
        sys.exit(1)
    
    model_pkl = run_dir / "model_multiclass.pkl"
    if not model_pkl.exists() and not args.no_beeswarm:
        print(f"Warning: Model file not found: {model_pkl}")
        print("Beeswarm plots will be skipped. Use --no-beeswarm to suppress this warning.")
        args.no_beeswarm = True
    
    print("=" * 70)
    print("GENERATING COMPREHENSIVE SHAP VISUALIZATIONS")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Top features: {args.top_n}")
    print(f"Plot format: {args.plot_format}")
    print(f"Bar charts only: {args.barcharts_only}")
    print(f"Skip beeswarm: {args.no_beeswarm}")
    print("=" * 70)
    
    if args.barcharts_only:
        # Generate only bar charts and heatmap (fast)
        print("\n1. Creating feature importance bar charts...")
        try:
            bar_chart_paths = create_feature_importance_barcharts(
                importance_csv=importance_csv,
                out_dir=run_dir,
                top_n=args.top_n,
                plot_format=args.plot_format
            )
            print(f"✓ Created {len(bar_chart_paths)} bar charts")
        except Exception as e:
            print(f"✗ Error creating bar charts: {e}")
        
        print("\n2. Creating feature importance heatmap...")
        try:
            heatmap_path = create_feature_importance_heatmap(
                importance_csv=importance_csv,
                out_dir=run_dir,
                top_n=args.top_n * 2,  # More features for heatmap
                plot_format=args.plot_format
            )
            print(f"✓ Created heatmap: {heatmap_path}")
        except Exception as e:
            print(f"✗ Error creating heatmap: {e}")
    
    else:
        # Generate comprehensive report
        if args.no_beeswarm or not model_pkl.exists():
            print("\nNote: Beeswarm plots will be skipped (model file missing or --no-beeswarm)")
            
            # Manual approach without beeswarm
            print("\n1. Creating feature importance bar charts...")
            try:
                bar_chart_paths = create_feature_importance_barcharts(
                    importance_csv=importance_csv,
                    out_dir=run_dir,
                    top_n=args.top_n,
                    plot_format=args.plot_format
                )
                print(f"✓ Created {len(bar_chart_paths)} bar charts")
            except Exception as e:
                print(f"✗ Error creating bar charts: {e}")
            
            print("\n2. Creating feature importance heatmap...")
            try:
                heatmap_path = create_feature_importance_heatmap(
                    importance_csv=importance_csv,
                    out_dir=run_dir,
                    top_n=args.top_n * 2,
                    plot_format=args.plot_format
                )
                print(f"✓ Created heatmap: {heatmap_path}")
            except Exception as e:
                print(f"✗ Error creating heatmap: {e}")
        
        else:
            # Full comprehensive report
            if not args.dataset:
                print("Warning: No dataset provided. Using default dataset path...")
                # Try to infer dataset path from run directory structure
                possible_datasets = [
                    "train_pc_1000/master",
                    "data/training/features",
                    run_dir.parent / "dataset"
                ]
                dataset_path = None
                for path in possible_datasets:
                    if Path(path).exists():
                        dataset_path = str(path)
                        break
                
                if not dataset_path:
                    print("Could not find dataset. Skipping beeswarm plots.")
                    args.no_beeswarm = True
                else:
                    args.dataset = dataset_path
                    print(f"Using dataset: {dataset_path}")
            
            if not args.no_beeswarm:
                try:
                    results = generate_comprehensive_shap_report(
                        importance_csv=importance_csv,
                        model_path=model_pkl,
                        dataset_path=args.dataset,
                        out_dir=run_dir,
                        top_n=args.top_n,
                        sample_size=args.sample_size,
                        plot_format=args.plot_format
                    )
                    
                    if 'summary_stats' in results:
                        stats = results['summary_stats']
                        print(f"\n📊 SHAP ANALYSIS SUMMARY:")
                        print(f"   Total features analyzed: {stats.get('total_features', 'N/A')}")
                        print(f"   Top overall feature: {stats.get('top_feature_overall', 'N/A')}")
                        print(f"   Top neither feature: {stats.get('top_feature_neither', 'N/A')}")
                        print(f"   Top donor feature: {stats.get('top_feature_donor', 'N/A')}")
                        print(f"   Top acceptor feature: {stats.get('top_feature_acceptor', 'N/A')}")
                        print(f"   Features with zero importance: {stats.get('features_with_zero_importance', 'N/A')}")
                        
                except Exception as e:
                    print(f"✗ Error generating comprehensive report: {e}")
                    print("Falling back to bar charts only...")
                    args.barcharts_only = True
    
    print("\n" + "=" * 70)
    print("SHAP VISUALIZATION GENERATION COMPLETE!")
    enhanced_viz_dir = run_dir / "enhanced_shap_viz"
    if enhanced_viz_dir.exists():
        print(f"Enhanced visualizations saved to: {enhanced_viz_dir}")
        
        # List generated files
        viz_files = list(enhanced_viz_dir.glob(f"*.{args.plot_format}"))
        if viz_files:
            print(f"Generated {len(viz_files)} visualization files:")
            for file in sorted(viz_files):
                print(f"  - {file.name}")
    else:
        print(f"Visualizations saved to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main() 