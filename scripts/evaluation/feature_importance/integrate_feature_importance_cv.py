#!/usr/bin/env python3
"""
Example integration of feature importance analysis into gene-wise CV workflow.

This script shows how to add comprehensive feature importance analysis
to the end of your run_gene_cv_sigmoid.py pipeline.
"""

def add_feature_importance_to_cv(
    dataset_path: str,
    cv_output_dir: str,
    subject: str = "gene_cv_analysis",
    top_k: int = 25,
    methods: list = None,
    include_shap: bool = True,
    verbose: bool = True
):
    """
    Add comprehensive feature importance analysis to gene-wise CV workflow.
    
    This function should be called at the end of your CV pipeline, after
    the model has been trained and saved.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset used in CV
    cv_output_dir : str
        Output directory from gene-wise CV (where model.pkl is saved)
    subject : str
        Subject name for analysis files
    top_k : int
        Number of top features to analyze
    methods : list, optional
        Methods to use. Default: all methods
    include_shap : bool
        Whether to include SHAP analysis
    verbose : bool
        Whether to print progress
    """
    from pathlib import Path
    import pandas as pd
    import joblib
    import os
    
    # Import the feature importance modules
    from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import (
        run_comprehensive_feature_importance_analysis
    )
    
    # Set up paths
    cv_dir = Path(cv_output_dir)
    model_path = cv_dir / "model_multiclass.pkl"
    feature_analysis_dir = cv_dir / "feature_importance_analysis"
    
    # Check if required files exist
    if not model_path.exists():
        print(f"[WARNING] Model file not found: {model_path}")
        print("Make sure gene-wise CV completed successfully")
        return None
    
    if verbose:
        print(f"\n{'='*60}")
        print("🔍 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_path}")
        print(f"CV Output: {cv_output_dir}")
        print(f"Analysis Output: {feature_analysis_dir}")
    
    # Create analysis output directory
    feature_analysis_dir.mkdir(exist_ok=True)
    
    # Set default methods if not provided
    if methods is None:
        methods = ['xgboost', 'hypothesis_testing', 'effect_sizes', 'mutual_info']
    
    try:
        # Load the trained model to check if it's compatible
        model = joblib.load(model_path)
        if verbose:
            print(f"✓ Loaded model: {type(model).__name__}")
        
        # Check if we have a feature manifest
        feature_manifest = cv_dir / "feature_manifest.csv"
        if feature_manifest.exists():
            features_df = pd.read_csv(feature_manifest)
            if verbose:
                print(f"✓ Found {len(features_df)} features in manifest")
        
        # Run comprehensive analysis
        results = run_comprehensive_feature_importance_analysis(
            model_path=str(model_path),
            data_path=dataset_path,
            output_dir=str(feature_analysis_dir),
            subject=subject,
            top_k=top_k,
            methods=methods,
            include_shap=include_shap,
            shap_max_samples=10000,  # Limit SHAP for memory efficiency
            verbose=1 if verbose else 0
        )
        
        if verbose:
            print(f"\n🎉 Feature importance analysis completed!")
            print(f"📁 Results saved to: {feature_analysis_dir}")
            
            # Print summary of consensus features
            if 'summary' in results and 'consensus_features' in results['summary']:
                consensus = results['summary']['consensus_features']
                if consensus:
                    print(f"\n🏆 Top consensus features (appearing in multiple methods):")
                    for i, feature in enumerate(consensus[:10], 1):
                        print(f"  {i:2d}. {feature}")
                else:
                    print("\n📝 No consensus features found across methods")
            
            # Print method agreement statistics
            if 'summary' in results and 'method_agreement' in results['summary']:
                agreements = results['summary']['method_agreement']
                if agreements:
                    print(f"\n🤝 Method Agreement (Jaccard Similarity):")
                    for agreement in agreements[:5]:  # Show top 5
                        method1 = agreement['method1']
                        method2 = agreement['method2']
                        jaccard = agreement['jaccard_similarity']
                        overlap = agreement['overlap']
                        print(f"  {method1} ↔ {method2}: {jaccard:.3f} ({overlap} features)")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Feature importance analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def integrate_into_run_gene_cv_sigmoid():
    """
    Example showing how to modify run_gene_cv_sigmoid.py to include feature importance.
    
    Add this code at the very end of your main() function in run_gene_cv_sigmoid.py,
    just before the final return statement.
    """
    integration_code = '''
    # ===================================================================
    # COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS (ADD TO END OF main())
    # ===================================================================
    
    # Skip feature importance if --skip-eval is enabled
    if not args.skip_eval:
        try:
            print("\\n" + "="*60)
            print("🔍 STARTING COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
            print("="*60)
            
            # Import the integration function
            from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import (
                run_comprehensive_feature_importance_analysis
            )
            
            # Set up parameters
            feature_subject = f"gene_cv_{args.dataset.replace('/', '_').replace('.', '_')}"
            feature_methods = ['xgboost', 'hypothesis_testing', 'effect_sizes', 'mutual_info']
            
            # Create feature analysis subdirectory
            feature_analysis_dir = out_dir / "feature_importance_analysis"
            feature_analysis_dir.mkdir(exist_ok=True)
            
            # Run comprehensive feature importance analysis
            feature_results = run_comprehensive_feature_importance_analysis(
                model_path=str(out_dir / "model_multiclass.pkl"),
                data_path=args.dataset,
                output_dir=str(feature_analysis_dir),
                subject=feature_subject,
                top_k=25,
                methods=feature_methods,
                include_shap=True,
                shap_max_samples=min(15000, diag_sample if diag_sample else 15000),
                verbose=1
            )
            
            # Log results summary
            if feature_results and 'summary' in feature_results:
                summary = feature_results['summary']
                
                # Print consensus features
                consensus_features = summary.get('consensus_features', [])
                if consensus_features:
                    print(f"\\n🏆 TOP CONSENSUS FEATURES (multiple methods agree):")
                    for i, feature in enumerate(consensus_features[:10], 1):
                        print(f"  {i:2d}. {feature}")
                
                # Print method agreement
                agreements = summary.get('method_agreement', [])
                if agreements:
                    print(f"\\n🤝 METHOD AGREEMENT SUMMARY:")
                    for agreement in agreements[:3]:  # Top 3 agreements
                        method1 = agreement['method1']
                        method2 = agreement['method2']
                        jaccard = agreement['jaccard_similarity']
                        overlap = agreement['overlap']
                        print(f"  {method1} ↔ {method2}: {jaccard:.3f} similarity ({overlap} overlapping features)")
            
            print(f"\\n✅ Feature importance analysis completed!")
            print(f"📁 Results saved to: {feature_analysis_dir}")
            print(f"📊 Generated plots and Excel file with comprehensive results")
            
        except Exception as e:
            print(f"\\n⚠️  Feature importance analysis failed: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
            print("Continuing with standard CV workflow...")
    '''
    
    print("To integrate feature importance into run_gene_cv_sigmoid.py:")
    print("1. Add this code at the END of the main() function")
    print("2. Place it just before the final return statement")
    print("3. Make sure it's within the 'if not args.skip_eval:' block")
    print("\nHere's the code to add:")
    print(integration_code)


# Command-line interface for standalone usage
def main():
    """Command-line interface for feature importance analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add comprehensive feature importance analysis to gene-wise CV results"
    )
    
    parser.add_argument(
        "dataset", 
        help="Path to the dataset used in CV"
    )
    
    parser.add_argument(
        "cv_output_dir", 
        help="Output directory from gene-wise CV"
    )
    
    parser.add_argument(
        "--subject", "-s",
        default="gene_cv_analysis",
        help="Subject name for analysis files"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=25,
        help="Number of top features to analyze"
    )
    
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        help="Feature importance methods to use (default: all)"
    )
    
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP analysis"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    results = add_feature_importance_to_cv(
        dataset_path=args.dataset,
        cv_output_dir=args.cv_output_dir,
        subject=args.subject,
        top_k=args.top_k,
        methods=args.methods,
        include_shap=not args.no_shap,
        verbose=args.verbose
    )
    
    if results:
        print("✅ Analysis completed successfully!")
    else:
        print("❌ Analysis failed!")
        exit(1)


if __name__ == "__main__":
    # Show integration instructions
    integrate_into_run_gene_cv_sigmoid()
    print("\n" + "="*60)
    print("Or run as standalone script:")
    print("python scripts/integrate_feature_importance_cv.py <dataset> <cv_output_dir>")
    print("="*60) 