#!/usr/bin/env python3
"""
Ensemble model analysis utility.

This script analyzes trained ensemble models to understand their structure,
calibration status, and performance characteristics.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional


def analyze_ensemble_model(model_path: str, output_file: Optional[str] = None) -> Dict:
    """
    Analyze a trained ensemble model.
    
    Args:
        model_path: Path to the trained model (.pkl file)
        output_file: Optional output file for detailed analysis
        
    Returns:
        Dictionary containing ensemble analysis results
    """
    model_path = Path(model_path)
    
    print("🤖 ENSEMBLE MODEL ANALYSIS")
    print("=" * 50)
    print(f"Analyzing model: {model_path}")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return {'error': 'Model file not found'}
    
    # Initialize results
    analysis = {
        'model_path': str(model_path),
        'model_info': {},
        'ensemble_structure': {},
        'calibration_info': {},
        'feature_info': {},
        'recommendations': [],
        'issues': []
    }
    
    try:
        # Load the model
        print(f"\n📊 LOADING MODEL:")
        print("-" * 20)
        
        with open(model_path, 'rb') as f:
            ensemble = pickle.load(f)
        
        model_type = type(ensemble).__name__
        print(f"  Model type: {model_type}")
        
        analysis['model_info'] = {
            'type': model_type,
            'file_size_mb': model_path.stat().st_size / (1024 * 1024)
        }
        
        print(f"  File size: {analysis['model_info']['file_size_mb']:.1f} MB")
        
        # Analyze ensemble structure
        print(f"\n🏗️ ENSEMBLE STRUCTURE:")
        print("-" * 25)
        
        if hasattr(ensemble, 'models'):
            n_models = len(ensemble.models)
            print(f"  Number of base models: {n_models}")
            
            # Analyze base model types
            if n_models > 0:
                base_model_types = {}
                for i, model in enumerate(ensemble.models):
                    model_type = type(model).__name__
                    base_model_types[model_type] = base_model_types.get(model_type, 0) + 1
                
                print(f"  Base model types:")
                for model_type, count in base_model_types.items():
                    print(f"    - {model_type}: {count}")
                
                analysis['ensemble_structure'] = {
                    'n_base_models': n_models,
                    'base_model_types': base_model_types
                }
                
                # Check if models have consistent parameters
                if hasattr(ensemble.models[0], 'get_params'):
                    try:
                        first_params = ensemble.models[0].get_params()
                        consistent_params = True
                        
                        for model in ensemble.models[1:]:
                            if hasattr(model, 'get_params'):
                                if model.get_params() != first_params:
                                    consistent_params = False
                                    break
                        
                        if consistent_params:
                            print(f"  ✅ All base models have consistent parameters")
                            analysis['recommendations'].append("✅ Base models have consistent hyperparameters")
                        else:
                            print(f"  ⚠️ Base models have different parameters")
                            analysis['issues'].append("Base models have inconsistent parameters")
                        
                        analysis['ensemble_structure']['consistent_params'] = consistent_params
                        
                    except Exception as e:
                        print(f"  ⚠️ Could not check parameter consistency: {e}")
            
        else:
            print(f"  ⚠️ Model does not have 'models' attribute")
            analysis['issues'].append("Model structure not recognized as ensemble")
        
        # Check feature information
        print(f"\n🔍 FEATURE INFORMATION:")
        print("-" * 25)
        
        feature_count = None
        feature_names = None
        
        if hasattr(ensemble, 'feature_names'):
            feature_names = ensemble.feature_names
            feature_count = len(feature_names)
        elif hasattr(ensemble, 'feature_names_'):
            feature_names = ensemble.feature_names_
            feature_count = len(feature_names)
        elif hasattr(ensemble, 'models') and len(ensemble.models) > 0:
            if hasattr(ensemble.models[0], 'feature_names_in_'):
                feature_names = ensemble.models[0].feature_names_in_.tolist()
                feature_count = len(feature_names)
        
        if feature_count:
            print(f"  Feature count: {feature_count}")
            analysis['feature_info'] = {
                'count': feature_count,
                'names_available': feature_names is not None
            }
            
            if feature_names:
                # Analyze feature types
                feature_types = {
                    'kmer': len([f for f in feature_names if 'mer_' in f]),
                    'spliceai': len([f for f in feature_names if 'spliceai' in f.lower()]),
                    'positional': len([f for f in feature_names if any(pos in f for pos in ['position', 'distance', 'offset'])]),
                    'other': 0
                }
                feature_types['other'] = feature_count - sum(feature_types.values())
                
                print(f"  Feature types:")
                for ftype, count in feature_types.items():
                    if count > 0:
                        print(f"    - {ftype}: {count}")
                
                analysis['feature_info']['types'] = feature_types
        else:
            print(f"  ⚠️ Feature count not available")
            analysis['issues'].append("Feature information not accessible")
        
        # Check calibration status
        print(f"\n🎯 CALIBRATION STATUS:")
        print("-" * 25)
        
        calibration_info = {}
        
        if hasattr(ensemble, 'calibrator'):
            print(f"  ✅ Model is calibrated")
            calibrator_type = type(ensemble.calibrator).__name__
            print(f"  Calibrator type: {calibrator_type}")
            
            calibration_info = {
                'is_calibrated': True,
                'calibrator_type': calibrator_type
            }
            
            analysis['recommendations'].append("✅ Model has probability calibration")
        else:
            print(f"  ℹ️ Model is not calibrated")
            calibration_info = {
                'is_calibrated': False,
                'calibrator_type': None
            }
            
            analysis['recommendations'].append("Consider adding probability calibration for better reliability")
        
        analysis['calibration_info'] = calibration_info
        
        # Check for prediction methods
        print(f"\n🔮 PREDICTION CAPABILITIES:")
        print("-" * 30)
        
        prediction_methods = {
            'predict': hasattr(ensemble, 'predict'),
            'predict_proba': hasattr(ensemble, 'predict_proba'),
            'decision_function': hasattr(ensemble, 'decision_function')
        }
        
        for method, available in prediction_methods.items():
            status = "✅" if available else "❌"
            print(f"  {status} {method}")
        
        analysis['model_info']['prediction_methods'] = prediction_methods
        
        if not prediction_methods['predict_proba']:
            analysis['issues'].append("Model does not support probability prediction")
        
        # Generate recommendations based on analysis
        if analysis['ensemble_structure'].get('n_base_models', 0) < 3:
            analysis['recommendations'].append("Consider using more base models for better ensemble performance")
        
        if feature_count and feature_count < 10:
            analysis['recommendations'].append("Low feature count may limit model performance")
        elif feature_count and feature_count > 1000:
            analysis['recommendations'].append("High feature count may indicate potential overfitting")
        
    except Exception as e:
        analysis['issues'].append(f"Failed to analyze model: {e}")
        print(f"❌ Failed to analyze model: {e}")
        return analysis
    
    # Generate overall assessment
    n_issues = len(analysis['issues'])
    n_recommendations = len(analysis['recommendations'])
    
    analysis['summary'] = {
        'n_issues': n_issues,
        'n_recommendations': n_recommendations,
        'overall_status': 'good' if n_issues == 0 else 'needs_attention',
        'analysis_score': max(0, 100 - n_issues * 15)
    }
    
    print(f"\n📈 ANALYSIS SUMMARY:")
    print("-" * 25)
    print(f"  Issues found: {n_issues}")
    print(f"  Recommendations: {n_recommendations}")
    print(f"  Analysis score: {analysis['summary']['analysis_score']}/100")
    print(f"  Overall status: {analysis['summary']['overall_status']}")
    
    # Show recommendations
    if analysis['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 20)
        for rec in analysis['recommendations']:
            print(f"  {rec}")
    
    # Show issues
    if analysis['issues']:
        print(f"\n⚠️ ISSUES DETECTED:")
        print("-" * 20)
        for issue in analysis['issues']:
            print(f"  - {issue}")
    else:
        print(f"\n✅ NO ISSUES DETECTED")
    
    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Ensemble analysis saved to: {output_path}")
    
    return analysis


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Ensemble model analysis utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ensemble model
  python ensemble_analyzer.py \\
    --model results/gene_cv_pc_5000_3mers_diverse_run1/model_multiclass.pkl \\
    --output ensemble_analysis.json

  # Quick ensemble analysis
  python ensemble_analyzer.py \\
    --model results/gene_cv_pc_5000_3mers_diverse_run1/model_multiclass.pkl
        """
    )
    
    parser.add_argument("--model", required=True,
                       help="Path to trained model (.pkl file)")
    parser.add_argument("--output", 
                       help="Output file for ensemble analysis (JSON format)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run ensemble analysis
    results = analyze_ensemble_model(args.model, args.output)
    
    if 'error' in results:
        print(f"❌ Ensemble analysis failed: {results['error']}")
        sys.exit(1)
    
    # Exit with appropriate code based on analysis
    analysis_score = results.get('summary', {}).get('analysis_score', 0)
    if analysis_score < 60:
        print(f"\n⚠️ Low analysis score ({analysis_score}/100). Review issues.")
        sys.exit(1)
    else:
        print(f"\n🎉 Ensemble analysis completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()




