#!/usr/bin/env python3
"""
Model calibration analysis utility.

This script analyzes model calibration results and provides insights into
calibration quality, overconfidence, and reliability of probability predictions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional


def analyze_calibration_results(results_dir: str, output_file: Optional[str] = None) -> Dict:
    """
    Analyze model calibration from results directory.
    
    Args:
        results_dir: Directory containing calibration analysis results
        output_file: Optional output file for detailed analysis
        
    Returns:
        Dictionary containing calibration analysis results
    """
    results_dir = Path(results_dir)
    
    print("🎯 MODEL CALIBRATION ANALYSIS")
    print("=" * 50)
    print(f"Analyzing calibration from: {results_dir}")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return {'error': 'Results directory not found'}
    
    # Look for calibration analysis directory
    calib_dir = results_dir / 'calibration_analysis'
    if not calib_dir.exists():
        print(f"❌ Calibration analysis directory not found: {calib_dir}")
        return {'error': 'Calibration analysis not found'}
    
    # Initialize results
    analysis = {
        'results_dir': str(results_dir),
        'calibration_dir': str(calib_dir),
        'calibration_metrics': {},
        'reliability_analysis': {},
        'recommendations': [],
        'issues': []
    }
    
    # Load calibration summary
    calib_summary_file = calib_dir / 'calibration_summary.json'
    if calib_summary_file.exists():
        try:
            with open(calib_summary_file) as f:
                calib_results = json.load(f)
            
            print(f"\n📊 CALIBRATION METRICS:")
            print("-" * 30)
            
            # Extract key metrics
            overall_error = calib_results.get('overall_calibration_error', 'N/A')
            overconfidence = calib_results.get('overconfidence_score', 'N/A')
            quality = calib_results.get('calibration_quality', 'N/A')
            
            analysis['calibration_metrics'] = {
                'overall_calibration_error': overall_error,
                'overconfidence_score': overconfidence,
                'calibration_quality': quality
            }
            
            print(f"  Overall Calibration Error: {overall_error}")
            print(f"  Overconfidence Score: {overconfidence}")
            print(f"  Calibration Quality: {quality}")
            
            # Analyze calibration quality
            if isinstance(overall_error, (int, float)):
                if overall_error < 0.05:
                    analysis['recommendations'].append("✅ Excellent calibration (error < 0.05)")
                elif overall_error < 0.1:
                    analysis['recommendations'].append("✅ Good calibration (error < 0.1)")
                elif overall_error < 0.2:
                    analysis['recommendations'].append("⚠️ Moderate calibration (error < 0.2)")
                else:
                    analysis['recommendations'].append("❌ Poor calibration (error >= 0.2)")
                    analysis['issues'].append(f"High calibration error: {overall_error:.3f}")
            
            if isinstance(overconfidence, (int, float)):
                if overconfidence > 0.1:
                    analysis['issues'].append(f"High overconfidence: {overconfidence:.3f}")
                    analysis['recommendations'].append("Consider temperature scaling or Platt scaling")
                else:
                    analysis['recommendations'].append("✅ Low overconfidence detected")
            
            # Store full calibration results
            analysis['calibration_metrics']['full_results'] = calib_results
            
        except Exception as e:
            analysis['issues'].append(f"Failed to load calibration summary: {e}")
            print(f"  ❌ Failed to load calibration summary: {e}")
    else:
        analysis['issues'].append("Calibration summary file not found")
        print(f"  ❌ Calibration summary not found: {calib_summary_file}")
    
    # Check for additional calibration files
    calib_files = {
        'reliability_diagram': 'reliability_diagram.png',
        'calibration_curve': 'calibration_curve.png',
        'per_class_calibration': 'per_class_calibration.json'
    }
    
    print(f"\n📁 CALIBRATION FILES:")
    print("-" * 25)
    
    for file_key, filename in calib_files.items():
        file_path = calib_dir / filename
        exists = file_path.exists()
        
        status = "✅" if exists else "❌"
        print(f"  {status} {filename}")
        
        if exists:
            analysis['calibration_metrics'][f'{file_key}_path'] = str(file_path)
        else:
            analysis['issues'].append(f"Missing calibration file: {filename}")
    
    # Load per-class calibration if available
    per_class_file = calib_dir / 'per_class_calibration.json'
    if per_class_file.exists():
        try:
            with open(per_class_file) as f:
                per_class_calib = json.load(f)
            
            print(f"\n🎯 PER-CLASS CALIBRATION:")
            print("-" * 30)
            
            for class_name, metrics in per_class_calib.items():
                if isinstance(metrics, dict):
                    error = metrics.get('calibration_error', 'N/A')
                    print(f"  {class_name}: {error}")
                    
                    if isinstance(error, (int, float)) and error > 0.15:
                        analysis['issues'].append(f"Poor calibration for {class_name}: {error:.3f}")
            
            analysis['calibration_metrics']['per_class'] = per_class_calib
            
        except Exception as e:
            analysis['issues'].append(f"Failed to load per-class calibration: {e}")
            print(f"  ❌ Failed to load per-class calibration: {e}")
    
    # Generate reliability analysis
    n_issues = len(analysis['issues'])
    n_recommendations = len(analysis['recommendations'])
    
    analysis['reliability_analysis'] = {
        'n_issues': n_issues,
        'n_recommendations': n_recommendations,
        'overall_status': 'good' if n_issues == 0 else 'needs_attention',
        'reliability_score': max(0, 100 - n_issues * 20)
    }
    
    print(f"\n📈 RELIABILITY ANALYSIS:")
    print("-" * 30)
    print(f"  Issues found: {n_issues}")
    print(f"  Reliability score: {analysis['reliability_analysis']['reliability_score']}/100")
    print(f"  Overall status: {analysis['reliability_analysis']['overall_status']}")
    
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
        print(f"\n✅ NO CALIBRATION ISSUES DETECTED")
    
    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Calibration analysis saved to: {output_path}")
    
    return analysis


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Model calibration analysis utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze model calibration
  python calibration_checker.py \\
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \\
    --output calibration_report.json

  # Quick calibration check
  python calibration_checker.py \\
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1
        """
    )
    
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing training results with calibration analysis")
    parser.add_argument("--output", 
                       help="Output file for calibration analysis (JSON format)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run calibration analysis
    results = analyze_calibration_results(args.results_dir, args.output)
    
    if 'error' in results:
        print(f"❌ Calibration analysis failed: {results['error']}")
        sys.exit(1)
    
    # Exit with appropriate code based on reliability
    reliability_score = results['reliability_analysis'].get('reliability_score', 0)
    if reliability_score < 60:
        print(f"\n⚠️ Low reliability score ({reliability_score}/100). Review calibration issues.")
        sys.exit(1)
    else:
        print(f"\n🎉 Calibration analysis completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()




