#!/usr/bin/env python3
"""
Extract Presentation Metrics

This script extracts key performance and diagnostic metrics from various
evaluation result directories to prepare for presentation slides.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.extract_presentation_metrics \
      --gene-cv-dir results/gene_cv_1000_run_15 \
      --ablation-dir results/enhanced_ablation \
      --output-dir results/presentation_metrics/
"""

import argparse
import json
import os
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    rocauc: float
    f1_score: float
    average_precision: float
    precision: float
    recall: float
    accuracy: float
    
@dataclass
class DiagnosticResults:
    """Container for diagnostic test results."""
    test_name: str
    status: str  # "pass", "warning", "fail"
    description: str
    key_metrics: Dict[str, float]
    interpretation: str

def extract_gene_cv_metrics(cv_dir: str) -> Dict[str, Any]:
    """Extract metrics from gene-aware cross-validation results."""
    cv_path = Path(cv_dir)
    
    metrics = {
        "test_type": "Gene-Aware Cross-Validation",
        "description": "Performance across genetically independent test sets",
        "status": "unknown"
    }
    
    # Look for common result files
    result_files = [
        "cross_validation_results.json",
        "cv_summary.json", 
        "performance_metrics.json",
        "final_results.json"
    ]
    
    results_data = None
    for file_name in result_files:
        file_path = cv_path / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    results_data = json.load(f)
                print(f"Found results in: {file_path}")
                break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    # Look for CSV files with metrics
    csv_files = list(cv_path.glob("*metrics*.csv")) + list(cv_path.glob("*results*.csv"))
    
    if csv_files and results_data is None:
        try:
            # Try to read the first CSV file
            df = pd.read_csv(csv_files[0])
            print(f"Found CSV results in: {csv_files[0]}")
            
            # Convert to dict-like structure
            if 'fold' in df.columns or 'cv_fold' in df.columns:
                # This looks like CV results
                metrics_cols = [col for col in df.columns if any(metric in col.lower() 
                               for metric in ['auc', 'f1', 'precision', 'recall', 'ap'])]
                
                results_data = {
                    "cv_results": df[metrics_cols].to_dict('records') if metrics_cols else {},
                    "summary": df[metrics_cols].describe().to_dict() if metrics_cols else {}
                }
        except Exception as e:
            print(f"Error reading CSV {csv_files[0]}: {e}")
    
    if results_data:
        # Extract key metrics
        try:
            # Try different possible structures
            if "cv_results" in results_data and "summary" in results_data:
                summary = results_data["summary"]
                metrics.update({
                    "donor_rocauc_mean": summary.get("donor_rocauc", {}).get("mean", "N/A"),
                    "donor_rocauc_std": summary.get("donor_rocauc", {}).get("std", "N/A"),
                    "acceptor_rocauc_mean": summary.get("acceptor_rocauc", {}).get("mean", "N/A"),
                    "acceptor_rocauc_std": summary.get("acceptor_rocauc", {}).get("std", "N/A"),
                    "n_folds": len(results_data.get("cv_results", [])),
                    "status": "pass"
                })
            elif "performance_metrics" in results_data:
                perf = results_data["performance_metrics"]
                metrics.update({
                    "donor_rocauc_mean": perf.get("donor", {}).get("rocauc", "N/A"),
                    "acceptor_rocauc_mean": perf.get("acceptor", {}).get("rocauc", "N/A"), 
                    "status": "pass"
                })
            else:
                # Try to extract any numeric metrics
                flat_metrics = {}
                for key, value in results_data.items():
                    if isinstance(value, (int, float)):
                        flat_metrics[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)):
                                flat_metrics[f"{key}_{subkey}"] = subvalue
                
                metrics.update(flat_metrics)
                metrics["status"] = "pass" if flat_metrics else "unknown"
                        
        except Exception as e:
            print(f"Error parsing results data: {e}")
            metrics["status"] = "error"
            metrics["error"] = str(e)
    else:
        metrics["status"] = "not_found"
        metrics["error"] = f"No results files found in {cv_dir}"
    
    return metrics

def extract_ablation_metrics(ablation_dir: str) -> Dict[str, Any]:
    """Extract metrics from ablation study results."""
    ablation_path = Path(ablation_dir)
    
    metrics = {
        "test_type": "Ablation Study",
        "description": "Performance impact of different feature categories",
        "status": "unknown"
    }
    
    # Look for ablation results
    result_files = [
        "ablation_results.json",
        "feature_importance.json",
        "ablation_summary.json"
    ]
    
    results_data = None
    for file_name in result_files:
        file_path = ablation_path / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    results_data = json.load(f)
                print(f"Found ablation results in: {file_path}")
                break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    # Look for CSV files
    csv_files = list(ablation_path.glob("*ablation*.csv")) + list(ablation_path.glob("*feature*.csv"))
    
    if csv_files and results_data is None:
        try:
            # Try to read ablation CSV
            df = pd.read_csv(csv_files[0])
            print(f"Found ablation CSV in: {csv_files[0]}")
            
            # Look for columns indicating ablation results
            if 'feature_set' in df.columns or 'ablated_features' in df.columns:
                results_data = {
                    "ablation_results": df.to_dict('records')
                }
        except Exception as e:
            print(f"Error reading ablation CSV {csv_files[0]}: {e}")
    
    if results_data:
        try:
            if "ablation_results" in results_data:
                ablation_data = results_data["ablation_results"]
                
                # Find baseline (all features) performance
                baseline = None
                for result in ablation_data:
                    if any(term in str(result.get('feature_set', '')).lower() 
                          for term in ['all', 'full', 'baseline', 'complete']):
                        baseline = result
                        break
                
                if baseline:
                    baseline_auc = baseline.get('rocauc', baseline.get('auc', None))
                    
                    # Calculate performance drops
                    performance_drops = []
                    for result in ablation_data:
                        if result != baseline:
                            ablated_auc = result.get('rocauc', result.get('auc', None))
                            if baseline_auc and ablated_auc:
                                drop = baseline_auc - ablated_auc
                                performance_drops.append({
                                    'feature_set': result.get('feature_set', 'unknown'),
                                    'performance_drop': drop,
                                    'relative_drop_pct': (drop / baseline_auc) * 100
                                })
                    
                    metrics.update({
                        "baseline_rocauc": baseline_auc,
                        "performance_drops": performance_drops,
                        "max_drop": max([p['performance_drop'] for p in performance_drops]) if performance_drops else 0,
                        "status": "pass"
                    })
                else:
                    metrics["status"] = "warning"
                    metrics["error"] = "Could not identify baseline performance"
            else:
                metrics["status"] = "warning" 
                metrics["error"] = "Unexpected ablation results format"
                
        except Exception as e:
            print(f"Error parsing ablation data: {e}")
            metrics["status"] = "error"
            metrics["error"] = str(e)
    else:
        metrics["status"] = "not_found"
        metrics["error"] = f"No ablation results found in {ablation_dir}"
    
    return metrics

def analyze_label_shuffling_expectations(class_distribution: Dict[str, float]) -> Dict[str, float]:
    """Calculate expected metrics for label shuffling test."""
    
    # For 3-class classification with random labels
    n_classes = len(class_distribution)
    
    expected = {
        "rocauc_binary": 0.50,  # Random chance for any binary classification
        "accuracy": 1.0 / n_classes,  # Random chance for multi-class
        "f1_macro": 1.0 / n_classes,  # Each class gets ~1/n precision and recall
        "f1_micro": 1.0 / n_classes,  # Same as accuracy for balanced prediction
    }
    
    # Average precision depends on class proportions
    for class_name, proportion in class_distribution.items():
        expected[f"ap_{class_name}"] = proportion
    
    return expected

def generate_presentation_summary(all_metrics: Dict[str, Any], output_dir: str) -> str:
    """Generate a presentation-ready summary of all metrics."""
    
    summary_lines = [
        "# Meta-Model Evaluation: Presentation Summary",
        "",
        f"Generated from analysis of evaluation results",
        "",
        "## 🎯 Key Performance Metrics",
        ""
    ]
    
    # Gene CV results
    if "gene_cv" in all_metrics:
        gene_cv = all_metrics["gene_cv"]
        status_emoji = "✅" if gene_cv["status"] == "pass" else "⚠️" if gene_cv["status"] == "warning" else "❌"
        
        summary_lines.extend([
            f"### {status_emoji} Gene-Aware Cross-Validation",
            f"- **Status**: {gene_cv['status']}",
            f"- **Description**: {gene_cv['description']}"
        ])
        
        if gene_cv["status"] == "pass":
            if "donor_rocauc_mean" in gene_cv:
                donor_mean = gene_cv.get('donor_rocauc_mean', 'N/A')
                donor_std = gene_cv.get('donor_rocauc_std', 0)
                acceptor_mean = gene_cv.get('acceptor_rocauc_mean', 'N/A')
                acceptor_std = gene_cv.get('acceptor_rocauc_std', 0)
                
                # Format numbers safely
                donor_str = f"{donor_mean:.3f} ± {donor_std:.3f}" if isinstance(donor_mean, (int, float)) else str(donor_mean)
                acceptor_str = f"{acceptor_mean:.3f} ± {acceptor_std:.3f}" if isinstance(acceptor_mean, (int, float)) else str(acceptor_mean)
                
                summary_lines.extend([
                    f"- **Donor ROCAUC**: {donor_str}",
                    f"- **Acceptor ROCAUC**: {acceptor_str}",
                    f"- **CV Folds**: {gene_cv.get('n_folds', 'N/A')}"
                ])
        else:
            summary_lines.append(f"- **Error**: {gene_cv.get('error', 'Unknown error')}")
        
        summary_lines.append("")
    
    # Ablation study results
    if "ablation" in all_metrics:
        ablation = all_metrics["ablation"]
        status_emoji = "✅" if ablation["status"] == "pass" else "⚠️" if ablation["status"] == "warning" else "❌"
        
        summary_lines.extend([
            f"### {status_emoji} Ablation Study",
            f"- **Status**: {ablation['status']}",
            f"- **Description**: {ablation['description']}"
        ])
        
        if ablation["status"] == "pass":
            baseline_auc = ablation.get('baseline_rocauc', 'N/A')
            max_drop = ablation.get('max_drop', 'N/A')
            
            # Format numbers safely
            baseline_str = f"{baseline_auc:.3f}" if isinstance(baseline_auc, (int, float)) else str(baseline_auc)
            max_drop_str = f"{max_drop:.3f}" if isinstance(max_drop, (int, float)) else str(max_drop)
            
            summary_lines.extend([
                f"- **Baseline ROCAUC**: {baseline_str}",
                f"- **Max Performance Drop**: {max_drop_str}",
                f"- **Feature Sets Tested**: {len(ablation.get('performance_drops', []))}"
            ])
            
            # Show top feature importance
            if "performance_drops" in ablation:
                drops = sorted(ablation["performance_drops"], 
                             key=lambda x: x['performance_drop'], reverse=True)
                summary_lines.append("- **Most Important Features**:")
                for drop in drops[:3]:  # Top 3
                    perf_drop = drop['performance_drop']
                    rel_drop = drop['relative_drop_pct']
                    drop_str = f"{perf_drop:.3f}" if isinstance(perf_drop, (int, float)) else str(perf_drop)
                    rel_str = f"{rel_drop:.1f}%" if isinstance(rel_drop, (int, float)) else str(rel_drop)
                    summary_lines.append(f"  - {drop['feature_set']}: -{drop_str} ({rel_str})")
        else:
            summary_lines.append(f"- **Error**: {ablation.get('error', 'Unknown error')}")
        
        summary_lines.append("")
    
    # Expected label shuffling results
    summary_lines.extend([
        "### 🎲 Expected Label Shuffling Results",
        "- **ROCAUC**: ~0.50 (random chance)",
        "- **F1-Score (macro)**: ~0.33 (3-class random)",
        "- **Accuracy**: ~0.33 (3-class random)",
        "- **AP**: ~class proportion",
        "",
        "## 🔍 Diagnostic Tests Status",
        ""
    ])
    
    # Create diagnostic summary table
    diagnostics = []
    if "gene_cv" in all_metrics:
        diagnostics.append(f"| Gene-Aware CV | {all_metrics['gene_cv']['status']} | Cross-validation on independent genes |")
    if "ablation" in all_metrics:
        diagnostics.append(f"| Ablation Study | {all_metrics['ablation']['status']} | Feature importance analysis |")
    
    diagnostics.extend([
        "| Label Shuffling | pending | Sanity check with random labels |",
        "| Chromosome CV | pending | Cross-validation on independent chromosomes |"
    ])
    
    summary_lines.extend([
        "| Test | Status | Description |",
        "|------|--------|-------------|"
    ])
    summary_lines.extend(diagnostics)
    
    summary_lines.extend([
        "",
        "## 📊 Presentation Readiness",
        ""
    ])
    
    # Calculate readiness score
    ready_tests = sum(1 for test, metrics in all_metrics.items() if metrics.get("status") == "pass")
    total_tests = len(all_metrics)
    readiness = (ready_tests / total_tests) * 100 if total_tests > 0 else 0
    
    readiness_emoji = "🟢" if readiness >= 80 else "🟡" if readiness >= 60 else "🔴"
    
    summary_lines.extend([
        f"{readiness_emoji} **Overall Readiness**: {readiness:.0f}% ({ready_tests}/{total_tests} tests complete)",
        "",
        "### Next Steps:",
        "- [ ] Run label shuffling test",
        "- [ ] Run chromosome-aware CV",
        "- [ ] Create visualizations from results",
        "- [ ] Prepare slide content with actual metrics",
        "",
        "### Files for Slides:",
    ])
    
    # List relevant files found
    for test_name, test_metrics in all_metrics.items():
        if test_metrics.get("status") == "pass":
            summary_lines.append(f"- **{test_name}**: Results available for visualization")
    
    # Save summary
    summary_path = Path(output_dir) / "presentation_summary.md"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    return str(summary_path)

def main():
    """Main function to extract metrics for presentation."""
    
    parser = argparse.ArgumentParser(description="Extract metrics for presentation slides")
    parser.add_argument("--gene-cv-dir", type=str, required=True,
                       help="Directory containing gene-aware CV results")
    parser.add_argument("--ablation-dir", type=str, required=True,
                       help="Directory containing ablation study results")
    parser.add_argument("--label-shuffle-dir", type=str, default=None,
                       help="Directory containing label shuffling results (optional)")
    parser.add_argument("--chromosome-cv-dir", type=str, default=None,
                       help="Directory containing chromosome-aware CV results (optional)")
    parser.add_argument("--output-dir", type=str, default="results/presentation_metrics",
                       help="Output directory for extracted metrics")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting metrics for presentation...")
    print(f"Output directory: {output_dir}")
    
    # Extract metrics from each source
    all_metrics = {}
    
    # Gene-aware CV
    if os.path.exists(args.gene_cv_dir):
        print(f"\n📊 Analyzing gene-aware CV results: {args.gene_cv_dir}")
        gene_cv_metrics = extract_gene_cv_metrics(args.gene_cv_dir)
        all_metrics["gene_cv"] = gene_cv_metrics
    else:
        print(f"⚠️ Gene CV directory not found: {args.gene_cv_dir}")
    
    # Ablation study
    if os.path.exists(args.ablation_dir):
        print(f"\n🔬 Analyzing ablation study results: {args.ablation_dir}")
        ablation_metrics = extract_ablation_metrics(args.ablation_dir)
        all_metrics["ablation"] = ablation_metrics
    else:
        print(f"⚠️ Ablation directory not found: {args.ablation_dir}")
    
    # Label shuffling (if available)
    if args.label_shuffle_dir and os.path.exists(args.label_shuffle_dir):
        print(f"\n🎲 Analyzing label shuffling results: {args.label_shuffle_dir}")
        # Implement label shuffling extraction when available
        all_metrics["label_shuffling"] = {
            "test_type": "Label Shuffling",
            "status": "available",
            "description": "Sanity check with randomized labels"
        }
    
    # Chromosome CV (if available)
    if args.chromosome_cv_dir and os.path.exists(args.chromosome_cv_dir):
        print(f"\n🧬 Analyzing chromosome-aware CV results: {args.chromosome_cv_dir}")
        # Implement chromosome CV extraction when available
        all_metrics["chromosome_cv"] = {
            "test_type": "Chromosome-Aware CV", 
            "status": "available",
            "description": "Cross-validation on independent chromosomes"
        }
    
    # Save raw metrics
    metrics_path = output_dir / "extracted_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print(f"\n💾 Saved raw metrics to: {metrics_path}")
    
    # Generate presentation summary
    print(f"\n📋 Generating presentation summary...")
    summary_path = generate_presentation_summary(all_metrics, str(output_dir))
    print(f"📄 Saved presentation summary to: {summary_path}")
    
    # Print quick summary
    print(f"\n✅ Extraction complete!")
    print(f"Found results for {len(all_metrics)} test types:")
    for test_name, metrics in all_metrics.items():
        status_emoji = "✅" if metrics["status"] == "pass" else "⚠️" if metrics["status"] == "warning" else "❌"
        print(f"  {status_emoji} {test_name}: {metrics['status']}")

if __name__ == "__main__":
    main() 