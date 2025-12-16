#!/usr/bin/env python3
"""
Quick Metrics Extraction for Presentation

Simple script to extract key metrics from your specific result formats.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.quick_metrics_extract
"""

import pandas as pd
import json
from pathlib import Path

def extract_gene_cv_metrics(cv_dir="results/gene_cv_1000_run_15"):
    """Extract gene CV metrics from the CSV file."""
    csv_path = Path(cv_dir) / "gene_cv_metrics.csv"
    
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Calculate statistics for meta-model performance
        results = {
            "test_type": "Gene-Aware Cross-Validation",
            "n_folds": len(df),
            
            # Meta-model AUC (assuming auc_meta is the meta-model performance)
            "meta_auc_mean": df['auc_meta'].mean(),
            "meta_auc_std": df['auc_meta'].std(),
            "meta_auc_min": df['auc_meta'].min(),
            "meta_auc_max": df['auc_meta'].max(),
            
            # Base model AUC for comparison
            "base_auc_mean": df['auc_base'].mean(),
            "base_auc_std": df['auc_base'].std(),
            
            # Improvement
            "auc_improvement": df['auc_meta'].mean() - df['auc_base'].mean(),
            
            # Average Precision
            "meta_ap_mean": df['ap_meta'].mean(),
            "meta_ap_std": df['ap_meta'].std(),
            "base_ap_mean": df['ap_base'].mean(),
            "ap_improvement": df['ap_meta'].mean() - df['ap_base'].mean(),
            
            # F1 scores
            "meta_f1_mean": df['meta_f1'].mean(),
            "meta_f1_std": df['meta_f1'].std(),
            "base_f1_mean": df['base_f1'].mean(),
            "f1_improvement": df['meta_f1'].mean() - df['base_f1'].mean(),
            
            # Accuracy
            "test_accuracy_mean": df['test_accuracy'].mean(),
            "test_accuracy_std": df['test_accuracy'].std(),
            
            "status": "success"
        }
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

def extract_ablation_metrics(ablation_dir="results/enhanced_ablation"):
    """Extract ablation study metrics from the CSV file."""
    csv_path = Path(ablation_dir) / "ablation_summary.csv"
    
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Find baseline (full model) performance
        full_row = df[df['mode'] == 'full']
        
        if len(full_row) == 0:
            return {"error": "Could not find 'full' baseline in ablation results"}
        
        baseline = full_row.iloc[0]
        
        # Calculate performance drops for each ablated model
        ablated_models = df[df['mode'] != 'full']
        
        performance_drops = []
        for _, row in ablated_models.iterrows():
            auc_drop = baseline['binary_roc_auc'] - row['binary_roc_auc']
            f1_drop = baseline['macro_f1'] - row['macro_f1']
            acc_drop = baseline['accuracy'] - row['accuracy']
            
            performance_drops.append({
                'ablation_mode': row['mode'],
                'auc_drop': auc_drop,
                'f1_drop': f1_drop,
                'accuracy_drop': acc_drop,
                'relative_auc_drop_pct': (auc_drop / baseline['binary_roc_auc']) * 100,
                'relative_f1_drop_pct': (f1_drop / baseline['macro_f1']) * 100
            })
        
        # Sort by AUC drop (most important features first)
        performance_drops.sort(key=lambda x: x['auc_drop'], reverse=True)
        
        results = {
            "test_type": "Ablation Study",
            "baseline_auc": baseline['binary_roc_auc'],
            "baseline_f1": baseline['macro_f1'],
            "baseline_accuracy": baseline['accuracy'],
            "baseline_n_features": baseline['n_features'],
            
            "performance_drops": performance_drops,
            "max_auc_drop": max([p['auc_drop'] for p in performance_drops]),
            "max_f1_drop": max([p['f1_drop'] for p in performance_drops]),
            
            "status": "success"
        }
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

def expected_shuffled_metrics():
    """Expected performance with shuffled labels."""
    return {
        "rocauc": 0.50,
        "f1_macro": 0.33,  # 1/3 for 3-class problem
        "accuracy": 0.33,
        "explanation": "With truly random labels, performance should drop to chance levels"
    }

def generate_presentation_summary():
    """Generate presentation summary with extracted metrics."""
    
    print("🎯 Extracting Meta-Model Metrics for Presentation")
    print("=" * 60)
    
    # Extract Gene CV metrics
    print("\n📊 Gene-Aware Cross-Validation...")
    gene_cv = extract_gene_cv_metrics()
    
    if gene_cv.get("status") == "success":
        print(f"✅ Successfully extracted gene CV metrics")
        print(f"   • Meta AUC: {gene_cv['meta_auc_mean']:.3f} ± {gene_cv['meta_auc_std']:.3f}")
        print(f"   • Base AUC: {gene_cv['base_auc_mean']:.3f} ± {gene_cv['base_auc_std']:.3f}")
        print(f"   • AUC Improvement: +{gene_cv['auc_improvement']:.3f}")
        print(f"   • Meta F1: {gene_cv['meta_f1_mean']:.3f} ± {gene_cv['meta_f1_std']:.3f}")
        print(f"   • F1 Improvement: +{gene_cv['f1_improvement']:.3f}")
        print(f"   • CV Folds: {gene_cv['n_folds']}")
    else:
        print(f"❌ Error: {gene_cv.get('error')}")
    
    # Extract Ablation metrics
    print("\n🔬 Ablation Study...")
    ablation = extract_ablation_metrics()
    
    if ablation.get("status") == "success":
        print(f"✅ Successfully extracted ablation metrics")
        print(f"   • Baseline AUC: {ablation['baseline_auc']:.3f}")
        print(f"   • Baseline F1: {ablation['baseline_f1']:.3f}")
        print(f"   • Max AUC Drop: {ablation['max_auc_drop']:.3f}")
        print(f"   • Feature Sets Tested: {len(ablation['performance_drops'])}")
        print(f"   • Most Important Ablation:")
        for i, drop in enumerate(ablation['performance_drops'][:3]):
            print(f"     {i+1}. {drop['ablation_mode']}: -{drop['auc_drop']:.3f} AUC ({drop['relative_auc_drop_pct']:.1f}%)")
    else:
        print(f"❌ Error: {ablation.get('error')}")
    
    # Expected shuffled results
    print("\n🎲 Expected Label Shuffling Results...")
    shuffled = expected_shuffled_metrics()
    print(f"   • Expected AUC: {shuffled['rocauc']:.2f} (random chance)")
    print(f"   • Expected F1: {shuffled['f1_macro']:.2f} (3-class random)")
    print(f"   • Expected Accuracy: {shuffled['accuracy']:.2f}")
    
    # Save detailed results
    output_dir = Path("results/presentation_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "gene_cv": gene_cv,
        "ablation": ablation,
        "expected_shuffled": shuffled,
        "summary": {
            "gene_cv_ready": gene_cv.get("status") == "success",
            "ablation_ready": ablation.get("status") == "success",
            "presentation_readiness": "🟢 Ready" if all(
                results.get("status") == "success" 
                for results in [gene_cv, ablation]
            ) else "🟡 Partial"
        }
    }
    
    # Save JSON (convert numpy types to native Python types)
    json_path = output_dir / "extracted_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create presentation-ready summary
    summary_lines = [
        "# 🎯 Meta-Model Presentation Metrics",
        "",
        "## Key Performance Results",
        ""
    ]
    
    if gene_cv.get("status") == "success":
        summary_lines.extend([
            "### ✅ Gene-Aware Cross-Validation",
            f"- **Meta-Model AUC**: {gene_cv['meta_auc_mean']:.3f} ± {gene_cv['meta_auc_std']:.3f}",
            f"- **Baseline AUC**: {gene_cv['base_auc_mean']:.3f} ± {gene_cv['base_auc_std']:.3f}",
            f"- **AUC Improvement**: +{gene_cv['auc_improvement']:.3f} ({(gene_cv['auc_improvement']/gene_cv['base_auc_mean']*100):.1f}%)",
            f"- **Meta-Model F1**: {gene_cv['meta_f1_mean']:.3f} ± {gene_cv['meta_f1_std']:.3f}",
            f"- **F1 Improvement**: +{gene_cv['f1_improvement']:.3f}",
            f"- **Cross-Validation Folds**: {gene_cv['n_folds']}",
            f"- **Consistency**: AUC range {gene_cv['meta_auc_min']:.3f} - {gene_cv['meta_auc_max']:.3f}",
            ""
        ])
    
    if ablation.get("status") == "success":
        summary_lines.extend([
            "### ✅ Ablation Study",
            f"- **Full Model AUC**: {ablation['baseline_auc']:.3f}",
            f"- **Full Model F1**: {ablation['baseline_f1']:.3f}",
            f"- **Total Features**: {ablation['baseline_n_features']:,}",
            f"- **Ablation Tests**: {len(ablation['performance_drops'])} feature sets",
            "",
            "**Feature Importance Ranking**:",
        ])
        
        for i, drop in enumerate(ablation['performance_drops']):
            summary_lines.append(
                f"{i+1}. **{drop['ablation_mode']}**: -{drop['auc_drop']:.3f} AUC "
                f"({drop['relative_auc_drop_pct']:.1f}% relative drop)"
            )
        summary_lines.append("")
    
    summary_lines.extend([
        "### 🎲 Diagnostic Test Expectations",
        "",
        "**Label Shuffling Test** (when run):",
        f"- Expected AUC: ~{shuffled['rocauc']:.2f} (random chance)",
        f"- Expected F1: ~{shuffled['f1_macro']:.2f} (3-class random)",
        f"- Expected Accuracy: ~{shuffled['accuracy']:.2f}",
        "",
        "## 📊 Slide-Ready Metrics",
        "",
        "### For Performance Slides:",
    ])
    
    if gene_cv.get("status") == "success":
        summary_lines.extend([
            f"- Meta-model achieves **{gene_cv['meta_auc_mean']:.1%}** AUC",
            f"- **{(gene_cv['auc_improvement']/gene_cv['base_auc_mean']*100):.1f}%** improvement over baseline",
            f"- Consistent across **{gene_cv['n_folds']} independent gene sets**",
        ])
    
    if ablation.get("status") == "success":
        summary_lines.extend([
            f"- All feature types contribute (**max drop: {ablation['max_auc_drop']:.1%}**)",
            f"- Most critical: **{ablation['performance_drops'][0]['ablation_mode']}** features",
        ])
    
    summary_lines.extend([
        "",
        "### For Validation Slides:",
        "- Systematic diagnostic testing framework",
        "- Gene-independent cross-validation ✅",
        "- Feature ablation analysis ✅", 
        "- Label shuffling test (pending)",
        "- Chromosome-aware CV (pending)",
        "",
        f"📁 **Detailed results saved to**: `{json_path}`"
    ])
    
    # Save markdown summary
    md_path = output_dir / "presentation_summary.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n💾 Detailed results saved to: {json_path}")
    print(f"📄 Presentation summary saved to: {md_path}")
    
    # Overall status
    if all(results.get("status") == "success" for results in [gene_cv, ablation]):
        print(f"\n🎉 Ready for presentation! Both CV and ablation results extracted successfully.")
    else:
        print(f"\n⚠️  Partial results - some tests need to be completed or files are missing.")
    
    return all_results

if __name__ == "__main__":
    results = generate_presentation_summary() 