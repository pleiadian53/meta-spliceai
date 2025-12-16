#!/usr/bin/env python3
"""
Demo script for splice site comparison visualization.

This script demonstrates the correct workflow:
1. Load training data (train_pc_1000/master) with hierarchical sampling
2. Find genes with good splice site representation
3. Compare base model (raw SpliceAI scores) vs meta model (improved predictions)
4. Both models operate on the SAME genes from training data
"""

import sys
from pathlib import Path
sys.path.append('.')

from meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer import SpliceSiteComparisonVisualizer

def main():
    """Main demonstration function."""
    
    print("🧬 Splice Site Comparison Visualizer Demo")
    print("=" * 60)
    print("Comparing Base Model (SpliceAI scores) vs Meta Model (improved predictions)")
    print("Both models operate on the SAME genes from training data")
    
    # Initialize visualizer
    viz = SpliceSiteComparisonVisualizer(verbose=True)
    
    # Load gene features for name mapping
    print("\n1. Loading gene features for gene name mapping...")
    gene_features = viz.load_gene_features('data/ensembl/spliceai_analysis/gene_features.tsv')
    
    # Load training data with hierarchical sampling (preserves all splice sites)
    print("\n2. Loading training data with hierarchical sampling...")
    print("   This preserves ALL splice sites while sampling complete genes")
    base_data = viz.load_dataset('train_pc_1000/master', sample_genes=100)
    
    # Show what's in the training data
    print(f"\n   Training data loaded:")
    print(f"   • Total positions: {len(base_data):,}")
    print(f"   • Total genes: {base_data['gene_id'].nunique()}")
    
    # Analyze splice site distribution
    splice_dist = base_data['splice_type'].value_counts()
    total = len(base_data)
    print(f"   • Splice site distribution:")
    for splice_type, count in splice_dist.items():
        pct = count / total * 100
        print(f"     - {splice_type}: {count:,} ({pct:.1f}%)")
    
    # Load meta model predictions (CV results contain both base and meta for same positions)
    print("\n3. Loading meta-model predictions from CV results...")
    print("   CV results contain both base and meta predictions for comparison")
    meta_data = viz.load_cv_results('results/gene_cv_1000_run_15/position_level_classification_results.tsv')
    meta_data_formatted = viz.format_cv_results_as_meta_data(meta_data)
    
    print(f"   • Meta predictions loaded: {len(meta_data_formatted):,} positions")
    print(f"   • Meta genes: {meta_data_formatted['gene_id'].nunique()}")
    
    # Find genes that exist in BOTH training data and have meta predictions
    print("\n4. Finding genes available in both datasets...")
    training_genes = set(base_data['gene_id'].unique())
    meta_genes = set(meta_data_formatted['gene_id'].unique())
    common_genes = training_genes & meta_genes
    
    print(f"   • Training data genes: {len(training_genes)}")
    print(f"   • Meta prediction genes: {len(meta_genes)}")
    print(f"   • Common genes (can be visualized): {len(common_genes)}")
    
    # Find genes with good splice site representation for visualization
    print("\n5. Finding genes with good splice site representation...")
    good_genes = []
    
    for gene_id in common_genes:
        # Get training data for this gene
        gene_subset = base_data[base_data['gene_id'] == gene_id]
        donor_count = (gene_subset['splice_type'] == 'donor').sum()
        acceptor_count = (gene_subset['splice_type'] == 'acceptor').sum()
        total_positions = len(gene_subset)
        
        # Only include genes with sufficient splice sites for good visualization
        if donor_count >= 3 and acceptor_count >= 3 and total_positions >= 50:
            gene_name = viz.get_gene_display_name(gene_id)
            good_genes.append({
                'gene_id': gene_id,
                'gene_name': gene_name,
                'donors': donor_count,
                'acceptors': acceptor_count,
                'total_positions': total_positions,
                'splice_sites': donor_count + acceptor_count
            })
    
    # Sort by total splice sites for best visualization candidates
    good_genes.sort(key=lambda x: x['splice_sites'], reverse=True)
    
    print(f"\n   Found {len(good_genes)} genes suitable for visualization:")
    for i, gene in enumerate(good_genes[:10]):
        print(f"   {i+1:2d}. {gene['gene_name']:<15} ({gene['gene_id']}): "
              f"{gene['donors']:2d} donors, {gene['acceptors']:2d} acceptors, "
              f"{gene['total_positions']:4d} total positions")
    
    if not good_genes:
        print("   ❌ No genes found with sufficient splice sites for visualization")
        return
    
    # Create output directory
    output_dir = Path('results/splice_comparison_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n6. Creating visualizations in {output_dir}...")
    
    # Example 1: Single gene detailed analysis
    best_gene = good_genes[0]
    print(f"\n   📊 Creating detailed plot for {best_gene['gene_name']}...")
    print(f"      Gene ID: {best_gene['gene_id']}")
    print(f"      Splice sites: {best_gene['donors']} donors + {best_gene['acceptors']} acceptors")
    
    # Get data for this specific gene
    gene_data_base = viz.get_gene_data(base_data, best_gene['gene_id'])
    gene_data_meta = viz.get_gene_data(meta_data_formatted, best_gene['gene_id'])
    
    print(f"      Base data positions: {len(gene_data_base)}")
    print(f"      Meta data positions: {len(gene_data_meta)}")
    
    # Create the comparison plot
    result = viz.create_gene_comparison_plot(
        gene_data_base, gene_data_meta, 
        best_gene['gene_id'], output_dir, 
        threshold=0.5
    )
    
    print(f"   ✅ Single gene plot saved: {result['plot_file']}")
    
    # Show meta-learning improvements
    changes = result['changes']
    n_rescued_donors = len(changes['rescued_donors'])
    n_rescued_acceptors = len(changes['rescued_acceptors'])
    n_eliminated_fp_donors = len(changes['eliminated_fp_donors'])
    n_eliminated_fp_acceptors = len(changes['eliminated_fp_acceptors'])
    
    total_improvements = n_rescued_donors + n_rescued_acceptors + n_eliminated_fp_donors + n_eliminated_fp_acceptors
    
    if total_improvements > 0:
        print(f"   🎯 Meta-learning improvements detected:")
        print(f"      • Rescued donor sites: {n_rescued_donors}")
        print(f"      • Rescued acceptor sites: {n_rescued_acceptors}")
        print(f"      • Eliminated FP donors: {n_eliminated_fp_donors}")
        print(f"      • Eliminated FP acceptors: {n_eliminated_fp_acceptors}")
        print(f"      • Total improvements: {total_improvements}")
    else:
        print(f"   📊 No improvements detected (may indicate good base model performance)")
    
    # Example 2: Multi-gene comparison (if we have enough good genes)
    if len(good_genes) >= 3:
        print(f"\n   📊 Creating multi-gene comparison plot...")
        
        # Select top 3 genes for side-by-side comparison
        selected_genes = [g['gene_id'] for g in good_genes[:3]]
        selected_names = [g['gene_name'] for g in good_genes[:3]]
        
        print(f"      Comparing: {', '.join(selected_names)}")
        
        result = viz.create_multi_gene_comparison(
            base_data, meta_data_formatted, 
            selected_genes, output_dir, 
            threshold=0.5
        )
        
        print(f"   ✅ Multi-gene plot saved: {result['plot_file']}")
    
    # Example 3: Generate summary report
    print(f"\n   📋 Generating summary report for top genes...")
    
    # Analyze improvements across multiple genes
    all_changes = {}
    total_genes_analyzed = 0
    
    for gene in good_genes[:10]:  # Analyze top 10 genes
        try:
            gene_data_base = viz.get_gene_data(base_data, gene['gene_id'])
            gene_data_meta = viz.get_gene_data(meta_data_formatted, gene['gene_id'])
            changes = viz.identify_splice_site_changes(gene_data_base, gene_data_meta, threshold=0.5)
            all_changes[gene['gene_id']] = changes
            total_genes_analyzed += 1
        except ValueError:
            continue  # Skip if gene not found in meta data
    
    report_file = viz.generate_summary_report(all_changes, output_dir)
    print(f"   ✅ Summary report saved: {report_file}")
    print(f"      Analyzed {total_genes_analyzed} genes")
    
    # Summary statistics
    total_rescued = sum(len(changes['rescued_donors']) + len(changes['rescued_acceptors']) 
                       for changes in all_changes.values())
    total_eliminated = sum(len(changes['eliminated_fp_donors']) + len(changes['eliminated_fp_acceptors']) 
                          for changes in all_changes.values())
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"\n📊 Overall Results:")
    print(f"   • Genes analyzed: {total_genes_analyzed}")
    print(f"   • Total rescued splice sites: {total_rescued}")
    print(f"   • Total eliminated false positives: {total_eliminated}")
    print(f"   • Output directory: {output_dir}")
    
    print(f"\n📁 Files created:")
    print(f"   • Individual gene plots: splice_comparison_*.png")
    if len(good_genes) >= 3:
        print(f"   • Multi-gene comparison: multi_gene_splice_comparison.png")
    print(f"   • Summary report: meta_learning_improvements_summary.txt")
    
    print(f"\n💡 Usage Tips:")
    print(f"   • All genes come from training data: train_pc_1000/master")
    print(f"   • Base model = raw SpliceAI scores in training data")
    print(f"   • Meta model = improved predictions using additional features")
    print(f"   • Both models operate on the SAME positions for fair comparison")
    
    return output_dir

if __name__ == "__main__":
    main() 