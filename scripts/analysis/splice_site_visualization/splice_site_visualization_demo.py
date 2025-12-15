#!/usr/bin/env python3
"""
Simple usage example for splice site comparison visualization.

Adapt this script for your own analysis by changing:
- gene_ids: List of genes from your training data
- output_dir: Where to save plots
- threshold: Probability threshold for splice site detection
"""

import sys
from pathlib import Path
sys.path.append('.')

from meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer import SpliceSiteComparisonVisualizer

def simple_visualization_example():
    """Simple example showing how to create splice site visualizations."""
    
    # Initialize visualizer
    viz = SpliceSiteComparisonVisualizer(verbose=True)
    
    # Load gene features (for gene name mapping)
    viz.load_gene_features('data/ensembl/spliceai_analysis/gene_features.tsv')
    
    # Load training data with hierarchical sampling
    print("Loading training data...")
    base_data = viz.load_dataset('train_pc_1000/master', sample_genes=50)
    
    # Load meta-model predictions from CV results
    print("Loading meta-model predictions...")
    meta_data = viz.load_cv_results('results/gene_cv_1000_run_15/position_level_classification_results.tsv')
    meta_data_formatted = viz.format_cv_results_as_meta_data(meta_data)
    
    # Find available genes
    common_genes = set(base_data['gene_id'].unique()) & set(meta_data_formatted['gene_id'].unique())
    print(f"Found {len(common_genes)} genes available for visualization")
    
    # Example genes with good splice sites (replace with your genes of interest)
    gene_ids = [
        'ENSG00000205592',  # MUC19 - 170 donors, 173 acceptors
        'ENSG00000142798',  # HSPG2 - 102 donors, 104 acceptors
        'ENSG00000198947',  # DMD - 100 donors, 94 acceptors
    ]
    
    # Filter to available genes
    available_gene_ids = [g for g in gene_ids if g in common_genes]
    print(f"Analyzing genes: {available_gene_ids}")
    
    # Create output directory
    output_dir = Path('results/splice_site_viz_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    for gene_id in available_gene_ids:
        print(f"\nCreating plot for {viz.get_gene_display_name(gene_id)}...")
        
        # Get gene data
        gene_data_base = viz.get_gene_data(base_data, gene_id)
        gene_data_meta = viz.get_gene_data(meta_data_formatted, gene_id)
        
        # Create comparison plot
        result = viz.create_gene_comparison_plot(
            gene_data_base, gene_data_meta, 
            gene_id, output_dir, 
            threshold=0.5  # Adjust threshold as needed
        )
        
        print(f"   Plot saved: {result['plot_file']}")
        
        # Show improvements
        changes = result['changes']
        improvements = (len(changes['rescued_donors']) + len(changes['rescued_acceptors']) + 
                       len(changes['eliminated_fp_donors']) + len(changes['eliminated_fp_acceptors']))
        print(f"   Meta-learning improvements: {improvements}")
    
    # Create multi-gene comparison if multiple genes
    if len(available_gene_ids) > 1:
        print(f"\nCreating multi-gene comparison plot...")
        result = viz.create_multi_gene_comparison(
            base_data, meta_data_formatted, 
            available_gene_ids, output_dir, 
            threshold=0.5
        )
        print(f"Multi-gene plot saved: {result['plot_file']}")
    
    print(f"\n✅ Visualization complete! Check {output_dir} for results.")

if __name__ == "__main__":
    simple_visualization_example() 