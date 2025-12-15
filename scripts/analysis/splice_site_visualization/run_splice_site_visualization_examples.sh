#!/bin/bash

# Splice Site Visualization Examples with Gene Discovery
# This script demonstrates the complete workflow from gene discovery to visualization

echo "🧬 Splice Site Visualization Examples with Gene Discovery"
echo "=========================================================="

# Step 1: Discover suitable genes for testing
echo ""
echo "Step 1: Discovering genes suitable for splice site visualization..."
python splice_site_gene_discovery.py \
  --dataset ../../../train_pc_1000/master \
  --cv-results ../../../results/gene_cv_1000_run_15/position_level_classification_results.tsv \
  --gene-features ../../../data/ensembl/spliceai_analysis/gene_features.tsv \
  --output discovered_genes_for_visualization.tsv \
  --top-n 20 \
  --verbose

echo ""
echo "✅ Gene discovery completed! Check discovered_genes_for_visualization.tsv for results."

echo ""
echo "📊 Generating global FP/FN summary..."
python scripts/analysis/splice_site_visualization/generate_global_fp_fn_summary.py \
  discovered_genes_for_visualization.tsv \
  --output global_fp_fn_summary.tsv

# Step 2: Example 1 - Single gene analysis using discovered top gene
echo ""
echo "Step 2: Single gene analysis using top discovered gene..."
TOP_GENE=$(tail -n +2 discovered_genes_for_visualization.tsv | head -n 1 | cut -f1)
TOP_GENE_NAME=$(tail -n +2 discovered_genes_for_visualization.tsv | head -n 1 | cut -f2)
echo "Using top gene: $TOP_GENE_NAME ($TOP_GENE)"

python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
  --dataset ../../../train_pc_1000/master \
  --genes $TOP_GENE \
  --cv-results ../../../results/gene_cv_1000_run_15/position_level_classification_results.tsv \
  --gene-features ../../../data/ensembl/spliceai_analysis/gene_features.tsv \
  --output-dir ../../../results/splice_site_viz_discovered_top_gene_${TOP_GENE_NAME} \
  --threshold 0.5 \
  --verbose

# Step 3: Example 2 - Multi-gene comparison using top 3 discovered genes
echo ""
echo "Step 3: Multi-gene comparison using top 3 discovered genes..."
TOP_3_GENES=$(tail -n +2 discovered_genes_for_visualization.tsv | head -n 3 | cut -f1 | tr '\n' ',' | sed 's/,$//')
TOP_3_NAMES=$(tail -n +2 discovered_genes_for_visualization.tsv | head -n 3 | cut -f2 | tr '\n' '_' | sed 's/_$//')
echo "Using top 3 genes: $TOP_3_GENES"

python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
  --dataset ../../../train_pc_1000/master \
  --genes $TOP_3_GENES \
  --cv-results ../../../results/gene_cv_1000_run_15/position_level_classification_results.tsv \
  --gene-features ../../../data/ensembl/spliceai_analysis/gene_features.tsv \
  --output-dir ../../../results/splice_site_viz_discovered_top3_${TOP_3_NAMES} \
  --threshold 0.3 \
  --verbose

# Step 4: Example 3 - Gene with most improvements (rescued FNs + eliminated FPs)
echo ""
echo "Step 4: Analyzing gene with most meta-learning improvements..."
# Sort by total_improvements column (last column) and get the gene with most improvements
BEST_IMPROVEMENT_GENE=$(tail -n +2 discovered_genes_for_visualization.tsv | sort -t$'\t' -k15 -nr | head -n 1 | cut -f1)
BEST_IMPROVEMENT_NAME=$(tail -n +2 discovered_genes_for_visualization.tsv | sort -t$'\t' -k15 -nr | head -n 1 | cut -f2)
IMPROVEMENT_COUNT=$(tail -n +2 discovered_genes_for_visualization.tsv | sort -t$'\t' -k15 -nr | head -n 1 | cut -f15)

if [ "$IMPROVEMENT_COUNT" != "0" ]; then
    echo "Using gene with most improvements: $BEST_IMPROVEMENT_NAME ($BEST_IMPROVEMENT_GENE) - $IMPROVEMENT_COUNT improvements"
    
    python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
      --dataset ../../../train_pc_1000/master \
      --genes $BEST_IMPROVEMENT_GENE \
      --cv-results ../../../results/gene_cv_1000_run_15/position_level_classification_results.tsv \
      --gene-features ../../../data/ensembl/spliceai_analysis/gene_features.tsv \
      --output-dir ../../../results/splice_site_viz_best_improvements_${BEST_IMPROVEMENT_NAME} \
      --threshold 0.3 \
      --verbose
else
    echo "No genes with meta-learning improvements found in this sample."
fi

# Step 5: Example 4 - Test specific gene by name (user-friendly)
echo ""
echo "Step 5: Testing specific gene by name using enhanced data loader..."
echo "Demonstrating gene name resolution: HSPG2"

python -c "
import sys
sys.path.append('.')
from splice_site_enhanced_data_loader import SpliceSiteDataLoader

# Load and test specific gene
loader = SpliceSiteDataLoader(verbose=True)
loader.load_metadata(
    gene_features_path='../../../data/ensembl/spliceai_analysis/gene_features.tsv'
)

# Resolve gene name to ID
resolved = loader.resolve_gene_identifiers(['HSPG2'])
print(f'\\n🧬 Resolved HSPG2 to: {resolved[0]}')

# Get gene info
info = loader.get_gene_display_info(resolved[0])
print(f'Gene info: {info}')

# Test data loading
data = loader.load_genes(['HSPG2'], '../../../train_pc_1000/master')
print(f'\\n📊 Loaded {len(data)} positions for HSPG2')
print(f'Splice sites: {(data[\"splice_type\"] == \"donor\").sum()} donors, {(data[\"splice_type\"] == \"acceptor\").sum()} acceptors')
"

# Visualize HSPG2 using the resolved gene ID
echo ""
echo "Now visualizing HSPG2..."
python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
  --dataset ../../../train_pc_1000/master \
  --genes ENSG00000142798 \
  --cv-results ../../../results/gene_cv_1000_run_15/position_level_classification_results.tsv \
  --gene-features ../../../data/ensembl/spliceai_analysis/gene_features.tsv \
  --output-dir ../../../results/splice_site_viz_gene_name_example_HSPG2 \
  --threshold 0.5 \
  --verbose

echo ""
echo "✅ All examples completed!"
echo ""
echo "📊 Generated visualizations:"
echo "  • Top discovered gene analysis"
echo "  • Multi-gene comparison (top 3)"
echo "  • Gene with most meta-learning improvements"
echo "  • Gene name resolution example (HSPG2)"
echo ""
echo "📁 Output directories:"
ls -d ../../../results/splice_site_viz_* 2>/dev/null | head -10
echo ""
echo "🧬 Top discovered genes (from discovered_genes_for_visualization.tsv):"
echo "Gene_Name (Gene_ID): Donors, Acceptors, Improvements"
tail -n +2 discovered_genes_for_visualization.tsv | head -5 | while IFS=$'\t' read -r gene_id gene_name donors acceptors total transcripts density score fp_d fn_d fp_a fn_a rescued_d elim_d rescued_a elim_a improvements; do
    echo "  • $gene_name ($gene_id): $donors donors, $acceptors acceptors, $improvements improvements"
done 