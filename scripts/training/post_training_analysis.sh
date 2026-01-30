#!/bin/bash
# Post-Training Analysis Convenience Script
# 
# This script helps you remember and run post-training analysis tools
# after completing CV training runs.

set -e

echo "🔍 MetaSpliceAI Post-Training Analysis Tools"
echo "=============================================="
echo ""

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Available commands:"
    echo "  compare-cv <run1> <run2> [output_dir]  - Compare two CV runs"
    echo "  help                                   - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 compare-cv results/gene_cv_run_1 results/gene_cv_run_2"
    echo "  $0 compare-cv results/run_1 results/run_2 analysis/comparison"
    echo ""
    exit 1
fi

COMMAND="$1"

case $COMMAND in
    "compare-cv")
        if [ $# -lt 3 ]; then
            echo "❌ Error: compare-cv requires at least 2 run directories"
            echo "Usage: $0 compare-cv <run1> <run2> [output_dir]"
            exit 1
        fi
        
        RUN1="$2"
        RUN2="$3"
        OUTPUT_DIR="${4:-cv_comparison_results}"
        
        echo "🔍 Comparing CV runs..."
        echo "  Run 1: $RUN1"
        echo "  Run 2: $RUN2"
        echo "  Output: $OUTPUT_DIR"
        echo ""
        
        # Check if run directories exist
        if [ ! -d "$RUN1" ]; then
            echo "❌ Error: Run 1 directory not found: $RUN1"
            exit 1
        fi
        
        if [ ! -d "$RUN2" ]; then
            echo "❌ Error: Run 2 directory not found: $RUN2"
            exit 1
        fi
        
        # Run the comparison
        echo "🚀 Running CV comparison analysis..."
        python -m meta_spliceai.splice_engine.meta_models.analysis.post_training.compare_cv_runs \
            --run1 "$RUN1" \
            --run2 "$RUN2" \
            --output "$OUTPUT_DIR"
        
        echo ""
        echo "✅ Comparison complete! Check the results:"
        echo "  📊 HTML Report: $OUTPUT_DIR/cv_comparison_report.html"
        echo "  📈 Visualization: $OUTPUT_DIR/cv_comparison_visualization.png"
        echo "  📋 JSON Data: $OUTPUT_DIR/comparison_results.json"
        ;;
    
    "help")
        echo "🔍 MetaSpliceAI Post-Training Analysis Tools"
        echo "=============================================="
        echo ""
        echo "These tools help you analyze and compare CV training results."
        echo ""
        echo "Available commands:"
        echo ""
        echo "📊 compare-cv <run1> <run2> [output_dir]"
        echo "   Compare two CV runs for reproducibility assessment"
        echo "   - run1: Path to first CV run results directory"
        echo "   - run2: Path to second CV run results directory"
        echo "   - output_dir: Output directory (default: cv_comparison_results)"
        echo ""
        echo "📋 help"
        echo "   Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 compare-cv results/gene_cv_pc_1000_3mers_run_2_more_genes results/gene_cv_pc_1000_3mers_run_3"
        echo "  $0 compare-cv results/run_1 results/run_2 analysis/comparison"
        echo ""
        echo "📚 Documentation:"
        echo "  See: meta_spliceai/splice_engine/meta_models/analysis/post_training/README.md"
        ;;
    
    *)
        echo "❌ Unknown command: $COMMAND"
        echo "Run '$0 help' for available commands"
        exit 1
        ;;
esac 