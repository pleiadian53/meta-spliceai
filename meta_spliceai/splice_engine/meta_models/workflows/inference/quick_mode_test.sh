#!/bin/bash

# 🧪 Quick Test for All Three Inference Modes
# 
# This script quickly tests all three inference modes (hybrid, base_only, meta_only)
# on both Scenario 1 (genes with unseen positions) and Scenario 2B (unprocessed genes)
# to validate proper integration of meta-model predictions.

set -e  # Exit on any error

echo "🧪 QUICK INFERENCE MODE TEST SUITE"
echo "=================================="

# Configuration
MODEL="results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl"
TRAINING_DATASET="train_pc_1000_3mers"
SCENARIO1_GENE="ENSG00000280739"    # Gene in training data (unseen positions)
SCENARIO2B_GENE="ENSG00000142611"   # Unseen test gene (not in training data)

# Create results directory
mkdir -p results/quick_inference_tests

echo ""
echo "📊 Test Configuration:"
echo "  Model: $MODEL"
echo "  Training Dataset: $TRAINING_DATASET"
echo "  Scenario 1 Gene (in training): $SCENARIO1_GENE"
echo "  Scenario 2B Gene (unseen test gene): $SCENARIO2B_GENE"

# Function to run a single test
run_test() {
    local test_name="$1"
    local gene="$2"
    local mode="$3"
    local output_dir="$4"
    
    echo ""
    echo "🧬 Running: $test_name"
    echo "   Gene: $gene, Mode: $mode"
    
    start_time=$(date +%s)
    
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
        --model "$MODEL" \
        --training-dataset "$TRAINING_DATASET" \
        --genes "$gene" \
        --output-dir "$output_dir" \
        --inference-mode "$mode" \
        --verbose > "$output_dir.log" 2>&1
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $? -eq 0 ]; then
        echo "   ✅ SUCCESS (${duration}s)"
        
        # Extract key metrics if available
        if [ -f "$output_dir/performance_report.txt" ]; then
            meta_usage=$(grep "Meta-model usage" "$output_dir/performance_report.txt" 2>/dev/null || echo "N/A")
            total_positions=$(grep "Total positions analyzed" "$output_dir/performance_report.txt" 2>/dev/null || echo "N/A")
            echo "   📊 $meta_usage"
            echo "   📊 $total_positions"
        fi
    else
        echo "   ❌ FAILED (${duration}s)"
        echo "   📋 Check log: $output_dir.log"
    fi
}

# Test Scenario 1: Gene with unseen positions (in training data)
echo ""
echo "🎯 SCENARIO 1: Gene with Unseen Positions ($SCENARIO1_GENE)"
echo "============================================================"

run_test "Scenario 1 - Hybrid Mode" "$SCENARIO1_GENE" "hybrid" "results/quick_inference_tests/s1_hybrid"
run_test "Scenario 1 - Base Only" "$SCENARIO1_GENE" "base_only" "results/quick_inference_tests/s1_base_only"
run_test "Scenario 1 - Meta Only" "$SCENARIO1_GENE" "meta_only" "results/quick_inference_tests/s1_meta_only"

# Test Scenario 2B: Completely unprocessed gene
echo ""
echo "🎯 SCENARIO 2B: Completely Unprocessed Gene ($SCENARIO2B_GENE)"
echo "=============================================================="

run_test "Scenario 2B - Hybrid Mode" "$SCENARIO2B_GENE" "hybrid" "results/quick_inference_tests/s2b_hybrid"
run_test "Scenario 2B - Base Only" "$SCENARIO2B_GENE" "base_only" "results/quick_inference_tests/s2b_base_only"
run_test "Scenario 2B - Meta Only" "$SCENARIO2B_GENE" "meta_only" "results/quick_inference_tests/s2b_meta_only"

# Generate summary report
echo ""
echo "📋 GENERATING SUMMARY REPORT"
echo "============================="

REPORT_FILE="results/quick_inference_tests/summary_report.txt"

cat > "$REPORT_FILE" << EOF
🧪 QUICK INFERENCE MODE TEST SUMMARY
==================================

Test Configuration:
- Model: $MODEL
- Training Dataset: $TRAINING_DATASET
- Scenario 1 Gene: $SCENARIO1_GENE (in training data)
- Scenario 2B Gene: $SCENARIO2B_GENE (unseen test gene)

Test Results:
EOF

# Check results and add to report
for result_dir in results/quick_inference_tests/*/; do
    if [ -d "$result_dir" ]; then
        test_name=$(basename "$result_dir")
        if [ -f "$result_dir/performance_report.txt" ]; then
            echo "✅ $test_name: SUCCESS" >> "$REPORT_FILE"
            
            # Extract key metrics
            success_rate=$(grep "Success rate" "$result_dir/performance_report.txt" 2>/dev/null | cut -d' ' -f3 || echo "N/A")
            meta_usage=$(grep "Meta-model usage" "$result_dir/performance_report.txt" 2>/dev/null | cut -d' ' -f3 || echo "N/A")
            total_positions=$(grep "Total positions analyzed" "$result_dir/performance_report.txt" 2>/dev/null | cut -d' ' -f4 || echo "N/A")
            
            echo "   Success Rate: $success_rate" >> "$REPORT_FILE"
            echo "   Meta-model Usage: $meta_usage" >> "$REPORT_FILE"
            echo "   Total Positions: $total_positions" >> "$REPORT_FILE"
        else
            echo "❌ $test_name: FAILED (no performance report)" >> "$REPORT_FILE"
        fi
    fi
done

cat >> "$REPORT_FILE" << EOF

Expected Validation Criteria:
=============================

✅ Mode Behavior Validation:
   - base_only: Meta-model usage = 0.0%
   - hybrid: Meta-model usage = 2-5% (selective)
   - meta_only: Meta-model usage = 100% (all positions)

✅ Scenario Validation:
   - Scenario 1: Fast processing (reuse existing artifacts)
   - Scenario 2B: Slower processing (generate new artifacts)

✅ Integration Validation:
   - All modes produce valid predictions
   - Hybrid combines base + meta appropriately
   - Default mode is hybrid

EOF

echo "📁 Summary report saved to: $REPORT_FILE"
echo ""
echo "🎉 QUICK TEST SUITE COMPLETED"
echo "============================="
echo "📊 Results available in: results/quick_inference_tests/"
echo "📋 Summary report: $REPORT_FILE"
echo ""
echo "To view results:"
echo "  cat $REPORT_FILE"
echo "  ls -la results/quick_inference_tests/"