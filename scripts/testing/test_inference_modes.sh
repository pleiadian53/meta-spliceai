#!/bin/bash
#
# Systematic Inference Workflow Testing
# Tests all 3 operational modes (base-only, hybrid, meta-only) on 2 scenarios
# 
# Scenarios:
#   1. Training genes (genes in training data)
#   2. Unseen genes (genes NOT in training data)
#
# Modes:
#   - base_only: SpliceAI predictions only
#   - hybrid: SpliceAI + selective meta-model for uncertain positions
#   - meta_only: Meta-model recalibration for all positions
#

set -e  # Exit on error

# Configuration
MODEL_PATH="results/meta_model_1000genes_3mers/model_multiclass.pkl"
TRAINING_DATASET="data/train_pc_1000_3mers"
GENE_LISTS_DIR="gene_lists"
RESULTS_BASE="results/inference_tests"
LOGS_DIR="logs/inference_tests"

# Create directories
mkdir -p "${LOGS_DIR}"
mkdir -p "${RESULTS_BASE}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║   🧪 SYSTEMATIC INFERENCE WORKFLOW TESTING                         ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Configuration:"
echo "   Model: ${MODEL_PATH}"
echo "   Training dataset: ${TRAINING_DATASET}"
echo "   Gene lists: ${GENE_LISTS_DIR}/"
echo "   Results: ${RESULTS_BASE}/"
echo "   Logs: ${LOGS_DIR}/"
echo ""

# Function to run a test
run_test() {
    local scenario=$1
    local mode=$2
    local gene_file=$3
    local test_name="${scenario}_${mode}"
    local output_dir="${RESULTS_BASE}/${test_name}"
    local log_file="${LOGS_DIR}/${test_name}.log"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}🧬 Test: ${test_name}${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Scenario: ${scenario}"
    echo "   Mode: ${mode}"
    echo "   Genes: $(cat ${gene_file} | wc -l | tr -d ' ') genes from ${gene_file}"
    echo "   Output: ${output_dir}"
    echo "   Log: ${log_file}"
    echo ""
    
    # Show genes being tested
    echo "   Testing genes:"
    cat "${gene_file}" | while read gene; do
        echo "     - ${gene}"
    done
    echo ""
    
    # Run inference workflow
    echo -e "${YELLOW}⏳ Running inference workflow...${NC}"
    start_time=$(date +%s)
    
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
        --model "${MODEL_PATH}" \
        --training-dataset "${TRAINING_DATASET}" \
        --genes-file "${gene_file}" \
        --output-dir "${output_dir}" \
        --inference-mode "${mode}" \
        --verbose \
        2>&1 | tee "${log_file}"
    
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Test completed successfully in ${duration}s${NC}"
        
        # Check for key indicators
        echo ""
        echo "📊 Validation checks:"
        
        # Check for incremental processing
        if grep -q "INCREMENTAL" "${log_file}"; then
            echo -e "   ${GREEN}✅ Incremental processing confirmed${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Incremental processing not detected${NC}"
        fi
        
        # Check for memory monitoring
        if grep -q "Peak memory" "${log_file}"; then
            peak_mem=$(grep "Peak memory" "${log_file}" | tail -1 | awk '{print $4, $5}')
            echo -e "   ${GREEN}✅ Memory monitoring active: ${peak_mem}${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Memory monitoring not detected${NC}"
        fi
        
        # Check for per-gene processing
        gene_count=$(grep "Gene [0-9]" "${log_file}" | wc -l | tr -d ' ')
        echo -e "   ${GREEN}✅ Per-gene processing: ${gene_count} genes${NC}"
        
        # Check output files
        if [ -d "${output_dir}/per_gene" ]; then
            per_gene_files=$(ls "${output_dir}/per_gene"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
            echo -e "   ${GREEN}✅ Per-gene files: ${per_gene_files} files${NC}"
        fi
        
        if [ -f "${output_dir}/combined_predictions.parquet" ]; then
            echo -e "   ${GREEN}✅ Combined predictions file created${NC}"
        fi
        
    else
        echo ""
        echo -e "${YELLOW}❌ Test failed with exit code ${exit_code}${NC}"
        echo "   Check log file for details: ${log_file}"
        return 1
    fi
}

# ============================================================================
# Phase 1: Base-Only Mode (SpliceAI predictions only)
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║   PHASE 1: BASE-ONLY MODE                                          ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

# Test 1a: Training genes, base-only
run_test "training" "base_only" "${GENE_LISTS_DIR}/training_genes.txt"

# Test 1b: Unseen genes, base-only
run_test "unseen" "base_only" "${GENE_LISTS_DIR}/unseen_genes.txt"

# ============================================================================
# Phase 2: Hybrid Mode (Selective meta-model)
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║   PHASE 2: HYBRID MODE                                             ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

# Test 2a: Training genes, hybrid
run_test "training" "hybrid" "${GENE_LISTS_DIR}/training_genes.txt"

# Test 2b: Unseen genes, hybrid
run_test "unseen" "hybrid" "${GENE_LISTS_DIR}/unseen_genes.txt"

# ============================================================================
# Phase 3: Meta-Only Mode (Full meta-model recalibration)
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║   PHASE 3: META-ONLY MODE                                          ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

# Test 3a: Training genes, meta-only
run_test "training" "meta_only" "${GENE_LISTS_DIR}/training_genes.txt"

# Test 3b: Unseen genes, meta-only
run_test "unseen" "meta_only" "${GENE_LISTS_DIR}/unseen_genes.txt"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║   🎉 ALL TESTS COMPLETE                                            ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Test Summary:"
echo "   Total tests run: 6 (2 scenarios × 3 modes)"
echo "   Results directory: ${RESULTS_BASE}/"
echo "   Logs directory: ${LOGS_DIR}/"
echo ""
echo "📁 Generated outputs:"
ls -lh "${RESULTS_BASE}" | tail -n +2
echo ""
echo "💡 Next steps:"
echo "   1. Review logs in ${LOGS_DIR}/"
echo "   2. Analyze results in ${RESULTS_BASE}/"
echo "   3. Compare performance across modes and scenarios"
echo ""

