#!/bin/bash
#
# Quick Test: Run OpenSpliceAI on Chromosome 21
#
# This script tests the chromosome selection feature on the smallest chromosome.
# Expected runtime: ~30 minutes
# Memory usage: ~2-3GB
#

set -e  # Exit on error

cd /Users/pleiadian53/work/meta-spliceai
source ~/.bash_profile
mamba activate surveyor

echo "========================================="
echo "Chromosome 21 Test - OpenSpliceAI"
echo "========================================="
echo ""
echo "This will test the base model pass on chromosome 21 only."
echo "Expected: ~227 genes, ~30 minutes runtime"
echo ""

# Run the test
python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --mode test \
    --coverage full_genome \
    --chromosomes 21 \
    --verbosity 1 \
    2>&1 | tee logs/openspliceai_chr21_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo ""
echo "Check outputs in:"
echo "  data/mane/GRCh38/openspliceai_eval/meta_models/"
echo ""

