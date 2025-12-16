#!/bin/bash

# 🧬 Enhanced Test Gene Finder
# 
# Flexible wrapper to identify test genes for all three inference scenarios
# with support for arbitrary gene counts and user-friendly display.

# Default values
SCENARIO1_COUNT=6
SCENARIO2A_COUNT=8
SCENARIO2B_COUNT=8
OUTPUT_FILE="$(dirname "$0")/test_genes.json"
VERBOSE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario1-count)
            SCENARIO1_COUNT="$2"
            shift 2
            ;;
        --scenario2a-count)
            SCENARIO2A_COUNT="$2"
            shift 2
            ;;
        --scenario2b-count)
            SCENARIO2B_COUNT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            echo "🧬 Enhanced Test Gene Finder"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --scenario1-count N    Number of Scenario 1 genes (default: 6)"
            echo "  --scenario2a-count N   Number of Scenario 2A genes (default: 8)"
            echo "  --scenario2b-count N   Number of Scenario 2B genes (default: 8)"
            echo "  --output FILE          Output JSON file (default: test_genes.json)"
            echo "  --verbose, -v          Enable verbose output"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use defaults (6, 8, 8)"
            echo "  $0 --scenario2b-count 20             # Focus on unseen genes"
            echo "  $0 --scenario1-count 10 --scenario2a-count 15 --scenario2b-count 25"
            echo "  $0 --output large_test_set.json --verbose"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🧬 FINDING TEST GENES FOR INFERENCE SCENARIOS"
echo "=============================================="
echo "Requested counts: Scenario1=$SCENARIO1_COUNT, Scenario2A=$SCENARIO2A_COUNT, Scenario2B=$SCENARIO2B_COUNT"
echo ""

# Run the gene identification utility
python "$(dirname "$0")/identify_test_genes.py" \
    --scenario1-count "$SCENARIO1_COUNT" \
    --scenario2a-count "$SCENARIO2A_COUNT" \
    --scenario2b-count "$SCENARIO2B_COUNT" \
    --output "$OUTPUT_FILE" \
    $VERBOSE

echo ""
echo "📋 QUICK REFERENCE COMMANDS"
echo "=========================="

# Parse the JSON output to create quick reference commands
if [ -f "$OUTPUT_FILE" ]; then
    python3 -c "
import json
import sys
from pathlib import Path

# Load results
json_file = Path('$OUTPUT_FILE')
if json_file.exists():
    with open(json_file) as f:
        results = json.load(f)
    
    # Extract genes from each scenario
    scenarios = {
        'scenario1': 'Genes in training (unseen positions)',
        'scenario2a': 'Unseen genes with artifacts', 
        'scenario2b': 'Unseen genes without artifacts'
    }
    
    for scenario_key, description in scenarios.items():
        if scenario_key in results and results[scenario_key]['genes']:
            all_genes = [g['gene_id'] for g in results[scenario_key]['genes']]
            count = len(all_genes)
            
            # Show first 3 for export, but mention total count
            export_genes = all_genes[:3]
            genes_str = ','.join(export_genes)
            
            print(f'# {description} ({count} genes total)')
            print(f'export TEST_{scenario_key.upper()}_GENES=\"{genes_str}\"')
            
            # If more than 3 genes, show how to get all
            if count > 3:
                all_genes_str = ','.join(all_genes)
                print(f'export TEST_{scenario_key.upper()}_ALL_GENES=\"{all_genes_str}\"')
            
            print()
            print(f'# Quick test command for {scenario_key} (first 3 genes):')
            print(f'python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\\\')
            print(f'    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\\\')
            print(f'    --training-dataset train_pc_1000_3mers \\\\')
            print(f'    --genes \$TEST_{scenario_key.upper()}_GENES \\\\')
            print(f'    --output-dir results/test_{scenario_key} \\\\')
            print(f'    --inference-mode hybrid \\\\')
            print(f'    --verbose')
            
            if count > 3:
                print()
                print(f'# Test all {count} {scenario_key} genes:')
                print(f'python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\\\')
                print(f'    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\\\')
                print(f'    --training-dataset train_pc_1000_3mers \\\\')
                print(f'    --genes \$TEST_{scenario_key.upper()}_ALL_GENES \\\\')
                print(f'    --output-dir results/test_{scenario_key}_all \\\\')
                print(f'    --inference-mode hybrid \\\\')
                print(f'    --enable-chunked-processing \\\\')
                print(f'    --verbose')
            
            print()
else:
    print('Error: JSON file not found')
"
else
    echo "❌ Error: Could not generate test genes"
fi

echo "📁 Full results saved to: $OUTPUT_FILE"
echo "🔧 For more options, run: python $(dirname "$0")/identify_test_genes.py --help"
echo "🔧 For help with this script, run: $0 --help"