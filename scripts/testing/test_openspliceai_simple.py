#!/usr/bin/env python
"""
Simple OpenSpliceAI Test - Quick validation with 5 genes
"""

import sys
sys.path.insert(0, '/Users/pleiadian53/work/meta-spliceai')

from meta_spliceai import run_base_model_predictions
from datetime import datetime

print("=" * 80)
print("OPENSPLICEAI SIMPLE TEST")
print("=" * 80)
print()

# Test with just 5 well-known genes
TEST_GENES = ['BRCA1', 'TP53', 'EGFR', 'MYC', 'KRAS']

print(f"Testing {len(TEST_GENES)} genes: {', '.join(TEST_GENES)}")
print()

# Run predictions
try:
    results = run_base_model_predictions(
        base_model='openspliceai',
        target_genes=TEST_GENES,
        do_extract_sequences=True,  # Force on-the-fly extraction
        test_mode=True,  # Use test mode for quick iteration
        verbosity=2
    )
    
    print("\n" + "=" * 80)
    print("✅ TEST SUCCESSFUL!")
    print("=" * 80)
    print(f"\nResults: {results}")
    
except Exception as e:
    print("\n" + "=" * 80)
    print("❌ TEST FAILED!")
    print("=" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

