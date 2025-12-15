#!/bin/bash
# Monitor the TN sampling fix test

cd /Users/pleiadian53/work/meta-spliceai

NEWEST_LOG=$(ls -t logs/openspliceai_chr21_tn_fix_test_*.log 2>/dev/null | head -1)

if [ -z "$NEWEST_LOG" ]; then
    echo "❌ No log file found"
    exit 1
fi

echo "=== Monitoring: $NEWEST_LOG ==="
echo ""

# Check if process is still running
if ps aux | grep -v grep | grep "run_full_genome.*21" > /dev/null; then
    echo "✅ Process is running"
    
    # Show progress
    echo ""
    echo "=== Progress ==="
    grep -E "Processing gene|/214" "$NEWEST_LOG" | tail -5
    
    # Show TN sampling messages
    echo ""
    echo "=== TN Sampling Activity ===" 
    grep -E "tn_sampling|TN sampling|Collected.*TN positions" "$NEWEST_LOG" | tail -10
    
    # Show any errors
    echo ""
    echo "=== Recent Errors (if any) ==="
    grep -i "error\|exception\|failed" "$NEWEST_LOG" | tail -5
    
    # Check file size of output
    echo ""
    echo "=== Output File Size ==="
    if [ -f "data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv" ]; then
        ls -lh data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv | awk '{print "Size: " $5 " (" $9 ")"}'
        wc -l data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv | awk '{print "Rows: " $1}'
    else
        echo "Output file not created yet"
    fi
else
    echo "❌ Process has stopped"
    echo ""
    echo "=== Final Output ==="
    tail -50 "$NEWEST_LOG"
fi


