#!/bin/bash

# Monitor meta-model training progress
# Usage: bash scripts/testing/monitor_training_progress.sh

LOG_FILE="logs/meta_training_1000genes_fresh.log"
OUTPUT_DIR="results/meta_model_1000genes_3mers_fresh"

clear
echo "════════════════════════════════════════════════════════════════"
echo "📊 META-MODEL TRAINING PROGRESS MONITOR"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   Training may not have started yet."
    exit 1
fi

# Get file size
LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
echo "📄 Log file: $LOG_FILE (Size: $LOG_SIZE)"
echo ""

# Check training stage
echo "🔍 Current Training Stage:"
echo "─────────────────────────────────────────────────────────────"

if grep -q "Gene-CV-Sigmoid] Fold" "$LOG_FILE" 2>/dev/null; then
    # Extract fold information
    LAST_FOLD=$(grep "Gene-CV-Sigmoid] Fold" "$LOG_FILE" | tail -1)
    echo "✅ Cross-Validation: $LAST_FOLD"
    
    # Count completed folds
    COMPLETED_FOLDS=$(grep -c "Gene-CV-Sigmoid] Fold" "$LOG_FILE")
    echo "   Completed folds: $COMPLETED_FOLDS/5"
elif grep -q "Global Feature Screening" "$LOG_FILE" 2>/dev/null; then
    echo "⏳ Feature Screening (preprocessing phase)"
elif grep -q "Dataset preparation completed" "$LOG_FILE" 2>/dev/null; then
    echo "⏳ Preparing for training..."
else
    echo "⏳ Initializing..."
fi

echo ""

# Check for key milestones
echo "📋 Training Milestones:"
echo "─────────────────────────────────────────────────────────────"

grep -q "Dataset preparation completed" "$LOG_FILE" && echo "✅ Dataset loaded: 99,858 positions from 543 genes" || echo "⏳ Loading dataset..."
grep -q "Features: 131" "$LOG_FILE" && echo "✅ Features: 131 (including 64 k-mers)" || echo "⏳ Preparing features..."
grep -q "Global Feature Screening" "$LOG_FILE" && echo "✅ Feature screening started" || echo "⏳ Feature screening..."

# Check if CV started
if grep -q "Running.*fold.*cross-validation" "$LOG_FILE" 2>/dev/null; then
    echo "✅ Cross-validation started"
    
    # Check individual fold completion
    for i in {1..5}; do
        if grep -q "Fold $i/5" "$LOG_FILE" 2>/dev/null; then
            echo "   ✅ Fold $i/5 completed"
        fi
    done
fi

# Check for model saving
grep -q "Production model training" "$LOG_FILE" && echo "✅ Production model training started" || echo "⏳ Production model training..."
grep -q "model_multiclass.pkl" "$LOG_FILE" && echo "✅ Model saved" || echo "⏳ Model saving..."

echo ""

# Check for errors
echo "⚠️  Errors/Warnings:"
echo "─────────────────────────────────────────────────────────────"
ERROR_COUNT=$(grep -i "error" "$LOG_FILE" 2>/dev/null | grep -v "error_artifact" | wc -l | tr -d ' ')
WARNING_COUNT=$(grep -i "warning" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Errors detected: $ERROR_COUNT"
    echo "   Last error:"
    grep -i "error" "$LOG_FILE" | grep -v "error_artifact" | tail -1
else
    echo "✅ No errors detected"
fi

if [ "$WARNING_COUNT" -gt 0 ]; then
    echo "⚠️  Warnings: $WARNING_COUNT (this is usually normal)"
else
    echo "✅ No warnings"
fi

echo ""

# Show recent output
echo "📝 Recent Output (last 20 lines):"
echo "─────────────────────────────────────────────────────────────"
tail -20 "$LOG_FILE" | grep -v "Fontconfig\|matplotlib\|pkg_resources" || tail -20 "$LOG_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "⏱️  Training typically takes 2-4 hours"
echo "🔄 Run this script again to check progress"
echo "📄 Full log: $LOG_FILE"
echo "📂 Output: $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════"

