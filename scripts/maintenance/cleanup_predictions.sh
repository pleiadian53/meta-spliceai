#!/bin/bash
#
# Cleanup Predictions Directory
#
# Issue: 12GB of predictions with duplicated artifacts (complete_base_predictions)
# Solution: Move artifacts to centralized location, remove duplicates
#
# Usage:
#   # Dry run (see what would be removed)
#   bash scripts/cleanup_predictions.sh --dry-run
#
#   # Execute cleanup
#   bash scripts/cleanup_predictions.sh
#
# Created: 2025-10-28

set -e

# Parse arguments
DRY_RUN=true
if [[ "$1" == "--execute" ]]; then
    DRY_RUN=false
fi

BASE_DIR="predictions"
ARTIFACTS_DIR="${BASE_DIR}/spliceai_eval/meta_models"

echo "================================================================================"
echo "PREDICTIONS DIRECTORY CLEANUP"
echo "================================================================================"
echo "Base directory: ${BASE_DIR}"
echo "Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN (no changes)" || echo "EXECUTE (will delete files)")"
echo ""

# Check if base directory exists
if [ ! -d "${BASE_DIR}" ]; then
    echo "❌ Predictions directory not found: ${BASE_DIR}"
    exit 1
fi

# Calculate current size
CURRENT_SIZE=$(du -sh "${BASE_DIR}" | cut -f1)
echo "Current size: ${CURRENT_SIZE}"
echo ""

echo "================================================================================"
echo "STEP 1: IDENTIFY DUPLICATED ARTIFACTS"
echo "================================================================================"
echo ""

# Find all complete_base_predictions directories
echo "Searching for complete_base_predictions directories..."
ARTIFACT_DIRS=$(find "${BASE_DIR}" -type d -name "complete_base_predictions" 2>/dev/null)

if [ -z "$ARTIFACT_DIRS" ]; then
    echo "✅ No duplicated artifact directories found"
else
    echo "Found duplicated artifact directories:"
    echo "$ARTIFACT_DIRS" | while read dir; do
        size=$(du -sh "$dir" | cut -f1)
        echo "  $dir ($size)"
    done
    
    ARTIFACT_COUNT=$(echo "$ARTIFACT_DIRS" | wc -l | tr -d ' ')
    echo ""
    echo "Total: $ARTIFACT_COUNT directories"
fi

echo ""
echo "================================================================================"
echo "STEP 2: IDENTIFY OLD TEST DIRECTORIES"
echo "================================================================================"
echo ""

# Find old test directories
OLD_TEST_DIRS=$(find "${BASE_DIR}" -maxdepth 1 -type d \( -name "*test*" -o -name "*diverse*" \) 2>/dev/null)

if [ -z "$OLD_TEST_DIRS" ]; then
    echo "✅ No old test directories found"
else
    echo "Found old test directories:"
    echo "$OLD_TEST_DIRS" | while read dir; do
        size=$(du -sh "$dir" | cut -f1)
        echo "  $dir ($size)"
    done
    
    OLD_TEST_COUNT=$(echo "$OLD_TEST_DIRS" | wc -l | tr -d ' ')
    echo ""
    echo "Total: $OLD_TEST_COUNT directories"
fi

echo ""
echo "================================================================================"
echo "STEP 3: CONSOLIDATE ARTIFACTS"
echo "================================================================================"
echo ""

# Create centralized artifacts directory
if [ "$DRY_RUN" = false ]; then
    mkdir -p "${ARTIFACTS_DIR}/complete_base_predictions"
    mkdir -p "${ARTIFACTS_DIR}/analysis_sequences"
    echo "✅ Created centralized artifacts directory: ${ARTIFACTS_DIR}"
else
    echo "[DRY RUN] Would create: ${ARTIFACTS_DIR}"
fi

# Move unique artifacts to centralized location
if [ -n "$ARTIFACT_DIRS" ]; then
    echo ""
    echo "Consolidating artifacts..."
    
    CONSOLIDATED=0
    echo "$ARTIFACT_DIRS" | while read dir; do
        if [ "$DRY_RUN" = false ]; then
            # Copy unique files to centralized location
            find "$dir" -type f -name "*.parquet" 2>/dev/null | while read file; do
                filename=$(basename "$file")
                target="${ARTIFACTS_DIR}/complete_base_predictions/${filename}"
                
                if [ ! -f "$target" ]; then
                    cp "$file" "$target"
                    echo "  ✅ Copied: $filename"
                fi
            done
            CONSOLIDATED=$((CONSOLIDATED + 1))
        else
            echo "  [DRY RUN] Would consolidate: $dir"
        fi
    done
    
    echo ""
    echo "Consolidated artifacts from $ARTIFACT_COUNT directories"
else
    echo "No artifacts to consolidate"
fi

echo ""
echo "================================================================================"
echo "STEP 4: REMOVE DUPLICATES"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = false ]; then
    # Remove duplicated artifact directories
    if [ -n "$ARTIFACT_DIRS" ]; then
        echo "Removing duplicated artifact directories..."
        echo "$ARTIFACT_DIRS" | while read dir; do
            rm -rf "$dir"
            echo "  ✅ Removed: $dir"
        done
    fi
    
    # Remove old test directories
    if [ -n "$OLD_TEST_DIRS" ]; then
        echo ""
        echo "Removing old test directories..."
        echo "$OLD_TEST_DIRS" | while read dir; do
            rm -rf "$dir"
            echo "  ✅ Removed: $dir"
        done
    fi
else
    if [ -n "$ARTIFACT_DIRS" ]; then
        echo "[DRY RUN] Would remove duplicated artifacts:"
        echo "$ARTIFACT_DIRS" | while read dir; do
            size=$(du -sh "$dir" | cut -f1)
            echo "  $dir ($size)"
        done
    fi
    
    if [ -n "$OLD_TEST_DIRS" ]; then
        echo ""
        echo "[DRY RUN] Would remove old test directories:"
        echo "$OLD_TEST_DIRS" | while read dir; do
            size=$(du -sh "$dir" | cut -f1)
            echo "  $dir ($size)"
        done
    fi
fi

echo ""
echo "================================================================================"
echo "STEP 5: SUMMARY"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes were made"
    echo ""
    echo "To execute cleanup, run:"
    echo "  bash scripts/cleanup_predictions.sh --execute"
else
    # Calculate new size
    NEW_SIZE=$(du -sh "${BASE_DIR}" | cut -f1)
    
    echo "Cleanup complete!"
    echo ""
    echo "Before: ${CURRENT_SIZE}"
    echo "After:  ${NEW_SIZE}"
    echo ""
    echo "✅ Artifacts centralized at: ${ARTIFACTS_DIR}"
    echo "✅ Duplicates removed"
    echo "✅ Old test directories removed"
fi

echo ""
echo "================================================================================"

