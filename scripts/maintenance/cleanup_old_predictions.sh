#!/bin/bash
#
# Cleanup old timestamped inference predictions to reclaim disk space.
#
# This script helps manage disk space by removing old timestamped prediction runs.
# After switching to mode-based directories (base_only, hybrid, meta_only),
# the old timestamped directories can be safely removed.
#
# Usage:
#   ./scripts/maintenance/cleanup_old_predictions.sh [--dry-run]
#
# Options:
#   --dry-run    Show what would be deleted without actually deleting
#

set -euo pipefail

PREDICTIONS_DIR="data/ensembl/spliceai_eval/meta_models/inference/predictions"
DRY_RUN=false

# Parse arguments
if [[ $# -gt 0 ]] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - No files will be deleted"
    echo
fi

# Check if predictions directory exists
if [[ ! -d "$PREDICTIONS_DIR" ]]; then
    echo "Predictions directory not found: $PREDICTIONS_DIR"
    exit 1
fi

echo "================================================================"
echo "CLEANUP OLD TIMESTAMPED INFERENCE PREDICTIONS"
echo "================================================================"
echo

# Show current disk usage
echo "Current disk usage:"
du -sh "$PREDICTIONS_DIR"
echo

# Find timestamped directories (pattern: *_YYYYMMDD_HHMMSS or enhanced_selective_inference_*)
echo "Looking for old timestamped directories..."
echo

TIMESTAMPED_DIRS=()
while IFS= read -r -d '' dir; do
    basename_dir=$(basename "$dir")
    # Match patterns like:
    # - enhanced_selective_inference_20251023_222311
    # - base_only_20251023_222311
    # - hybrid_20251023_222311
    # - meta_only_20251023_222311
    if [[ "$basename_dir" =~ _[0-9]{8}_[0-9]{6}$ ]]; then
        TIMESTAMPED_DIRS+=("$dir")
    fi
done < <(find "$PREDICTIONS_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

if [[ ${#TIMESTAMPED_DIRS[@]} -eq 0 ]]; then
    echo "No old timestamped directories found."
    echo "Nothing to clean up!"
    exit 0
fi

echo "Found ${#TIMESTAMPED_DIRS[@]} timestamped directories:"
echo

# Show what will be deleted with sizes
total_size=0
for dir in "${TIMESTAMPED_DIRS[@]}"; do
    size=$(du -sh "$dir" | cut -f1)
    echo "  - $(basename "$dir") : $size"
done

echo
echo "================================================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo
    echo "DRY RUN - Would delete ${#TIMESTAMPED_DIRS[@]} directories"
    echo "Run without --dry-run to actually delete these directories"
    exit 0
fi

# Confirm deletion
echo
read -p "Delete all ${#TIMESTAMPED_DIRS[@]} timestamped directories? [y/N] " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Delete directories
echo
echo "Deleting timestamped directories..."
for dir in "${TIMESTAMPED_DIRS[@]}"; do
    echo "  Removing: $(basename "$dir")"
    rm -rf "$dir"
done

echo
echo "Done!"
echo

# Show new disk usage
echo "New disk usage:"
du -sh "$PREDICTIONS_DIR"
echo

# Keep only mode-based directories
echo "Remaining directories:"
ls -1 "$PREDICTIONS_DIR" | while read -r dir; do
    if [[ -d "$PREDICTIONS_DIR/$dir" ]]; then
        size=$(du -sh "$PREDICTIONS_DIR/$dir" 2>/dev/null | cut -f1)
        echo "  - $dir : $size"
    fi
done

echo
echo "================================================================"
echo "CLEANUP COMPLETE"
echo "================================================================"

