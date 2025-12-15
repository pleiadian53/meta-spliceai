#!/bin/bash
# Final Cleanup: Fix hardcoded paths and remaining references

echo "🧹 FINAL CLEANUP"
echo "==============="
echo ""

# Fix hardcoded paths in old system format
echo "1. Fixing hardcoded old system paths..."
find . \( -name '*.py' -o -name '*.md' -o -name '*.sh' \) -type f \
    ! -path './data/*' ! -path './__pycache__/*' ! -path './.git/*' ! -path './backup_*/*' \
    -exec sed -i '' 's|/Users/pleiadian53/work/meta-spliceai/meta_spliceai|/Users/pleiadian53/work/meta-spliceai/meta_spliceai|g' {} \;

# Fix any remaining meta_spliceai in comments or docstrings
echo "2. Fixing remaining meta_spliceai references..."
find . \( -name '*.py' -o -name '*.md' \) -type f \
    ! -path './data/*' ! -path './__pycache__/*' ! -path './.git/*' ! -path './backup_*/*' \
    -exec sed -i '' 's|meta-spliceai/meta_spliceai|meta-spliceai/meta_spliceai|g' {} \;

# Fix MetaSpliceAI title references
echo "3. Updating title case references..."
find . \( -name '*.py' -o -name '*.md' \) -type f \
    ! -path './data/*' ! -path './__pycache__/*' ! -path './.git/*' ! -path './backup_*/*' \
    -exec sed -i '' 's|MetaSpliceAI|MetaSpliceAI|g' {} \;

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Running verification..."
./scripts/verify_rename.sh

