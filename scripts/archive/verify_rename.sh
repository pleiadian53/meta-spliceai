#!/bin/bash
# Verify Package Rename Completion
# Checks for any remaining references to the old package name

echo "🔍 VERIFYING PACKAGE RENAME"
echo "==========================="
echo ""

# Exclude patterns
EXCLUDE_ARGS="! -path './data/*' ! -path './__pycache__/*' ! -path './.git/*' ! -path './backup_*/*'"

echo "📊 Counting remaining references to 'meta_spliceai'..."
echo ""

# Count Python files
PYTHON_COUNT=$(eval "find . -name '*.py' -type f $EXCLUDE_ARGS" | xargs grep -l "meta_spliceai" 2>/dev/null | wc -l | tr -d ' ')
echo "Python files: $PYTHON_COUNT"

# Count documentation files
DOC_COUNT=$(eval "find . \( -name '*.md' -o -name '*.txt' \) -type f $EXCLUDE_ARGS" | xargs grep -l "meta_spliceai" 2>/dev/null | wc -l | tr -d ' ')
echo "Documentation files: $DOC_COUNT"

# Count shell scripts
SHELL_COUNT=$(eval "find . -name '*.sh' -type f $EXCLUDE_ARGS" | xargs grep -l "meta_spliceai" 2>/dev/null | wc -l | tr -d ' ')
echo "Shell scripts: $SHELL_COUNT"

TOTAL=$((PYTHON_COUNT + DOC_COUNT + SHELL_COUNT))
echo ""
echo "Total files with old references: $TOTAL"
echo ""

if [ $TOTAL -eq 0 ]; then
    echo "✅ SUCCESS: No remaining references to 'meta_spliceai' found!"
    echo ""
    echo "📁 Verifying new package structure..."
    if [ -d "meta_spliceai" ]; then
        echo "✅ meta_spliceai directory exists"
        echo ""
        echo "Testing import..."
        /Users/pleiadian53/miniforge3-new/bin/mamba run -n surveyor python -c "import meta_spliceai; print('✅ Import successful!')" 2>/dev/null
    else
        echo "❌ meta_spliceai directory not found!"
    fi
else
    echo "⚠️  Found $TOTAL files with remaining references"
    echo ""
    echo "Sample files (first 10):"
    eval "find . \( -name '*.py' -o -name '*.md' -o -name '*.sh' \) -type f $EXCLUDE_ARGS" | xargs grep -l "meta_spliceai" 2>/dev/null | head -10
fi

echo ""
echo "📊 PACKAGE RENAME SUMMARY"
echo "========================="
echo "Old package: meta_spliceai"
echo "New package: meta_spliceai"
echo "Remaining references: $TOTAL"
echo ""

