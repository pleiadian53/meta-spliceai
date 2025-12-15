#!/bin/bash
# Preview Package Renaming Changes
# 
# This script analyzes what would be changed when renaming meta_spliceai to meta_spliceai

OLD_PACKAGE="meta_spliceai"
NEW_PACKAGE="meta_spliceai"

echo "🔍 PACKAGE RENAME IMPACT ANALYSIS"
echo "=================================="
echo "Renaming: $OLD_PACKAGE → $NEW_PACKAGE"
echo ""

echo "📊 FILES CONTAINING PACKAGE REFERENCES:"
echo "======================================="

echo ""
echo "🐍 Python files (.py):"
PYTHON_FILES=$(find . -name "*.py" -type f -exec grep -l "$OLD_PACKAGE" {} \; 2>/dev/null)
PYTHON_COUNT=$(echo "$PYTHON_FILES" | wc -l)
echo "   Total: $PYTHON_COUNT files"
echo "$PYTHON_FILES" | head -10
if [ $PYTHON_COUNT -gt 10 ]; then
    echo "   ... and $((PYTHON_COUNT - 10)) more files"
fi

echo ""
echo "📚 Documentation files (.md, .txt, .rst):"
DOC_FILES=$(find . \( -name "*.md" -o -name "*.txt" -o -name "*.rst" \) -type f -exec grep -l "$OLD_PACKAGE\|meta-spliceai\|MetaSpliceAI" {} \; 2>/dev/null)
DOC_COUNT=$(echo "$DOC_FILES" | wc -l)
echo "   Total: $DOC_COUNT files"
echo "$DOC_FILES" | head -5
if [ $DOC_COUNT -gt 5 ]; then
    echo "   ... and $((DOC_COUNT - 5)) more files"
fi

echo ""
echo "🐚 Shell scripts (.sh):"
SHELL_FILES=$(find . -name "*.sh" -type f -exec grep -l "$OLD_PACKAGE\|meta-spliceai" {} \; 2>/dev/null)
SHELL_COUNT=$(echo "$SHELL_FILES" | wc -l)
echo "   Total: $SHELL_COUNT files"
echo "$SHELL_FILES"

echo ""
echo "📋 SAMPLE REFERENCES TO UPDATE:"
echo "==============================="

echo ""
echo "🔸 Import statements:"
find . -name "*.py" -exec grep -h "from $OLD_PACKAGE" {} \; 2>/dev/null | head -3
find . -name "*.py" -exec grep -h "import $OLD_PACKAGE" {} \; 2>/dev/null | head -3

echo ""
echo "🔸 Command line references:"
find . \( -name "*.py" -o -name "*.sh" -o -name "*.md" \) -exec grep -h "python -m $OLD_PACKAGE" {} \; 2>/dev/null | head -3

echo ""
echo "🔸 Module references:"
find . -name "*.py" -exec grep -h "$OLD_PACKAGE\." {} \; 2>/dev/null | head -3

echo ""
echo "📁 DIRECTORIES TO RENAME:"
echo "========================="
find . -type d -name "*splice*" | grep -v __pycache__ | sort

echo ""
echo "⚡ ESTIMATED IMPACT:"
echo "==================="
TOTAL_PYTHON_REFS=$(find . -name "*.py" -exec grep -c "$OLD_PACKAGE" {} \; 2>/dev/null | awk '{sum += $1} END {print sum}')
TOTAL_ALL_REFS=$(find . \( -name "*.py" -o -name "*.md" -o -name "*.sh" -o -name "*.txt" \) -exec grep -c "$OLD_PACKAGE\|meta-spliceai\|MetaSpliceAI" {} \; 2>/dev/null | awk '{sum += $1} END {print sum}')

echo "🔢 Estimated references to update:"
echo "   Python code references: $TOTAL_PYTHON_REFS"
echo "   Total references: $TOTAL_ALL_REFS"
echo ""
echo "🎯 Ready to proceed with renaming?"
echo "   Run: ./scripts/rename_package.sh --dry-run    # Preview mode"
echo "   Run: ./scripts/rename_package.sh             # Execute renaming"