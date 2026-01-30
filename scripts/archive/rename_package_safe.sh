#!/bin/bash
# Safe Package Renaming Script: meta-spliceai → meta-spliceai
# 
# This script systematically renames the meta_spliceai package to meta_spliceai
# while EXCLUDING data directories and model paths that should remain unchanged.
#
# Usage: ./scripts/rename_package_safe.sh [--dry-run]

set -e  # Exit on any error

# Configuration
OLD_PACKAGE_NAME="meta_spliceai"
OLD_PACKAGE_HYPHEN="meta-spliceai" 
OLD_PACKAGE_TITLE="MetaSpliceAI"

NEW_PACKAGE_NAME="meta_spliceai"
NEW_PACKAGE_HYPHEN="meta-spliceai"
NEW_PACKAGE_TITLE="MetaSpliceAI"

DRY_RUN=false
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"

# Directories and patterns to EXCLUDE from renaming
EXCLUDE_DIRS=(
    "./data/ensembl/spliceai*"
    "./data/models/spliceai"
    "./data/spliceai_analysis"
    "./__pycache__"
    "./.git"
    "./backup_*"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

echo "🔄 SAFE PACKAGE RENAMING SCRIPT"
echo "==============================="
echo "Old package name: $OLD_PACKAGE_NAME"
echo "New package name: $NEW_PACKAGE_NAME"
echo "Dry run mode: $DRY_RUN"
echo ""
echo "📁 Directories EXCLUDED from renaming:"
for exclude in "${EXCLUDE_DIRS[@]}"; do
    echo "   ✗ $exclude"
done
echo ""

# Function to check if path should be excluded
should_exclude() {
    local path="$1"
    for exclude in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$path" == $exclude* ]]; then
            return 0  # Should exclude
        fi
    done
    return 1  # Should not exclude
}

# Function to execute or show command based on dry-run mode
execute_or_show() {
    local cmd="$1"
    local description="$2"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $description: $cmd"
    else
        echo "✅ $description"
        eval "$cmd"
    fi
}

# Function to safely replace text in files
safe_replace() {
    local old_text="$1"
    local new_text="$2"
    local file="$3"
    
    # Check if file should be excluded
    if should_exclude "$file"; then
        echo "   [SKIP] $file (in excluded directory)"
        return 0
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Replace '$old_text' → '$new_text' in $file"
    else
        # Use sed with backup for safety
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|$old_text|$new_text|g" "$file"
        else
            # Linux
            sed -i "s|$old_text|$new_text|g" "$file"
        fi
    fi
}

# Step 1: Create backup
if [ "$DRY_RUN" = false ]; then
    echo "📦 Creating backup..."
    mkdir -p "../$BACKUP_DIR"
    
    # Copy only non-data files
    echo "   Backing up source code..."
    rsync -av --exclude='data/' --exclude='*.pyc' --exclude='__pycache__' --exclude='.git/' . "../$BACKUP_DIR/" > /dev/null
    
    echo "✅ Backup created at ../$BACKUP_DIR"
fi

# Step 2: Find all files that need updating (EXCLUDING data directories)
echo ""
echo "🔍 ANALYZING FILES TO UPDATE"
echo "============================"

# Build find command with exclusions
FIND_EXCLUDE=""
for exclude in "${EXCLUDE_DIRS[@]}"; do
    FIND_EXCLUDE="$FIND_EXCLUDE ! -path '$exclude/*'"
done

# Find Python files with package references (excluding data dirs)
PYTHON_FILES=$(eval "find . -name '*.py' -type f $FIND_EXCLUDE" | xargs grep -l "$OLD_PACKAGE_NAME" 2>/dev/null || true)
PYTHON_COUNT=$(echo "$PYTHON_FILES" | grep -c '.' || echo "0")

# Find documentation files (excluding data dirs)
DOC_FILES=$(eval "find . \( -name '*.md' -o -name '*.txt' -o -name '*.rst' \) -type f $FIND_EXCLUDE" | xargs grep -l "$OLD_PACKAGE_NAME\|$OLD_PACKAGE_HYPHEN\|$OLD_PACKAGE_TITLE" 2>/dev/null || true)
DOC_COUNT=$(echo "$DOC_FILES" | grep -c '.' || echo "0")

# Find shell scripts (excluding data dirs)
SHELL_FILES=$(eval "find . -name '*.sh' -type f $FIND_EXCLUDE" | xargs grep -l "$OLD_PACKAGE_NAME\|$OLD_PACKAGE_HYPHEN" 2>/dev/null || true)
SHELL_COUNT=$(echo "$SHELL_FILES" | grep -c '.' || echo "0")

# Find configuration files (excluding data dirs)
CONFIG_FILES=$(eval "find . \( -name '*.yml' -o -name '*.yaml' -o -name '*.json' -o -name '*.cfg' -o -name '*.ini' -o -name 'setup.py' -o -name 'requirements.txt' \) -type f $FIND_EXCLUDE" | xargs grep -l "$OLD_PACKAGE_NAME\|$OLD_PACKAGE_HYPHEN" 2>/dev/null || true)
CONFIG_COUNT=$(echo "$CONFIG_FILES" | grep -c '.' || echo "0")

echo "📊 Files to update:"
echo "   Python files: $PYTHON_COUNT"
echo "   Documentation: $DOC_COUNT"  
echo "   Shell scripts: $SHELL_COUNT"
echo "   Config files: $CONFIG_COUNT"

# Step 3: Update Python imports and references
echo ""
echo "🐍 UPDATING PYTHON FILES"
echo "========================"

if [ -n "$PYTHON_FILES" ] && [ "$PYTHON_COUNT" -gt 0 ]; then
    echo "$PYTHON_FILES" | while read -r file; do
        if [ -f "$file" ] && ! should_exclude "$file"; then
            echo "📝 Updating: $file"
            
            # Update imports
            safe_replace "from $OLD_PACKAGE_NAME" "from $NEW_PACKAGE_NAME" "$file"
            safe_replace "import $OLD_PACKAGE_NAME" "import $NEW_PACKAGE_NAME" "$file"
            safe_replace "$OLD_PACKAGE_NAME\\." "$NEW_PACKAGE_NAME." "$file"
            
            # Update command line references
            safe_replace "python -m $OLD_PACKAGE_NAME" "python -m $NEW_PACKAGE_NAME" "$file"
            
            # Update string references (but not in comments about data paths)
            safe_replace "\"$OLD_PACKAGE_NAME\"" "\"$NEW_PACKAGE_NAME\"" "$file"
            safe_replace "'$OLD_PACKAGE_NAME'" "'$NEW_PACKAGE_NAME'" "$file"
        fi
    done
fi

# Step 4: Update documentation files
echo ""
echo "📚 UPDATING DOCUMENTATION"
echo "========================="

if [ -n "$DOC_FILES" ] && [ "$DOC_COUNT" -gt 0 ]; then
    echo "$DOC_FILES" | while read -r file; do
        if [ -f "$file" ] && ! should_exclude "$file"; then
            echo "📝 Updating: $file"
            
            # Update package names in all forms
            safe_replace "$OLD_PACKAGE_NAME" "$NEW_PACKAGE_NAME" "$file"
            safe_replace "$OLD_PACKAGE_HYPHEN" "$NEW_PACKAGE_HYPHEN" "$file"
            safe_replace "$OLD_PACKAGE_TITLE" "$NEW_PACKAGE_TITLE" "$file"
            
            # Update command references
            safe_replace "python -m $OLD_PACKAGE_NAME" "python -m $NEW_PACKAGE_NAME" "$file"
        fi
    done
fi

# Step 5: Update shell scripts
echo ""
echo "🐚 UPDATING SHELL SCRIPTS"
echo "========================="

if [ -n "$SHELL_FILES" ] && [ "$SHELL_COUNT" -gt 0 ]; then
    echo "$SHELL_FILES" | while read -r file; do
        if [ -f "$file" ] && ! should_exclude "$file"; then
            echo "📝 Updating: $file"
            
            # Update package references
            safe_replace "$OLD_PACKAGE_NAME" "$NEW_PACKAGE_NAME" "$file"
            safe_replace "$OLD_PACKAGE_HYPHEN" "$NEW_PACKAGE_HYPHEN" "$file"
            
            # Update paths
            safe_replace "python -m $OLD_PACKAGE_NAME" "python -m $NEW_PACKAGE_NAME" "$file"
        fi
    done
fi

# Step 6: Update configuration files
echo ""
echo "⚙️ UPDATING CONFIGURATION FILES"
echo "==============================="

if [ -n "$CONFIG_FILES" ] && [ "$CONFIG_COUNT" -gt 0 ]; then
    echo "$CONFIG_FILES" | while read -r file; do
        if [ -f "$file" ] && ! should_exclude "$file"; then
            echo "📝 Updating: $file"
            
            # Update package references
            safe_replace "$OLD_PACKAGE_NAME" "$NEW_PACKAGE_NAME" "$file"
            safe_replace "$OLD_PACKAGE_HYPHEN" "$NEW_PACKAGE_HYPHEN" "$file"
        fi
    done
fi

# Step 7: Rename the main package directory
echo ""
echo "📁 RENAMING MAIN PACKAGE DIRECTORY"
echo "==================================="

if [ -d "$OLD_PACKAGE_NAME" ]; then
    execute_or_show "mv '$OLD_PACKAGE_NAME' '$NEW_PACKAGE_NAME'" "Rename main package directory"
else
    echo "⚠️  Main package directory '$OLD_PACKAGE_NAME' not found"
fi

# Step 8: Update any remaining directory references in code
echo ""
echo "🔍 UPDATING REMAINING REFERENCES"
echo "================================"

# Update any hardcoded paths that might reference the old directory structure
eval "find . -name '*.py' -type f $FIND_EXCLUDE" | xargs grep -l "/$OLD_PACKAGE_NAME/" 2>/dev/null | while read -r file; do
    if [ -f "$file" ] && ! should_exclude "$file"; then
        echo "📝 Updating directory paths in: $file"
        safe_replace "/$OLD_PACKAGE_NAME/" "/$NEW_PACKAGE_NAME/" "$file"
    fi
done

# Step 9: Final verification
echo ""
echo "🎯 VERIFICATION"
echo "==============="

if [ "$DRY_RUN" = false ]; then
    echo "🔍 Checking for any remaining old references (excluding data/)..."
    REMAINING=$(eval "find . \( -name '*.py' -o -name '*.md' -o -name '*.sh' \) -type f $FIND_EXCLUDE" | xargs grep -l "$OLD_PACKAGE_NAME" 2>/dev/null | wc -l)
    
    if [ "$REMAINING" -eq 0 ]; then
        echo "✅ All references successfully updated!"
    else
        echo "⚠️  Warning: Found $REMAINING files that may still contain old references"
        echo "   (These might be intentional references in comments or documentation)"
        echo ""
        echo "   Files with remaining references:"
        eval "find . \( -name '*.py' -o -name '*.md' -o -name '*.sh' \) -type f $FIND_EXCLUDE" | xargs grep -l "$OLD_PACKAGE_NAME" 2>/dev/null | head -10
    fi
    
    echo ""
    echo "📊 FINAL STRUCTURE:"
    echo "==================="
    echo "Main package directory:"
    ls -ld $NEW_PACKAGE_NAME 2>/dev/null || echo "   [Not found - may need to check]"
    
    echo ""
    echo "Data directories (unchanged):"
    ls -ld data/ensembl/spliceai* data/models/spliceai 2>/dev/null || echo "   [No spliceai data directories found]"
    
else
    echo "🔍 This was a dry run. No changes were made."
    echo "To execute the renaming, run: $0"
fi

echo ""
echo "🎉 PACKAGE RENAMING COMPLETE!"
echo "============================="
echo "Old name: $OLD_PACKAGE_NAME"
echo "New name: $NEW_PACKAGE_NAME"

if [ "$DRY_RUN" = false ]; then
    echo "Backup available at: ../$BACKUP_DIR"
    echo ""
    echo "🚀 NEXT STEPS:"
    echo "1. Verify the changes: cd $NEW_PACKAGE_NAME && ls"
    echo "2. Test imports: python -c 'import $NEW_PACKAGE_NAME'"
    echo "3. Run tests to ensure everything works"
    echo "4. Data directories remain at: data/ensembl/spliceai*, data/models/spliceai"
    echo "5. Consider updating Git remote if renaming the repository"
fi

