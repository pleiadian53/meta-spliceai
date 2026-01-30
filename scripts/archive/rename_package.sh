#!/bin/bash
# Package Renaming Script: meta-spliceai → meta-spliceai
# 
# This script systematically renames the meta_spliceai package to meta_spliceai
# across all files, directories, and references in the codebase.
#
# Usage: ./scripts/rename_package.sh [--dry-run] [--new-name package_name]

set -e  # Exit on any error

# Configuration
OLD_PACKAGE_NAME="meta_spliceai"
OLD_PACKAGE_HYPHEN="meta-spliceai" 
OLD_PACKAGE_TITLE="MetaSpliceAI"
OLD_DIR_NAME="meta-spliceai"

NEW_PACKAGE_NAME="${2:-meta_spliceai}"
NEW_PACKAGE_HYPHEN="meta-spliceai"
NEW_PACKAGE_TITLE="MetaSpliceAI"
NEW_DIR_NAME="meta-spliceai"

DRY_RUN=false
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --new-name)
            NEW_PACKAGE_NAME="$2"
            NEW_PACKAGE_HYPHEN=$(echo "$2" | tr '_' '-')
            NEW_PACKAGE_TITLE=$(echo "$2" | sed 's/_/ /g' | sed 's/\b\w/\U&/g' | tr -d ' ')
            NEW_DIR_NAME="$NEW_PACKAGE_HYPHEN"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--new-name package_name]"
            exit 1
            ;;
    esac
done

echo "🔄 PACKAGE RENAMING SCRIPT"
echo "=========================="
echo "Old package name: $OLD_PACKAGE_NAME"
echo "New package name: $NEW_PACKAGE_NAME"
echo "Old directory name: $OLD_DIR_NAME"  
echo "New directory name: $NEW_DIR_NAME"
echo "Dry run mode: $DRY_RUN"
echo ""

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
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Replace '$old_text' → '$new_text' in $file"
    else
        # Use sed with backup for safety
        sed -i.bak "s|$old_text|$new_text|g" "$file" && rm "$file.bak"
    fi
}

# Step 1: Create backup
if [ "$DRY_RUN" = false ]; then
    echo "📦 Creating backup..."
    mkdir -p "../$BACKUP_DIR"
    cp -r . "../$BACKUP_DIR/"
    echo "✅ Backup created at ../$BACKUP_DIR"
fi

# Step 2: Find all files that need updating
echo ""
echo "🔍 ANALYZING FILES TO UPDATE"
echo "============================"

# Find Python files with package references
PYTHON_FILES=$(find . -name "*.py" -type f -exec grep -l "$OLD_PACKAGE_NAME" {} \; 2>/dev/null)
PYTHON_COUNT=$(echo "$PYTHON_FILES" | wc -l)

# Find documentation files
DOC_FILES=$(find . \( -name "*.md" -o -name "*.txt" -o -name "*.rst" \) -type f -exec grep -l "$OLD_PACKAGE_NAME\|$OLD_PACKAGE_HYPHEN\|$OLD_PACKAGE_TITLE" {} \; 2>/dev/null)
DOC_COUNT=$(echo "$DOC_FILES" | wc -l)

# Find shell scripts
SHELL_FILES=$(find . -name "*.sh" -type f -exec grep -l "$OLD_PACKAGE_NAME\|$OLD_PACKAGE_HYPHEN" {} \; 2>/dev/null)
SHELL_COUNT=$(echo "$SHELL_FILES" | wc -l)

# Find configuration files
CONFIG_FILES=$(find . \( -name "*.yml" -o -name "*.yaml" -o -name "*.json" -o -name "*.cfg" -o -name "*.ini" -o -name "setup.py" -o -name "requirements.txt" \) -type f -exec grep -l "$OLD_PACKAGE_NAME\|$OLD_PACKAGE_HYPHEN" {} \; 2>/dev/null)
CONFIG_COUNT=$(echo "$CONFIG_FILES" | wc -l)

echo "📊 Files to update:"
echo "   Python files: $PYTHON_COUNT"
echo "   Documentation: $DOC_COUNT"  
echo "   Shell scripts: $SHELL_COUNT"
echo "   Config files: $CONFIG_COUNT"

# Step 3: Update Python imports and references
echo ""
echo "🐍 UPDATING PYTHON FILES"
echo "========================"

if [ -n "$PYTHON_FILES" ]; then
    echo "$PYTHON_FILES" | while read -r file; do
        if [ -f "$file" ]; then
            echo "📝 Updating: $file"
            
            # Update imports
            safe_replace "from $OLD_PACKAGE_NAME" "from $NEW_PACKAGE_NAME" "$file"
            safe_replace "import $OLD_PACKAGE_NAME" "import $NEW_PACKAGE_NAME" "$file"
            safe_replace "$OLD_PACKAGE_NAME\\." "$NEW_PACKAGE_NAME." "$file"
            
            # Update command line references
            safe_replace "python -m $OLD_PACKAGE_NAME" "python -m $NEW_PACKAGE_NAME" "$file"
            
            # Update string references
            safe_replace "\"$OLD_PACKAGE_NAME" "\"$NEW_PACKAGE_NAME" "$file"
            safe_replace "'$OLD_PACKAGE_NAME" "'$NEW_PACKAGE_NAME" "$file"
        fi
    done
fi

# Step 4: Update documentation files
echo ""
echo "📚 UPDATING DOCUMENTATION"
echo "========================="

if [ -n "$DOC_FILES" ]; then
    echo "$DOC_FILES" | while read -r file; do
        if [ -f "$file" ]; then
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

if [ -n "$SHELL_FILES" ]; then
    echo "$SHELL_FILES" | while read -r file; do
        if [ -f "$file" ]; then
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

if [ -n "$CONFIG_FILES" ]; then
    echo "$CONFIG_FILES" | while read -r file; do
        if [ -f "$file" ]; then
            echo "📝 Updating: $file"
            
            # Update package references
            safe_replace "$OLD_PACKAGE_NAME" "$NEW_PACKAGE_NAME" "$file"
            safe_replace "$OLD_PACKAGE_HYPHEN" "$NEW_PACKAGE_HYPHEN" "$file"
        fi
    done
fi

# Step 7: Rename the main package directory
echo ""
echo "📁 RENAMING DIRECTORIES"
echo "======================="

if [ -d "$OLD_PACKAGE_NAME" ]; then
    execute_or_show "mv '$OLD_PACKAGE_NAME' '$NEW_PACKAGE_NAME'" "Rename main package directory"
fi

# Step 8: Update any remaining directory references
echo ""
echo "🔍 UPDATING REMAINING REFERENCES"
echo "================================"

# Update any hardcoded paths that might reference the old directory structure
find . -name "*.py" -type f -exec grep -l "/$OLD_PACKAGE_NAME/" {} \; 2>/dev/null | while read -r file; do
    if [ -f "$file" ]; then
        echo "📝 Updating directory paths in: $file"
        safe_replace "/$OLD_PACKAGE_NAME/" "/$NEW_PACKAGE_NAME/" "$file"
    fi
done

# Step 9: Final verification
echo ""
echo "🎯 VERIFICATION"
echo "==============="

if [ "$DRY_RUN" = false ]; then
    echo "🔍 Checking for any remaining old references..."
    REMAINING=$(find . -name "*.py" -o -name "*.md" -o -name "*.sh" | xargs grep -l "$OLD_PACKAGE_NAME" 2>/dev/null | wc -l)
    
    if [ "$REMAINING" -eq 0 ]; then
        echo "✅ All references successfully updated!"
    else
        echo "⚠️  Warning: Found $REMAINING files that may still contain old references"
        echo "   Please review these files manually:"
        find . -name "*.py" -o -name "*.md" -o -name "*.sh" | xargs grep -l "$OLD_PACKAGE_NAME" 2>/dev/null
    fi
    
    echo ""
    echo "📊 FINAL STRUCTURE:"
    echo "==================="
    ls -la | grep -E "$NEW_PACKAGE_NAME|$OLD_PACKAGE_NAME" || echo "No package directories visible (may need to check subdirectories)"
    
else
    echo "🔍 This was a dry run. No changes were made."
    echo "To execute the renaming, run: $0 --new-name $NEW_PACKAGE_NAME"
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
    echo "1. Test your code with the new package name"
    echo "2. Update any external documentation or README files"
    echo "3. Consider updating your Git repository name if applicable"
    echo "4. Update any CI/CD pipelines or deployment scripts"
fi