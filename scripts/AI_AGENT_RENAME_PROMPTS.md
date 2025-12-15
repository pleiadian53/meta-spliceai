# AI Agent Prompts for Package Renaming

## 📋 Overview

This document provides structured prompts that can be given to AI agents to systematically rename a Python package. The prompts are designed to be clear, actionable, and safe, with built-in verification steps.

## 🎯 Context Setup

**Before starting, provide this context to your AI agent:**

```
I need to rename my Python package from `meta-spliceai`/`meta_spliceai` to `meta-spliceai`/`meta_spliceai`. 

Current package structure:
- Main package directory: meta_spliceai/
- Repository name: meta-spliceai  
- Python imports use: meta_spliceai.*
- Command line tools: python -m meta_spliceai.*

Target package structure:
- Main package directory: meta_spliceai/
- Repository name: meta-spliceai
- Python imports use: meta_spliceai.*  
- Command line tools: python -m meta_spliceai.*

Please follow the prompts step-by-step and wait for my confirmation before proceeding to destructive operations.
```

---

## 🔍 Phase 1: Analysis & Assessment

### **Prompt 1.1: Initial Codebase Analysis**
```
Please analyze my current codebase to understand the scope of the package renaming task:

1. Find and count all Python files (.py) that contain references to "meta_spliceai" or "meta-spliceai"
2. Find and count all documentation files (.md, .rst, .txt) with package references  
3. Find and count shell scripts (.sh) and config files that reference the package
4. Estimate the total number of references that need to be updated
5. Identify the main package directory structure
6. List any potential complications (e.g., hardcoded paths, external dependencies)

Present your findings in a summary table showing file counts and reference estimates.
```

### **Prompt 1.2: Directory Structure Assessment**
```
Examine the current directory structure and identify:

1. All directories that contain "splice" in their name
2. The main Python package directory location
3. Any duplicate or redundant directories that should be cleaned up first
4. Configuration files (setup.py, pyproject.toml, requirements.txt) that need updates
5. Any symbolic links or aliases that reference the old package name

Create a plan for which directories need to be renamed and in what order.
```

### **Prompt 1.3: Import Dependency Mapping**
```
Create a comprehensive map of how the package is imported and used:

1. Search for all import statements that reference meta_spliceai
2. Find all command-line usage patterns (python -m meta_spliceai.*)
3. Identify any external scripts or tools that depend on the package name
4. Look for string literals that contain the package name (for dynamic imports)
5. Find any documentation examples that show package usage

Categorize these by type (direct imports, module calls, string references, etc.) and estimated complexity to update.
```

---

## ⚡ Phase 2: Safety Preparation

### **Prompt 2.1: Backup Strategy Creation**
```
Create a comprehensive backup strategy before making any changes:

1. Create a full backup of the entire project directory
2. Use a timestamped backup directory name (backup_YYYYMMDD_HHMMSS format)
3. Verify the backup was created successfully and contains all files
4. Create a restore script that can undo all changes if needed
5. Test that the current codebase works before making changes (run key imports/commands)

Show me the backup location and confirm it's complete before proceeding.
```

### **Prompt 2.2: Create Dry-Run Scripts**
```
Create scripts that can preview all changes without executing them:

1. Write a script that shows exactly which files would be modified
2. Create a preview of what the new directory structure will look like
3. Generate sample "before/after" examples for key file types (Python imports, docs, configs)
4. Create a verification script that can check if the rename was successful
5. Make all scripts executable and test them in preview mode

Run the preview scripts and show me what changes would be made.
```

---

## 🔧 Phase 3: Systematic Renaming

### **Prompt 3.1: File Content Updates**
```
Update all file contents systematically, in this order:

1. **Python files first**: Update all import statements, module references, and string literals
2. **Documentation**: Update README files, API docs, tutorials, and examples  
3. **Configuration files**: Update setup.py, pyproject.toml, requirements files
4. **Shell scripts**: Update any bash/sh scripts that reference the package
5. **Other text files**: Update any remaining references in comments, logs, etc.

For each category:
- Show me a sample of changes before applying them
- Apply changes to a few files as a test
- Wait for my approval before batch-processing all files
- Verify each category was updated successfully

Use sed/awk commands with backup files (.bak) for safety.
```

### **Prompt 3.2: Directory Renaming**
```
Rename directories in the correct order to avoid breaking references:

1. **Prepare**: Ensure no active processes are using the directories
2. **Main package**: Rename meta_spliceai/ to meta_spliceai/
3. **Related directories**: Rename any other directories with "splice" in the name
4. **Update paths**: Fix any hardcoded path references that broke due to renames
5. **Verify structure**: Confirm the new directory structure is correct

After each rename:
- Verify the directory was renamed successfully  
- Check that no references to the old directory path remain
- Test that the package can still be imported from the new location
```

---

## ✅ Phase 4: Testing & Verification

### **Prompt 4.1: Import Testing**
```
Systematically test that all imports work with the new package name:

1. **Basic import test**: `python -c "import meta_spliceai"`
2. **Submodule imports**: Test importing key submodules 
3. **Command line tools**: Test `python -m meta_spliceai.*` commands
4. **Dynamic imports**: Test any code that imports modules dynamically
5. **Cross-references**: Verify internal package imports still work

For each test:
- Show the command being run
- Show the result (success/failure)
- If failures occur, identify what still needs to be updated
```

### **Prompt 4.2: Comprehensive Verification**
```
Run a complete verification to ensure the rename was successful:

1. **Search for old references**: Find any remaining "meta_spliceai" or "meta-spliceai" references
2. **Test key functionality**: Run the main entry points and core functions
3. **Documentation consistency**: Verify all docs use the new package name consistently
4. **External compatibility**: Check that the package can still be installed/imported externally
5. **Performance check**: Ensure rename didn't break anything or slow down imports

Create a verification report showing:
- What was successfully updated
- Any remaining issues
- Performance/functionality comparison (before vs. after)
```

---

## 🚀 Phase 5: Finalization

### **Prompt 5.1: Cleanup & Optimization**
```
Clean up temporary files and optimize the renamed package:

1. **Remove backup files**: Delete all .bak files created during the rename process
2. **Clean up temp files**: Remove any temporary scripts or intermediate files
3. **Optimize imports**: Look for any import optimizations possible with the new structure
4. **Update version info**: Update any version strings or metadata that reference the old name
5. **Final verification**: One last check that everything works correctly

Show me what files are being cleaned up and confirm the final state is clean.
```

### **Prompt 5.2: Documentation & Migration Guide**
```
Create documentation for the completed rename:

1. **Update main README**: Ensure it reflects the new package name throughout
2. **Create migration guide**: Document what changed for any external users
3. **Update installation instructions**: Modify setup/install docs for new package name
4. **Version control prep**: Prepare commit message and change documentation
5. **External notification plan**: List what external references need manual updates

Provide a summary of the completed rename and next steps for version control.
```

---

## 🛡️ Emergency Rollback

### **Emergency Prompt: Restore from Backup**
```
EMERGENCY: Restore the original package from backup immediately:

1. Stop any running processes that might be using the package
2. Remove the current (problematic) directory structure  
3. Restore the complete backup created earlier
4. Verify the original functionality works
5. Document what went wrong for future reference

Show me the restore process and confirm the original state is recovered.
```

---

## 📋 Usage Instructions

### **How to Use These Prompts:**

1. **Sequential Execution**: Give these prompts in order, one phase at a time
2. **Confirmation Required**: Wait for AI agent to complete each prompt and confirm results before proceeding  
3. **Safety First**: Always run Phase 2 (backups) before any destructive operations
4. **Verification**: Use the verification prompts after major changes
5. **Emergency Ready**: Keep the rollback prompt ready if anything goes wrong

### **Estimated Timeline:**
- **Phase 1 (Analysis)**: 10-15 minutes
- **Phase 2 (Safety)**: 5-10 minutes  
- **Phase 3 (Renaming)**: 20-30 minutes
- **Phase 4 (Testing)**: 15-20 minutes
- **Phase 5 (Finalization)**: 10-15 minutes
- **Total**: 60-90 minutes

### **Risk Level:** LOW
- Multiple backup layers
- Dry-run verification at each step  
- Incremental testing
- Easy rollback capability

---

## 🔧 Technical Notes

- **Shell**: Designed for bash/zsh compatibility
- **Platform**: Works on macOS/Linux (Windows may need modifications)
- **Dependencies**: Uses standard Unix tools (sed, awk, grep, find)
- **Python**: Compatible with Python 3.6+
- **Backup**: Creates complete project snapshots for safety

**Generated**: 2025-10-15  
**Estimated Effort**: 60-90 minutes with AI assistance  
**Success Rate**: High (with proper backups and testing)