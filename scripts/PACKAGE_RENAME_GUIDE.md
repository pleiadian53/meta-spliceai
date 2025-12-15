# Package Renaming Guide: meta-spliceai → meta-spliceai

## 📋 Overview

This guide provides a comprehensive approach to rename the `meta-spliceai`/`meta_spliceai` package to `meta-spliceai`/`meta_spliceai` across the entire codebase, including all dependencies, documentation, and configuration files.

## 🎯 Renaming Strategy

### **Package Name Transformations**
| Context | Old Name | New Name |
|---------|----------|----------|
| Python Package | `meta_spliceai` | `meta_spliceai` |
| Directory Name | `meta-spliceai` | `meta-spliceai` |
| Display Name | `MetaSpliceAI` | `MetaSpliceAI` |
| Command Line | `python -m meta_spliceai` | `python -m meta_spliceai` |

## 📊 Impact Analysis

Based on the current codebase analysis:

- **🐍 Python files**: 443 files with references
- **📚 Documentation**: 390 files with references  
- **🐚 Shell scripts**: 18 files with references
- **📋 Total references**: ~5,266 references to update
- **🔢 Python code refs**: ~1,869 import/module references

## 🛠️ Available Tools

### 1. **Preview Script** (`preview_rename.sh`)
Analyzes the scope of changes without making modifications:

```bash
./scripts/preview_rename.sh
```

**Shows**:
- Files that will be modified
- Sample references that will be updated
- Directories that will be renamed
- Estimated impact statistics

### 2. **Renaming Script** (`rename_package.sh`)
Comprehensive automated renaming with safety features:

```bash
# Dry run (preview mode - safe to run)
./scripts/rename_package.sh --dry-run

# Execute the renaming
./scripts/rename_package.sh

# Custom package name
./scripts/rename_package.sh --new-name custom_package_name
```

## 🔧 What Gets Updated

### **Python Code Changes**
```python
# OLD → NEW
from meta_spliceai.system import config
from meta_spliceai.system import config

import meta_spliceai.utils as utils  
import meta_spliceai.utils as utils

meta_spliceai.module.function()
meta_spliceai.module.function()

"meta_spliceai" → "meta_spliceai"
```

### **Command Line Changes**
```bash
# OLD → NEW
python -m meta_spliceai.splice_engine.training
python -m meta_spliceai.splice_engine.training
```

### **Documentation Changes**
```markdown
# OLD → NEW
MetaSpliceAI system → MetaSpliceAI system
meta-spliceai → meta-spliceai
meta_spliceai → meta_spliceai
```

### **Directory Renaming**
```bash
# Main package directory
meta_spliceai/ → meta_spliceai/

# Related directories (if they contain package references)
scripts/splice_* → scripts/meta_*
docs/splice_* → docs/meta_*
```

## 🚨 Safety Features

### **Automatic Backup**
- Creates timestamped backup: `../backup_YYYYMMDD_HHMMSS/`
- Complete copy of entire project before changes
- Easy restoration if needed

### **Dry Run Mode**
- Preview all changes without executing
- Safe to run multiple times
- Shows exact transformations that will occur

### **File Safety**
- Creates `.bak` files during sed operations
- Removes backup files only after successful replacement
- Preserves file permissions and structure

### **Error Handling**
- Exits on first error (`set -e`)
- Validates file existence before modification
- Comprehensive error reporting

## 📋 Step-by-Step Process

### **Phase 1: Analysis & Planning**
```bash
# 1. Analyze current state
./scripts/preview_rename.sh

# 2. Run dry-run to preview changes
./scripts/rename_package.sh --dry-run

# 3. Review the planned changes carefully
```

### **Phase 2: Safe Execution**
```bash
# 4. Execute the renaming (creates backup automatically)
./scripts/rename_package.sh

# 5. Verify the changes
ls -la  # Check directory structure
```

### **Phase 3: Testing & Validation**
```bash
# 6. Test imports
python -c "import meta_spliceai; print('✅ Import successful')"

# 7. Test key functionality
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid --help

# 8. Run existing tests
pytest tests/
```

## 🔍 Verification Checklist

After renaming, verify these areas:

### **✅ Python Package Structure**
- [ ] Main package directory renamed: `meta_spliceai/`
- [ ] All imports work: `import meta_spliceai`
- [ ] Module paths updated: `meta_spliceai.splice_engine.*`
- [ ] Command line tools work: `python -m meta_spliceai.*`

### **✅ Documentation**
- [ ] README files updated
- [ ] API documentation reflects new names
- [ ] Tutorial examples use new package name
- [ ] Installation instructions updated

### **✅ Configuration & Scripts**
- [ ] Shell scripts use new package name
- [ ] Configuration files updated
- [ ] Environment setup scripts work
- [ ] CI/CD pipelines updated (if applicable)

### **✅ External References**
- [ ] Setup.py/pyproject.toml updated
- [ ] Requirements files reference correct package
- [ ] Docker files use new package name
- [ ] Documentation links still work

## ⚠️ Potential Issues & Solutions

### **Issue 1: Import Errors**
```python
# Error: ModuleNotFoundError: No module named 'meta_spliceai'
# Solution: Check for missed references
find . -name "*.py" -exec grep -l "meta_spliceai" {} \;
```

### **Issue 2: Command Line Tools Broken**
```bash
# Error: python -m meta_spliceai.* not found
# Solution: Update command references
grep -r "python -m meta_spliceai" . --include="*.sh" --include="*.md"
```

### **Issue 3: Data Path References**
```python
# Error: Path not found errors
# Solution: Check for hardcoded paths
grep -r "/meta_spliceai/" . --include="*.py"
```

## 🔄 Rollback Plan

If issues occur, you can restore from backup:

```bash
# 1. Navigate to parent directory
cd ..

# 2. Remove problematic renamed directory
rm -rf meta-spliceai

# 3. Restore from backup
cp -r backup_YYYYMMDD_HHMMSS/ meta-spliceai/

# 4. Navigate back
cd meta-spliceai/
```

## 🚀 Post-Rename Tasks

### **Immediate (Required)**
1. **Test core functionality**
2. **Update version control** (if using Git)
3. **Update project README**
4. **Verify all imports work**

### **Soon After (Recommended)**
1. **Update external documentation**
2. **Notify collaborators** of package name change
3. **Update any published packages** (PyPI, conda, etc.)
4. **Update CI/CD pipelines**

### **Eventually (Optional)**
1. **Update repository name** (GitHub, GitLab, etc.)
2. **Update domain/URLs** if applicable
3. **Update academic papers/publications**
4. **Create migration guide** for external users

## 🎯 Success Metrics

The renaming is successful when:

- ✅ **All imports work**: No `ModuleNotFoundError`s
- ✅ **Commands execute**: All `python -m meta_spliceai.*` work
- ✅ **Tests pass**: Existing functionality preserved
- ✅ **Documentation consistent**: All references updated
- ✅ **No mixed references**: No old package names remain

## 📞 Troubleshooting

### **Common Commands for Debugging**
```bash
# Find remaining old references
grep -r "meta_spliceai" . --exclude-dir=backup_*

# Check Python import paths
python -c "import sys; print('\n'.join(sys.path))"

# Test specific module imports
python -c "from meta_spliceai.splice_engine import meta_models"

# Verify command line tools
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid --help
```

---

**Generated**: 2025-10-15  
**Tools**: `preview_rename.sh`, `rename_package.sh`  
**Estimated Time**: 15-30 minutes (depending on testing)  
**Risk Level**: Low (with backup and dry-run safety features)