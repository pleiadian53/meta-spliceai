# Gene Type Consistency Guide

**Quick reference for maintaining gene type consistency across strategic training workflow**

## TL;DR - Gene Type Answers

### **Q1: How to specify `--gene-types` consistently?**

**Answer:** Use **identical syntax** in both tools:

```bash
# ✅ CONSISTENT APPROACH
# Step 1: Strategic selection
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    meta-optimized \
    --count 2000 \
    --gene-types protein_coding \
    --output strategic_pc.txt

# Step 2: Training dataset creation  
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --gene-types protein_coding \
    --gene-ids-file strategic_pc.txt \
    --output-dir train_pc_7000_strategic
```

**Key Point:** Both tools use the **exact same `--gene-types protein_coding`** specification.

### **Q2: Does `prepare_gene_lists.py` support gene type selection?**

**Answer:** **✅ YES** - `prepare_gene_lists.py` now **FULLY SUPPORTS** gene type filtering!

**Enhanced Functionality:**
```bash
# Now supports gene type filtering!
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --training 20 \
    --unseen 30 \
    --gene-types protein_coding \
    --study-name "evaluation"
# Result: Only protein-coding genes selected for both training and unseen sets
```

**Complete Gene Type Consistency:** All three tools now support identical gene type filtering!

---

## Gene Type Syntax Reference

### **Single Gene Type**
```bash
--gene-types protein_coding
```

### **Multiple Gene Types**
```bash
--gene-types protein_coding lncRNA
--gene-types protein_coding lncRNA pseudogene
```

### **All Tools Comparison**

| Tool | Gene Type Support | Example |
|------|------------------|---------|
| `strategic_gene_selector.py` | ✅ **Full Support** | `--gene-types protein_coding` |
| `incremental_builder.py` | ✅ **Full Support** | `--gene-types protein_coding` |
| `prepare_gene_lists.py` | ✅ **Full Support** | `--gene-types protein_coding` |

---

## Complete Example Workflow

### **Protein-Coding Focused Training**

```bash
# 1. Strategic gene selection (protein-coding only)
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    meta-optimized \
    --count 2000 \
    --gene-types protein_coding \
    --output strategic_pc_2000.txt \
    --verbose

# 2. Training dataset creation (protein-coding only)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --subset-policy random \
    --gene-types protein_coding \
    --gene-ids-file strategic_pc_2000.txt \
    --output-dir train_pc_7000_strategic \
    --batch-size 500 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --verbose

# 3. Verification (should be 100% protein-coding)
python -c "
import polars as pl
manifest = pl.read_csv('train_pc_7000_strategic/gene_manifest.csv')
print(f'Total genes: {len(manifest):,}')
print('Gene type distribution:')
print(manifest['gene_type'].value_counts())
"

# 4. Evaluation gene lists (⚠️ no gene type filtering available)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --training 20 \
    --unseen 30 \
    --study-name "pc_evaluation" \
    --training-dataset train_pc_7000_strategic \
    --verbose
```

### **Multi-Type Training**

```bash
# 1. Strategic selection (protein-coding + lncRNA)
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    meta-optimized \
    --count 1500 \
    --gene-types protein_coding lncRNA \
    --output strategic_pc_lnc_1500.txt \
    --verbose

# 2. Training dataset (protein-coding + lncRNA)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 3500 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --gene-ids-file strategic_pc_lnc_1500.txt \
    --output-dir train_pc_lnc_5000_strategic \
    --run-workflow \
    --kmer-sizes 3 \
    --verbose
```

---

## Validation Commands

### **Check Gene Type Consistency**

```bash
# Verify strategic genes match expected types
python -c "
from meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector import StrategicGeneSelector
import polars as pl

# Load strategic genes
with open('strategic_pc_2000.txt') as f:
    strategic_genes = [line.strip() for line in f]

# Check their types
selector = StrategicGeneSelector(verbose=False)
gene_df = selector.gene_characteristics_df.filter(
    pl.col('gene_id').is_in(strategic_genes)
)

print(f'Strategic genes: {len(strategic_genes)}')
print('Gene types in strategic selection:')
print(gene_df['gene_type'].value_counts())
"

# Verify training dataset consistency
python -c "
import polars as pl
manifest = pl.read_csv('train_pc_7000_strategic/gene_manifest.csv')
expected_types = ['protein_coding']  # Adjust as needed

actual_types = set(manifest['gene_type'].unique())
expected_set = set(expected_types)

if actual_types == expected_set:
    print('✅ Gene types are consistent')
else:
    print('❌ Gene type mismatch!')
    print(f'Expected: {expected_set}')
    print(f'Actual: {actual_types}')
    print(f'Extra types: {actual_types - expected_set}')
    print(f'Missing types: {expected_set - actual_types}')
"
```

---

## Common Patterns

### **Pattern 1: Pure Protein-Coding**
```bash
# Both tools use identical specification
--gene-types protein_coding
```
**Result:** 100% protein-coding genes in training dataset

### **Pattern 2: Protein-Coding + lncRNA**
```bash
# Both tools use identical specification
--gene-types protein_coding lncRNA
```
**Result:** Mix of protein-coding and lncRNA genes

### **Pattern 3: Comprehensive RNA**
```bash
# Both tools use identical specification  
--gene-types protein_coding lncRNA miRNA
```
**Result:** Multiple RNA gene types

---

## Troubleshooting

### **Issue: Strategic genes don't match training dataset types**

**Cause:** Different `--gene-types` specifications

**Solution:**
```bash
# ❌ WRONG - Inconsistent gene types
strategic_gene_selector.py --gene-types protein_coding
incremental_builder.py --gene-types protein_coding lncRNA  # Different!

# ✅ CORRECT - Consistent gene types
strategic_gene_selector.py --gene-types protein_coding
incremental_builder.py --gene-types protein_coding  # Same!
```

### **Issue: Want to use ALL genes of specific types**

**Solution:** Use the new `--subset-policy all` option:

```bash
# Use ALL protein-coding genes (~20,089 genes)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --subset-policy all \
    --gene-types protein_coding \
    --output-dir train_pc_all_3mers \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3

# Use ALL protein-coding + lncRNA genes (~39,347 genes)  
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --subset-policy all \
    --gene-types protein_coding lncRNA \
    --output-dir train_pc_lncrna_all_3mers \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3
```

**Benefits:**
- Maximum gene coverage for comprehensive training
- No need to specify `--n-genes` (automatically uses all available)
- Ignores `--gene-ids-file` with clear warning
- Perfect for final production models

### **Issue: Evaluation genes include unexpected types**

**Status:** **✅ RESOLVED** - `prepare_gene_lists.py` now supports gene type filtering!

**New Solution:**
```bash
# Now supports gene type filtering directly!
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --training 20 \
    --unseen 30 \
    --gene-types protein_coding \
    --study-name "pc_evaluation" \
    --training-dataset train_pc_7000_strategic \
    --verbose

# Result: Only protein-coding genes in both training and unseen evaluation sets
```

**Old Workaround (no longer needed):**
```bash
# This manual filtering is no longer necessary
# prepare_gene_lists.py now handles gene type filtering automatically
```

---

## Future Enhancement Proposal

### **Enhanced `prepare_gene_lists.py`**

```python
# Proposed command-line interface
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --training 20 \
    --unseen 30 \
    --gene-types protein_coding \  # NEW PARAMETER
    --study-name "pc_evaluation" \
    --training-dataset train_pc_7000_strategic \
    --verbose
```

This would ensure complete gene type consistency across the entire workflow from training to evaluation.

---

## Summary

**Key Takeaways:**

1. **✅ Strategic Gene Selector**: Full gene type support
2. **✅ Incremental Builder**: Full gene type support  
3. **❌ Prepare Gene Lists**: No gene type support (enhancement needed)
4. **🔑 Consistency Rule**: Use identical `--gene-types` in both supported tools
5. **⚠️ Evaluation Limitation**: Manual filtering required for gene type consistency

**Recommended Workflow:**
1. Use consistent `--gene-types` in strategic selection and training
2. Verify training dataset gene type composition
3. Apply manual filtering to evaluation gene lists if needed
4. Document your gene type choices for reproducibility
