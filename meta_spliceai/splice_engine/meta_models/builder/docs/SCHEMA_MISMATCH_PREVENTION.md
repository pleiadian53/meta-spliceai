# Schema Mismatch Prevention Guide

**Date:** August 2025  
**Status:** ✅ **ACTIVE PREVENTION**  

## 🚨 **Problem Summary**

Schema mismatches occur when batch files in a dataset have inconsistent column schemas, particularly with k-mer features. This causes training scripts to fail with `polars.exceptions.SchemaError`.

### **Common Causes**
1. **Ambiguous k-mers**: Some batches contain `3mer_NNN`, `3mer_TCN` etc. while others don't
2. **Inconsistent filtering**: Different batches processed with different quality filters
3. **Sequence quality variations**: Some genes have sequences with 'N' nucleotides
4. **Processing timing**: Batches created at different times with different logic

## 🔧 **Prevention Strategies**

### **1. Pre-Dataset Creation Validation**

**Use the validation script before training:**
```bash
# Validate existing dataset
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py \
    --dataset train_pc_5000_3mers_diverse/master

# Auto-fix issues if found
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py \
    --dataset train_pc_5000_3mers_diverse/master --fix
```

### **2. Enhanced Sequence Featurizer**

**Fixed in `sequence_featurizer.py`:**
- ✅ **Consistent filtering**: All k-mer sizes now use `is_valid_kmer()` filtering
- ✅ **Strict N-filtering**: Any k-mer with 'N' is automatically excluded
- ✅ **Standardized output**: Only clean A,C,G,T k-mers are included

### **3. Dataset Creation Best Practices**

**When creating new datasets:**

```python
# Always use consistent filtering
featurize_gene_sequences(
    df=gene_df,
    kmer_sizes=[3, 5],  # Specify exact k-mer sizes
    filter_invalid_kmers=True,  # CRITICAL: Always enable
    verbose=True
)
```

**Quality Control Checklist:**
- [ ] All sequences pre-filtered for ambiguous nucleotides
- [ ] Consistent k-mer filtering applied across all batches
- [ ] Schema validation run after dataset creation
- [ ] Test loading with training script before full training

### **4. Expected K-mer Counts**

**Standard k-mer counts (no ambiguous nucleotides):**
- **1-mers**: 4 (A, C, G, T)
- **2-mers**: 16 (AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT)
- **3-mers**: 64 (AAA through TTT)
- **4-mers**: 256
- **5-mers**: 1,024
- **6-mers**: 4,096

**If you see different counts, investigate!**

## 🛠️ **Detection and Fixing**

### **Quick Detection**
```bash
# Check for schema issues
python -c "
import pandas as pd
import glob

batch_files = glob.glob('dataset/master/batch_*.parquet')
column_counts = []
kmer_counts = []

for batch in batch_files:
    df = pd.read_parquet(batch)
    kmer_cols = [col for col in df.columns if col.startswith('3mer_')]
    column_counts.append(len(df.columns))
    kmer_counts.append(len(kmer_cols))

print(f'Column counts: {min(column_counts)}-{max(column_counts)}')
print(f'3-mer counts: {min(kmer_counts)}-{max(kmer_counts)}')

if len(set(column_counts)) > 1 or len(set(kmer_counts)) > 1:
    print('⚠️ SCHEMA INCONSISTENCY DETECTED!')
else:
    print('✅ Schema is consistent')
"
```

### **Manual Fixing**
```python
import pandas as pd
import glob

# Find problematic batches
batch_files = glob.glob('dataset/master/batch_*.parquet')
for batch_file in batch_files:
    df = pd.read_parquet(batch_file)
    kmer_cols = [col for col in df.columns if col.startswith('3mer_')]
    
    # Check for ambiguous k-mers
    ambiguous = [col for col in kmer_cols if 'N' in col]
    if ambiguous:
        print(f"Fixing {batch_file}: {ambiguous}")
        
        # Remove ambiguous k-mers
        df_fixed = df.drop(columns=ambiguous)
        df_fixed.to_parquet(batch_file, index=False)
        print(f"  ✅ Fixed: removed {len(ambiguous)} columns")
```

## 📋 **Integration with Workflow**

### **Automated Validation in CI/CD**
```yaml
# Add to your CI pipeline
- name: Validate Dataset Schema
  run: |
    python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py \
      --dataset ${{ env.DATASET_PATH }}
```

### **Pre-Training Check**
```bash
# Always run before training
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py \
    --dataset train_pc_5000_3mers_diverse/master

# If issues found, fix them
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py \
    --dataset train_pc_5000_3mers_diverse/master --fix
```

## 🎯 **Key Takeaways**

1. **Always validate schemas** before training
2. **Use consistent filtering** in sequence featurization
3. **Expect standard k-mer counts** (64 for 3-mers)
4. **Automate validation** in your workflow
5. **Fix issues early** before they cause training failures

## 📞 **Support**

If you encounter schema issues:
1. Run the validation script to identify problems
2. Use `--fix` flag to auto-resolve
3. Check sequence quality in your input data
4. Verify k-mer filtering settings in featurization

---

**Remember**: Schema consistency is critical for successful training. Always validate before training! 🎯





