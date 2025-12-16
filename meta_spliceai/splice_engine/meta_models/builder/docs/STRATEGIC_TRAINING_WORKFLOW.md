# Strategic Training Dataset Creation Workflow

**Complete guide for creating gene-type-focused training datasets with strategic enhancements**

## Overview

This workflow combines three powerful tools to create optimized training datasets:

1. **`strategic_gene_selector.py`** - Advanced gene selection based on characteristics
2. **`incremental_builder.py`** - Training dataset creation with gene type filtering  
3. **`prepare_gene_lists.py`** - Evaluation gene list preparation (inference workflow)

## Key Features

- ✅ **Gene Type Consistency** across all tools
- ✅ **Strategic Enhancement** with characteristic-based selection
- ✅ **Automatic Deduplication** when combining random + strategic genes
- ✅ **Enhanced Manifests** with comprehensive gene characteristics
- ✅ **End-to-End Workflow** from training to evaluation

---

## Gene Selection Strategies

### **Available Subset Policies**

| Policy | Description | Use Case | Gene Count |
|--------|-------------|----------|------------|
| `random` | Random selection for diversity | Balanced training | `n_genes` |
| `error_total` | Genes with most prediction errors | Error-focused training | `n_genes` |
| `error_fp` | Genes with most false positives | FP reduction training | `n_genes` |
| `error_fn` | Genes with most false negatives | FN reduction training | `n_genes` |
| `custom` | Use only provided gene list | Strategic-only training | `len(gene_file)` |
| `all` | **NEW**: All available genes | Maximum coverage | All available |

### **Gene Count Logic & Overlap Handling**

The system intelligently handles overlaps between strategic files and random selection:

```bash
# Scenario 1: Strategic < n_genes (normal case)
--n-genes 7000 --gene-ids-file strategic_2000.txt
# Result: Exactly 7000 genes (2000 strategic + 5000 random)

# Scenario 2: Strategic > n_genes (strategic takes precedence)  
--n-genes 1000 --gene-ids-file strategic_2000.txt
# Result: 2000 genes (all strategic, no random)

# Scenario 3: All genes (ignores n_genes and gene files)
--subset-policy all --gene-types protein_coding
# Result: ~20,089 genes (all protein-coding)
```

**Formula:** `final_count = max(n_genes, len(strategic_genes))` for non-`all` policies

---

## Gene Type Specification

### **Consistent `--gene-types` Usage**

All three tools use **identical gene type syntax** for consistency:

```bash
# Single gene type
--gene-types protein_coding

# Multiple gene types  
--gene-types protein_coding lncRNA

# All available types
--gene-types protein_coding lncRNA pseudogene miRNA TEC
```

### **Available Gene Types**

Based on Ensembl gene features (`data/ensembl/spliceai_analysis/gene_features.tsv`):

| Gene Type | Description | Typical Count |
|-----------|-------------|---------------|
| `protein_coding` | Protein-coding genes | ~20,000 |
| `lncRNA` | Long non-coding RNA | ~15,000 |
| `pseudogene` | Non-functional gene copies | ~14,000 |
| `processed_pseudogene` | Processed pseudogenes | ~10,000 |
| `unprocessed_pseudogene` | Unprocessed pseudogenes | ~2,000 |
| `miRNA` | MicroRNA genes | ~4,000 |
| `TEC` | To be Experimentally Confirmed | ~1,000 |

---

## Complete Workflow Example

### **Phase 1: Strategic Gene Selection**

Create strategic gene lists focusing on protein-coding genes:

```bash
# Activate environment
mamba activate surveyor

# 1. Meta-optimized protein-coding genes (best for meta-model)
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    meta-optimized \
    --count 1000 \
    --gene-types protein_coding \
    --output strategic_meta_optimized_pc.txt \
    --verbose

# 2. High splice density protein-coding genes  
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    high-density \
    --count 600 \
    --min-density 12.0 \
    --gene-types protein_coding \
    --output strategic_high_density_pc.txt \
    --verbose

# 3. Length-stratified protein-coding genes
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    length-strata \
    --ranges 20000,50000 50000,150000 150000,500000 \
    --counts 200,200,200 \
    --gene-types protein_coding \
    --output-dir strategic_length_pc \
    --verbose

# 4. Combine all strategic selections
cat strategic_meta_optimized_pc.txt \
    strategic_high_density_pc.txt \
    strategic_length_pc/all_length_strata.txt > strategic_combined_pc.txt

# 5. Remove duplicates
sort strategic_combined_pc.txt | uniq > strategic_final_2000_pc.txt

echo "Strategic gene selection complete: $(wc -l < strategic_final_2000_pc.txt) genes"
```

### **Phase 2: Training Dataset Creation**

Create the combined training dataset with consistent gene types:

#### **Option A: Strategic + Random Combination (Recommended)**

```bash
# Build 5000 random + 2000 strategic = exactly 7000 total protein-coding genes
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 7000 \
    --subset-policy random \
    --gene-types protein_coding \
    --gene-ids-file strategic_final_2000_pc.txt \
    --output-dir train_pc_7000_3mers_strategic \
    --batch-size 500 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --verbose
```

#### **Option B: All Available Genes (Maximum Coverage)**

```bash
# Use ALL protein-coding genes (~20,089 genes)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --subset-policy all \
    --gene-types protein_coding \
    --output-dir train_pc_all_3mers \
    --batch-size 500 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --verbose
```

#### **Option C: Strategic Genes Only**

```bash
# Use only strategic genes (custom selection)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --gene-types protein_coding \
    --gene-ids-file strategic_final_2000_pc.txt \
    --output-dir train_pc_strategic_only_3mers \
    --batch-size 500 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --verbose
```

**Key Points:**
- `--gene-types protein_coding` ensures all genes are protein-coding
- `--gene-ids-file` adds strategic genes to the selection
- `--batch-rows 20000` prevents out-of-memory (OOM) errors with strategic gene sets
- **Smart overlap handling**: Strategic genes are always included, additional genes fill remaining slots
- **Gene count logic**: `final_count = max(n_genes, strategic_count)`
- Enhanced manifest includes comprehensive gene characteristics

### **Phase 3: Verification**

Verify the training dataset composition:

```bash
# Check enhanced manifest
python -c "
import polars as pl
manifest = pl.read_csv('train_pc_7000_3mers_strategic/gene_manifest.csv')
print(f'📊 TRAINING DATASET VERIFICATION')
print(f'=' * 50)
print(f'Total genes: {len(manifest):,}')
print(f'Gene type distribution:')
type_counts = manifest['gene_type'].value_counts()
for row in type_counts.iter_rows():
    gene_type, count = row
    print(f'  {gene_type}: {count:,}')

print(f'\\n📏 Length statistics:')
length_stats = manifest['gene_length']
print(f'  Mean: {length_stats.mean():.0f} bp')
print(f'  Median: {length_stats.median():.0f} bp')
print(f'  Range: {length_stats.min():,} - {length_stats.max():,} bp')

print(f'\\n🧬 Splice density statistics:')
density_stats = manifest['splice_density_per_kb']
print(f'  Mean: {density_stats.mean():.2f} sites/kb')
print(f'  Median: {density_stats.median():.2f} sites/kb')
print(f'  Range: {density_stats.min():.2f} - {density_stats.max():.2f} sites/kb')
"
```

---

## Evaluation Workflow

### **Phase 4: Evaluation Gene List Preparation**

**✅ Gene Type Filtering Supported:** `prepare_gene_lists.py` now supports gene type filtering with the `--gene-types` parameter, maintaining consistency with `strategic_gene_selector.py` and `incremental_builder.py`.

```bash
# Gene type filtering now supported (protein-coding only)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --training 20 \
    --unseen 30 \
    --gene-types protein_coding \
    --study-name "pc_strategic_evaluation" \
    --training-dataset train_pc_7000_3mers_strategic \
    --verbose

# Multiple gene types
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --training 15 \
    --unseen 25 \
    --gene-types protein_coding lncRNA \
    --study-name "pc_lnc_strategic_evaluation" \
    --training-dataset train_pc_7000_3mers_strategic \
    --verbose
```

**Result:** Gene type-consistent evaluation sets that match the training data composition.

### **Gene Type Consistency Achieved**

All three tools now support consistent `--gene-types` filtering:

- ✅ `strategic_gene_selector.py` - Strategic gene selection with gene type filtering
- ✅ `incremental_builder.py` - Training dataset creation with gene type filtering  
- ✅ `prepare_gene_lists.py` - Evaluation gene list preparation with gene type filtering

**Implementation Details:**

```python
# Now implemented and functional
def select_training_genes(self, training_genes: Set[str], count: int,
                         gene_types: Optional[List[str]] = None,  # ✅ IMPLEMENTED
                         min_length: int = 10000, max_length: int = 500000) -> List[str]:
    """Select diverse training genes with optional gene type filtering."""
    gene_features = self.load_all_genes()
    
    # Apply gene type filter if specified
    if gene_types:
        training_gene_features = training_gene_features[
            training_gene_features['gene_type'].isin(gene_types)
        ]
    training_gene_features = gene_features[gene_features['gene_id'].isin(training_genes)]
    
    # Apply gene type filter if specified
    if gene_types:
        training_gene_features = training_gene_features[
            training_gene_features['gene_type'].isin(gene_types)
        ]
    
    # Continue with existing length and diversity logic...
```

---

## Gene Type Consistency Matrix

| Tool | Gene Type Support | Syntax | Notes |
|------|------------------|--------|-------|
| `strategic_gene_selector.py` | ✅ Full | `--gene-types protein_coding` | All strategies support filtering |
| `incremental_builder.py` | ✅ Full | `--gene-types protein_coding` | All policies support filtering (including `all`) |
| `prepare_gene_lists.py` | ✅ Full | `--gene-types protein_coding` | **Recently enhanced** |

---

## Memory Management

### **Critical: `--batch-rows` Parameter**

**Always specify `--batch-rows 20000` for strategic training workflows** to avoid out-of-memory (OOM) errors.

**Why this matters:**
- **Default**: `--batch-rows 500000` (500K rows) - **HIGH OOM RISK**
- **Recommended**: `--batch-rows 20000` (20K rows) - **Safe for strategic workflows**
- **Strategic gene sets** often include high splice density genes that use more memory
- **Feature enrichment** and k-mer extraction increase memory per row

**Memory Usage Comparison:**
```bash
# ❌ RISKY - May cause OOM with strategic genes
--batch-rows 500000  # Default, ~8-12 GB RAM peak

# ✅ SAFE - Recommended for strategic workflows  
--batch-rows 20000   # ~1-2 GB RAM peak
```

**Performance Impact:**
- Slightly slower I/O due to more frequent disk flushes
- **Much better than OOM crashes!**
- Negligible impact on total runtime

---

## Best Practices

### **1. Gene Type Consistency**

Always use the same `--gene-types` specification across tools:

```bash
# ✅ CONSISTENT - All protein-coding
strategic_gene_selector.py --gene-types protein_coding
incremental_builder.py --gene-types protein_coding
# prepare_gene_lists.py (needs enhancement)

# ❌ INCONSISTENT - Mixed types
strategic_gene_selector.py --gene-types protein_coding
incremental_builder.py --gene-types protein_coding lncRNA  # Different!
```

### **2. Strategic Selection Approaches**

Choose the approach that best fits your training objectives:

#### **A. Strategic + Random Combination**
```bash
# Conservative enhancement (20% strategic)
--n-genes 5000 --gene-ids-file strategic_1000.txt  # = 5000 total

# Moderate enhancement (40% strategic)  
--n-genes 5000 --gene-ids-file strategic_2000.txt  # = 5000 total

# Aggressive enhancement (60% strategic)
--n-genes 5000 --gene-ids-file strategic_3000.txt  # = 5000 total
```

#### **B. Maximum Coverage (All Genes)**
```bash
# Use all available genes of specified types
--subset-policy all --gene-types protein_coding     # ~20,089 genes
--subset-policy all --gene-types protein_coding lncRNA  # ~39,347 genes
```

#### **C. Strategic-Only Training**
```bash
# Use only carefully selected strategic genes
--gene-ids-file strategic_optimized.txt  # Auto-detects custom mode
```

### **3. Verification Steps**

Always verify your training dataset:

1. **Gene count**: Matches expected total
2. **Gene types**: 100% consistency with specification
3. **Characteristics**: Strategic genes show expected properties
4. **Deduplication**: No overlap between random and strategic

### **4. Documentation**

Document your training dataset creation:

```bash
# Create dataset documentation
echo "# Training Dataset: train_pc_7000_3mers_strategic

## Configuration
- Random genes: 5000 (protein_coding)
- Strategic genes: 2000 (protein_coding)
- Total genes: ~7000 (after deduplication)
- Gene types: protein_coding only
- k-mer sizes: 3
- Created: $(date)

## Strategic Selection
- Meta-optimized: 1000 genes
- High splice density: 600 genes  
- Length-stratified: 600 genes

## Verification
$(python -c "import polars as pl; manifest = pl.read_csv('train_pc_7000_3mers_strategic/gene_manifest.csv'); print(f'Actual genes: {len(manifest):,}'); print(f'Gene types: {manifest[\"gene_type\"].value_counts()}')")
" > train_pc_7000_3mers_strategic/DATASET_INFO.md
```

---

## Troubleshooting

### **Common Issues**

1. **Out of Memory (OOM) Errors**
   - **Problem**: Process crashes with memory errors during dataset building
   - **Solution**: Always use `--batch-rows 20000` for strategic workflows
   - **Prevention**: Never use default `--batch-rows` with strategic gene sets

2. **Gene Type Mismatch**
   - **Problem**: Strategic genes don't match incremental builder gene types
   - **Solution**: Use identical `--gene-types` in both tools

3. **Lower Than Expected Gene Count**
   - **Problem**: Deduplication removes more genes than expected
   - **Solution**: Increase strategic gene count to compensate

4. **Strategic Selection Fails**
   - **Problem**: Not enough genes meet strategic criteria
   - **Solution**: Relax criteria (lower density thresholds, broader length ranges)

4. **Evaluation Gene Type Inconsistency**
   - **Problem**: `prepare_gene_lists.py` includes non-protein-coding genes
   - **Solution**: Manual filtering or wait for enhancement

### **Validation Commands**

```bash
# Check strategic gene characteristics
python -c "
import polars as pl
from meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector import StrategicGeneSelector

selector = StrategicGeneSelector(verbose=True)
with open('strategic_final_2000_pc.txt') as f:
    genes = [line.strip() for line in f]

stats = selector.get_gene_statistics(genes)
print('Strategic genes statistics:')
for key, value in stats.items():
    print(f'  {key}: {value}')
"

# Verify training dataset gene types
python -c "
import polars as pl
manifest = pl.read_csv('train_pc_7000_3mers_strategic/gene_manifest.csv')
non_pc = manifest.filter(pl.col('gene_type') != 'protein_coding')
if len(non_pc) > 0:
    print(f'⚠️ Found {len(non_pc)} non-protein-coding genes!')
    print(non_pc['gene_type'].value_counts())
else:
    print('✅ All genes are protein-coding')
"
```

---

## Future Enhancements

### **Priority 1: `prepare_gene_lists.py` Gene Type Support**

Add `--gene-types` parameter to maintain consistency:

```bash
# Proposed future syntax
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --training 20 \
    --unseen 30 \
    --gene-types protein_coding \  # NEW
    --study-name "pc_strategic_evaluation" \
    --training-dataset train_pc_7000_3mers_strategic \
    --verbose
```

### **Priority 2: Integrated Workflow Script**

Create a single script that orchestrates the entire workflow:

```bash
# Proposed integrated workflow
python -m meta_spliceai.splice_engine.meta_models.builder.integrated_strategic_workflow \
    --base-genes 5000 \
    --strategic-genes 2000 \
    --gene-types protein_coding \
    --output-dir train_pc_7000_strategic \
    --evaluation-genes 50 \
    --study-name strategic_evaluation
```

### **Priority 3: Gene Type Templates**

Pre-defined configurations for common use cases:

```bash
# Protein-coding focused
--template protein_coding_focused

# Multi-type comprehensive  
--template comprehensive_rna

# Pseudogene analysis
--template pseudogene_analysis
```

---

## Summary

This workflow provides a comprehensive approach to creating strategically enhanced, gene-type-consistent training datasets. The key is maintaining `--gene-types` consistency across `strategic_gene_selector.py` and `incremental_builder.py`, while being aware that `prepare_gene_lists.py` currently lacks gene type filtering capabilities.

**Next Steps:**
1. Use this workflow to create your protein-coding focused training dataset
2. Enhance `prepare_gene_lists.py` with gene type filtering
3. Validate training dataset composition before proceeding to model training
4. Document your specific configuration for reproducibility
