# Enhanced Incremental Builder Configuration for Noncoding Regulatory Variants

## Current Command Analysis
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 7000 \
    --subset-policy random \
    --gene-types protein_coding \
    --gene-ids-file strategic_combined_pc.txt \
    --output-dir train_pc_7000_3mers_opt \
    --batch-size 250 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --verbose 2>&1 | tee -a logs/train_pc_7000_3mers_opt.log
```

## Key Limitations for Noncoding Regulatory Patterns

### 1. **Gene Type Restriction**
- `--gene-types protein_coding` excludes crucial noncoding genes
- Missing: lncRNAs, miRNAs, regulatory RNAs that affect splicing

### 2. **K-mer Size Limitation** 
- `--kmer-sizes 3` may miss longer regulatory motifs
- Regulatory elements often span 6-12 nucleotides

### 3. **Missing Regulatory Context**
- No explicit inclusion of regulatory regions
- No consideration of chromatin accessibility
- No tissue-specific expression patterns

## Enhanced Configuration for Noncoding Regulatory Patterns

### **Option 1: Comprehensive Gene Types (Recommended)**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 10000 \
    --subset-policy meta_optimized \
    --gene-types protein_coding,lncRNA,miRNA,snoRNA,snRNA,misc_RNA \
    --gene-ids-file strategic_combined_comprehensive.txt \
    --output-dir train_comprehensive_regulatory_5mers \
    --batch-size 200 \
    --batch-rows 15000 \
    --run-workflow \
    --kmer-sizes 3,5 \
    --include-regulatory-regions \
    --regulatory-window 2000 \
    --include-chromatin-features \
    --verbose 2>&1 | tee -a logs/train_comprehensive_regulatory.log
```

### **Option 2: All Gene Types (Maximum Coverage)**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 12000 \
    --subset-policy all \
    --output-dir train_all_genes_regulatory_kmers \
    --batch-size 150 \
    --batch-rows 12000 \
    --run-workflow \
    --kmer-sizes 3,5 \
    --include-regulatory-regions \
    --regulatory-window 5000 \
    --include-tissue-expression \
    --include-chromatin-features \
    --include-conservation-scores \
    --verbose 2>&1 | tee -a logs/train_all_genes_regulatory.log
```

### **Option 3: Regulatory-Focused Training**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 8000 \
    --subset-policy high_density \
    --gene-types protein_coding,lncRNA,misc_RNA \
    --gene-ids-file regulatory_splice_targets.txt \
    --output-dir train_regulatory_focused_7mers \
    --batch-size 200 \
    --batch-rows 15000 \
    --run-workflow \
    --kmer-sizes 3,5 \
    --focus-regulatory-splicing \
    --include-enhancer-regions \
    --include-silencer-regions \
    --regulatory-window 3000 \
    --include-tissue-expression \
    --verbose 2>&1 | tee -a logs/train_regulatory_focused.log
```

## New Parameters Needed (Implementation Required)

### **Regulatory Region Parameters**
```bash
--include-regulatory-regions     # Include known regulatory elements
--regulatory-window 2000         # bp window around splice sites for regulatory context
--include-enhancer-regions       # Include splice enhancer sequences  
--include-silencer-regions       # Include splice silencer sequences
--focus-regulatory-splicing      # Prioritize regulatory splice patterns
```

### **Genomic Context Parameters**
```bash
--include-chromatin-features     # Include chromatin accessibility data
--include-conservation-scores    # Include phylogenetic conservation
--include-tissue-expression      # Include tissue-specific expression patterns
--include-variant-annotations    # Include known pathogenic variant annotations
```

### **Advanced K-mer Parameters**
```bash
--kmer-sizes 3,5,7,9            # Multiple k-mer sizes for regulatory motifs
--regulatory-kmer-focus         # Weight k-mers in regulatory regions more heavily
--motif-aware-kmers            # Use known splice regulatory motifs as features
```

## Implementation Priority for Noncoding Support

### **Phase 1: Immediate Improvements (1-2 weeks)**
1. **Expand gene types**: Remove `protein_coding` restriction
2. **Add longer k-mers**: Include 5-mers and 7-mers for regulatory motifs
3. **Increase regulatory window**: Expand context around splice sites

```bash
# Quick improvement - use this now:
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 8000 \
    --subset-policy random \
    --gene-types protein_coding,lncRNA,misc_RNA \
    --gene-ids-file strategic_combined_pc.txt \
    --output-dir train_expanded_5_7mers \
    --batch-size 200 \
    --batch-rows 15000 \
    --run-workflow \
    --kmer-sizes 3,5 \
    --verbose 2>&1 | tee -a logs/train_expanded_kmers.log
```

### **Phase 2: Regulatory Features (2-4 weeks)**
1. Implement `--include-regulatory-regions` parameter
2. Add chromatin accessibility features (ENCODE data)
3. Include tissue-specific expression patterns
4. Add conservation scores (phyloP, phastCons)

### **Phase 3: Advanced Regulatory Modeling (4-8 weeks)**
1. Implement splice regulatory motif detection
2. Add enhancer/silencer region identification  
3. Include variant pathogenicity annotations
4. Implement tissue-specific splice pattern modeling

## Expected Benefits for Noncoding Regulatory Patterns

### **Improved Detection of:**
1. **Cryptic splice sites** activated by regulatory variants
2. **Exon skipping** caused by silencer mutations
3. **Intron retention** from regulatory disruption
4. **Alternative splicing** changes from noncoding variants
5. **Tissue-specific splicing** alterations

### **Better Performance on:**
1. **Deep intronic variants** affecting splicing
2. **UTR variants** with regulatory effects
3. **lncRNA variants** affecting nearby gene splicing
4. **Enhancer/silencer variants** in noncoding regions
5. **Structural variants** disrupting regulatory domains

## Validation Strategy

### **Test Datasets Needed:**
1. **ClinVar splice variants**: Known pathogenic splice-affecting variants
2. **GTEx splice QTLs**: Variants affecting splicing in population data
3. **Deep intronic variants**: Variants >50bp from splice sites
4. **Regulatory region variants**: Variants in known enhancers/silencers
5. **Tissue-specific datasets**: Variants with tissue-specific splice effects

### **Evaluation Metrics:**
1. **Regulatory variant recall**: % of known regulatory splice variants detected
2. **Deep intronic performance**: Accuracy on variants >100bp from splice sites  
3. **Tissue specificity**: Correlation with tissue-specific splice changes
4. **Novel variant discovery**: Identification of previously unknown splice-affecting variants

## Dataset Post-Processing: Gene Type Filtering

### **Use Case: Iterative Dataset Refinement**

After initial dataset creation with diverse gene types, you may need to filter to specific gene types based on:
1. **Quality Analysis**: Remove unreliable gene type annotations
2. **Performance Analysis**: Focus on gene types that improve meta-model performance
3. **Production Requirements**: Use only high-confidence gene annotations

### **Dataset Filtering Utility**

**Location**: `meta_spliceai/splice_engine/meta_models/builder/filter_dataset_by_gene_type.py`

**Purpose**: Filter existing training datasets to keep only specified gene types while preserving all position-centric training instances and enhanced manifest information.

### **Usage Examples**

#### **Filter to Reliable Gene Types (Recommended)**
```bash
# Keep only protein-coding and lncRNA genes (most reliable)
python -m meta_spliceai.splice_engine.meta_models.builder.filter_dataset_by_gene_type \
    train_pc_5000_3mers_diverse \
    train_pc_lnc_2779_3mers_filtered \
    --gene-types protein_coding lncRNA
```

#### **Filter to Protein-Coding Only**
```bash
# Maximum splice site density and reliability
python -m meta_spliceai.splice_engine.meta_models.builder.filter_dataset_by_gene_type \
    train_diverse_dataset \
    train_protein_coding_only \
    --gene-types protein_coding
```

#### **Custom Gene Type Selection**
```bash
# Include transcribed pseudogenes for evolutionary analysis
python -m meta_spliceai.splice_engine.meta_models.builder.filter_dataset_by_gene_type \
    input_dataset \
    filtered_dataset \
    --gene-types protein_coding lncRNA transcribed_unprocessed_pseudogene
```

### **Gene Type Reliability Guide**

| Gene Type | Reliability | Splice Density | Recommended Use |
|-----------|-------------|----------------|-----------------|
| `protein_coding` | ✅ **High** | High (124.6 avg sites) | **Always include** |
| `lncRNA` | ✅ **High** | Medium (20.4 avg sites) | **Recommended** |
| `transcribed_unprocessed_pseudogene` | ⚠️ **Medium** | Medium (14.6 avg sites) | Research only |
| `processed_pseudogene` | ❌ **Low** | Low (2.4 avg sites) | **Avoid** |
| `unprocessed_pseudogene` | ❌ **Low** | Low (7.2 avg sites) | **Avoid** |
| `IG_V_gene`, `TR_V_gene` | ⚠️ **Specialized** | Low (2.0 avg sites) | Immunology research |
| `artifact` | ❌ **Poor** | Low (2.0 avg sites) | **Remove** |

### **Filtering Impact Analysis**

**Example: protein_coding + lncRNA filtering**
- **Genes Retained**: 89.3% (2,779 / 3,111)
- **Records Retained**: 98.0% (572,832 / 584,379) 
- **Splice Sites Retained**: 99.0% (211,944 / 214,192)
- **Quality Improvement**: Removes low-reliability annotations while preserving training data

### **Integration with Training Workflow**

```bash
# Step 1: Create diverse dataset for exploration
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --gene-types protein_coding,lncRNA,processed_pseudogene,unprocessed_pseudogene \
    --output-dir train_diverse_exploration \
    --kmer-sizes 3,5 \
    --verbose

# Step 2: Analyze gene type performance (manual analysis)
# ... train meta-model and evaluate performance by gene type ...

# Step 3: Filter to optimal gene types based on analysis
python -m meta_spliceai.splice_engine.meta_models.builder.filter_dataset_by_gene_type \
    train_diverse_exploration \
    train_optimized_filtered \
    --gene-types protein_coding lncRNA

# Step 4: Train production meta-model on filtered dataset
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_optimized_filtered/master \
    --out-dir results/production_model \
    --row-cap 0 \
    --verbose
```

### **Future Enhancements**

**Planned Features** (to be implemented):
1. **Error-Based Filtering**: Remove genes/positions with high prediction errors
2. **Confidence-Based Filtering**: Keep only high-confidence splice site predictions
3. **Tissue-Specific Filtering**: Filter based on tissue-specific expression patterns
4. **Variant-Impact Filtering**: Focus on genes with known disease-associated variants

## Conclusion

Your current command is optimized for protein-coding gene splicing patterns but will miss crucial noncoding regulatory variants that represent the majority of pathogenic splice-affecting mutations. The enhanced configurations above will significantly improve the model's ability to detect and predict the effects of noncoding regulatory variants on splicing.

**Immediate Action**: Use the Phase 1 quick improvement command to start capturing longer regulatory motifs while planning implementation of the advanced regulatory features.

**Dataset Management**: Use the gene type filtering utility to iteratively refine datasets based on performance analysis and production requirements.

