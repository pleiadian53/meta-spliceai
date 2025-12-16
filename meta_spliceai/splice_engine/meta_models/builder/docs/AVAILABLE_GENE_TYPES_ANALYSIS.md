# Available Gene Types Analysis for Noncoding Regulatory Enhancement

**Analysis of gene types available in `data/ensembl/spliceai_analysis/gene_features.tsv` for regulatory splice prediction**

---

## Gene Type Distribution in Current Dataset

Based on analysis of `data/ensembl/spliceai_analysis/gene_features.tsv`:

| Gene Type | Count | Percentage | Relevance for Splice Regulation |
|-----------|-------|------------|----------------------------------|
| **protein_coding** | 20,089 | 31.8% | ✅ **High** - Primary splice targets |
| **lncRNA** | 19,258 | 30.5% | ✅ **High** - Regulatory RNAs affecting nearby gene splicing |
| **processed_pseudogene** | 10,144 | 16.1% | ⚠️ **Low** - Generally not transcribed |
| **unprocessed_pseudogene** | 2,602 | 4.1% | ⚠️ **Low** - Generally not transcribed |
| **misc_RNA** | 2,217 | 3.5% | ✅ **Medium** - Various regulatory RNAs |
| **snRNA** | 1,910 | 3.0% | ✅ **High** - Splicing machinery components |
| **miRNA** | 1,879 | 3.0% | ✅ **Medium** - Post-transcriptional regulation |
| **TEC** | 1,052 | 1.7% | ⚠️ **Unknown** - "To be Experimentally Confirmed" |
| **transcribed_unprocessed_pseudogene** | 962 | 1.5% | ✅ **Low-Medium** - May have regulatory function |
| **snoRNA** | 942 | 1.5% | ✅ **Medium** - RNA modification, some splice regulation |
| **transcribed_processed_pseudogene** | 513 | 0.8% | ✅ **Low-Medium** - May have regulatory function |
| **rRNA_pseudogene** | 497 | 0.8% | ❌ **Very Low** - Ribosomal RNA related |
| **Others** | <200 each | <0.3% each | Various |

**Total Genes**: 63,142

---

## Recommended Gene Type Combinations for Splice Regulation

### **Option 1: Conservative Regulatory Focus (Recommended)**
```bash
--gene-types protein_coding,lncRNA,miRNA,snoRNA,snRNA,misc_RNA
```
- **Total genes**: ~46,295 (73.3% of dataset)
- **Rationale**: Includes all gene types with established splice regulatory roles
- **Benefits**: High-confidence regulatory elements, manageable dataset size

### **Option 2: Expanded Regulatory Coverage**
```bash
--gene-types protein_coding,lncRNA,miRNA,snoRNA,snRNA,misc_RNA,transcribed_unprocessed_pseudogene,transcribed_processed_pseudogene
```
- **Total genes**: ~47,770 (75.6% of dataset)
- **Rationale**: Includes transcribed pseudogenes that may have regulatory functions
- **Benefits**: Captures potential novel regulatory elements

### **Option 3: Maximum Coverage (Research/Discovery)**
```bash
--gene-types protein_coding,lncRNA,miRNA,snoRNA,snRNA,misc_RNA,transcribed_unprocessed_pseudogene,transcribed_processed_pseudogene,TEC
```
- **Total genes**: ~48,822 (77.3% of dataset)
- **Rationale**: Includes experimental/uncertain gene types for discovery
- **Benefits**: Maximum potential regulatory coverage, good for research

### **Option 4: Protein-Coding + lncRNA Only (Focused)**
```bash
--gene-types protein_coding,lncRNA
```
- **Total genes**: ~39,347 (62.3% of dataset)
- **Rationale**: Focus on the two most abundant and well-characterized types
- **Benefits**: Simpler model, faster training, well-understood biology

---

## Gene Type Functional Relevance for Splice Regulation

### **High Relevance (Include in all configurations)**

#### **protein_coding (20,089 genes)**
- Primary targets of splice regulation
- Contain canonical and alternative splice sites
- Essential for training splice prediction models

#### **lncRNA (19,258 genes)**
- **Critical for regulatory enhancement**
- Many lncRNAs regulate splicing of nearby protein-coding genes
- Can act as splice enhancers or silencers
- Examples: MALAT1, NEAT1 affect splice site selection

#### **snRNA (1,910 genes)**
- **Essential splice machinery components**
- U1, U2, U4, U5, U6 snRNAs are core spliceosome components
- Variants in snRNA genes can affect global splicing patterns
- Critical for understanding splice mechanism

### **Medium Relevance (Include for comprehensive coverage)**

#### **miRNA (1,879 genes)**
- Post-transcriptional regulators
- Some miRNAs affect splice site selection
- Can influence alternative splicing patterns
- Relevant for tissue-specific splice regulation

#### **misc_RNA (2,217 genes)**
- Heterogeneous group of regulatory RNAs
- May include novel splice regulatory elements
- Worth including for discovery potential

#### **snoRNA (942 genes)**
- Primarily involved in RNA modification
- Some snoRNAs have splice regulatory functions
- Can affect splice site recognition through RNA modifications

### **Low Relevance (Consider excluding)**

#### **Pseudogenes (processed_pseudogene, unprocessed_pseudogene)**
- Generally not transcribed (processed pseudogenes)
- May lack regulatory function
- Large number (12,746 total) may add noise
- **Exception**: Transcribed pseudogenes may have regulatory roles

#### **TEC (1,052 genes)**
- "To be Experimentally Confirmed"
- Unknown functional relevance
- May include both regulatory and non-regulatory elements

---

## Updated Incremental Builder Commands

### **Immediate Use (Conservative Regulatory)**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 10000 \
    --subset-policy meta_optimized \
    --gene-types protein_coding,lncRNA,miRNA,snoRNA,snRNA,misc_RNA \
    --gene-ids-file strategic_regulatory_genes.txt \
    --output-dir train_regulatory_enhanced_kmers \
    --batch-size 200 \
    --batch-rows 15000 \
    --run-workflow \
    --kmer-sizes 3,5 \
    --verbose 2>&1 | tee -a logs/train_regulatory_enhanced.log
```

### **Research/Discovery Mode (Maximum Coverage)**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 12000 \
    --subset-policy all \
    --gene-types protein_coding,lncRNA,miRNA,snoRNA,snRNA,misc_RNA,transcribed_unprocessed_pseudogene,transcribed_processed_pseudogene \
    --output-dir train_comprehensive_regulatory \
    --batch-size 150 \
    --batch-rows 12000 \
    --run-workflow \
    --kmer-sizes 3,5 \
    --verbose 2>&1 | tee -a logs/train_comprehensive_regulatory.log
```

### **Focused lncRNA + Protein-Coding**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 8000 \
    --subset-policy random \
    --gene-types protein_coding,lncRNA \
    --gene-ids-file strategic_combined_pc.txt \
    --output-dir train_focused_lncRNA_pc \
    --batch-size 250 \
    --batch-rows 15000 \
    --run-workflow \
    --kmer-sizes 3,5 \
    --verbose 2>&1 | tee -a logs/train_focused_lncRNA_pc.log
```

---

## Impact Analysis

### **Adding lncRNAs (19,258 genes)**
- **Benefit**: Captures ~30% more potential regulatory elements
- **Impact**: Major improvement in noncoding regulatory pattern detection
- **Cost**: ~2x increase in dataset size

### **Adding Small RNAs (miRNA, snoRNA, snRNA, misc_RNA: ~6,948 genes)**
- **Benefit**: Captures specialized regulatory mechanisms
- **Impact**: Improved detection of RNA-mediated splice regulation
- **Cost**: ~11% increase in dataset size

### **Adding Transcribed Pseudogenes (~1,475 genes)**
- **Benefit**: May capture novel regulatory elements
- **Impact**: Potential discovery of new regulatory mechanisms
- **Cost**: ~2% increase in dataset size, possible noise

---

## Validation Strategy

### **Gene Type-Specific Validation**
1. **Protein-coding genes**: Standard splice site prediction metrics
2. **lncRNAs**: Correlation with nearby gene splice changes
3. **snRNAs**: Global splicing pattern effects
4. **miRNAs**: Tissue-specific splice regulation
5. **Small RNAs**: Specialized regulatory mechanism detection

### **Comparative Analysis**
- Compare models trained with different gene type combinations
- Measure improvement in noncoding variant prediction
- Assess tissue-specific splice prediction accuracy

---

## Recommendations

### **For Immediate Implementation**
Use **Option 1 (Conservative Regulatory Focus)**:
- Includes all high-confidence splice-regulatory gene types
- Manageable dataset size for training
- Clear biological rationale for each gene type

### **For Research/Discovery**
Use **Option 2 (Expanded Regulatory Coverage)**:
- Includes transcribed pseudogenes for potential novel discoveries
- Comprehensive coverage of regulatory elements
- Good balance of coverage vs. noise

### **For Production/Clinical Use**
Start with **Option 4 (Protein-coding + lncRNA)**:
- Focus on well-characterized gene types
- Faster training and inference
- Lower risk of false positives from poorly characterized elements

---

## Next Steps

1. **Immediate**: Test Option 1 command with available gene types
2. **Short-term**: Validate performance improvement from including lncRNAs
3. **Medium-term**: Implement tissue-specific expression filtering
4. **Long-term**: Add chromatin accessibility and regulatory motif features

This analysis ensures that the noncoding regulatory enhancement uses only gene types that are actually available in the current dataset, maximizing the potential for improved splice prediction while maintaining biological relevance.

