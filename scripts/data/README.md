# Splice Site Analysis Scripts

## 📁 Overview

This directory contains comprehensive analysis scripts for understanding the splice site annotation dataset (`splice_sites.tsv`) used in the MetaSpliceAI project. These scripts provide both quick insights and detailed biological analysis to support computational biology research.

**Documentation**: All findings and statistics are documented in `docs/data/splice_sites/splice_site_annotations.md`

**Directory Structure**: This directory mirrors `docs/data/` for easy cross-referencing:
- `scripts/data/splice_sites/` ↔ `docs/data/splice_sites/`
- Each script references its corresponding documentation
- Each documentation file lists the scripts used to generate its findings

## 🛠️ Available Analysis Scripts

### 1. **Quick Analysis** (`quick_splice_analysis.sh`)
**Purpose**: Fast, dependency-free analysis using standard Unix tools  
**Runtime**: ~30 seconds  
**Dependencies**: bash, awk, bc, sort, uniq  

```bash
# Run quick analysis
./scripts/data/quick_splice_analysis.sh

# With custom file path
./scripts/data/quick_splice_analysis.sh /path/to/custom/splice_sites.tsv
```

**Provides**:
- Basic dataset statistics (genes, transcripts, splice sites)
- Chromosome distribution analysis
- Top genes by splice site count
- Splice site type breakdown
- Gene complexity distribution

### 2. **Gene Pattern Analysis** (`analyze_gene_patterns.sh`)
**Purpose**: Deep dive into gene-level biological patterns  
**Runtime**: ~2-3 minutes  
**Dependencies**: bash, awk, bc, sort  

```bash
# Run detailed gene analysis
./scripts/data/analyze_gene_patterns.sh
```

**Provides**:
- Alternative splicing complexity rankings
- Splice site density analysis
- Gene size distribution statistics
- Chromosome-specific gene characteristics
- Isoform diversity patterns

### 3. **Comprehensive Analysis** (`analyze_splice_sites.py`)
**Purpose**: Complete statistical and biological analysis with Python  
**Runtime**: ~5-10 minutes  
**Dependencies**: pandas, numpy  

```bash
# Full analysis with all features
python scripts/data/analyze_splice_sites.py

# Quick mode (skip complex computations)
python scripts/data/analyze_splice_sites.py --quick

# Custom input/output
python scripts/data/analyze_splice_sites.py -i custom_file.tsv -o output_dir/
```

**Provides**:
- All features from quick analysis
- Detailed positional pattern analysis
- Gene complexity scoring
- Statistical distributions
- Exportable summary reports

### 4. **Documentation Verification** (`splice_sites/verify_splice_sites.py`)
**Purpose**: Validate documentation accuracy and generate reproducible statistics  
**Runtime**: ~1-2 minutes  
**Dependencies**: pandas  
**Documentation**: `docs/data/splice_sites/splice_site_annotations.md`

```bash
# Generate verification report
python scripts/data/splice_sites/verify_splice_sites.py \
    --tsv data/ensembl/splice_sites.tsv \
    --top-n 10

# Save JSON summary
python scripts/data/splice_sites/verify_splice_sites.py \
    --tsv data/ensembl/splice_sites.tsv \
    --top-n 10 \
    --json-out docs/data/splice_sites/verification_summary.json
```

**Provides**:
- Schema validation (column names, types, order)
- Value validation (allowed site_type, strand values)
- Coordinate sanity checks (start < end, position within bounds)
- Summary statistics matching documentation
- Markdown and JSON output formats

## 📊 Key Findings from Analysis

### **Dataset Overview**
- **Total splice sites**: 2,829,398
- **Unique genes**: 39,291  
- **Unique transcripts**: 227,977
- **Genome coverage**: 35 chromosomes
- **Alternative splicing**: 5.8 transcripts per gene average

### **Most Complex Genes (by Isoforms)**
| Gene ID | Isoforms | Splice Sites | Biological Significance |
|---------|----------|--------------|------------------------|
| ENSG00000179818 | 295 | 2,696 | Extreme alternative splicing |
| ENSG00000215386 | 257 | 2,360 | Complex regulatory patterns |
| ENSG00000109339 | 192 | 3,704 | High splice site density |

### **Genes with Most Splice Sites**
| Gene ID | Splice Sites | Transcripts | Sites/Transcript |
|---------|--------------|-------------|------------------|
| ENSG00000155657 | 7,614 | 96 | 79.3 |
| ENSG00000145362 | 7,514 | 129 | 58.2 |
| ENSG00000224078 | 5,546 | 140 | 39.6 |

### **Gene Complexity Distribution**
- **Simple genes** (≤10 sites): 17,921 genes (45.6%)
- **Moderate genes** (11-100 sites): 14,115 genes (35.9%)
- **Complex genes** (>100 sites): 7,255 genes (18.4%)

### **Chromosomal Distribution Patterns**
- **Chr 1**: 267,284 sites (9.4%) - Highest splice site density
- **Chr 2**: 226,764 sites (8.0%) 
- **Chr 3**: 186,760 sites (6.6%)
- Perfect donor-acceptor balance: 50.0% / 50.0%

## 📁 Output Files

All analysis scripts generate organized output in `scripts/data/output/`:

```
scripts/data/output/
├── splice_sites_summary.txt              # Comprehensive summary report
├── splice_sites_quick_summary.txt        # Quick analysis results  
├── gene_analysis_report.txt              # Detailed gene patterns
└── [Additional analysis files...]
```

## 🧬 Biological Interpretations

### **Alternative Splicing Complexity**
The dataset reveals extensive alternative splicing in human transcriptome:
- **18.4% of genes** show high complexity (>100 splice sites)
- **Top genes** have 100-300+ isoforms, indicating sophisticated regulatory mechanisms
- **Average 5.8 transcripts per gene** consistent with known human transcriptome complexity

### **Splice Site Balance and Quality**
- **Perfect 1:1 donor-acceptor ratio** indicates complete, high-quality annotations
- **No orphaned splice sites** confirms proper intron-exon boundary definitions
- **Comprehensive coverage** across all chromosomes including sex chromosomes and mitochondria

### **Gene Family Patterns**
Analysis suggests several biological patterns:
- **Large genes** tend to have more isoforms (alternative splicing correlation)
- **Gene-dense chromosomes** (Chr 1, 2, 3) show highest splice site counts
- **Regulatory complexity** varies significantly across gene families

## ⚙️ Usage in MetaSpliceAI Workflows

### **Training Data Preparation**
```bash
# Analyze dataset before meta-model training
./scripts/data/quick_splice_analysis.sh

# Use findings to inform gene selection strategies
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/gene_cv_analysis
```

### **Quality Control**
```bash
# Validate dataset integrity
wc -l data/ensembl/splice_sites.tsv  
head -1 data/ensembl/splice_sites.tsv  # Check header
tail -n +2 data/ensembl/splice_sites.tsv | cut -f7 | sort -u | wc -l  # Gene count
```

### **Gene Selection for Analysis**
Based on analysis results, genes can be categorized for different analysis purposes:
- **High-complexity genes**: ENSG00000179818, ENSG00000215386 (alternative splicing studies)
- **High splice site density**: ENSG00000155657, ENSG00000145362 (splice prediction challenges)
- **Simple genes**: For baseline/control analysis

## 🚀 Integration with Documentation

These analysis results directly support:
- **`docs/data/splice_sites/splice_site_annotations.md`**: Primary dataset documentation
- **Meta-model training workflows**: Gene selection and validation
- **Quality assurance**: Dataset integrity verification
- **Research planning**: Identifying interesting gene families for study

### **Bidirectional Cross-References**
- **Scripts → Docs**: Each script header includes `Documentation:` field pointing to relevant markdown files
- **Docs → Scripts**: Each documentation file includes `Reproducibility` section with exact commands
- **Parallel Structure**: `scripts/data/splice_sites/` mirrors `docs/data/splice_sites/` for easy navigation

## 📝 Reproducibility

All scripts are designed for reproducibility:
- **Version controlled**: All scripts tracked in git
- **Standardized paths**: Default paths for MetaSpliceAI environment  
- **Cross-platform**: Compatible with macOS and Linux
- **Documented**: Clear usage instructions and biological interpretations

## 🔄 Regular Updates

These scripts should be re-run when:
- **New Ensembl releases** are incorporated (e.g., updating from v109 to v112)
- **Splice site annotations** are updated or refined
- **Additional analysis** is needed for specific research questions
- **Quality control** checks are required

---

**Generated**: 2025-10-04  
**Data Source**: `/Users/pleiadian53/work/meta-spliceai/data/ensembl/splice_sites.tsv`  
**Scripts Version**: 1.0  
**Compatible with**: MetaSpliceAI analysis workflows