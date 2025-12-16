# Data Documentation

## 📁 Overview

This directory contains comprehensive documentation for all data files, datasets, and data organization in the MetaSpliceAI project.

**🚀 Start Here**: [`DATA_LAYOUT_MASTER_GUIDE.md`](DATA_LAYOUT_MASTER_GUIDE.md) - Complete guide to all data directories and datasets

## 🗂️ Directory Structure

Documentation is organized by topic:

```
docs/data/
├── README.md                          # This file
├── DATA_LAYOUT_MASTER_GUIDE.md        # ⭐ MASTER GUIDE - Start here!
│
└── splice_sites/                      # Splice site data documentation
    ├── splice_site_annotations.md     # Schema, statistics, interpretation
    ├── SCHEMA_STANDARDIZATION.md      # Schema standardization guide
    └── verification_summary.json      # Generated statistics (gitignored)
```

## 📖 Quick Links

### Essential Guides
- **[Data Layout Master Guide](DATA_LAYOUT_MASTER_GUIDE.md)** - Complete data organization reference
- **[Splice Site Schema](splice_sites/SCHEMA_STANDARDIZATION.md)** - Schema standardization guide
- **[Base Model Data Mapping](../base_models/BASE_MODEL_DATA_MAPPING.md)** - Which data to use with which model

### By Data Type
- **Base Model Data**: [SpliceAI (GRCh37)](DATA_LAYOUT_MASTER_GUIDE.md#spliceai-grch37) | [OpenSpliceAI (GRCh38)](DATA_LAYOUT_MASTER_GUIDE.md#openspliceai-grch38-mane)
- **Training Datasets**: [Training Data Guide](DATA_LAYOUT_MASTER_GUIDE.md#training-and-evaluation-datasets)
- **Case Studies**: [Variant Databases Guide](DATA_LAYOUT_MASTER_GUIDE.md#case-study-data)
- **Model Weights**: [Pre-trained Models](DATA_LAYOUT_MASTER_GUIDE.md#pre-trained-model-weights)

## 🔗 Parallel Structure with Scripts

This directory mirrors `scripts/data/` for easy cross-referencing:

| Documentation | Analysis Scripts |
|---------------|------------------|
| `docs/data/splice_sites/` | `scripts/data/splice_sites/` |
| Future: `docs/data/transcripts/` | Future: `scripts/data/transcripts/` |
| Future: `docs/data/genes/` | Future: `scripts/data/genes/` |

### **Bidirectional References**
- **Documentation → Scripts**: Each doc includes a "Reproducibility" section with exact commands to regenerate findings
- **Scripts → Documentation**: Each script header includes a "Documentation:" field pointing to relevant docs

## 📊 Available Documentation

### **Splice Site Annotations**
- **File**: `splice_sites/splice_site_annotations.md`
- **Data Source**: `data/ensembl/splice_sites.tsv`
- **Analysis Scripts**: 
  - `scripts/data/splice_sites/verify_splice_sites.py`
  - `scripts/data/quick_splice_analysis.sh`
  - `scripts/data/analyze_gene_patterns.sh`
  - `scripts/data/analyze_splice_sites.py`

**Key Statistics**:
- 2,829,398 total splice sites
- 39,291 unique genes
- 227,977 unique transcripts
- Perfect 50/50 donor/acceptor balance

## 🚀 Usage

### **Finding Documentation**
1. Navigate to the topic subdirectory (e.g., `splice_sites/`)
2. Read the markdown documentation
3. Use the "Reproducibility" section to run analysis scripts

### **Verifying Documentation**
```bash
# Verify splice site documentation
python scripts/data/splice_sites/verify_splice_sites.py \
    --tsv data/ensembl/splice_sites.tsv \
    --top-n 10
```

### **Updating Documentation**
1. Run the relevant analysis script from `scripts/data/`
2. Compare output to documented statistics
3. Update markdown files if discrepancies found
4. Commit both script and documentation changes together

## 📝 Documentation Standards

Each data documentation file should include:
- **Overview**: Purpose and source of the data
- **Schema**: Column names, types, descriptions, examples
- **Statistics**: Counts, distributions, quality metrics
- **Biological Interpretation**: What the data means scientifically
- **Reproducibility**: Exact commands to regenerate findings
- **References**: Source data, processing tools, related files

## 🔄 Maintenance

Documentation should be updated when:
- **Data sources change**: New Ensembl releases, updated annotations
- **Schema evolves**: New columns, changed formats
- **Analysis reveals issues**: Data quality problems, unexpected patterns
- **Scripts are modified**: New analysis capabilities, changed outputs

---

**Last Updated**: 2025-10-04  
**Maintained By**: MetaSpliceAI Team  
**Related**: `scripts/data/README.md`
