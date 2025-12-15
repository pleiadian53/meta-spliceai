# Variant Analysis Pipeline Documentation

This directory contains comprehensive documentation for MetaSpliceAI's variant analysis pipelines, focusing on VCF preprocessing, normalization, and preparation for delta score computation.

## 📋 Document Organization

### 🚀 **Core Pipeline Documentation**

#### **Complete ClinVar Pipeline**
- **[COMPLETE_CLINVAR_PIPELINE_README.md](COMPLETE_CLINVAR_PIPELINE_README.md)** - **MAIN REFERENCE** for complete ClinVar processing pipeline
  - Raw VCF → WT/ALT ready data in one command
  - Comprehensive preprocessing and normalization
  - Ready for delta score analysis

#### **Pipeline Solution Summary**
- **[PIPELINE_SOLUTION_SUMMARY.md](PIPELINE_SOLUTION_SUMMARY.md)** - Complete solution summary for ClinVar pipeline automation
  - Problem solved: Raw ClinVar VCF → WT/ALT ready data
  - One-command solution with production features
  - Comprehensive validation and testing

### 🧬 **Technical Implementation Guides**

#### **Context Window Strategy**
- **[CONTEXT_WINDOW_STRATEGY.md](CONTEXT_WINDOW_STRATEGY.md)** - Two-context approach for splice variant analysis
  - Large context (±5000bp) for sequence construction
  - Small window (±50bp) for impact localization
  - OpenSpliceAI integration best practices

#### **Enhanced Mechanisms**
- **[ENHANCED_MECHANISMS_SUMMARY.md](ENHANCED_MECHANISMS_SUMMARY.md)** - Enhanced splice mechanism classification
  - Granular splice mechanism categories
  - Improved accuracy for variant classification
  - Universal VCF parser enhancements

### 🔧 **Analysis Tools**

#### **VCF Column Documentation Tool**
- **[README_VCF_COLUMN_DOCUMENTER.md](README_VCF_COLUMN_DOCUMENTER.md)** - Comprehensive VCF column analysis and documentation tool
  - **Entry Point**: `meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py`
  - **Purpose**: Analyze and document VCF column values, meanings, and possible values
  - **Outputs**: JSON (structured), Markdown (human-readable), CSV (summary)
  - **Features**: ClinVar-specific knowledge, value enumeration, statistical analysis
  - **Usage**: `python meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py --vcf data/ensembl/clinvar/vcf/clinvar.vcf.gz --output-dir docs/`

---

## 🎯 **Pipeline Overview**

### **Complete Workflow**
```
Raw ClinVar VCF → Chromosome Filter → Normalization → Universal Parsing → WT/ALT Sequences
```

### **Key Features**
- ✅ **One-command execution**: Complete pipeline automation
- ✅ **Production-ready**: Robust error handling and validation
- ✅ **Flexible configuration**: Multiple modes and parameters
- ✅ **Delta score ready**: Perfect for OpenSpliceAI/SpliceAI analysis
- ✅ **Comprehensive logging**: Full progress tracking and statistics

---

## 🚀 **Quick Start**

### **Basic Usage**
```bash
# Complete pipeline in one command
python run_complete_clinvar_pipeline.py \
    data/clinvar_20250831.vcf.gz \
    results/complete_pipeline/
```

### **With Custom Reference**
```bash
python run_complete_clinvar_pipeline.py \
    data/clinvar_20250831.vcf.gz \
    results/complete_pipeline/ \
    --reference data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

### **Test Mode**
```bash
python run_complete_clinvar_pipeline.py \
    data/clinvar_20250831.vcf.gz \
    results/test/ \
    --max-variants 1000
```

---

## 📊 **Pipeline Steps**

### **Step 0: Data Preparation** ✅ **AUTOMATED**
- Chromosome filtering (1-22, X, Y, MT)
- Input validation and integrity checks
- Smart detection of pre-filtered data

### **Step 1: VCF Normalization** ✅ **ENHANCED**
- bcftools normalization with multiallelic splitting
- Left-alignment with reference genome
- Compression and indexing

### **Step 2: Universal VCF Parsing** ✅ **INTEGRATED**
- Comprehensive splice variant detection
- ClinVar annotation parsing
- Quality filtering and clinical significance classification

### **Step 3: WT/ALT Sequence Construction** ✅ **READY**
- Reference and alternative sequence extraction
- Configurable context sizes
- Delta score calculation preparation

---

## 🔧 **Technical Details**

### **Context Window Strategy**
The pipeline implements a two-context approach:

1. **Large Context (±5000bp)**: For reliable OpenSpliceAI predictions
2. **Small Window (±50bp)**: For localized impact assessment

### **Enhanced Mechanisms**
- **8 granular categories**: From basic 4-category system
- **Improved accuracy**: Better splice mechanism classification
- **Universal parser integration**: Seamless workflow integration

### **Production Features**
- **Robust error handling**: Graceful failure management
- **Comprehensive validation**: Every step validated
- **Performance optimization**: Multi-threading and memory efficiency
- **Multiple output formats**: TSV, Parquet, JSON

---

## 📚 **Related Documentation**

### **Variant Splicing Analysis**
- `../variant_splicing/` - OpenSpliceAI integration and delta score analysis
- `../tutorials/` - Step-by-step tutorials for VCF processing

### **Universal VCF Parser**
- `../UNIVERSAL_VCF_PARSER_GUIDE.md` - Complete parser documentation
- `../VCF_ANALYSIS_TOOLS_GUIDE.md` - VCF analysis tools reference

### **Workflow Integration**
- `../../workflows/` - Complete workflow implementations
- `../../examples/` - Working examples and demos

---

## 🎯 **Use Cases**

### **1. Clinical Variant Analysis**
- Pathogenic splice variant identification
- Clinical significance assessment
- Disease-specific analysis

### **2. Research Dataset Preparation**
- Comprehensive variant datasets
- Research-grade preprocessing
- Delta score calculation preparation

### **3. High-Throughput Analysis**
- Large-scale variant processing
- Production pipeline deployment
- Automated analysis workflows

---

## 🔍 **Validation & Testing**

### **Pipeline Validation**
- ✅ Input validation (VCF integrity, reference accessibility)
- ✅ Step validation (normalization, parsing, sequences)
- ✅ Output validation (file integrity, data completeness)

### **Testing Support**
- Test mode with limited variants
- Comprehensive error reporting
- Performance monitoring

---

## 📋 **Next Steps**

After running the pipeline, your data is ready for:

1. **Delta Score Calculation**:
   ```python
   # Use WT/ALT sequences with OpenSpliceAI
   wt_scores = openspliceai.predict(wt_sequences)
   alt_scores = openspliceai.predict(alt_sequences)
   delta_scores = alt_scores - wt_scores
   ```

2. **Splice Site Analysis**:
   ```python
   # Filter to splice-affecting variants
   splice_variants = df[df['is_splice_affecting'] == True]
   ```

3. **Clinical Interpretation**:
   ```python
   # Focus on pathogenic variants
   pathogenic = df[df['is_pathogenic'] == True]
   ```

---

## 🤝 **Support**

For issues or questions:
1. Check the troubleshooting sections in individual documents
2. Review the complete pipeline README for detailed usage
3. Examine the solution summary for implementation details
4. Check the context window strategy for technical guidance

---

**Author**: MetaSpliceAI Team  
**Date**: 2025-09-12  
**Version**: 1.0.0
