# VCF Variant Analysis Documentation Reference

## 📍 Primary Documentation Location

The comprehensive **VCF Variant Analysis Workflow Documentation** is now located at:

**`meta_spliceai/splice_engine/case_studies/docs/VCF_VARIANT_ANALYSIS_WORKFLOW.md`**

> **Updated Location**: Moved from `docs/development/` to case studies docs for better organization and maintenance alongside related tools and tutorials.

This document provides the complete technical guide for VCF variant analysis in MetaSpliceAI, including:

- Complete workflow architecture from raw VCF to splice impact assessment
- VCF preprocessing pipeline with bcftools normalization
- WT/ALT sequence construction logic with detailed examples
- OpenSpliceAI integration and delta score computation
- Meta-model enhancement and feature engineering
- Alternative splice site prediction and cryptic site detection
- Comprehensive usage examples and troubleshooting

## 🔗 Related Documentation in case_studies/docs/

### Enhanced VCF Coordinate Validation (NEW)
- **`tutorials/CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md`** - Complete hands-on tutorial with enhanced coordinate validation
- **Enhanced VCF Coordinate Verifier** (`tools/vcf_coordinate_verifier.py`) - Production-ready coordinate system validation with:
  - 95%+ consistency scoring and clear pass/fail criteria
  - Complex indel normalization using variant standardizer
  - Strand-aware variant verification with gene context
  - Direct genome browser integration (UCSC, Ensembl, IGV)
  - Systematic validation for reliable variant analysis pipelines

### Core VCF Analysis
- **`VCF_ANALYSIS_TOOLS_GUIDE.md`** - bcftools and tabix usage guide
- **`VCF_TO_ALTERNATIVE_SPLICE_SITES_WORKFLOW.md`** - VCF to meta-model training pipeline

### OpenSpliceAI Integration
- **`OPENSPLICEAI_VARIANT_ANALYSIS_Q8_Q9.md`** - OpenSpliceAI variant capabilities analysis
- **`variant_splicing/OPENSPLICEAI_VARIANT_ANALYSIS_GUIDE.md`** - Delta score technical guide
- **`variant_splicing/DELTA_SCORE_IMPLEMENTATION_GUIDE.md`** - Implementation details

### Biological Context
- **`VARIANT_SPLICING_BIOLOGY_Q10_Q12.md`** - Splice variant biology and mechanisms
- **`ENHANCED_ALTERNATIVE_SPLICING_SUMMARY.md`** - Alternative splicing patterns

### Implementation Guides
- **`IMPLEMENTATION_GUIDE.md`** - General case studies implementation
- **`DELTA_SCORE_BRIDGE_IMPLEMENTATION.md`** - Delta score bridge architecture

## 🎯 Quick Navigation

### For VCF Coordinate Validation (START HERE)
1. **Tutorial**: `tutorials/CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md` - Complete hands-on guide
2. **Tool**: `tools/vcf_coordinate_verifier.py` - Enhanced coordinate validation with normalization
3. **Command**: `python vcf_coordinate_verifier.py --validate-coordinates --vcf file.vcf.gz --fasta genome.fa`
   - **Example**: `--vcf clinvar_20250831.vcf.gz --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa` 

### For VCF Processing
1. **Main workflow**: `VCF_VARIANT_ANALYSIS_WORKFLOW.md` (comprehensive technical guide)
2. **Tools reference**: `VCF_ANALYSIS_TOOLS_GUIDE.md`
3. **Implementation**: `workflows/vcf_preprocessing.py`

### For Variant Analysis
1. **Complete workflow**: `VCF_VARIANT_ANALYSIS_WORKFLOW.md`
2. **OpenSpliceAI integration**: `variant_splicing/OPENSPLICEAI_VARIANT_ANALYSIS_GUIDE.md`
3. **Practical examples**: `examples/clinvar_openspliceai_workflow.py`

### For Meta-Model Integration
1. **Training pipeline**: `VCF_TO_ALTERNATIVE_SPLICE_SITES_WORKFLOW.md`
2. **Biological context**: `VARIANT_SPLICING_BIOLOGY_Q10_Q12.md`
3. **Enhanced features**: `ENHANCED_ALTERNATIVE_SPLICING_SUMMARY.md`

## 📋 Document Relationships

```
VCF_VARIANT_ANALYSIS_WORKFLOW.md (MAIN - Comprehensive Technical Guide)
├── tutorials/CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md (NEW - Hands-on Tutorial)
├── tools/vcf_coordinate_verifier.py (NEW - Enhanced Validation Tool)
├── VCF_ANALYSIS_TOOLS_GUIDE.md (tools reference)
├── VCF_TO_ALTERNATIVE_SPLICE_SITES_WORKFLOW.md (training pipeline)
├── OPENSPLICEAI_VARIANT_ANALYSIS_Q8_Q9.md (capabilities)
├── variant_splicing/
│   ├── OPENSPLICEAI_VARIANT_ANALYSIS_GUIDE.md (technical)
│   ├── DELTA_SCORE_IMPLEMENTATION_GUIDE.md (implementation)
│   └── OPENSPLICEAI_TECHNICAL_FAQ.md (troubleshooting)
└── VARIANT_SPLICING_BIOLOGY_Q10_Q12.md (biology)
```

## 🆕 Recent Enhancements (September 2025)

### Enhanced Coordinate Validation System
- **🔍 VCF Coordinate Verifier**: Production-ready tool for systematic coordinate validation
- **📊 Consistency Scoring**: 95%+ consistency indicates reliable coordinate system
- **🧬 Complex Indel Support**: Integrated variant standardizer for normalization handling
- **🌐 Genome Browser Integration**: Direct links to UCSC, Ensembl, and IGV for variant investigation

### Smart Path Resolution
- **📁 Enhanced Path Resolver**: Auto-resolves filenames to standard locations
- **🚫 No Silent Fallbacks**: Only finds exact matches, prevents wrong file usage
- **🔍 Helpful Error Messages**: Shows available files when specified files not found

### Strand-Aware Analysis
- **🧬 Gene Context Support**: Handles plus/minus strand interpretation
- **🔄 Complement Calculation**: Automatic conversion for minus-strand genes
- **📍 Biological Accuracy**: Matches genome browser displays (e.g., UCSC)

### Validation Test Results (September 2025)
- **✅ ClinVar + GRCh38**: 100% coordinate system consistency achieved
- **✅ Complex Indel Support**: Variant standardizer resolves VCF normalization differences
- **✅ Idempotency Confirmed**: Safe to run on already-processed VCF files
- **✅ All VCF Versions**: Original, reheadered, and main_chroms files all show perfect consistency

## 🚀 Getting Started

### **Recommended Workflow (Updated September 2025)**

1. **Start with coordinate validation**: `tutorials/CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md`
2. **Validate your data**: Use enhanced VCF coordinate verifier for 100% consistency
   - **Example**: `--vcf clinvar_20250831.vcf.gz --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa`
3. **Follow main workflow**: `VCF_VARIANT_ANALYSIS_WORKFLOW.md` (comprehensive technical guide)
4. **Use enhanced tools**: Leverage smart path resolution and strand-aware analysis
5. **Understand biology**: Review `VARIANT_SPLICING_BIOLOGY_Q10_Q12.md`

### **Quick Commands for Common Tasks**

```bash
# Validate coordinate system consistency (ESSENTIAL before variant analysis)
python tools/vcf_coordinate_verifier.py \
  --vcf file.vcf.gz \
  --fasta genome.fa \
  --validate-coordinates
# Example: --vcf clinvar_20250831.vcf.gz --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa
# ✅ TESTED: Achieves 100% consistency with ClinVar + GRCh38

# Investigate specific variant with gene context
python tools/vcf_coordinate_verifier.py \
  --verify-position chr:pos:ref:alt \
  --fasta genome.fa \
  --gene-strand [+|-] --gene-name GENE
# Example: --verify-position chr1:94062595:G:A --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa --gene-strand - --gene-name ABCA4
# ✅ TESTED: Shows G→A genomic becomes C→T in gene context (minus strand)

# Quick verification with enhanced normalization (handles complex indels)
python tools/vcf_coordinate_verifier.py \
  --vcf file.vcf.gz \
  --fasta genome.fa \
  --variants 100
# Example: --vcf clinvar_20250831.vcf.gz --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa
# ✅ TESTED: 100% consistency with complex indel normalization
```

---

**Note**: All documentation is now centralized in `meta_spliceai/splice_engine/case_studies/docs/` for easier access and maintenance. The enhanced coordinate validation system is essential for reliable variant analysis pipelines.
