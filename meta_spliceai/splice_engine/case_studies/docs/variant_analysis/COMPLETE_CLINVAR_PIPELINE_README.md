# Complete ClinVar Pipeline: Raw VCF → WT/ALT Ready Data

## 🎯 **Overview**

The Complete ClinVar Pipeline automates the entire process from raw ClinVar VCF download to data ready for WT/ALT sequence construction and delta score calculations. This eliminates manual steps and ensures consistent, reproducible results.

### **What This Pipeline Does**

```
Raw ClinVar VCF → Filtered & Normalized → Comprehensively Parsed → WT/ALT Ready
```

1. **📋 Data Preparation**: Filter to main chromosomes, validate input
2. **🔧 VCF Normalization**: bcftools normalization with multiallelic splitting  
3. **🧬 Universal Parsing**: Comprehensive splice variant detection
4. **🧪 Sequence Construction**: WT/ALT sequences for delta score analysis

---

## 🚀 **Quick Start**

### **Basic Usage**
```bash
# Simple one-command pipeline
python meta_spliceai/splice_engine/case_studies/entry_points/run_complete_clinvar_pipeline.py \
    data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    results/complete_pipeline/
```

### **With Custom Reference**
```bash
python meta_spliceai/splice_engine/case_studies/entry_points/run_complete_clinvar_pipeline.py \
    data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    results/complete_pipeline/ \
    --reference data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

### **Test Mode (Limited Variants)**
```bash
python meta_spliceai/splice_engine/case_studies/entry_points/run_complete_clinvar_pipeline.py \
    data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    results/test_pipeline/ \
    --max-variants 1000 \
    --research-mode
```

---

## 📊 **Pipeline Steps Explained**

### **Step 0: Data Preparation**
- ✅ **Chromosome Filtering**: Keep only main chromosomes (1-22, X, Y, MT)
- ✅ **Input Validation**: Verify VCF structure and content
- ✅ **Smart Detection**: Skip filtering if already done

```python
# Handles both chr1 and 1 formats automatically
# Filters out patches, alternate loci, unplaced contigs
```

### **Step 1: VCF Normalization** (Uses `vcf_preprocessing.py`)
- ✅ **Multiallelic Splitting**: `-m -both` (splits complex variants)
- ✅ **Left-Alignment**: `-f reference.fa` (standardizes indel positions)
- ✅ **Compression & Indexing**: Creates `.vcf.gz` and `.tbi`
- ✅ **Validation**: Verifies normalization success

### **Step 2: Universal VCF Parsing** (Uses `universal_vcf_parser.py`)
- ✅ **Comprehensive Splice Detection**: All SO terms + keywords
- ✅ **ClinVar Annotation Parsing**: CLNSIG, MC, CLNDN, etc.
- ✅ **Quality Filtering**: Configurable quality thresholds
- ✅ **Clinical Significance**: Pathogenic/benign classification

### **Step 3: WT/ALT Sequence Construction**
- ✅ **Reference Sequences**: Extract context around variants
- ✅ **Alternative Sequences**: Construct ALT sequences
- ✅ **Configurable Context**: Default 50bp, adjustable
- ✅ **Validation**: Ensure sequence integrity

---

## 🛠️ **Installation & Requirements**

### **Dependencies**
```bash
# Core tools (must be in PATH)
bcftools  # For VCF normalization
tabix     # For indexing
bgzip     # For compression

# Python packages
pandas
numpy
pysam
pathlib
```

### **Reference Data**
The pipeline auto-detects reference FASTA from common locations:
- `/data/reference/GRCh38.fa`
- `/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa`
- Or specify with `--reference`

---

## 📖 **Usage Examples**

### **1. Production Pipeline**
```bash
# Full pipeline for production analysis
python run_complete_clinvar_pipeline.py \
    data/clinvar_20250831.vcf.gz \
    results/production/ \
    --threads 8 \
    --research-mode
```

**Output**: 
- `clinvar_wt_alt_ready.tsv` - Tab-separated data
- `clinvar_wt_alt_ready.parquet` - Efficient binary format
- `pipeline_summary.json` - Complete statistics

### **2. Pathogenic Variants Only**
```bash
# Focus on clinically relevant variants
python run_complete_clinvar_pipeline.py \
    data/clinvar_20250831.vcf.gz \
    results/pathogenic/ \
    --pathogenic-only \
    --splice-detection comprehensive
```

### **3. Research Mode (All Variants)**
```bash
# Include all variants for research
python run_complete_clinvar_pipeline.py \
    data/clinvar_20250831.vcf.gz \
    results/research/ \
    --research-mode \
    --splice-detection permissive \
    --sequence-context 100
```

### **4. Quick Test**
```bash
# Test with small dataset
python run_complete_clinvar_pipeline.py \
    data/clinvar_20250831.vcf.gz \
    results/test/ \
    --max-variants 1000 \
    --no-sequences  # Skip sequence construction for speed
```

---

## 🐍 **Programmatic Usage**

### **Python API**
```python
from meta_spliceai.splice_engine.case_studies.workflows.complete_clinvar_pipeline import (
    CompleteClinVarPipeline, CompletePipelineConfig
)

# Create configuration
config = CompletePipelineConfig(
    input_vcf=Path("data/clinvar_20250831.vcf.gz"),
    output_dir=Path("results/api_run/"),
    research_mode=True,
    include_sequences=True,
    max_variants=5000
)

# Run pipeline
pipeline = CompleteClinVarPipeline(config)
results = pipeline.run_complete_pipeline()

# Access results
summary = results['summary']
output_files = results['output_files']
```

### **Load Results for Analysis**
```python
import pandas as pd

# Load processed data
df = pd.read_parquet("results/api_run/clinvar_wt_alt_ready.parquet")

# Filter splice-affecting variants
splice_variants = df[df['is_splice_affecting'] == True]

# Extract sequences for delta score calculation
wt_sequences = splice_variants['ref_sequence'].tolist()
alt_sequences = splice_variants['alt_sequence'].tolist()
```

---

## 📋 **Configuration Options**

### **Core Options**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_vcf` | Required | Raw ClinVar VCF file |
| `output_dir` | Required | Output directory |
| `reference_fasta` | Auto-detect | Reference genome FASTA |

### **Processing Options**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `research_mode` | False | Include all variants vs. splice-focused |
| `pathogenic_only` | False | Only pathogenic/likely pathogenic |
| `splice_detection_mode` | comprehensive | strict/comprehensive/permissive |
| `max_variants` | None | Limit for testing |

### **Sequence Options**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_sequences` | True | Extract WT/ALT sequences |
| `sequence_context` | 50 | Sequence context size (bp) |

### **Performance Options**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `threads` | 4 | Number of threads |
| `memory_gb` | 8 | Memory limit |
| `chunk_size` | 10000 | Processing chunk size |

---

## 📊 **Output Files**

### **Primary Outputs**
- **`clinvar_wt_alt_ready.tsv`** - Tab-separated data (human-readable)
- **`clinvar_wt_alt_ready.parquet`** - Binary format (efficient loading)
- **`pipeline_summary.json`** - Complete pipeline statistics

### **Intermediate Files**
- **`*_main_chroms.vcf.gz`** - Chromosome-filtered VCF
- **`*_normalized.vcf.gz`** - Normalized VCF
- **`parsing/`** - Universal parser outputs

### **Key Data Columns**
```
# Core variant information
chrom, pos, ref, alt, id, qual, filter

# Clinical annotations
clinical_significance, review_status, disease, molecular_consequence

# Splice analysis
is_splice_affecting, splice_impact_level, splice_terms

# WT/ALT sequences (if include_sequences=True)
ref_sequence, alt_sequence, sequence_start, sequence_end
variant_position_in_sequence
```

---

## 🔍 **Pipeline Validation**

The pipeline includes comprehensive validation:

### **Input Validation**
- ✅ VCF file exists and is readable
- ✅ Reference FASTA is accessible
- ✅ Required tools (bcftools, tabix) available

### **Step Validation**
- ✅ Chromosome filtering success
- ✅ Normalization completeness (no multiallelic sites remaining)
- ✅ Parsing statistics (variants processed, splice detection rate)
- ✅ Sequence construction success rate

### **Output Validation**
- ✅ File integrity checks
- ✅ Column completeness validation
- ✅ Sequence quality validation

---

## 🎯 **Use Cases**

### **1. Clinical Variant Analysis**
```bash
# Focus on pathogenic splice variants
python run_complete_clinvar_pipeline.py input.vcf.gz results/ --pathogenic-only
```

### **2. Research Dataset Preparation**
```bash
# Comprehensive dataset for research
python run_complete_clinvar_pipeline.py input.vcf.gz results/ --research-mode
```

### **3. Delta Score Calculation Prep**
```bash
# Ready for OpenSpliceAI/SpliceAI analysis
python run_complete_clinvar_pipeline.py input.vcf.gz results/ --sequence-context 100
```

### **4. High-Throughput Analysis**
```bash
# Optimized for large datasets
python run_complete_clinvar_pipeline.py input.vcf.gz results/ --threads 16 --memory 32
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **"bcftools not found"**
```bash
# Install bcftools
conda install -c bioconda bcftools
# or
sudo apt-get install bcftools
```

#### **"Reference FASTA not found"**
```bash
# Specify reference explicitly
python run_complete_clinvar_pipeline.py input.vcf.gz results/ \
    --reference /path/to/GRCh38.fa
```

#### **"Memory error during processing"**
```bash
# Reduce memory usage
python run_complete_clinvar_pipeline.py input.vcf.gz results/ \
    --max-variants 10000 \
    --memory 4
```

#### **"No splice variants found"**
- Check `--splice-detection` mode (try `permissive`)
- Verify input VCF has proper annotations
- Use `--research-mode` to include all variants

### **Performance Tips**

1. **Use more threads**: `--threads 8`
2. **Limit variants for testing**: `--max-variants 1000`
3. **Skip sequences if not needed**: `--no-sequences`
4. **Use research mode for comprehensive analysis**: `--research-mode`

---

## 🎉 **Success Metrics**

After successful completion, you should see:

```
🎉 COMPLETE CLINVAR PIPELINE SUMMARY
============================================================
📁 Input:  data/clinvar_20250831.vcf.gz
📁 Output: results/complete_pipeline
⏱️  Runtime: 45.2s

📊 Results:
   • Total variants processed: 2,157,891
   • Splice-affecting variants: 1,234,567
   • Pathogenic variants: 45,123
   • Variants with WT/ALT sequences: 1,234,567
   • Sequence success rate: 100.0%

📄 Output files:
   • TSV: results/complete_pipeline/clinvar_wt_alt_ready.tsv
   • PARQUET: results/complete_pipeline/clinvar_wt_alt_ready.parquet

✅ Pipeline completed successfully!
```

---

## 📚 **Next Steps**

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
1. Check the troubleshooting section above
2. Review the demo scripts: `demo_complete_pipeline.py`
3. Examine intermediate files for debugging
4. Check the pipeline summary JSON for detailed statistics

---

**Author**: MetaSpliceAI Team  
**Date**: 2025-09-12  
**Version**: 1.0.0
