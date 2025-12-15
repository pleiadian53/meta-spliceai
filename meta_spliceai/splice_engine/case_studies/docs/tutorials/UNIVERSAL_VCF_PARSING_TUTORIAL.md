# Universal VCF Parsing Tutorial: From Any VCF to WT/ALT Sequences

## Overview

This tutorial demonstrates how to use the Universal VCF Parser to process **any VCF file** for splice variant analysis, regardless of annotation system or data source. This is **Step 2.5** in the enhanced workflow - bridging basic parsing and sequence construction.

**What you'll learn**:
- 🚀 **Universal VCF parsing** for any annotation system  
- 🧬 **Advanced splice detection** using comprehensive SO terms
- 🔧 **Configurable filtering** for different research questions
- 🎯 **Preparation for WT/ALT** sequence construction

**What you get**: Production-ready parsed variants optimized for delta score calculations.

**Enhanced 6-Step Workflow Position**:
```
Step 1: VCF Normalization → Step 2: Basic Parsing → Step 2.5: Universal Parsing → 
Step 3: WT/ALT Construction → Step 4: Delta Scoring → Step 5: Analysis
```

## Path Conventions

- **`<META_SPLICEAI_ROOT>`**: Project root directory
- **`<OUTPUT_DIR>`**: Output directory for results
- **`<VCF_FILE>`**: Any normalized VCF file (ClinVar, research data, etc.)

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start Examples](#quick-start-examples)
3. [Annotation System Support](#annotation-system-support)
4. [Splice Detection Modes](#splice-detection-modes)
5. [Advanced Configuration](#advanced-configuration)
6. [Integration with Existing Workflows](#integration-with-existing-workflows)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps to WT/ALT Sequences](#next-steps-to-wtalt-sequences)

## Prerequisites

### Required Tools
```bash
# Activate environment
mamba activate surveyor

# Required packages (should already be installed)
# - pysam (VCF parsing)
# - pandas (data handling)  
# - pyfaidx (optional, for sequence extraction)
```

### Input Requirements
- ✅ **Normalized VCF file** (preferably from Step 1)
- ✅ **Reference FASTA** (optional, for sequence extraction)
- ✅ **Known annotation system** (ClinVar, VEP, SnpEff, etc.)

## Quick Start Examples

### **Example 1: ClinVar VCF Parsing**
```bash
cd <META_SPLICEAI_ROOT>
python meta_spliceai/splice_engine/case_studies/workflows/universal_vcf_parser.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz \
    --output-dir results/clinvar_parsed \
    --annotation-system clinvar \
    --splice-detection comprehensive \
    --max-variants 1000
```

**Expected Output**:
```
✅ Successfully parsed 1000 variants
🧬 Found 976 splice-affecting variants (97.6%)
⚠️  Found 10 pathogenic variants (1.0%)
```

**Note**: The high splice detection rate (97.6%) demonstrates the comprehensive SO term detection working effectively. This includes direct splice sites, intronic variants, exonic variants with potential splice effects, and UTR regulatory elements.

### **Example 2: VEP-Annotated VCF with Sequences**
```bash
python meta_spliceai/splice_engine/case_studies/workflows/universal_vcf_parser.py \
    --vcf your_vep_annotated.vcf.gz \
    --output-dir results/vep_parsed \
    --annotation-system vep \
    --reference-fasta data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --include-sequences \
    --splice-detection comprehensive
```

### **Example 3: Research Mode (All Fields)**
```bash
python meta_spliceai/splice_engine/case_studies/workflows/universal_vcf_parser.py \
    --vcf research_variants.vcf.gz \
    --output-dir results/research_parsed \
    --splice-detection permissive \
    --extract-all-info \
    --output-format parquet
```

## Annotation System Support

### **1. ClinVar Annotations**
```python
from meta_spliceai.splice_engine.case_studies.workflows.universal_vcf_parser import create_clinvar_parser

parser = create_clinvar_parser(
    splice_detection="comprehensive",
    include_uncertain=True,
    include_sequences=False
)
variants = parser.parse_vcf("clinvar.vcf.gz")
```

**Extracted Fields**:
- `clinical_significance` (CLNSIG)
- `molecular_consequence` (MC)
- `disease` (CLNDN)
- `review_status` (CLNREVSTAT)
- `is_pathogenic` (computed flag)

### **2. VEP Annotations**
```python
parser = create_vep_parser(
    splice_detection="comprehensive",
    pathogenicity_filter=["HIGH", "MODERATE"]
)
variants = parser.parse_vcf("variants_vep.vcf.gz")
```

**Extracted Fields**:
- `VEP_Consequence`
- `VEP_Gene`
- `VEP_Transcript`
- `VEP_HGVSc`
- `VEP_HGVSp`

### **3. Custom Annotations**
```python
config = VCFParsingConfig(
    info_fields=['CADD_phred', 'SpliceAI_pred', 'custom_score'],
    annotation_system=AnnotationSystem.CUSTOM,
    custom_splice_keywords=['cryptic_site', 'ese_disruption']
)
parser = UniversalVCFParser(config)
```

## Splice Detection Modes

### **Strict Mode**: Core Splice Sites Only
```python
config = VCFParsingConfig(splice_detection_mode=SpliceDetectionMode.STRICT)
# Detects: splice_donor_variant, splice_acceptor_variant, splice_region_variant
```

### **Comprehensive Mode**: Extended Coverage (Recommended)
```python
config = VCFParsingConfig(splice_detection_mode=SpliceDetectionMode.COMPREHENSIVE)
# Detects: Core sites + intronic + exonic + UTR variants
```

### **Permissive Mode**: Research Applications
```python
config = VCFParsingConfig(splice_detection_mode=SpliceDetectionMode.PERMISSIVE)
# Detects: All potential splice-affecting variants
```

### **Custom Mode**: User-Defined Criteria
```python
config = VCFParsingConfig(
    splice_detection_mode=SpliceDetectionMode.CUSTOM,
    custom_splice_terms=['SO:0001575', 'SO:0001574'],  # Only donor/acceptor
    custom_splice_keywords=['branch_point', 'polypyrimidine']
)
```

## Advanced Configuration

### **Sequence Extraction**
```python
config = VCFParsingConfig(
    include_sequences=True,
    sequence_context=200,  # ±200bp context
    reference_fasta=Path("GRCh38.fa")
)
```

**Output includes**:
- `ref_sequence`: WT sequence with context
- `alt_sequence`: ALT sequence with variant applied
- `variant_position_in_sequence`: Position within extracted sequence

### **Performance Tuning**
```python
config = VCFParsingConfig(
    chunk_size=50000,  # Process 50K variants at a time
    max_variants=100000,  # Limit for testing
    output_format="parquet"  # More efficient than TSV
)
```

### **Quality Filtering**
```python
config = VCFParsingConfig(
    apply_quality_filter=True,
    min_quality_score=20.0,
    exclude_failed_filters=True,
    pathogenicity_filter=['Pathogenic', 'Likely_pathogenic']
)
```

## Integration with Existing Workflows

### **Replace Tutorial Scripts**

#### **Before** (Tutorial):
```python
from meta_spliceai.splice_engine.case_studies.examples.vcf_clinvar_tutorial import parse_clinvar_vcf

# Limited to ClinVar, basic splice detection
variants = parse_clinvar_vcf("clinvar.vcf.gz", max_variants=1000)
```

#### **After** (Universal):
```python
from meta_spliceai.splice_engine.case_studies.workflows.universal_vcf_parser import create_clinvar_parser

# Comprehensive, production-ready
parser = create_clinvar_parser(include_sequences=True)
variants = parser.parse_vcf("clinvar.vcf.gz")
```

### **Enhance ClinVar Workflow**

#### **Complete Enhanced Workflow** (Recommended):
```python
# Use the pre-built Enhanced ClinVar Workflow
from meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow import create_enhanced_clinvar_workflow

enhanced_workflow = create_enhanced_clinvar_workflow(
    input_vcf="clinvar.vcf.gz",
    output_dir="results/enhanced_clinvar",
    use_universal_parser=True  # Enhanced Step 2
)

# Run complete 5-step pipeline with enhanced parsing
results = enhanced_workflow.run_complete_workflow()
```

**See the complete Step 2.5 tutorial**: `../CLINVAR_WORKFLOW_STEP_2.5_TUTORIAL.md`

#### **Custom Step 2 Enhancement**:
```python
# For custom integration only
def step2_enhanced_parsing(self, normalized_vcf: Path) -> pd.DataFrame:
    """Enhanced Step 2 using Universal VCF Parser."""
    parser = create_clinvar_parser(
        include_sequences=True,
        splice_detection="comprehensive"
    )
    return parser.parse_vcf(normalized_vcf)
```

### **OpenSpliceAI Result Reproduction**
```python
# Parse OpenSpliceAI results
parser = create_vep_parser()  # or custom config
openspliceai_variants = parser.parse_vcf("openspliceai_results.vcf.gz")

# Extract delta scores for comparison
delta_scores = openspliceai_variants[['chrom', 'pos', 'SpliceAI_pred']]
```

## Troubleshooting

### **Common Issues**

#### **1. "No splice-affecting variants found"**
**Cause**: Annotation system mismatch or strict detection mode
**Solution**: 
```python
# Try different detection mode
config = VCFParsingConfig(splice_detection_mode=SpliceDetectionMode.PERMISSIVE)

# Check available INFO fields
parser = UniversalVCFParser(config)
# Look at header_info to see available fields
```

#### **2. "Unknown annotation system"**
**Cause**: VCF uses non-standard annotations
**Solution**:
```python
# Use custom annotation system
config = VCFParsingConfig(
    annotation_system=AnnotationSystem.CUSTOM,
    info_fields=['YOUR_CUSTOM_FIELD', 'ANOTHER_FIELD']
)
```

#### **3. "Sequence extraction failed"**  
**Cause**: Reference FASTA not available or chromosome naming mismatch
**Solution**:
```bash
# Check FASTA availability
ls data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa

# Verify chromosome naming consistency
bcftools view -H your.vcf.gz | cut -f1 | sort | uniq | head -5
grep ">" GRCh38.fa | head -5
```

### **Debug Commands**
```bash
# Check VCF header for annotation fields
bcftools view -h your.vcf.gz | grep "##INFO"

# Test with small sample
python universal_vcf_parser.py --vcf your.vcf.gz --max-variants 10 --verbose

# Check parsing statistics
cat results/parsing_stats.json
```

## Next Steps to WT/ALT Sequences

After completing universal parsing, you have **analysis-ready variants** and are **2 steps away** from delta score calculations:

### **Step 3: WT/ALT Sequence Construction**
```python
# Your parsed variants are ready for sequence construction
from meta_spliceai.splice_engine.case_studies.workflows.sequence_construction import SequenceConstructor

constructor = SequenceConstructor(
    reference_fasta="GRCh38.fa",
    context_size=5000  # Large context for reliable basewise scores (OpenSpliceAI default)
)

wt_alt_pairs = constructor.construct_sequences(parsed_variants)
```

**Context Size Guidelines:**
- **Large context (5000bp)**: For OpenSpliceAI sequence construction to get reliable basewise predictions
- **Small delta window (±50bp)**: For impact summarization around the variant (handled by OpenSpliceAI internally)

```python
# OpenSpliceAI handles both contexts automatically:
# 1. Uses full 10kb window (±5000bp) for model predictions
# 2. Reports delta scores with ±50bp coverage window for localized effects
# 3. This reduces false positives from distant fluctuations
# 4. Matches OpenSpliceAI's standard variant analysis pipeline
```

### **Step 4: Delta Score Calculation**
```python
# Ready for OpenSpliceAI/SpliceAI scoring
from meta_spliceai.splice_engine.case_studies.workflows.delta_scoring import DeltaScoreCalculator

calculator = DeltaScoreCalculator()
delta_scores = calculator.compute_delta_scores(wt_alt_pairs)
```

### **Current Workflow Position**
```
✅ Step 1: VCF Normalization (done)
✅ Step 2: Basic Parsing (done) 
✅ Step 2.5: Universal Parsing (YOU ARE HERE)
🎯 Step 3: WT/ALT Construction (NEXT)
⏭️ Step 4: Delta Score Calculation
⏭️ Step 5: Analysis and Evaluation
```

**You're ready to move to WT/ALT sequence construction!** 🎯

## Comparison: Tutorial vs Production

| Feature | `vcf_clinvar_tutorial.py` | `universal_vcf_parser.py` |
|---------|---------------------------|---------------------------|
| **Purpose** | 🎓 Educational | 🚀 Production |
| **Scope** | ClinVar only | Any VCF file |
| **Splice Detection** | Basic keywords | Comprehensive SO terms |
| **Annotation Systems** | ClinVar only | ClinVar, VEP, SnpEff, Custom |
| **Output Formats** | TSV only | TSV, Parquet, JSON |
| **Sequence Extraction** | No | Yes (optional) |
| **Error Handling** | Basic | Production-grade |
| **Performance** | Not optimized | Chunk processing |
| **Configurability** | Fixed | Highly configurable |

**Recommendation**: Use **tutorial script for learning**, **universal parser for production** workflows and broader research applications.

