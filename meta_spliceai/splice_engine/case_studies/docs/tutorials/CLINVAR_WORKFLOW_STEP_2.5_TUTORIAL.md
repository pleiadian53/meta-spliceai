# ClinVar Workflow Step 2.5 Tutorial: Enhanced Parsing with Universal VCF Parser

## Overview

**Step 2.5** represents the **Enhanced ClinVar Workflow** that upgrades the basic Step 2 parsing with the **Universal VCF Parser** for comprehensive splice variant detection. This step bridges basic VCF parsing and advanced analysis while maintaining the complete 5-step evaluation framework.

**What Step 2.5 provides**:
- ✅ **Enhanced Step 2**: Universal VCF parsing with comprehensive splice detection
- ✅ **Complete pipeline**: Maintains all 5 workflow steps with upgraded parsing
- ✅ **Learning progression**: Clear step from basic (Step 2) to enhanced (Step 2.5) parsing
- ✅ **Production-ready**: Robust error handling and performance optimization

**Step 2.5 Position in Complete Workflow**:
```
Step 1: VCF Normalization → Step 2.5: Enhanced Universal Parsing → Step 3: OpenSpliceAI → Step 4: Delta Parsing → Step 5: Evaluation
                              ↑
                    (Replaces basic Step 2)
```

**Learning Path**:
```
Steps 1-2: Basic ClinVar processing → Step 2.5: Enhanced parsing → Steps 3-5: Advanced analysis
```

## Path Conventions

- **`<META_SPLICEAI_ROOT>`**: Project root directory
- **`<OUTPUT_DIR>`**: Output directory for results
- **`<CLINVAR_VCF>`**: ClinVar VCF file (preferably main chromosomes version)

## 🔗 Essential Resources

- **[VCF Analysis Tools Guide](../VCF_ANALYSIS_TOOLS_GUIDE.md)**: Comprehensive guide to bcftools, tabix, and bgzip (essential for troubleshooting)
- **[ClinVar Workflow Steps 1-2 Tutorial](CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md)**: Prerequisite basic workflow
- **[Universal VCF Parsing Tutorial](UNIVERSAL_VCF_PARSING_TUTORIAL.md)**: Advanced parsing techniques

## Table of Contents

1. [Learning Prerequisites](#learning-prerequisites)
2. [Step 2 vs Step 2.5 Comparison](#step-2-vs-step-25-comparison)
3. [Quick Start](#quick-start)
4. [Usage Examples](#usage-examples)
5. [Python API](#python-api)
6. [Command Line Interface](#command-line-interface)
7. [Integration Examples](#integration-examples)
8. [Troubleshooting](#troubleshooting)

## Learning Prerequisites

### **Recommended Learning Path**

Before using Step 2.5, you should understand:

1. **Step 1-2 Basics**: Complete `CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md`
   - VCF normalization concepts
   - Basic ClinVar parsing
   - Understanding of splice variant detection

2. **Universal Parser Concepts**: Review `UNIVERSAL_VCF_PARSING_TUTORIAL.md`
   - SO term-based splice detection
   - Annotation system support
   - Advanced parsing features

### **When to Use Step 2.5**

- ✅ **Production workflows**: Need robust, comprehensive parsing
- ✅ **Research applications**: Require detailed splice variant classification
- ✅ **Multi-source data**: Working with different annotation systems
- ✅ **Publication-quality analysis**: Need comprehensive splice detection

- ❌ **Learning basics**: Start with Steps 1-2 tutorial first
- ❌ **Quick experiments**: Basic Step 2 may be sufficient

## Step 2 vs Step 2.5 Comparison

| Feature | Step 2 (Basic) | Step 2.5 (Enhanced) |
|---------|----------------|----------------------|
| **Tutorial** | `CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md` | `CLINVAR_WORKFLOW_STEP_2.5_TUTORIAL.md` |
| **Purpose** | 🎓 Learning basic parsing | 🚀 Production-ready parsing |
| **Splice Detection** | Basic keyword matching | **Comprehensive SO terms + keywords** |
| **Parsing Method** | Simple bcftools + text parsing | **Universal VCF Parser** |
| **Annotation Support** | ClinVar only | **ClinVar + VEP + SnpEff + Custom** |
| **Output Detail** | Basic variant fields | **Splice confidence + mechanisms** |
| **Error Handling** | Basic | **Production-grade** |
| **Performance** | Not optimized | **Chunk processing + optimization** |
| **Learning Curve** | Easy | Moderate |

### **Upgrade Path: Step 2 → Step 2.5**

```python
# Step 2: Basic parsing (from Steps 1-2 tutorial)
def step2_filter_and_parse(self, normalized_vcf: Path) -> pd.DataFrame:
    # Basic bcftools parsing
    result = subprocess.run(["bcftools", "view", "-H", str(normalized_vcf)])
    # Simple text parsing...
    
# Step 2.5: Enhanced parsing (this tutorial)
def step2_filter_and_parse(self, normalized_vcf: Path) -> pd.DataFrame:
    # Universal VCF Parser with comprehensive splice detection
    parser = create_clinvar_parser(splice_detection="comprehensive")
    return parser.parse_vcf(normalized_vcf)
```

## Quick Start

### **Complete Enhanced Workflow**
```python
from meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow import create_enhanced_clinvar_workflow

# Create enhanced workflow with Universal VCF Parser
workflow = create_enhanced_clinvar_workflow(
    input_vcf="data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz",
    output_dir="results/enhanced_clinvar",
    use_universal_parser=True  # Enhanced Step 2
)

# Run complete 5-step pipeline
results = workflow.run_complete_workflow()
```

### **Step-by-Step Execution**
```python
# Step 1: VCF Normalization (same as original)
normalized_vcf = workflow.step1_normalize_vcf()

# Step 2.5: Enhanced Universal Parsing (NEW)
enhanced_variants = workflow.step2_filter_and_parse(normalized_vcf)

# Steps 3-5: OpenSpliceAI scoring, delta parsing, evaluation (same as original)
scored_vcf = workflow.step3_openspliceai_scoring(normalized_vcf)
delta_scores = workflow.step4_parse_delta_scores(scored_vcf)
evaluation = workflow.step5_evaluation(delta_scores, enhanced_variants)
```

## Usage Examples

### **Example 1: Basic Enhanced Workflow**
```python
from meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow import EnhancedClinVarWorkflow
from meta_spliceai.splice_engine.case_studies.workflows.clinvar_variant_analysis import ClinVarAnalysisConfig
from pathlib import Path

# Configure enhanced workflow
config = ClinVarAnalysisConfig(
    input_vcf=Path("clinvar_20250831_main_chroms.vcf.gz"),
    output_dir=Path("results/enhanced_analysis"),
    genome_build="GRCh38",
    apply_splice_filter=False  # Avoid evaluation bias
)

# Create enhanced workflow
enhanced_workflow = EnhancedClinVarWorkflow(
    config=config,
    use_universal_parser=True  # Enable enhanced parsing
)

# Run complete workflow
results = enhanced_workflow.run_complete_workflow()

print(f"Results saved to: {config.output_dir}")
print(f"Enhanced parsing detected: {results.get('enhanced_splice_count', 'N/A')} splice variants")
```

### **Example 2: Individual Step Execution**
```python
# Run only enhanced Step 2 for better parsing
enhanced_workflow = EnhancedClinVarWorkflow(config, use_universal_parser=True)

# Assume you have a normalized VCF from Step 1
normalized_vcf = Path("step1_normalized.vcf.gz")

# Run enhanced parsing
enhanced_variants = enhanced_workflow.step2_filter_and_parse(normalized_vcf)

print(f"Enhanced parsing results:")
print(f"  Total variants: {len(enhanced_variants)}")
if 'affects_splicing' in enhanced_variants.columns:
    splice_count = enhanced_variants['affects_splicing'].sum()
    print(f"  Splice-affecting: {splice_count} ({splice_count/len(enhanced_variants)*100:.1f}%)")
if 'splice_confidence' in enhanced_variants.columns:
    confidence_dist = enhanced_variants['splice_confidence'].value_counts()
    print(f"  Confidence distribution: {confidence_dist.to_dict()}")
```

### **Example 3: Fallback to Original Parsing**
```python
# Create workflow that falls back to original parsing if needed
enhanced_workflow = EnhancedClinVarWorkflow(
    config=config,
    use_universal_parser=False  # Disable enhanced parsing
)

# This will use the original ClinVar parsing logic
variants = enhanced_workflow.step2_filter_and_parse(normalized_vcf)
```

## Python API

### **Factory Function**
```python
def create_enhanced_clinvar_workflow(
    input_vcf: Union[str, Path],
    output_dir: Union[str, Path],
    use_universal_parser: bool = True,
    **kwargs
) -> EnhancedClinVarWorkflow
```

**Parameters**:
- `input_vcf`: Input ClinVar VCF file
- `output_dir`: Output directory for results
- `use_universal_parser`: Whether to use Universal VCF Parser for enhanced Step 2
- `**kwargs`: Additional configuration options (passed to `ClinVarAnalysisConfig`)

### **Enhanced Configuration Options**
```python
# All original ClinVar workflow options plus:
config = ClinVarAnalysisConfig(
    # Original options
    input_vcf=Path("clinvar.vcf.gz"),
    output_dir=Path("results/"),
    genome_build="GRCh38",
    apply_splice_filter=False,
    
    # OpenSpliceAI options
    openspliceai_model="spliceai",
    distance=50,
    
    # Evaluation options
    stratify_by_variant_type=True,
    stratify_by_distance=True,
    stratify_by_review_status=True
)

# Enhanced workflow
enhanced_workflow = EnhancedClinVarWorkflow(
    config=config,
    use_universal_parser=True  # Enhanced parsing
)
```

## Command Line Interface

### **Basic Usage**
```bash
cd <META_SPLICEAI_ROOT>

# Run complete enhanced workflow
python -m meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow \
    --input-vcf data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz \
    --output-dir results/enhanced_clinvar \
    --use-universal-parser
```

### **Step-by-Step Execution**
```bash
# Run only enhanced Step 2
python -m meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow \
    --input-vcf step1_normalized.vcf.gz \
    --output-dir results/ \
    --step 2 \
    --use-universal-parser
```

### **Advanced Options**
```bash
# Run with specific configuration
python -m meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow \
    --input-vcf clinvar_20250831_main_chroms.vcf.gz \
    --output-dir results/enhanced_analysis \
    --genome-build GRCh38 \
    --use-universal-parser \
    --no-splice-filter \
    --threads 8
```

### **Disable Enhancement (Fallback)**
```bash
# Use original parsing instead of enhanced
python -m meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow \
    --input-vcf clinvar.vcf.gz \
    --output-dir results/ \
    --no-universal-parser  # Disable enhanced parsing
```

## Comparison with Original Workflow

| Feature | Original ClinVar Workflow | Enhanced ClinVar Workflow |
|---------|---------------------------|---------------------------|
| **VCF Normalization** | ✅ Step 1: bcftools norm | ✅ Step 1: Same |
| **VCF Parsing** | ✅ Step 2: Basic parsing | ✅ **Step 2.5: Universal parsing** |
| **Splice Detection** | Basic keyword matching | **Comprehensive SO terms** |
| **OpenSpliceAI Integration** | ✅ Step 3: Same | ✅ Step 3: Same |
| **Delta Score Parsing** | ✅ Step 4: Same | ✅ Step 4: Same |
| **Evaluation** | ✅ Step 5: PR-AUC | ✅ Step 5: Same |
| **Annotation Systems** | ClinVar only | **ClinVar + Universal support** |
| **Backward Compatibility** | N/A | ✅ **Can fall back to original** |

### **Enhanced Step 2 Benefits**:
- 🧬 **Comprehensive splice detection**: Uses SO terms + keywords
- 📊 **Better statistics**: Splice confidence levels and mechanisms
- 🔧 **Configurable**: Supports different annotation systems
- 📈 **Improved sensitivity**: Detects more splice-affecting variants

## Integration Examples

### **Replace Original Workflow**
```python
# OLD: Original ClinVar workflow
from meta_spliceai.splice_engine.case_studies.workflows.clinvar_variant_analysis import create_clinvar_analysis_workflow

original_workflow = create_clinvar_analysis_workflow(
    input_vcf="clinvar.vcf.gz",
    output_dir="results/"
)

# NEW: Enhanced workflow with Universal parsing
from meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow import create_enhanced_clinvar_workflow

enhanced_workflow = create_enhanced_clinvar_workflow(
    input_vcf="clinvar.vcf.gz",
    output_dir="results/",
    use_universal_parser=True  # Enhanced parsing
)
```

### **Custom Enhanced Workflow**
```python
class CustomEnhancedWorkflow(EnhancedClinVarWorkflow):
    """Custom enhanced workflow with additional processing."""
    
    def step2_filter_and_parse(self, normalized_vcf: Path) -> pd.DataFrame:
        # Use enhanced parsing
        enhanced_variants = super().step2_filter_and_parse(normalized_vcf)
        
        # Add custom processing
        enhanced_variants = self._add_custom_annotations(enhanced_variants)
        
        return enhanced_variants
    
    def _add_custom_annotations(self, variants: pd.DataFrame) -> pd.DataFrame:
        # Add custom annotations here
        variants['custom_score'] = 0.5  # Placeholder
        return variants
```

### **Batch Processing Multiple VCFs**
```python
vcf_files = [
    "clinvar_20250831_main_chroms.vcf.gz",
    "research_variants.vcf.gz",
    "validation_set.vcf.gz"
]

for vcf_file in vcf_files:
    workflow = create_enhanced_clinvar_workflow(
        input_vcf=vcf_file,
        output_dir=f"results/{Path(vcf_file).stem}",
        use_universal_parser=True
    )
    
    results = workflow.run_complete_workflow()
    print(f"Completed: {vcf_file}")
```

## Troubleshooting

### **Common Issues**

#### **1. Import Errors**
**Error**: `ModuleNotFoundError: No module named 'clinvar_variant_analysis'`
**Solution**: Use module import syntax:
```bash
python -m meta_spliceai.splice_engine.case_studies.workflows.enhanced_clinvar_workflow --help
```

#### **2. Enhanced Parsing Fails**
**Error**: Universal parser encounters unknown annotations
**Solution**: Fall back to original parsing:
```python
enhanced_workflow = EnhancedClinVarWorkflow(
    config=config,
    use_universal_parser=False  # Disable enhanced parsing
)
```

#### **3. Missing Dependencies**
**Error**: `ModuleNotFoundError: No module named 'pysam'`
**Solution**: Install required packages:
```bash
mamba activate surveyor
mamba install -c bioconda pysam bcftools
```

### **Debug Enhanced Parsing**
```python
# Enable verbose logging for enhanced parsing
import logging
logging.basicConfig(level=logging.DEBUG)

# Check enhanced parsing statistics
enhanced_variants = workflow.step2_filter_and_parse(normalized_vcf)

if 'affects_splicing' in enhanced_variants.columns:
    print(f"Enhanced splice detection: {enhanced_variants['affects_splicing'].sum()} variants")
    
if 'splice_confidence' in enhanced_variants.columns:
    confidence_dist = enhanced_variants['splice_confidence'].value_counts()
    print(f"Confidence distribution: {confidence_dist}")
```

## Expected Output

### **Enhanced Parsing Output**
```
=== Step 2 (Enhanced): Universal VCF Parsing ===
Enhanced Step 2 completed: 1000 variants parsed
Universal parser detected 847 splice-affecting variants (84.7%)
Splice confidence distribution: {'low': 623, 'medium': 156, 'high': 68}
```

### **Complete Workflow Results**
```
Enhanced workflow completed successfully. Results saved to results/enhanced_clinvar

Files generated:
├── step1_normalized.vcf.gz                    # Normalized VCF
├── step2_enhanced_filtered_variants.tsv       # Enhanced parsing results
├── step3_openspliceai_scored.vcf.gz          # OpenSpliceAI scores
├── step4_delta_scores.tsv                    # Delta scores
└── step5_evaluation_results.json             # Final evaluation
```

The Enhanced ClinVar Workflow provides the **best of both worlds**: comprehensive universal parsing with the complete evaluation framework! 🚀
