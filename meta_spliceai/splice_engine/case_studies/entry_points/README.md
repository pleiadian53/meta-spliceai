# MetaSpliceAI Case Studies Entry Points

**Purpose**: Command-line entry points for MetaSpliceAI case study tools  
**Location**: `meta_spliceai/splice_engine/case_studies/entry_points/`  
**Version**: 1.0.0

---

## 🎯 **Overview**

This directory contains executable scripts that provide easy command-line access to MetaSpliceAI's case study tools. These scripts are designed to be run directly and handle all the necessary imports and configuration.

**Important**: Always run these scripts from the project root directory using the full path:
```bash
# From project root (/home/bchiu/work/meta-spliceai/)
python meta_spliceai/splice_engine/case_studies/entry_points/script_name.py [args]
```

**Convenience Wrappers**: For easier access, convenience wrappers are available in the `scripts/` directory:
```bash
# Using convenience wrapper (recommended for daily use)
python scripts/run_clinvar_pipeline.py [args]
```

## 🚀 **Available Entry Points**

### **1. ClinVar Pipeline (Consolidated)**
- **Script**: `run_clinvar_pipeline.py`
- **Project Root Wrapper**: `run_clinvar_pipeline.py` (in project root)
- **Purpose**: Raw VCF → WT/ALT ready data for delta score computation
- **Documentation**: [Complete ClinVar Pipeline README](../docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)

#### **Key Features**
- ✅ **Systematic Path Discovery**: Automatically finds VCF and reference files
- ✅ **Simple Commands**: Run from project root with short commands
- ✅ **Multiple Input Formats**: Filename, date, or full path
- ✅ **Genomic Resources Integration**: Uses MetaSpliceAI's resource management
- ✅ **Delta Score Ready**: Perfect for both base models and meta models
- ✅ **Production-ready**: Robust error handling and validation

#### **Quick Start**

**Option 1: Using Convenience Wrapper (Recommended)**
```bash
# Simple usage with systematic discovery
python scripts/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/clinvar_pipeline

# With specific reference genome
python scripts/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/clinvar_pipeline \
    --reference Homo_sapiens.GRCh38.dna.primary_assembly.fa

# Using date format
python scripts/run_clinvar_pipeline.py 20250831 results/clinvar_pipeline

# Research mode with all variants
python scripts/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/research --research-mode

# Test mode with limited variants
python scripts/run_clinvar_pipeline.py 20250831 results/test --max-variants 1000

# Pathogenic variants only
python scripts/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/pathogenic --pathogenic-only
```

**Option 2: Direct Entry Point Access**
```bash
# Simple usage with systematic discovery (run from project root)
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/clinvar_pipeline

# With specific reference genome
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/clinvar_pipeline \
    --reference Homo_sapiens.GRCh38.dna.primary_assembly.fa

# Using date format
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py 20250831 results/clinvar_pipeline
```

#### **Next Steps After Pipeline**
After running this pipeline, your data will be ready for:
- **Base Model Delta Scores**: SpliceAI, OpenSpliceAI
- **Meta Model Delta Scores**: MetaSpliceAI inference workflow
- **Recalibrated Scores**: Per-nucleotide splice site scores with reduced errors

### **2. VCF Column Documentation Tool**
- **Script**: `run_vcf_column_documenter.py`
- **Purpose**: Analyze and document VCF column values and meanings
- **Documentation**: [VCF Column Documenter README](../docs/variant_analysis/README_VCF_COLUMN_DOCUMENTER.md)

#### **Quick Start**
```bash
# Basic usage
python meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    --output-dir data/ensembl/clinvar/vcf/docs/

# With sample size limit
python meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    --output-dir data/ensembl/clinvar/vcf/docs/ \
    --max-variants 50000

# JSON only output
python meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    --output-dir data/ensembl/clinvar/vcf/docs/ \
    --formats json
```

#### **Key Features**
- ✅ **Comprehensive analysis**: All VCF columns and values
- ✅ **ClinVar-specific knowledge**: Built-in field meanings
- ✅ **Multiple outputs**: JSON, Markdown, CSV formats
- ✅ **Structured paths**: Supports `data/` relative paths
- ✅ **Statistical analysis**: Value counts and frequencies

---

## 🔧 **Usage Patterns**

### **Pattern 1: Direct Execution**
```bash
# Run from project root
python meta_spliceai/splice_engine/case_studies/entry_points/run_complete_clinvar_pipeline.py --help
```

### **Pattern 2: Add to PATH**
```bash
# Add entry points to PATH
export PATH=$PATH:$(pwd)/meta_spliceai/splice_engine/case_studies/entry_points

# Now run directly
run_complete_clinvar_pipeline.py --help
run_vcf_column_documenter.py --help
```

### **Pattern 3: Create Symlinks**
```bash
# Create symlinks in project root
ln -s meta_spliceai/splice_engine/case_studies/entry_points/run_complete_clinvar_pipeline.py .
ln -s meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py .

# Run from project root
./run_complete_clinvar_pipeline.py --help
./run_vcf_column_documenter.py --help
```

---

## 📋 **Common Options**

### **Complete ClinVar Pipeline Options**
| Option | Description | Default |
|--------|-------------|---------|
| `--reference`, `-r` | Reference FASTA file | Auto-detected |
| `--research-mode` | Include all variants | False |
| `--max-variants` | Limit variants for testing | None |
| `--no-sequences` | Skip WT/ALT sequence construction | False |
| `--pathogenic-only` | Only pathogenic variants | False |
| `--threads` | Number of threads | 4 |
| `--quiet` | Reduce output verbosity | False |

### **VCF Column Documenter Options**
| Option | Description | Default |
|--------|-------------|---------|
| `--vcf`, `-v` | Input VCF file | Required |
| `--output-dir`, `-o` | Output directory | Required |
| `--max-variants` | Maximum variants to analyze | None |
| `--sample-size` | Sample size for enumeration | 10000 |
| `--formats` | Output formats | json markdown |
| `--verbose` | Enable verbose output | False |

---

## 🎯 **Use Cases**

### **1. Data Preparation**
```bash
# Prepare ClinVar data for analysis
run_complete_clinvar_pipeline.py \
    data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    results/analysis/ \
    --research-mode
```

### **2. VCF Exploration**
```bash
# Understand VCF structure before processing
run_vcf_column_documenter.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    --output-dir data/ensembl/clinvar/vcf/docs/ \
    --max-variants 10000
```

### **3. Testing & Development**
```bash
# Quick test with limited data
run_complete_clinvar_pipeline.py \
    data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    results/test/ \
    --max-variants 1000 \
    --no-sequences
```

### **4. Production Analysis**
```bash
# Full production pipeline
run_complete_clinvar_pipeline.py \
    data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    results/production/ \
    --threads 8 \
    --research-mode
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **"Module not found" errors**
```bash
# Ensure you're running from project root
cd /path/to/meta-spliceai
python meta_spliceai/splice_engine/case_studies/entry_points/run_complete_clinvar_pipeline.py --help
```

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
run_complete_clinvar_pipeline.py input.vcf.gz output/ --reference /path/to/GRCh38.fa
```

### **Getting Help**
```bash
# Get help for any script
python meta_spliceai/splice_engine/case_studies/entry_points/run_complete_clinvar_pipeline.py --help
python meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py --help
```

---

## 📚 **Related Documentation**

- **[Complete ClinVar Pipeline README](../docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)** - Detailed pipeline documentation
- **[VCF Column Documenter README](../docs/variant_analysis/README_VCF_COLUMN_DOCUMENTER.md)** - Column analysis tool documentation
- **[Variant Analysis Pipeline README](../docs/variant_analysis/README.md)** - Overview of variant analysis tools
- **[Case Studies Main README](../docs/README.md)** - Complete case studies documentation

---

## 🤝 **Adding New Entry Points**

To add a new entry point script:

1. **Create the script** in this directory
2. **Make it executable**: `chmod +x script_name.py`
3. **Add shebang**: `#!/usr/bin/env python3`
4. **Handle imports**: Use the shared project root utility
5. **Update this README** with usage examples
6. **Test thoroughly** with various options

### **Template for New Entry Points**
```python
#!/usr/bin/env python3
"""
Your Tool Entry Point

Usage Examples:
    python your_tool.py --input input_file --output output_dir/

Author: MetaSpliceAI Team
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the path using systematic detection
from project_root_utils import setup_entry_point_imports
setup_entry_point_imports(__file__)

from meta_spliceai.splice_engine.case_studies.your_module import YourClass

def main():
    parser = argparse.ArgumentParser(description="Your Tool Description")
    # Add arguments here
    args = parser.parse_args()
    
    # Your implementation here
    pass

if __name__ == "__main__":
    sys.exit(main())
```

### **Project Root Detection**

The entry points use a systematic approach to detect the project root directory:

- **✅ Robust Detection**: Looks for common project markers (`.git`, `setup.py`, `pyproject.toml`, `requirements.txt`, `meta_spliceai`)
- **✅ No Hardcoded Paths**: No need to count directory levels
- **✅ Fallback Support**: Graceful fallback if markers aren't found
- **✅ Shared Utility**: `project_root_utils.py` provides reusable functions

### **Available Utilities**

```python
from project_root_utils import (
    find_project_root,           # Find project root directory
    setup_project_imports,      # Set up imports for any script
    setup_entry_point_imports,  # Alias for entry point scripts
    get_project_root,           # Get project root as Path object
    validate_project_structure  # Validate project structure
)
```

---

**Author**: MetaSpliceAI Team  
**Date**: 2025-01-12  
**Version**: 1.0.0
