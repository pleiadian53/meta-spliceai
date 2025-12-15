# VCF Column Documentation Tool

A comprehensive tool for analyzing and documenting VCF column values, their meanings, and possible values in both structured (JSON) and human-readable (Markdown) formats.

## 🎯 **Purpose**

This tool addresses the need to understand VCF column structures, especially for ClinVar and other complex VCF files, by providing:

1. **Column Definitions**: Clear descriptions of what each column represents
2. **Value Enumeration**: Complete list of possible values in the dataset
3. **Value Meanings**: Human-readable explanations of what each value means
4. **Statistical Analysis**: Value counts, frequencies, and data quality metrics
5. **Multiple Output Formats**: JSON for programmatic use, Markdown for human reading

## 🚀 **Key Features**

### **Comprehensive Column Analysis**
- ✅ **Standard VCF columns**: CHROM, POS, ID, REF, ALT, QUAL, FILTER
- ✅ **ClinVar INFO fields**: CLNSIG, CLNREVSTAT, MC, CLNDN, etc.
- ✅ **FORMAT fields**: GT, DP, GQ, etc.
- ✅ **Custom fields**: Any additional fields in the VCF

### **Intelligent Value Analysis**
- ✅ **Value enumeration**: All possible values in the dataset
- ✅ **Frequency analysis**: Count and percentage of each value
- ✅ **Data type inference**: Automatic detection of string, numeric, integer types
- ✅ **Null value analysis**: Missing data statistics

### **ClinVar-Specific Knowledge**
- ✅ **Pre-defined field meanings**: Built-in knowledge of ClinVar field semantics
- ✅ **Value interpretation**: Human-readable explanations of clinical significance, review status, etc.
- ✅ **Molecular consequence mapping**: Understanding of Sequence Ontology terms

### **Multiple Output Formats**
- ✅ **JSON**: Structured data for programmatic use
- ✅ **Markdown**: Human-readable documentation
- ✅ **CSV**: Summary table for quick reference

## 📋 **Usage**

### **Command Line Interface**

```bash
# Basic usage
python vcf_column_documenter.py --vcf clinvar.vcf.gz --output-dir docs/

# With sample size limit
python vcf_column_documenter.py --vcf clinvar.vcf.gz --output-dir docs/ --max-variants 50000

# JSON only output
python vcf_column_documenter.py --vcf clinvar.vcf.gz --output-dir docs/ --formats json

# Custom sample size
python vcf_column_documenter.py --vcf clinvar.vcf.gz --output-dir docs/ --sample-size 20000
```

### **Programmatic Usage**

```python
from vcf_column_documenter import VCFColumnDocumenter, VCFDocumentationConfig

# Create configuration
config = VCFDocumentationConfig(
    input_vcf=Path("clinvar.vcf.gz"),
    output_dir=Path("docs/"),
    max_variants=10000,
    output_formats=['json', 'markdown']
)

# Create documenter and analyze
documenter = VCFColumnDocumenter(config)
documentation = documenter.analyze_vcf_columns()

# Access specific column information
clnsig_doc = documentation['CLNSIG']
print(f"CLNSIG has {clnsig_doc.unique_count} unique values")
print(f"Top values: {list(clnsig_doc.value_counts.keys())[:5]}")

# Generate outputs
documenter.save_documentation()
```

## 📊 **Output Examples**

### **JSON Output Structure**

```json
{
  "metadata": {
    "vcf_file": "clinvar.vcf.gz",
    "analysis_date": "2025-01-12T10:30:00",
    "total_columns": 25,
    "sample_size": 10000
  },
  "columns": {
    "CLNSIG": {
      "name": "CLNSIG",
      "description": "Clinical significance",
      "data_type": "string",
      "source": "INFO",
      "is_required": false,
      "statistics": {
        "total_values": 8,
        "null_count": 150,
        "unique_count": 8,
        "example_values": ["Pathogenic", "Likely_pathogenic", "Uncertain_significance"]
      },
      "possible_values": ["Pathogenic", "Likely_pathogenic", "Uncertain_significance", "Benign"],
      "value_counts": {
        "Pathogenic": 4500,
        "Likely_pathogenic": 2000,
        "Uncertain_significance": 3000,
        "Benign": 500
      }
    }
  }
}
```

### **Markdown Output Sample**

```markdown
# VCF Column Documentation

**VCF File**: `clinvar.vcf.gz`
**Analysis Date**: 2025-01-12 10:30:00
**Total Columns**: 25
**Sample Size**: 10,000 variants

## Column Summary

| Column | Type | Source | Required | Description |
|--------|------|--------|----------|-------------|
| `CHROM` | string | VCF | ✅ | Chromosome or contig name |
| `CLNSIG` | string | INFO | ❌ | Clinical significance |
| `MC` | string | INFO | ❌ | Molecular consequence |

## CLNSIG

**Description**: Clinical significance
**Data Type**: string
**Source**: INFO
**Required**: No

### Statistics
- **Total Values**: 8
- **Unique Values**: 8
- **Null Values**: 150

### Value Distribution
| Value | Count | Percentage |
|-------|-------|------------|
| `Pathogenic` | 4,500 | 45.0% |
| `Likely_pathogenic` | 2,000 | 20.0% |
| `Uncertain_significance` | 3,000 | 30.0% |

### Value Meanings
- **`Pathogenic`**: Variant is known to cause disease
- **`Likely_pathogenic`**: Variant is likely to cause disease
- **`Uncertain_significance`**: Clinical significance is uncertain
```

## 🔧 **Configuration Options**

### **VCFDocumentationConfig Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_vcf` | Path | Required | Input VCF file path |
| `output_dir` | Path | Required | Output directory for documentation |
| `max_variants` | int | None | Maximum variants to analyze (None = all) |
| `sample_size` | int | 10000 | Sample size for value enumeration |
| `include_rare_values` | bool | True | Include rare values in analysis |
| `min_frequency` | float | 0.001 | Minimum frequency for rare values (0.1%) |
| `output_formats` | List[str] | ['json', 'markdown'] | Output formats to generate |
| `verbose` | bool | True | Enable verbose logging |

## 📋 **Understanding "Required" Fields**

### **What "Required" Means**

The "Required" column in the documentation indicates whether a field is **mandatory in the VCF specification**, not in the parsed output format:

- ✅ **Required**: Fields that are mandatory in VCF specification (CHROM, POS, REF, ALT)
- ❌ **Not Required**: Optional fields that may or may not be present in VCF records

This is based on the VCF 4.2 specification where certain fields are mandatory for valid VCF files. The tool automatically detects and marks these fields based on the VCF standard.

### **Examples**
- **CHROM, POS, REF, ALT**: Always required in every VCF record
- **QUAL, FILTER**: Optional but commonly present
- **INFO fields (CLNSIG, MC, etc.)**: Optional, depends on the VCF source
- **FORMAT fields (GT, DP, etc.)**: Optional, only present if samples are included

## 🧬 **ClinVar-Specific Features**

### **Pre-defined Field Knowledge**

The tool includes built-in knowledge of ClinVar-specific fields:

- **CLNSIG**: Clinical significance values and meanings
- **CLNREVSTAT**: Review status explanations
- **MC**: Molecular consequence (Sequence Ontology terms)
- **ORIGIN**: Allele origin descriptions
- **SSR**: Somatic status interpretations

### **Value Meaning Mappings**

```python
CLINVAR_VALUE_MEANINGS = {
    'CLNSIG': {
        'Pathogenic': 'Variant is known to cause disease',
        'Likely_pathogenic': 'Variant is likely to cause disease',
        'Uncertain_significance': 'Clinical significance is uncertain',
        # ... more mappings
    },
    'CLNREVSTAT': {
        'practice_guideline': 'Reviewed by practice guideline',
        'expert_panel': 'Reviewed by expert panel',
        # ... more mappings
    }
}
```

## 🔍 **Use Cases**

### **1. Data Exploration**
- Understand VCF structure before processing
- Identify available fields and their contents
- Assess data quality and completeness

### **2. Documentation Generation**
- Create comprehensive field documentation
- Generate data dictionaries for teams
- Maintain up-to-date field specifications

### **3. Workflow Integration**
- Inform parsing decisions based on field availability
- Validate field contents before processing
- Generate metadata for downstream tools

### **4. Quality Assurance**
- Identify missing or problematic fields
- Assess data completeness
- Validate field value distributions

## 📚 **Integration with MetaSpliceAI**

### **Enhanced ClinVar Workflow Integration**

```python
# Use documentation to inform parsing decisions
from vcf_column_documenter import VCFColumnDocumenter, VCFDocumentationConfig

# Analyze VCF structure first
config = VCFDocumentationConfig(input_vcf=vcf_path, output_dir=Path("temp"))
documenter = VCFColumnDocumenter(config)
documentation = documenter.analyze_vcf_columns()

# Check if required fields are present
required_fields = ['CLNSIG', 'MC', 'TYPE']
missing_fields = [field for field in required_fields if field not in documentation]

if missing_fields:
    print(f"Warning: Missing fields: {missing_fields}")
else:
    print("All required fields present - proceeding with parsing")
```

### **Universal VCF Parser Enhancement**

The documentation tool can be used to enhance the Universal VCF Parser by:

1. **Dynamic field detection**: Automatically discover available fields
2. **Field validation**: Ensure expected fields are present
3. **Value validation**: Check for expected value ranges
4. **Documentation generation**: Create field documentation automatically

## 🚀 **Performance Considerations**

### **Memory Usage**
- **Small VCFs** (< 1M variants): Full analysis in memory
- **Large VCFs** (> 1M variants): Use `max_variants` parameter to limit analysis
- **Sample-based analysis**: Uses configurable sample size for value enumeration

### **Processing Time**
- **Fast analysis**: ~1-2 minutes for 10K variants
- **Scalable**: Linear scaling with sample size
- **Efficient**: Uses pandas for fast value counting

### **Recommended Settings**
- **Development**: `max_variants=10000`, `sample_size=5000`
- **Production**: `max_variants=100000`, `sample_size=20000`
- **Full analysis**: `max_variants=None`, `sample_size=50000`

## 🔧 **Dependencies**

- **pandas**: Data manipulation and analysis
- **pysam**: VCF file reading
- **numpy**: Numerical operations
- **pathlib**: File path handling
- **json**: JSON output generation

## 📝 **Examples**

See `examples/vcf_column_documentation_example.py` for comprehensive usage examples including:

1. Basic ClinVar VCF documentation
2. Programmatic access to documentation
3. ClinVar-specific analysis
4. Workflow integration examples

## 🤝 **Contributing**

To add support for new VCF formats or field types:

1. **Add field definitions** to `CLINVAR_COLUMN_DEFINITIONS`
2. **Add value meanings** to `CLINVAR_VALUE_MEANINGS`
3. **Extend data type inference** in `_infer_data_type()`
4. **Add format-specific handlers** if needed

---

**Author**: MetaSpliceAI Team  
**Date**: 2025-01-12  
**Version**: 1.0.0

