# MetaSpliceAI Case Studies Infrastructure

This package provides comprehensive infrastructure for validating the MetaSpliceAI meta-learning model against disease-specific splice mutation databases and well-characterized splice-altering variants.

## 🎯 Overview

The case studies infrastructure is designed to:

- **Ingest** splice mutation data from multiple curated databases
- **Standardize** variant representations across different formats (HGVS, VCF, BED)
- **Validate** meta-model performance on real-world disease mutations
- **Analyze** individual mutations and mutation cohorts in detail
- **Compare** base model (SpliceAI) vs meta-model predictions

## 📊 Supported Databases

### 1. **SpliceVarDB** (Sullivan et al. 2024)
- **>50,000 experimentally validated splice variants**
- Comprehensive annotations including validation methods
- Ideal for cryptic splice site detection validation

### 2. **MutSpliceDB** (NCI)
- **TCGA and CCLE validated splice mutations**
- Cancer-specific splice alterations with RNA evidence
- Therapeutic target annotations (e.g., capmatinib for MET)

### 3. **DBASS5/DBASS3**
- **Cryptic splice site activation database**
- Strength scores and fold-change data
- Classic examples like CFTR pseudoexons

### 4. **ClinVar**
- **Clinical significance annotations**
- Pathogenic/benign classification
- Multiple submitter validation status

## 🏗️ Architecture

```
case_studies/
├── data_sources/          # Database ingestion modules
│   ├── base.py           # Common ingestion infrastructure  
│   ├── splicevardb.py    # SpliceVarDB ingester
│   ├── mutsplicedb.py    # MutSpliceDB ingester
│   ├── dbass.py          # DBASS ingester
│   └── clinvar.py        # ClinVar ingester
├── formats/              # Data format handling
│   ├── hgvs_parser.py    # HGVS notation parser
│   ├── variant_standardizer.py  # Coordinate standardization
│   └── annotation_converter.py  # Format conversion
├── workflows/            # Analysis workflows
│   ├── disease_validation.py    # Disease-specific validation
│   ├── mutation_analysis.py     # Individual mutation analysis
│   └── cryptic_site_validation.py  # Cryptic site detection
└── examples/             # Usage examples
    └── run_disease_validation_example.py
```

## 🚀 Quick Start

### Basic Usage

```python
from pathlib import Path
from meta_spliceai.splice_engine.case_studies.workflows.disease_validation import DiseaseValidationWorkflow

# Initialize workflow
work_dir = Path("./case_study_results")
meta_model_path = Path("./trained_model.pkl")  # Optional
workflow = DiseaseValidationWorkflow(work_dir, meta_model_path)

# Run disease-specific validation
results = workflow.run_disease_specific_validation(
    diseases=["lung_cancer", "breast_cancer", "cystic_fibrosis"],
    databases=["SpliceVarDB", "MutSpliceDB"],
    min_rna_evidence=10
)

# Print results
for disease, result in results.items():
    print(f"{disease}: {result.meta_accuracy:.3f} accuracy, {result.improvement:+.3f} improvement")
```

### Command Line Usage

```bash
# Run comprehensive disease validation
python examples/run_disease_validation_example.py \
    --work-dir ./results \
    --meta-model ./model.pkl \
    --comprehensive \
    --diseases lung_cancer breast_cancer \
    --databases SpliceVarDB MutSpliceDB

# Run specific case studies
python examples/run_disease_validation_example.py \
    --work-dir ./results \
    --specific-cases

# Test database ingestion
python examples/run_disease_validation_example.py \
    --work-dir ./test \
    --demo-ingestion
```

## 🔬 Key Case Studies

### 1. **MET Exon 14 Skipping** 
- **Clinical relevance**: FDA-approved therapies (capmatinib, tepotinib)
- **Mechanism**: Splice site mutations causing oncogenic exon skipping
- **Validation**: TCGA lung cancer cohort with RNA evidence

```python
from meta_spliceai.splice_engine.case_studies.workflows.disease_validation import run_met_exon14_case_study

result = run_met_exon14_case_study(work_dir="./met_analysis")
print(f"MET exon 14 accuracy: {result.accuracy:.3f}")
```

### 2. **CFTR Cryptic Pseudoexon** 
- **Classic example**: c.3718-2477C>T deep intronic mutation
- **Mechanism**: Activates cryptic pseudoexon causing cystic fibrosis
- **Therapeutic**: Target for antisense therapy

```python
from meta_spliceai.splice_engine.case_studies.workflows.mutation_analysis import analyze_cftr_cryptic_exon

result = analyze_cftr_cryptic_exon(work_dir="./cftr_analysis")
print(f"Cryptic sites detected: {len(result.cryptic_sites_detected)}")
```

### 3. **BRCA1/BRCA2 Splice Variants**
- **Clinical impact**: Hereditary breast/ovarian cancer
- **Validation**: Minigene assays and functional studies
- **Therapeutic**: PARP inhibitor sensitivity

### 4. **Neurodegenerative Disease Variants**
- **UNC13A cryptic exon** in ALS/FTD
- **STMN2 cryptic exon** in ALS  
- **MAPT exon 10** regulation in FTD

## 📋 Validation Metrics

### Model Performance
- **Accuracy**: Overall classification accuracy
- **Sensitivity/Specificity**: For pathogenic variant detection
- **F1 Score**: Balanced performance measure
- **Top-k Accuracy**: Gene-level ranking performance

### Clinical Relevance
- **Pathogenic vs Benign**: Clinical significance accuracy
- **Event Type Breakdown**: Cryptic vs canonical site performance
- **Experimental Validation**: Correlation with RNA evidence

### Comparative Analysis
- **Base vs Meta**: Improvement over SpliceAI
- **Error Analysis**: False positive/negative characterization
- **Confidence Calibration**: Prediction reliability

## 🛠️ Data Ingestion

### Individual Database Ingestion

```python
from meta_spliceai.splice_engine.case_studies.data_sources import SpliceVarDBIngester

# Initialize ingester
ingester = SpliceVarDBIngester(output_dir="./splicevardb_data")

# Download and process data
result = ingester.ingest(force_refresh=False)

print(f"Ingested {len(result.mutations)} mutations")
print(f"Created {len(result.splice_sites_df)} splice site annotations")

# Filter for high-quality mutations
validated_mutations = ingester.get_validated_mutations_only(result.mutations)
pathogenic_mutations = ingester.get_pathogenic_mutations_only(result.mutations)
```

### HGVS Parsing

```python
from meta_spliceai.splice_engine.case_studies.formats import HGVSParser

parser = HGVSParser()

# Parse splice site mutation
variant = parser.parse("c.3718-2477C>T")
print(f"Splice site type: {parser.get_splice_site_type(variant)}")
print(f"Is splice variant: {parser.is_splice_site_variant(variant)}")

# Batch parsing
variants = parser.parse_batch([
    "c.4357+1G>A",
    "c.3028+1G>A", 
    "c.1466-2A>G"
])
stats = parser.get_parsing_statistics(variants)
print(f"Success rate: {stats['success_rate']:.2%}")
```

### Coordinate Standardization

```python
from meta_spliceai.splice_engine.case_studies.formats import VariantStandardizer

standardizer = VariantStandardizer(reference_genome="GRCh38")

# Standardize from VCF format
std_variant = standardizer.standardize_from_vcf(
    chrom="7", pos=117199644, ref="C", alt="T"
)

# Convert to different formats
vcf_format = standardizer.to_vcf_format(std_variant)
bed_format = standardizer.to_bed_format(std_variant)
```

## 🔍 Detailed Mutation Analysis

### Single Mutation Analysis

```python
from meta_spliceai.splice_engine.case_studies.workflows.mutation_analysis import MutationAnalysisWorkflow

workflow = MutationAnalysisWorkflow(work_dir="./mutation_analysis")

# Analyze individual mutation
result = workflow.analyze_single_mutation(mutation)

print(f"Score improvement: {result.score_improvement:+.3f}")
print(f"Cryptic sites: {len(result.cryptic_sites_detected)}")
print(f"Experimental evidence: {result.experimental_evidence}")
```

### Cohort Analysis

```python
# Analyze related mutations together
cohort_results = workflow.analyze_mutation_cohort(mutations)

print(f"Success rate: {cohort_results['analysis_success_rate']:.2%}")
print(f"Mean improvement: {cohort_results['mean_score_improvement']:+.3f}")
print(f"Novel cryptic sites: {cohort_results['novel_cryptic_sites']}")
```

## 📈 Expected Outcomes

### Performance Improvements
- **5-15% accuracy improvement** over base SpliceAI
- **Enhanced cryptic site detection** for deep intronic variants
- **Better calibration** for clinical decision-making

### Clinical Validation
- **High correlation** with experimental evidence (>80% agreement)
- **Improved pathogenic variant classification** 
- **Reduced false positive rate** for clinical variants

### Research Applications
- **Novel cryptic site discovery** in patient cohorts
- **Therapeutic target identification** for antisense drugs
- **Biomarker development** for disease progression

## 🤝 Integration with Existing Pipeline

The case studies infrastructure integrates seamlessly with the existing MetaSpliceAI meta-model pipeline:

1. **Training Data**: Uses same feature engineering pipeline
2. **Base Predictions**: Compatible with existing SpliceAI workflow
3. **Meta-Model**: Applies trained meta-model for enhanced predictions
4. **Evaluation**: Reuses existing evaluation and visualization tools

## 📝 Output Formats

### Validation Results
- `validation_summary.json`: Aggregate performance metrics
- `detailed_predictions.tsv`: Per-mutation prediction results
- `false_positives.tsv` / `false_negatives.tsv`: Error analysis
- `disease_comparison.tsv`: Cross-disease performance

### Analysis Artifacts
- `splice_sites.tsv`: Compatible splice site annotations
- `gene_features.tsv`: Gene-level feature summaries
- `mutations.json`: Detailed mutation metadata
- `cohort_detailed_results.csv`: Comprehensive analysis results

### Visualizations
- ROC/PR curves for model comparison
- Feature importance heatmaps
- Error distribution analysis
- Clinical significance breakdown

## 🔧 Configuration

### Database Sources
Configure database URLs and access methods in individual ingester classes. Most databases provide demo data for development and testing.

### Reference Genome
Default is GRCh38. Update `VariantStandardizer` reference genome parameter for different builds.

### Validation Thresholds
- `min_rna_evidence`: Minimum RNA support reads (default: 5)
- `leakage_threshold`: Feature correlation threshold (default: 0.95)
- `validation_stringency`: Experimental validation requirements

## 🚨 Troubleshooting

### Common Issues

1. **Download Failures**: Most ingesters fall back to demo data for development
2. **Memory Constraints**: Use `memory_optimize=True` flag for large datasets  
3. **Missing Dependencies**: Install `requests`, `pandas`, `numpy` for full functionality
4. **Coordinate Mismatches**: Verify reference genome build consistency

### Debug Mode
Enable verbose logging for detailed error tracking:

```python
workflow = DiseaseValidationWorkflow(work_dir, verbose=True)
```

## 📚 References

### Key Publications
- **SpliceVarDB**: Sullivan et al. (2024) "Comprehensive database of splice variants"
- **MutSpliceDB**: NCI TCGA/CCLE splice mutation analysis
- **DBASS**: Database of aberrant splice sites
- **MET Exon 14**: Paik et al. (2020) "Clinical implications of MET exon 14 skipping"
- **CFTR Cryptic Exon**: Igreja et al. (2016) "Correction of a cystic fibrosis splice defect"

### Database Links
- [SpliceVarDB](https://splicevardb.org) - Experimental splice variant database
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) - Clinical variant significance
- [DBASS](http://www.dbass.org.uk/) - Aberrant splice site database

---

## 💡 Getting Started

1. **Install dependencies**: `pip install pandas numpy requests`
2. **Run basic demo**: `python examples/run_disease_validation_example.py --work-dir ./test --demo-ingestion`
3. **Try specific cases**: `python examples/run_disease_validation_example.py --work-dir ./results --specific-cases`
4. **Full validation**: `python examples/run_disease_validation_example.py --work-dir ./comprehensive --comprehensive --meta-model ./model.pkl`

For questions or support, please refer to the main MetaSpliceAI documentation or create an issue in the project repository. 