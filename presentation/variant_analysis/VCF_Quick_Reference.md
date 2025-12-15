# MetaSpliceAI VCF Analysis: Quick Reference Guide
## Essential Commands and Workflows

---

## Slide 1: Quick Start Checklist
### Get Started in 5 Minutes

**✅ Prerequisites**:
```bash
mamba activate surveyor
mamba install -c bioconda bcftools pysam
```

**✅ Essential Files**:
- VCF file: `clinvar_20250831.vcf.gz`
- Reference FASTA: `Homo_sapiens.GRCh38.dna.primary_assembly.fa`

**✅ Quick Validation**:
```bash
python vcf_coordinate_verifier.py \
    --vcf clinvar_20250831.vcf.gz \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --validate-coordinates
```

**✅ Expected Result**: `COORDINATE SYSTEM CONSISTENCY: 95.0%+`

---

## Slide 2: Command Patterns
### Smart Path Resolution

**🎯 Simple Filenames** (Recommended):
```bash
# System automatically finds files in standard locations
--vcf clinvar_20250831.vcf.gz
--fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

**📁 Relative Paths**:
```bash
--vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz
--fasta data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

**🗂️ Absolute Paths**:
```bash
--vcf /home/user/data/clinvar_20250831.vcf.gz
--fasta /home/user/data/GRCh38.fa
```

---

## Slide 3: Coordinate System Validation
### Essential Quality Check

**Basic Validation** (10 variants):
```bash
python vcf_coordinate_verifier.py \
    --vcf clinvar_20250831.vcf.gz \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --variants 10
```

**Comprehensive Validation** (100+ variants):
```bash
python vcf_coordinate_verifier.py \
    --vcf clinvar_20250831.vcf.gz \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --validate-coordinates
```

**Single Variant Check**:
```bash
python vcf_coordinate_verifier.py \
    --verify-position chr1:94062595:G:A \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

---

## Slide 4: Strand-Aware Analysis
### Gene Context Verification

**With Gene Context**:
```bash
python vcf_coordinate_verifier.py \
    --verify-position chr1:94062595:G:A \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --gene-strand - \
    --gene-name ABCA4
```

**Expected Output**:
```
Gene Context:
==============================
Gene: ABCA4
Strand: -
Gene REF: C (complement of genomic G)
Gene variant: C→T
```

**Genome Browser Links**:
- UCSC: `https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=chr1:94062545-94062645`
- Ensembl: `https://www.ensembl.org/Homo_sapiens/Location/View?r=1:94062545-94062645`

---

## Slide 5: VCF Preprocessing
### Production-Ready Normalization

**Python API**:
```python
from vcf_preprocessing import preprocess_clinvar_vcf

normalized_vcf = preprocess_clinvar_vcf(
    input_vcf="clinvar_20250831.vcf.gz",
    output_dir="processed/",
    reference_fasta="Homo_sapiens.GRCh38.dna.primary_assembly.fa"
)
```

**What It Does**:
- ✅ **Multiallelic splitting**: Complex variants → simple records
- ✅ **Left-alignment**: Consistent indel positioning
- ✅ **Indexing**: Creates tabix index for fast queries
- ✅ **Validation**: Checks format compliance

---

## Slide 6: Variant Standardization
### Individual Variant Processing

**Python API**:
```python
from variant_standardizer import VariantStandardizer

standardizer = VariantStandardizer()
variant = standardizer.standardize_from_vcf(
    chrom="chr1",
    pos=12345,
    ref="A",
    alt="G"
)

print(f"Variant type: {variant.variant_type}")
print(f"Normalized: {variant.normalized_representation}")
```

**Use Cases**:
- 🔍 **Debugging**: Understand variant representation issues
- 🧪 **Testing**: Validate individual variants
- 🔄 **Conversion**: Transform between coordinate systems

---

## Slide 7: Sequence-Centric Analysis
### Direct Splice Site Prediction

**Python API**:
```python
from sequence_inference import predict_splice_scores

# Predict for arbitrary sequence
results = predict_splice_scores(
    sequence="ATCGATCGATC...",
    model_path="meta_model.pkl",
    gene_id="ENSG00000012048"
)

print(f"Donor scores: {results['donor_scores']}")
print(f"Acceptor scores: {results['acceptor_scores']}")
```

**Variant Delta Scores**:
```python
from sequence_inference import compute_variant_delta_scores

delta_results = compute_variant_delta_scores(
    wt_sequence="ATCGATC...",
    alt_sequence="ATCAATC...",
    model_path="meta_model.pkl"
)
```

---

## Slide 8: Troubleshooting Guide
### Common Issues and Solutions

**❌ "File not found"**:
```bash
# Check file exists
ls -la clinvar_20250831.vcf.gz

# Try absolute path
--vcf /full/path/to/file.vcf.gz
```

**❌ "Coordinate mismatch"**:
```bash
# Check genome build compatibility
python vcf_coordinate_verifier.py --validate-coordinates

# Expected: 95%+ consistency for same build
```

**❌ "bcftools not found"**:
```bash
mamba install -c bioconda bcftools
```

**❌ "Import error"**:
```bash
mamba install -c bioconda pysam
```

---

## Slide 9: Performance Expectations
### Processing Benchmarks

**ClinVar Dataset** (3.6M variants):

| Operation | Time | Memory | Output Size |
|-----------|------|--------|-------------|
| Coordinate validation (100 variants) | 30s | <1GB | Report |
| VCF normalization | 3-4 min | 2-4GB | 171MB |
| Variant parsing | 1-2 min | 2-4GB | 584MB |

**Optimization Tips**:
- 💾 **SSD storage**: Faster I/O for large files
- 🧠 **8GB+ RAM**: Avoid memory swapping
- 🔄 **Batch processing**: For very large datasets

---

## Slide 10: Output Interpretation
### Understanding Results

**Coordinate Validation Report**:
```
COORDINATE SYSTEM CONSISTENCY: 95.0%
✅ COORDINATE SYSTEM ASSESSMENT: CONSISTENT

≥95%: Excellent - proceed with analysis
80-94%: Minor issues - check chromosome naming
<80%: Major problems - verify genome build
```

**Verification Details**:
```
Total variants processed: 100
Reference allele matches: 95
  - Basic matches: 90
  - Normalized matches: 5
Reference allele mismatches: 5
```

---

## Slide 11: Integration Workflows
### End-to-End Processing

**Complete Validation Pipeline**:
```bash
# 1. Coordinate validation
python vcf_coordinate_verifier.py --validate-coordinates \
    --vcf clinvar.vcf.gz --fasta GRCh38.fa

# 2. VCF preprocessing (if validation passes)
python -c "
from vcf_preprocessing import preprocess_clinvar_vcf
preprocess_clinvar_vcf('clinvar.vcf.gz', 'output/')
"

# 3. Sequence analysis
python sequence_analysis.py --vcf output/normalized.vcf.gz
```

**Python Workflow**:
```python
# Complete pipeline in Python
from vcf_coordinate_verifier import VCFCoordinateVerifier
from vcf_preprocessing import VCFPreprocessor

# Validate
verifier = VCFCoordinateVerifier("GRCh38.fa")
results = verifier.verify_vcf_variants("input.vcf.gz", 100)

# Process if validation passes
if verifier.calculate_consistency(results) >= 95:
    preprocessor = VCFPreprocessor(config)
    normalized = preprocessor.normalize_vcf()
```

---

## Slide 12: Advanced Usage
### Power User Features

**Custom Validation Parameters**:
```bash
python vcf_coordinate_verifier.py \
    --vcf input.vcf.gz \
    --fasta reference.fa \
    --variants 500 \
    --enable-normalization \
    --output detailed_report.tsv
```

**Batch Variant Verification**:
```python
# Process multiple positions
positions = [
    ("chr1", 12345, "A", "G"),
    ("chr2", 67890, "T", "C"),
    ("chrX", 11111, "G", "A")
]

for chrom, pos, ref, alt in positions:
    result = verifier.verify_single_position(chrom, pos, ref)
    print(f"{chrom}:{pos} {ref}→{alt}: {result['match']}")
```

---

## Slide 13: Error Recovery
### Handling Difficult Cases

**Enable All Recovery Strategies**:
```bash
python vcf_coordinate_verifier.py \
    --vcf problematic.vcf.gz \
    --fasta reference.fa \
    --enable-normalization \
    --variants 200
```

**Manual Investigation**:
```bash
# Check specific problematic variant
python vcf_coordinate_verifier.py \
    --verify-position chr1:12345:ATCG:A \
    --fasta reference.fa \
    --enable-normalization
```

**Expected Recovery**:
```
Verification Method: normalized
Normalization Note: Found expected sequence at position 12340 (offset: -5)
Likely Cause: VCF left-alignment difference
```

---

## Slide 14: Quality Assurance
### Validation Checklist

**✅ Pre-Analysis Checklist**:
1. **File integrity**: VCF and FASTA files exist and readable
2. **Coordinate consistency**: ≥95% validation success rate
3. **Genome build match**: Same reference genome version
4. **Chromosome naming**: Consistent naming convention
5. **Index files**: Proper tabix/fai indices present

**✅ Post-Processing Checklist**:
1. **Output files**: All expected outputs generated
2. **File sizes**: Reasonable output file sizes
3. **Variant counts**: Counts match expectations
4. **No multiallelic**: Normalization eliminated multiallelic sites

---

## Slide 15: Common Command Combinations
### Frequently Used Patterns

**Standard ClinVar Processing**:
```bash
# Quick validation
python vcf_coordinate_verifier.py \
    --vcf clinvar_20250831.vcf.gz \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --validate-coordinates

# If validation passes (≥95%), proceed with normalization
python -c "
from vcf_preprocessing import preprocess_clinvar_vcf
preprocess_clinvar_vcf('clinvar_20250831.vcf.gz', 'processed/')
"
```

**Investigation Workflow**:
```bash
# 1. Identify problematic variants
python vcf_coordinate_verifier.py --validate-coordinates \
    --output problems.tsv --vcf input.vcf.gz --fasta ref.fa

# 2. Investigate specific cases
grep "False" problems.tsv | head -5

# 3. Manual verification
python vcf_coordinate_verifier.py \
    --verify-position chr1:12345:A:G --fasta ref.fa
```

---

## Slide 16: API Reference
### Essential Python Functions

**Coordinate Verification**:
```python
verifier = VCFCoordinateVerifier(fasta_path)
result = verifier.verify_single_position(chrom, pos, ref)
# Returns: {'match': bool, 'status': str, 'context': str, ...}

batch_results = verifier.verify_vcf_variants(vcf_path, max_variants)
# Returns: pandas.DataFrame with verification results

report = verifier.generate_report(batch_results)
# Returns: str with formatted validation report
```

**VCF Preprocessing**:
```python
config = VCFPreprocessingConfig(input_vcf, output_vcf)
preprocessor = VCFPreprocessor(config)
normalized_path = preprocessor.normalize_vcf()
# Returns: Path to normalized VCF file

stats = preprocessor.get_normalization_stats()
# Returns: dict with processing statistics
```

---

## Slide 17: Configuration Examples
### Customizing Processing Parameters

**VCF Preprocessing Config**:
```python
config = VCFPreprocessingConfig(
    input_vcf=Path("input.vcf.gz"),
    output_vcf=Path("output.vcf.gz"),
    reference_fasta=Path("GRCh38.fa"),  # Optional - auto-detected
    split_multiallelics=True,           # Default: True
    left_align=True,                    # Default: True
    min_qual=10.0,                      # Optional quality filter
    threads=4,                          # Default: 1
    create_index=True                   # Default: True
)
```

**Coordinate Verifier Options**:
```python
verifier = VCFCoordinateVerifier(
    fasta_path="GRCh38.fa",
    enable_normalization=True  # Default: True
)
```

---

## Slide 18: Monitoring and Logging
### Tracking Processing Progress

**Enable Verbose Logging**:
```bash
python vcf_coordinate_verifier.py \
    --vcf input.vcf.gz \
    --fasta reference.fa \
    --validate-coordinates \
    --verbose
```

**Python Logging Setup**:
```python
import logging
logging.basicConfig(level=logging.INFO)

# Now all operations will show progress
verifier = VCFCoordinateVerifier("reference.fa")
results = verifier.verify_vcf_variants("input.vcf.gz", 100)
```

**Expected Log Output**:
```
INFO - Loading reference FASTA: reference.fa
INFO - Loaded 25 chromosomes
INFO - Opening VCF file: input.vcf.gz
INFO - Processed 100 variants...
```

---

## Slide 19: Best Practices
### Recommended Workflows

**🏆 Production Workflow**:
1. **Always validate coordinates first** (avoid silent failures)
2. **Use simple filenames** when possible (easier commands)
3. **Enable normalization** for complex indels
4. **Save validation reports** for audit trails
5. **Check consistency scores** before proceeding

**⚡ Development Workflow**:
1. **Start with small samples** (--variants 10)
2. **Use verbose logging** for debugging
3. **Investigate failures** with single variant checks
4. **Test with known good data** first

**📊 Analysis Workflow**:
1. **Validate → Normalize → Analyze** (never skip validation)
2. **Document parameters** used for reproducibility
3. **Save intermediate files** for troubleshooting
4. **Monitor processing times** for performance optimization

---

## Slide 20: Quick Reference Summary
### Essential Commands at a Glance

**🚀 Quick Start**:
```bash
python vcf_coordinate_verifier.py \
    --vcf clinvar_20250831.vcf.gz \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --validate-coordinates
```

**🔍 Single Variant**:
```bash
python vcf_coordinate_verifier.py \
    --verify-position chr1:94062595:G:A \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --gene-strand - --gene-name ABCA4
```

**🔄 VCF Processing**:
```python
from vcf_preprocessing import preprocess_clinvar_vcf
normalized = preprocess_clinvar_vcf("input.vcf.gz", "output/")
```

**📊 Success Criteria**:
- Coordinate consistency ≥95%
- No import/dependency errors  
- Reasonable processing times
- Expected output file sizes

---

*This quick reference guide provides all essential commands and patterns for effective MetaSpliceAI VCF analysis workflows.*
