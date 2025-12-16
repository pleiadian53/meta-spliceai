# Enhanced Splice Site Annotations

## 📋 Overview

This document describes the **enhanced splice site annotation file** (`splice_sites_enhanced.tsv`) which extends the original `splice_sites.tsv` with additional metadata columns useful for splice site prediction, variant analysis, and transcript-level studies.

**Generated**: 2025-10-08  
**Test Script**: `tests/test_enhanced_splice_sites.py`  
**Validation**: ✅ PASSED - Perfect consistency with original file

---

## 🆕 What's New?

The enhanced file adds **6 new columns** while maintaining **100% consistency** with the original file on all common columns:

| New Column | Type | Description | Use Case |
|------------|------|-------------|----------|
| `gene_name` | string | Human-readable gene symbol | Variant reporting, clinical interpretation |
| `exon_id` | string | Ensembl exon ID | Exon-level analysis, tracking specific exons |
| `exon_number` | integer | Exon number (from GTF) | GTF compatibility, exon numbering |
| `exon_rank` | integer | Exon rank in transcription order | Alternative splicing analysis, exon ordering |
| `gene_biotype` | string | Gene biotype classification | Gene filtering, biotype-specific analysis |
| `transcript_biotype` | string | Transcript biotype classification | Transcript filtering, NMD prediction |

---

## 📊 File Statistics

### Basic Metrics
- **Total splice sites**: 2,829,398 (identical to original)
- **Unique genes**: 39,291
- **Unique transcripts**: 227,977
- **Unique exons**: 775,526
- **Donor sites**: 1,414,699 (50.0%)
- **Acceptor sites**: 1,414,699 (50.0%)

### Column Completeness
- **gene_name**: 95.39% complete (130,574 null values, 4.61%)
- **All other new columns**: 100% complete (no null values)

---

## 🔍 Column Details

### 1. `gene_name` (Human-Readable Gene Symbol)

**Purpose**: Provides human-readable gene symbols for clinical interpretation and reporting.

**Statistics**:
- Unique values: 26,610
- Null values: 130,574 (4.61%)
- Sample values: `ENTREP3`, `ZNF93`, `RNMT`, `FRG2GP`, `RHOF`

**Use Cases**:
- **Variant Reporting**: Display gene names in variant analysis reports
- **Clinical Interpretation**: Map variants to known disease genes
- **Literature Search**: Link predictions to published studies
- **User Interface**: Show gene names instead of Ensembl IDs

**Example**:
```python
# Filter splice sites for a specific gene
gene_sites = df[df['gene_name'] == 'BRCA1']

# Variant report
print(f"Variant affects splice site in {row['gene_name']} (exon {row['exon_rank']})")
```

---

### 2. `exon_id` (Ensembl Exon ID)

**Purpose**: Unique identifier for each exon, enabling exon-level tracking and analysis.

**Statistics**:
- Unique values: 775,526
- Null values: 0 (0.00%)
- Sample values: `ENSE00003969440`, `ENSE00003775159`, `ENSE00002255156`

**Use Cases**:
- **Exon-Level Analysis**: Track specific exons across transcripts
- **Variant Mapping**: Map variants to specific exons
- **Alternative Splicing**: Identify exons involved in alternative splicing
- **Cross-Reference**: Link to Ensembl exon annotations

**Example**:
```python
# Find all splice sites for a specific exon
exon_sites = df[df['exon_id'] == 'ENSE00003969440']

# Count splice sites per exon
exon_counts = df.groupby('exon_id').size()
```

---

### 3. `exon_number` (GTF Exon Number)

**Purpose**: Exon number as annotated in the GTF file.

**Statistics**:
- Unique values: 365
- Range: 1 to 365
- Null values: 0 (0.00%)

**Use Cases**:
- **GTF Compatibility**: Match exon numbering from original GTF
- **Exon Position**: Understand exon position in transcript
- **Validation**: Cross-check with GTF annotations

**Note**: This is the exon number from the GTF `exon_number` attribute, which may differ from transcription order on the negative strand.

---

### 4. `exon_rank` (Transcription Order)

**Purpose**: Exon rank in transcription order (1-based), accounting for strand direction.

**Statistics**:
- Unique values: 365
- Range: 1 to 365
- Null values: 0 (0.00%)

**Use Cases**:
- **Alternative Splicing**: Identify exon skipping, retention patterns
- **Exon Ordering**: Understand exon order in transcription direction
- **Splice Site Pairing**: Match donor/acceptor sites by exon rank
- **Variant Impact**: Predict impact based on exon position

**Example**:
```python
# Find first exon splice sites (no acceptor, only donor)
first_exon_donors = df[(df['exon_rank'] == 1) & (df['site_type'] == 'donor')]

# Find last exon splice sites (no donor, only acceptor)
last_exon_acceptors = df[
    (df['exon_rank'] == df.groupby('transcript_id')['exon_rank'].transform('max')) &
    (df['site_type'] == 'acceptor')
]

# Exon skipping analysis
skipped_exons = df[df['exon_rank'].isin([3, 5, 7])]  # Analyze specific exons
```

**Key Difference from `exon_number`**:
- `exon_rank`: Always in transcription order (5' → 3')
- `exon_number`: From GTF, may not reflect transcription order

---

### 5. `gene_biotype` (Gene Classification)

**Purpose**: Classify genes by their biotype (protein-coding, lncRNA, pseudogene, etc.).

**Statistics**:
- Unique values: 18 biotypes
- Null values: 0 (0.00%)
- Sample values: `pseudogene`, `IG_C_pseudogene`, `IG_V_gene`, `unprocessed_pseudogene`, `TEC`

**Common Biotypes**:
- `protein_coding`: Protein-coding genes
- `lncRNA`: Long non-coding RNA
- `pseudogene`: Pseudogenes
- `IG_*_gene`: Immunoglobulin genes
- `TR_*_gene`: T-cell receptor genes
- `miRNA`: MicroRNA genes

**Use Cases**:
- **Gene Filtering**: Focus on protein-coding genes or specific biotypes
- **Biotype-Specific Models**: Train models for specific gene classes
- **Functional Analysis**: Understand functional implications
- **Quality Control**: Filter out pseudogenes or low-confidence annotations

**Example**:
```python
# Filter protein-coding genes only
protein_coding = df[df['gene_biotype'] == 'protein_coding']

# Analyze lncRNA splice sites
lncrna_sites = df[df['gene_biotype'] == 'lncRNA']

# Exclude pseudogenes
no_pseudogenes = df[~df['gene_biotype'].str.contains('pseudogene')]
```

---

### 6. `transcript_biotype` (Transcript Classification)

**Purpose**: Classify transcripts by their biotype, enabling transcript-level filtering.

**Statistics**:
- Unique values: 24 biotypes
- Null values: 0 (0.00%)
- Sample values: `processed_pseudogene`, `IG_V_gene`, `TR_V_pseudogene`, `nonsense_mediated_decay`, `pseudogene`

**Common Biotypes**:
- `protein_coding`: Protein-coding transcripts
- `nonsense_mediated_decay`: NMD transcripts
- `retained_intron`: Retained intron transcripts
- `processed_transcript`: Processed transcripts
- `lncRNA`: Long non-coding RNA transcripts

**Use Cases**:
- **Transcript Filtering**: Focus on canonical protein-coding transcripts
- **NMD Prediction**: Identify transcripts subject to nonsense-mediated decay
- **Alternative Splicing**: Analyze non-canonical transcript isoforms
- **Quality Control**: Filter out low-confidence transcript annotations

**Example**:
```python
# Filter canonical protein-coding transcripts
canonical = df[df['transcript_biotype'] == 'protein_coding']

# Identify NMD transcripts
nmd_transcripts = df[df['transcript_biotype'] == 'nonsense_mediated_decay']

# Analyze retained intron events
retained_intron = df[df['transcript_biotype'] == 'retained_intron']
```

---

## 🎯 Use Cases

### 1. Variant Analysis

**Scenario**: A variant is predicted to affect a splice site. Report which exon is affected.

```python
# Given a variant position
variant_pos = 43124194
variant_gene = 'BRCA1'

# Find affected splice sites
affected_sites = df[
    (df['gene_name'] == variant_gene) &
    (df['position'] == variant_pos)
]

# Report
for _, site in affected_sites.iterrows():
    print(f"Variant affects {site['site_type']} site in {site['gene_name']}")
    print(f"  Exon: {site['exon_rank']} (Exon ID: {site['exon_id']})")
    print(f"  Transcript: {site['transcript_id']} ({site['transcript_biotype']})")
```

### 2. Exon Skipping Detection

**Scenario**: Identify potential exon skipping events by analyzing exon ranks.

```python
# Find transcripts with non-consecutive exon ranks (potential skipping)
transcript_exons = df.groupby('transcript_id')['exon_rank'].apply(list)

for tx_id, exon_ranks in transcript_exons.items():
    sorted_ranks = sorted(exon_ranks)
    expected = list(range(1, max(sorted_ranks) + 1))
    
    if sorted_ranks != expected:
        skipped = set(expected) - set(sorted_ranks)
        print(f"{tx_id}: Potential exon skipping at ranks {skipped}")
```

### 3. Biotype-Specific Training

**Scenario**: Train separate models for protein-coding vs. non-coding genes.

```python
# Split by gene biotype
protein_coding_sites = df[df['gene_biotype'] == 'protein_coding']
lncrna_sites = df[df['gene_biotype'] == 'lncRNA']

# Train biotype-specific models
model_pc = train_model(protein_coding_sites)
model_lnc = train_model(lncrna_sites)
```

### 4. Clinical Variant Reporting

**Scenario**: Generate a clinical report for a splice site variant.

```python
def generate_variant_report(variant_pos, gene_name):
    site = df[
        (df['gene_name'] == gene_name) &
        (df['position'] == variant_pos)
    ].iloc[0]
    
    report = f"""
    Splice Site Variant Report
    ==========================
    Gene: {site['gene_name']} ({site['gene_id']})
    Gene Biotype: {site['gene_biotype']}
    
    Affected Site:
    - Type: {site['site_type']}
    - Position: chr{site['chrom']}:{site['position']}
    - Strand: {site['strand']}
    
    Transcript Context:
    - Transcript: {site['transcript_id']}
    - Transcript Biotype: {site['transcript_biotype']}
    - Exon Rank: {site['exon_rank']}
    - Exon ID: {site['exon_id']}
    
    Clinical Interpretation:
    - This variant affects a {site['site_type']} splice site
    - Located in exon {site['exon_rank']} of the transcript
    - May lead to exon skipping or cryptic splice site activation
    """
    return report
```

---

## 🔬 Validation Results

### Test: Consistency with Original File

**Test Script**: `tests/test_enhanced_splice_sites.py`

**Results**:
```
✅ PASS: Enhanced file is consistent with original file
   - Both files have matching core columns
   - Enhanced file adds 6 new columns
   - New columns: exon_id, exon_number, exon_rank, gene_biotype, gene_name, transcript_biotype
```

**Validation Checks**:
1. ✅ **Row Count**: 2,829,398 (identical)
2. ✅ **Common Columns**: Perfect match on all 8 common columns
3. ✅ **Site Type Distribution**: Perfect 1:1 donor/acceptor ratio
4. ✅ **Chromosome Distribution**: Identical distributions
5. ✅ **Gene/Transcript Counts**: Identical (39,291 genes, 227,977 transcripts)

---

## 📁 File Format

### Location
- **Original**: `data/ensembl/splice_sites.tsv`
- **Enhanced**: `data/ensembl/splice_sites_enhanced.tsv`

### Format Specifications
- **Delimiter**: Tab-separated (TSV)
- **Encoding**: UTF-8
- **Header**: Single header row with column names
- **Coordinate System**:
  - `start`/`end`: 0-based (BED format)
  - `position`: 1-based (standard genomic coordinates)

### File Size
- **Original**: ~200-300 MB
- **Enhanced**: ~350-450 MB (due to additional columns)

---

## 🔄 Generation

### How to Generate

```bash
# Run the test script to generate enhanced file
cd /Users/pleiadian53/work/meta-spliceai
mamba activate surveyor
python tests/test_enhanced_splice_sites.py
```

### Requirements
- GTF file: `data/ensembl/Homo_sapiens.GRCh38.112.gtf`
- gffutils database: `data/ensembl/annotations.db`
- Python packages: `polars`, `pandas`, `gffutils`, `tqdm`

### Processing Time
- **Database loading**: ~5 seconds (if already exists)
- **Extraction**: ~60 seconds (227,977 transcripts)
- **Saving**: ~10 seconds
- **Total**: ~75 seconds

---

## 🚀 Future Enhancements

Potential additional columns for future versions:

1. **Splice Site Sequences**:
   - `donor_sequence`: Donor consensus sequence (e.g., GT)
   - `acceptor_sequence`: Acceptor consensus sequence (e.g., AG)
   - `context_sequence`: Extended context (e.g., ±50 bp)

2. **Conservation Scores**:
   - `phylop_score`: PhyloP conservation score
   - `phastcons_score`: PhastCons conservation score

3. **Functional Annotations**:
   - `is_canonical`: Whether site uses canonical GT-AG dinucleotides
   - `is_constitutive`: Whether site is constitutively spliced
   - `psi_score`: Percent spliced in (PSI) score

4. **Structural Information**:
   - `distance_to_next_exon`: Distance to next exon
   - `intron_length`: Length of adjacent intron
   - `exon_length`: Length of adjacent exon

5. **Clinical Annotations**:
   - `clinvar_variants`: Known ClinVar variants at this site
   - `disease_association`: Disease associations from databases

---

## 📚 References

### Related Documentation
- [Splice Site Annotations](./splice_site_annotations.md) - Original file documentation
- [Genomic Resources Setup](../../meta_spliceai/system/docs/GENOMIC_RESOURCES_SETUP_SUMMARY.md) - System setup guide

### Source Code
- Test script: `tests/test_enhanced_splice_sites.py`
- Extraction function: `extract_enhanced_splice_sites()`
- Comparison function: `compare_splice_site_files()`

### Data Sources
- **Ensembl GTF**: Human genome annotation (GRCh38/hg38)
- **Version**: Release 112
- **Download**: `ftp://ftp.ensembl.org/pub/release-112/gtf/homo_sapiens/`

---

**Last Updated**: 2025-10-08  
**Validation Status**: ✅ PASSED  
**File Version**: 1.0
