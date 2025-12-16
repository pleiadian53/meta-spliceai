# Answers to Splice Site Annotation Questions

## Questions Asked

1. **Does the current annotation file have all the required columns?**
2. **Is it true that only the GTF file is needed to derive the annotation file?**
3. **What other additional columns would be useful?**

---

## 1. Does the current annotation file have all required columns?

### Current Columns in `splice_sites.tsv`

The existing `splice_sites.tsv` has **8 columns**:

| Column | Type | Description |
|--------|------|-------------|
| `chrom` | string | Chromosome identifier |
| `start` | integer | Start coordinate (0-based) |
| `end` | integer | End coordinate (0-based) |
| `position` | integer | Splice site position (1-based) |
| `strand` | string | DNA strand (+/-) |
| `site_type` | string | donor or acceptor |
| `gene_id` | string | Ensembl gene ID |
| `transcript_id` | string | Ensembl transcript ID |

### Answer: **Partially Complete**

The current file has the **minimum required columns** for basic splice site prediction:
- ✅ Genomic coordinates (chrom, start, end, position)
- ✅ Strand information
- ✅ Site type (donor/acceptor)
- ✅ Gene and transcript identifiers

However, it **lacks several useful columns** for advanced analysis:
- ❌ `gene_name`: Human-readable gene symbol
- ❌ `exon_id`: Exon identifier
- ❌ `exon_number`/`exon_rank`: Exon position information
- ❌ `gene_biotype`/`transcript_biotype`: Biotype classifications

### Recommendation

For **variant analysis** and **clinical reporting**, the enhanced file (`splice_sites_enhanced.tsv`) is recommended as it includes:
- `gene_name` for human-readable reporting
- `exon_id` for tracking specific exons
- `exon_rank` for understanding exon position
- Biotype information for filtering and classification

---

## 2. Is it true that only the GTF file is needed to derive the annotation file?

### Answer: **Yes, GTF file is sufficient**

The splice site annotation file can be derived **entirely from the GTF file**. Here's why:

### What the GTF File Contains

The GTF file provides all necessary information:

1. **Exon Coordinates**: Start and end positions of each exon
2. **Gene Annotations**: Gene IDs, gene names, gene biotypes
3. **Transcript Annotations**: Transcript IDs, transcript biotypes
4. **Exon Metadata**: Exon IDs, exon numbers
5. **Strand Information**: Which strand the gene is on

### Splice Site Derivation Logic

Splice sites are derived from **exon boundaries**:

```
For positive strand (+):
  - Donor site: exon_end + 1 (except last exon)
  - Acceptor site: exon_start - 1 (except first exon)

For negative strand (-):
  - Donor site: exon_start - 1 (except last exon in transcription order)
  - Acceptor site: exon_end + 1 (except first exon in transcription order)
```

### What About the FASTA File?

The **FASTA file is NOT required** for generating the annotation file. However, it **is useful** for:

1. **Sequence Extraction**: Get actual nucleotide sequences at splice sites
2. **Consensus Validation**: Verify GT-AG dinucleotides
3. **Context Sequences**: Extract flanking sequences for feature engineering
4. **Sequence-Based Features**: Generate k-mer features, GC content, etc.

### Summary

| File | Required for Annotation? | Used for |
|------|-------------------------|----------|
| **GTF** | ✅ **Yes** | Coordinates, gene/transcript IDs, biotypes, exon metadata |
| **FASTA** | ❌ No | Sequences, consensus validation, feature engineering |

---

## 3. What other additional columns would be useful?

### Currently Added in Enhanced File

The enhanced file (`splice_sites_enhanced.tsv`) adds **6 new columns**:

1. ✅ `gene_name`: Human-readable gene symbol
2. ✅ `exon_id`: Ensembl exon ID
3. ✅ `exon_number`: Exon number from GTF
4. ✅ `exon_rank`: Exon rank in transcription order
5. ✅ `gene_biotype`: Gene biotype classification
6. ✅ `transcript_biotype`: Transcript biotype classification

### Additional Useful Columns (Future Enhancements)

Here are **additional columns** that would be valuable for splice site prediction and variant analysis:

---

#### **A. Sequence-Based Columns** (Require FASTA)

| Column | Description | Use Case |
|--------|-------------|----------|
| `consensus_dinucleotide` | Dinucleotide at splice site (GT/AG/GC/AT) | Validate canonical sites, identify non-canonical |
| `donor_motif` | 9-mer donor motif (e.g., MAG\|GTRAGT) | Feature for ML models |
| `acceptor_motif` | 23-mer acceptor motif (e.g., YYYYYYYYYYYYNCAG\|G) | Feature for ML models |
| `context_sequence` | ±50 bp context sequence | Sequence-based feature engineering |
| `gc_content` | GC content in ±50 bp window | Sequence composition feature |

**Example**:
```python
# Identify non-canonical splice sites
non_canonical = df[~df['consensus_dinucleotide'].isin(['GT-AG', 'GC-AG'])]
```

---

#### **B. Structural Columns** (Derived from GTF)

| Column | Description | Use Case |
|--------|-------------|----------|
| `exon_length` | Length of the exon | Exon size analysis, NMD prediction |
| `intron_length` | Length of adjacent intron | Intron retention, splicing efficiency |
| `distance_to_next_site` | Distance to next splice site | Exon definition, splice site pairing |
| `is_first_exon` | Boolean: first exon in transcript | Special handling for first exons |
| `is_last_exon` | Boolean: last exon in transcript | Special handling for last exons |
| `is_internal_exon` | Boolean: internal exon | Focus on internal exons |
| `total_exons_in_transcript` | Total number of exons | Transcript complexity |

**Example**:
```python
# Analyze short introns (potential retained introns)
short_introns = df[df['intron_length'] < 100]

# Focus on internal exons only
internal_sites = df[df['is_internal_exon'] == True]
```

---

#### **C. Conservation Scores** (Require External Data)

| Column | Description | Use Case |
|--------|-------------|----------|
| `phylop_score` | PhyloP conservation score | Identify conserved sites |
| `phastcons_score` | PhastCons conservation score | Identify conserved regions |
| `gerp_score` | GERP++ conservation score | Constraint-based conservation |

**Example**:
```python
# Identify highly conserved splice sites
conserved_sites = df[df['phylop_score'] > 2.0]
```

---

#### **D. Functional Annotations** (Derived from GTF + Analysis)

| Column | Description | Use Case |
|--------|-------------|----------|
| `is_canonical` | Boolean: GT-AG or GC-AG | Filter canonical sites |
| `is_constitutive` | Boolean: always spliced | Identify constitutive vs. alternative |
| `psi_score` | Percent spliced in (PSI) | Alternative splicing quantification |
| `tissue_specificity` | Tissue-specific expression | Tissue-specific splicing |
| `has_overlapping_genes` | Boolean: overlapping genes | Handle complex loci |

**Example**:
```python
# Focus on canonical, constitutive sites
canonical_constitutive = df[
    (df['is_canonical'] == True) &
    (df['is_constitutive'] == True)
]
```

---

#### **E. Clinical Annotations** (Require External Databases)

| Column | Description | Use Case |
|--------|-------------|----------|
| `clinvar_pathogenic_count` | Number of pathogenic ClinVar variants | Prioritize clinically relevant sites |
| `spliceai_delta_score` | SpliceAI delta score | Pre-computed base model score |
| `disease_associations` | Known disease associations | Clinical interpretation |
| `dbscsnv_ada_score` | dbscSNV ADA score | Splice site prediction score |
| `dbscsnv_rf_score` | dbscSNV RF score | Splice site prediction score |

**Example**:
```python
# Prioritize sites with known pathogenic variants
pathogenic_sites = df[df['clinvar_pathogenic_count'] > 0]
```

---

#### **F. Meta-Model Features** (Derived from Predictions)

| Column | Description | Use Case |
|--------|-------------|----------|
| `spliceai_donor_score` | SpliceAI donor probability | Base model feature |
| `spliceai_acceptor_score` | SpliceAI acceptor probability | Base model feature |
| `openspliceai_donor_score` | OpenSpliceAI donor probability | Alternative base model |
| `pangolin_score` | Pangolin splice site score | Ensemble feature |
| `mmsplice_delta_psi` | MMSplice delta PSI | Variant effect prediction |

**Example**:
```python
# Combine base model scores for meta-learning
df['ensemble_score'] = (
    df['spliceai_donor_score'] +
    df['openspliceai_donor_score'] +
    df['pangolin_score']
) / 3
```

---

### Priority Recommendations

Based on your use case (**variant analysis** and **splice site prediction**), here are the **top priority columns** to add:

#### **Tier 1: Essential for Variant Analysis** (Already Added ✅)
1. ✅ `gene_name` - Human-readable gene symbol
2. ✅ `exon_id` - Exon identifier
3. ✅ `exon_rank` - Exon position in transcription order

#### **Tier 2: High Value for Prediction** (Require FASTA)
4. `consensus_dinucleotide` - Validate canonical sites
5. `donor_motif` / `acceptor_motif` - Sequence features
6. `context_sequence` - Full sequence context

#### **Tier 3: Structural Features** (Derived from GTF)
7. `exon_length` - Exon size
8. `intron_length` - Intron size
9. `is_first_exon` / `is_last_exon` - Exon position flags

#### **Tier 4: Advanced Features** (External Data)
10. `phylop_score` - Conservation
11. `clinvar_pathogenic_count` - Clinical relevance
12. `spliceai_delta_score` - Base model score

---

## Summary

### Question 1: Does the current file have all required columns?
**Answer**: It has the **minimum required columns** but lacks useful metadata like `gene_name`, `exon_id`, and biotype information. The enhanced file addresses this.

### Question 2: Is only the GTF file needed?
**Answer**: **Yes**, the GTF file alone is sufficient for generating the annotation file. The FASTA file is useful but not required for coordinates and metadata.

### Question 3: What other columns would be useful?
**Answer**: Many additional columns would be valuable:
- **Sequence-based**: consensus dinucleotides, motifs, context sequences
- **Structural**: exon/intron lengths, exon position flags
- **Conservation**: PhyloP, PhastCons scores
- **Clinical**: ClinVar annotations, disease associations
- **Prediction**: Base model scores (SpliceAI, Pangolin, etc.)

The enhanced file already adds the **most critical columns** (gene_name, exon_id, exon_rank, biotypes). Future enhancements can add sequence-based and conservation features.

---

**Generated**: 2025-10-08  
**Test Results**: ✅ All validations passed  
**Enhanced File**: `data/ensembl/splice_sites_enhanced.tsv`
