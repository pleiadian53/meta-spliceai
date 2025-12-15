# MetaSpliceAI Feature Wish List

This document tracks future enhancements and analysis features that would improve the MetaSpliceAI system but are not currently critical for core functionality.

---

## 1. Branch Point and Non-Canonical Intron Analysis

**Priority**: Medium  
**Status**: Planned  
**Estimated Effort**: 2-3 days

### Context

Current consensus motif analysis validates donor (GT) and acceptor (AG) sites with their immediate context (9-mer for donors, 25-mer for acceptors including polypyrimidine tract). However, **branch point** analysis and comprehensive characterization of **non-canonical introns** would provide additional features for the meta-model and improve variant impact prediction.

### Why This Feature Is Helpful

1. **Branch Point Identification**:
   - The branch point adenosine (typically 18-40 nt upstream of acceptor AG) is critical for lariat formation during splicing
   - Mutations disrupting branch point consensus (YRAY or YURAY motifs) can cause aberrant splicing even without affecting the AG dinucleotide
   - Distance and strength of branch point correlate with splice site usage and alternative splicing patterns
   - **Use Case**: Improve meta-model's ability to distinguish strong vs weak acceptor sites; better predict cryptic acceptor activation when canonical branch point is disrupted

2. **Non-Canonical Intron Characterization**:
   - **U12-type (AT-AC) introns**: Represent <0.5% of introns but use distinct spliceosome (U12-dependent) with different consensus sequences
   - **GC-AG introns**: Currently identified (~1.12% of donors) but not fully characterized
   - Different regulatory mechanisms and tissue-specific expression patterns
   - **Use Case**: Reduce false positives when predicting cryptic sites; correctly handle variants in genes with U12-type introns (e.g., some neurological disease genes)

3. **Splice Site Strength Scoring**:
   - Combine donor context + acceptor context + branch point strength + intron length
   - Create composite "splice site strength" scores for training meta-models
   - **Use Case**: Recalibrate base model predictions based on context strength; prioritize variants predicted to activate "strong" cryptic sites

### Where to Implement

**Recommended Location**: `scripts/data/splice_sites/`

**Rationale**:
- This is primarily an **analysis and feature extraction** task, not core workflow functionality
- Similar to `analyze_consensus_motifs.py`, which validates and characterizes splice sites
- Output would be **additional TSV files** with branch point annotations and intron classifications
- These derived features can be optionally loaded by the meta-model builder

**Proposed Structure**:
```
scripts/data/splice_sites/
├── analyze_consensus_motifs.py      (✅ exists - donor/acceptor consensus)
├── analyze_branch_points.py         (🆕 new - branch point identification)
├── analyze_noncanonical_introns.py  (🆕 new - U12-type, GC-AG characterization)
└── generate_splice_strength_scores.py (🆕 new - composite scoring)
```

**Alternative Location**: `meta_spliceai/splice_engine/meta_models/feature_extraction/`
- If these features become tightly integrated into the meta-model training pipeline
- Would require more substantial refactoring
- Defer this decision until features are proven useful in analysis scripts

### Implementation Approach

1. **Branch Point Analysis** (`analyze_branch_points.py`):
   - Extend acceptor motif extraction to 50-mer or 70-mer
   - Search for YRAY motif (Y=C/T, R=A/G, A=branch point adenosine)
   - Score branch point strength based on:
     - Motif match quality
     - Distance from AG (optimal: 20-40 nt)
     - Surrounding sequence context
   - Output: `branch_points.tsv` with columns:
     - `gene_id`, `transcript_id`, `acceptor_position`, `branch_point_position`
     - `branch_point_motif`, `branch_point_score`, `distance_to_ag`

2. **Non-Canonical Intron Analysis** (`analyze_noncanonical_introns.py`):
   - Identify AT-AC introns (search for AT donors + AC acceptors in same transcript)
   - Characterize GC-AG introns (already identified, add context analysis)
   - Analyze sequence features distinguishing U2 vs U12 introns:
     - U12 donor consensus: RTATCCTT (positions -3 to +5)
     - U12 branch point: TCCTTAAC (stronger consensus than U2)
   - Output: `noncanonical_introns.tsv` with columns:
     - `intron_id`, `gene_id`, `transcript_id`, `intron_type` (U2_GT-AG, U2_GC-AG, U12_AT-AC)
     - `donor_sequence`, `acceptor_sequence`, `branch_point_sequence`
     - `confidence_score`, `gene_biotype`

3. **Splice Strength Scoring** (`generate_splice_strength_scores.py`):
   - Integrate outputs from consensus, branch point, and intron type analyses
   - Compute position-specific weight matrices (PWM) for donor and acceptor
   - Calculate MaxEntScan-style scores (or similar)
   - Output: Enhanced `splice_sites_with_strength.tsv` including:
     - All original columns from `splice_sites_enhanced.tsv`
     - `donor_strength_score`, `acceptor_strength_score`
     - `branch_point_score`, `intron_type`, `composite_splice_strength`

### Integration with Meta-Model

**Current State**: Meta-model uses base SpliceAI scores + genomic features

**Enhanced State**: Meta-model could optionally use:
```python
features = [
    'donor_score',           # from SpliceAI
    'acceptor_score',        # from SpliceAI
    'donor_strength',        # 🆕 from PWM / context analysis
    'acceptor_strength',     # 🆕 from PWM / context analysis
    'branch_point_score',    # 🆕 from branch point analysis
    'intron_type',           # 🆕 U2_GT-AG, U2_GC-AG, U12_AT-AC
    'polypyrimidine_score',  # 🆕 from acceptor analysis
    # ... other genomic features
]
```

**Backward Compatibility**: These would be **optional features**; meta-model should work without them if files not present.

### Dependencies

- `pyfaidx`: sequence extraction (already used)
- `polars` or `pandas`: data processing (already used)
- `scipy` or `numpy`: scoring functions (already available)
- Ensembl GTF + FASTA: base genomic data (already available)

### Validation

Test against known examples:
1. **Branch point mutations**: e.g., IVS1-32 (C→T) in *ABCC7/CFTR* disrupts branch point
2. **U12-type introns**: e.g., *RNPC3*, *ZRSR2* genes with known AT-AC introns
3. **GC-AG introns**: Enriched in immunoglobulin genes

### References

- **Branch Points**:
  - Gao et al. (2008). "Human branch point consensus sequence is yUnAy." *Nucleic Acids Research*.
  - Corvelo et al. (2010). "Genome-wide association between branch point properties and alternative splicing." *PLoS Computational Biology*.

- **U12-type Introns**:
  - Turunen et al. (2013). "The significant other: splicing by the minor spliceosome." *Wiley Interdiscip Rev RNA*.
  - Madan et al. (2015). "Aberrant splicing of U12-type introns is the hallmark of ZRSR2 mutant myelodysplastic syndrome." *Nature Communications*.

---

## 2. Exonic/Intronic Splicing Enhancer and Silencer (ESE/ISS) Motif Analysis

**Priority**: Medium-Low  
**Status**: Planned  
**Estimated Effort**: 3-5 days

### Context

Splice sites are regulated not only by consensus sequences but also by auxiliary cis-regulatory elements:
- **ESE** (Exonic Splicing Enhancers): Promote exon inclusion
- **ESS** (Exonic Splicing Silencers): Promote exon skipping
- **ISE** (Intronic Splicing Enhancers): Strengthen nearby splice sites
- **ISS** (Intronic Splicing Silencers): Weaken nearby splice sites

### Why This Feature Is Helpful

1. **Variant Impact Prediction**: Synonymous variants in exons can disrupt ESE motifs and cause exon skipping
2. **Cryptic Site Prediction**: ESE near a weak cryptic donor/acceptor can activate it
3. **Meta-Model Features**: Density and type of regulatory motifs within ±50 nt of splice sites

### Where to Implement

**Location**: `scripts/data/splice_sites/analyze_regulatory_motifs.py`

**Output**: `regulatory_motifs.tsv` with counts of ESE/ESS/ISE/ISS motifs per exon/intron

### Resources

- **Databases**: ESEfinder, RESCUE-ESE, FAS-hex
- **Experimental Data**: CLIP-seq peaks for SR proteins (SRSF1, SRSF2) and hnRNPs

---

## 3. RNA Secondary Structure Analysis at Splice Sites

**Priority**: Low  
**Status**: Exploratory  
**Estimated Effort**: 5-7 days

### Context

RNA secondary structure can influence splice site accessibility:
- Strong stem-loops near splice sites can sequester them
- Structure changes due to variants can affect splicing

### Why This Feature Is Helpful

- Explain cases where strong consensus is not used (structure occlusion)
- Predict impact of variants on local RNA structure

### Where to Implement

**Location**: `scripts/data/splice_sites/analyze_rna_structure.py`

**Dependencies**: ViennaRNA package (RNAfold)

**Complexity**: High - structure prediction is computationally expensive and less reliable

---

## 4. Tissue-Specific and Disease-Associated Splice Site Analysis

**Priority**: Medium  
**Status**: Research Phase  
**Estimated Effort**: 1-2 weeks

### Context

Compare splice site characteristics across:
- Constitutive vs alternative exons
- Tissue-specific exons (brain, heart, muscle)
- Disease-associated variants (ClinVar, SpliceVarDB)

### Why This Feature Is Helpful

- Train context-aware meta-models (tissue-specific, disease-specific)
- Benchmark MetaSpliceAI on known pathogenic splice variants

### Where to Implement

**Location**: 
- Analysis scripts: `scripts/analysis/tissue_specificity/`
- Meta-model integration: `meta_spliceai/splice_engine/meta_models/context_models/`

---

## 5. Integration with Experimental Splice Data

**Priority**: High (for validation)  
**Status**: Planned  
**Estimated Effort**: 1-2 weeks

### Context

Validate predictions against:
- RNA-seq data (GTEx, ENCODE)
- CLIP-seq for splicing factors
- Massively parallel splice reporter assays (MPSA)

### Why This Feature Is Helpful

- Benchmark meta-model performance on real usage data
- Calibrate splice site strength scores
- Identify systematic errors in predictions

### Where to Implement

**Location**: 
- Data processing: `scripts/data/experimental/`
- Benchmarking: `meta_spliceai/splice_engine/meta_models/benchmarks/`

---

## Summary Table

| Feature | Priority | Location | Effort | Dependencies | Status |
|---------|----------|----------|--------|--------------|--------|
| **Branch Point + Non-canonical Introns** | Medium | `scripts/data/splice_sites/` | 2-3 days | pyfaidx, polars | Planned |
| ESE/ISS Motif Analysis | Medium-Low | `scripts/data/splice_sites/` | 3-5 days | ESE databases | Planned |
| RNA Secondary Structure | Low | `scripts/data/splice_sites/` | 5-7 days | ViennaRNA | Exploratory |
| Tissue/Disease Context | Medium | `scripts/analysis/` | 1-2 weeks | GTEx, ClinVar | Research |
| Experimental Data Integration | High | `scripts/data/experimental/` | 1-2 weeks | RNA-seq, CLIP-seq | Planned |

---

## How to Contribute

If you'd like to implement any of these features:

1. Create a new branch: `git checkout -b feature/branch-point-analysis`
2. Implement the script in the appropriate location
3. Add unit tests and validation against known examples
4. Update this document with implementation status
5. Submit a pull request with documentation

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-11  
**Maintained By**: MetaSpliceAI Development Team

