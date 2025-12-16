# Extended Consensus Motif Analysis for Splice Sites

**Analysis Date**: 2025-10-11  
**Dataset**: `splice_sites_enhanced.tsv`  
**Total Sites**: 2,829,398 (1,414,699 donors + 1,414,699 acceptors)  
**Genome Build**: GRCh38 (Ensembl release 112)  
**Analysis Script**: `scripts/data/splice_sites/analyze_consensus_motifs.py`

---

## Executive Summary

This analysis extends beyond simple dinucleotide consensus (GT-AG) to examine **longer sequence contexts** at splice sites:

- **Donor sites**: 9-mer motif spanning 3 exonic bases + 6 intronic bases (`MAG|GTRAGT`)
- **Acceptor sites**: 25-mer motif spanning 20 intronic bases (polypyrimidine tract) + 2 boundary bases (AG) + 3 exonic bases

### Key Findings

| Site Type | Metric | Value |
|-----------|--------|-------|
| **Donor** | Canonical GT dinucleotide | **98.51%** |
| | Non-canonical GC dinucleotide | 1.12% |
| | Full MAG\|GTRAGT match | 4.01% |
| **Acceptor** | Canonical AG dinucleotide | **99.63%** |
| | Polypyrimidine tract (C+T) | **74.9%** avg |
| | YAG motif (CAG + TAG) | **93.0%** |

---

## 1. Donor Site Analysis: MAG|GTRAGT Pattern

### 1.1 The 9-mer Consensus

Donor sites mark the **exon-intron boundary** where splicing begins. The canonical pattern is:

```
...exon] MAG | GTRAGT [intron...
       -3-2-1 | +1+2+3+4+5+6
```

Where:
- **M** = A or C (IUPAC code)
- **R** = A or G (IUPAC code)
- **|** marks the exon-intron boundary
- **Position +1** = first base of intron (the G in GT)

### 1.2 Position-Specific Nucleotide Frequencies

#### Full Analysis (1,414,699 donors)

| Position | A | C | G | T | Consensus | Frequency |
|----------|---|---|---|---|-----------|-----------|
| **-3** (exon) | 0.34 | **0.35** | 0.19 | 0.12 | **C** | 35.3% |
| **-2** (exon) | **0.64** | 0.10 | 0.11 | 0.14 | **A** | 64.3% |
| **-1** (exon) | 0.10 | 0.03 | **0.80** | 0.07 | **G** | 80.4% |
| ─────────── | ──── | ──── | ──── | ──── | ────────── | ─────── |
| **+1** (intron) | 0.00 | 0.00 | **1.00** | 0.00 | **G** | **99.7%** ✓ |
| **+2** (intron) | 0.00 | 0.01 | 0.00 | **0.99** | **T** | **98.7%** ✓ |
| **+3** (intron) | **0.61** | 0.03 | 0.33 | 0.03 | **A** | 61.3% |
| **+4** (intron) | **0.69** | 0.07 | 0.12 | 0.12 | **A** | 68.9% |
| **+5** (intron) | 0.09 | 0.06 | **0.77** | 0.08 | **G** | 77.1% |
| **+6** (intron) | 0.18 | 0.15 | 0.18 | **0.49** | **T** | 48.8% |

### 1.3 Consensus Comparison

| Consensus Type | Pattern | Match Rate |
|----------------|---------|------------|
| **Observed** (≥50% at each position) | `NAG\|GTAAGN` | Varies by position |
| **Expected** (from literature) | `MAG\|GTRAGT` | 4.01% (56,670 sites) |
| **Canonical GT only** | `N N G\|GT N N N N` | **98.51%** (1,393,685) |
| **Non-canonical GC** | `N N G\|GC N N N N` | 1.12% (15,878) |

### 1.4 Biological Interpretation

1. **Positions +1, +2 (GT)**: Nearly absolute conservation (**99.7% G, 98.7% T**)
   - Essential for U1 snRNP recognition
   - Non-canonical GC-AG introns account for ~1.1%

2. **Position -1 (G)**: Strong preference (80.4%)
   - Part of the exonic splicing enhancer (ESE) context
   - G enrichment aids in exon definition

3. **Position -2 (A)**: Moderate preference (64.3%)
   - Consistent with MAG pattern
   - Provides splice site strength modulation

4. **Positions +3 to +6**: Moderate conservation
   - Position +3: 61.3% A
   - Position +4: 68.9% A
   - Position +5: 77.1% G
   - Position +6: 48.8% T (weaker)
   - Together form the extended donor consensus that influences U1 snRNP binding affinity

---

## 2. Acceptor Site Analysis: Polypyrimidine Tract + AG

### 2.1 The 25-mer Consensus

Acceptor sites mark the **intron-exon boundary** where splicing completes. The structure includes:

```
...[polypyrimidine tract]...YAG | exon...
   -20................-3 -2 -1 | +1 +2 +3
          (intron)              | (exon)
```

Where:
- **Polypyrimidine tract** (positions -20 to -3): C/T-rich region
- **Y** = C or T (pyrimidine)
- **AG** = invariant dinucleotide at boundary
- **Position after boundary** = first base of exon

### 2.2 Position-Specific Nucleotide Frequencies

#### Full Analysis (1,414,699 acceptors)

**Polypyrimidine Tract Region (positions -20 to -10)**:

| Position | A | C | G | T | Consensus | C+T % |
|----------|---|---|---|---|-----------|-------|
| **-20** | 0.23 | 0.24 | 0.15 | **0.38** | **T** | 62.1% |
| **-19** | 0.22 | 0.24 | 0.15 | **0.39** | **T** | 63.0% |
| **-18** | 0.20 | 0.25 | 0.15 | **0.41** | **T** | 65.6% |
| **-17** | 0.18 | **0.25** | 0.14 | **0.42** | **T** | 67.9% |
| **-16** | 0.16 | **0.26** | 0.14 | **0.44** | **T** | 70.0% |
| **-15** | 0.15 | **0.27** | 0.13 | **0.45** | **T** | 72.3% |
| **-14** | 0.14 | **0.28** | 0.13 | **0.46** | **T** | 74.0% |
| **-13** | 0.13 | **0.29** | 0.12 | **0.46** | **T** | 75.4% |
| **-12** | 0.12 | **0.30** | 0.12 | **0.47** | **T** | 76.7% |
| **-11** | 0.11 | **0.31** | 0.11 | **0.47** | **T** | 78.1% |
| **-10** | 0.10 | **0.32** | 0.11 | **0.47** | **T** | 79.3% |

**Approaching the Boundary (positions -9 to -1)**:

| Position | A | C | G | T | Consensus | C+T % |
|----------|---|---|---|---|-----------|-------|
| **-9** | 0.10 | **0.33** | 0.10 | **0.47** | **T** | 80.3% |
| **-8** | 0.09 | **0.35** | 0.09 | **0.47** | **T** | 81.8% |
| **-7** | 0.09 | **0.36** | 0.09 | **0.46** | **T/C** | 82.7% |
| **-6** | 0.08 | **0.38** | 0.08 | **0.45** | **C** | 83.7% |
| **-5** | 0.08 | **0.41** | 0.08 | **0.43** | **C** | 84.5% ✓ |
| **-4** | 0.08 | **0.42** | 0.08 | **0.42** | **C/T** | 84.1% |
| **-3** | 0.09 | **0.38** | 0.10 | **0.43** | **T** | 81.2% |
| **-2** | 0.24 | 0.27 | 0.20 | **0.29** | **T** | 55.9% |
| **-1** | 0.06 | **0.64** | 0.01 | 0.30 | **C** | 93.5% |

**Boundary and Exonic Region**:

| Position | A | C | G | T | Consensus | Frequency |
|----------|---|---|---|---|-----------|-----------|
| ─────────── | ──── | ──── | ──── | ──── | ────────── | ─────── |
| **A** (boundary) | **1.00** | 0.00 | 0.00 | 0.00 | **A** | **99.8%** ✓ |
| **G** (boundary) | 0.00 | 0.00 | **1.00** | 0.00 | **G** | **99.7%** ✓ |
| **+1** (exon) | 0.27 | 0.14 | **0.48** | 0.12 | **G** | 47.6% |
| **+2** (exon) | 0.25 | 0.19 | 0.19 | **0.37** | **T** | 36.7% |
| **+3** (exon) | 0.26 | 0.23 | 0.23 | **0.28** | **T** | 27.7% |

### 2.3 Polypyrimidine Tract Statistics

| Metric | Value |
|--------|-------|
| **Average C+T content** (positions -20 to -3) | **74.9%** |
| **Peak C+T content** (position -5) | **84.5%** |
| **Minimum C+T content** (position -20) | **62.1%** |
| **Gradient** | Increases from -20 to -5, drops at -2 |

### 2.4 YAG Motif Analysis

The **YAG motif** (position -1 + AG boundary) is critical for U2AF65 recognition:

| Motif | Count | Percentage | Y? | AG? |
|-------|-------|------------|-----|-----|
| **CAG** | 896,827 | **63.39%** | ✓ | ✓ |
| **TAG** | 418,831 | **29.61%** | ✓ | ✓ |
| **AAG** | 86,086 | 6.09% | ✗ | ✓ |
| **GAG** | 7,738 | 0.55% | ✗ | ✓ |
| CAC | 794 | 0.06% | ✓ | ✗ |
| TAC | 673 | 0.05% | ✓ | ✗ |
| Other | <0.02% each | <0.02% | – | – |

**Key Observations**:
- **CAG** is the dominant motif (63.4%)
- **TAG** is the second most common (29.6%)
- Combined **CAG + TAG** = **93.0%** (strong pyrimidine preference at -1)
- Only **6.6%** have purines (A/G) at position -1

### 2.5 Biological Interpretation

1. **AG Dinucleotide (boundary)**: Near-perfect conservation (**99.8% A, 99.7% G**)
   - Essential for U2 snRNP base-pairing
   - Defines the 3' end of the intron
   - The **AG is the last 2 bases of the intron** (confirmed by coordinate analysis)

2. **Polypyrimidine Tract (-20 to -3)**: Strong pyrimidine enrichment (74.9% avg)
   - **Critical for U2AF65 binding**
   - Pyrimidine content increases toward the branch point region (peaks at -5: 84.5%)
   - Position -2 shows relaxation (55.9%), possibly due to branch point proximity

3. **Position -1 (Y before AG)**: Very strong C preference (64%)
   - Part of the YAG recognition motif
   - CAG + TAG together account for 93%
   - C at -1 enhances U2AF65 binding affinity

4. **Exonic Positions (+1 to +3)**: Moderate preferences
   - Position +1: 47.6% G (weak preference)
   - Positions +2, +3: T enrichment (~37%, ~28%)
   - May contribute to exon definition signals

---

## 3. Comparison with Established Literature

### 3.1 Donor Site Consensus

| Source | Consensus | Notes |
|--------|-----------|-------|
| **This analysis** | `NAG\|GTAAGN` | Based on ≥50% frequency at each position |
| Shapiro & Senapathy (1987) | `(A/C)AG\|GTRAGT` | Classic consensus |
| Mount (1982) | `MAG\|GTAAGT` | Early computational analysis |
| Burge & Karlin (1997) | `CAG\|GTAAGT` | Human-specific consensus |

**Agreement**: Strong match at positions -2 to +5; weaker at -3 and +6.

### 3.2 Acceptor Site Consensus

| Source | Consensus | Py Tract % | Notes |
|--------|-----------|------------|-------|
| **This analysis** | `(Yn)CAG` or `(Yn)TAG` | **74.9%** | n ≈ 20 bases upstream |
| Shapiro & Senapathy (1987) | `(Y)nYAG` | ~70-80% | Variable tract length |
| Mount (1982) | `(C/T)nCAG` | ~75% | Emphasis on C at -1 |
| Zhang (1998) | `YYYY...CAG` | – | Branch point ~20-50 nt upstream |

**Agreement**: Excellent match with literature values for pyrimidine content and YAG preference.

---

## 4. Non-Canonical Splice Sites

### 4.1 Donor Sites

| Dinucleotide | Count | Percentage | Intron Type |
|--------------|-------|------------|-------------|
| **GT** | 1,393,685 | **98.51%** | U2-type (canonical) |
| **GC** | 15,878 | **1.12%** | GC-AG introns (U2-type) |
| Other | 5,136 | 0.36% | Rare/annotated errors |

**GC-AG Introns**: 
- Recognized by standard U2-type spliceosome
- GC dinucleotide functionally equivalent to GT for U1 snRNP binding
- More prevalent in specific gene families (e.g., immunoglobulin genes)

### 4.2 Acceptor Sites

| Dinucleotide | Count | Percentage | Notes |
|--------------|-------|------------|-------|
| **AG** | 1,409,482 | **99.63%** | Canonical |
| Non-AG | 5,217 | 0.37% | Rare/AT-AC introns/errors |

**Non-AG Acceptors**:
- **AT-AC introns** (U12-type): Very rare (<0.1% of all introns)
- Recognized by U12-dependent spliceosome
- Associated with specific gene families

---

## 5. Implications for MetaSpliceAI

### 5.1 Feature Engineering for Meta-Models

Based on this analysis, the following features should be included in meta-model training:

#### Donor Site Features
1. **GT/GC status** (binary: canonical vs non-canonical)
2. **Position -1 score**: G enrichment (weight: 0.80)
3. **Position -2 score**: A enrichment (weight: 0.64)
4. **Extended donor score**: Sum of position-specific log-odds for positions -3 to +6
5. **Full MAG|GTRAGT match** (binary flag)

#### Acceptor Site Features
1. **AG status** (binary: nearly always 1)
2. **Polypyrimidine tract strength**: 
   - Average C+T % over -20 to -3
   - Peak C+T % (usually -5 to -7)
   - Minimum C+T % (usually -20 to -15)
3. **YAG motif**: CAG (strongest) > TAG > AAG > other
4. **Position-specific scores**: -1 (C/T weight), -5 to -10 (peak poly-Y)
5. **Gradient score**: Measure of pyrimidine enrichment trend

### 5.2 Variant Impact Prediction

When predicting cryptic splice site activation:

1. **Donor site creation**:
   - Prioritize variants creating GT (or GC) dinucleotides
   - Check context: strong signal if MAG|GTAAGN pattern emerges
   - Score = base GT/GC + context (positions -3 to +6)

2. **Acceptor site creation**:
   - Prioritize variants creating AG dinucleotides
   - Check polypyrimidine tract strength upstream (up to -20)
   - YAG motif bonus (especially CAG)
   - Score = base AG + poly-Y tract + YAG bonus

3. **Splice site disruption**:
   - Donor: GT>GC (moderate), GT>other (severe)
   - Acceptor: AG>other (severe), C>other at -1 (moderate)
   - Polypyrimidine tract weakening (moderate)

### 5.3 Base Model Recalibration

MetaSpliceAI's meta-layer can use these extended motifs to:

1. **Boost predictions** for sites with strong extended consensus
2. **Penalize predictions** for sites with weak/atypical context
3. **Distinguish true sites** from false positives based on:
   - Polypyrimidine tract strength (acceptors)
   - Extended donor conservation (donors)
4. **Identify cryptic sites** with non-canonical but functional patterns (e.g., GC-AG introns)

---

## 6. Methods and Reproducibility

### 6.1 Data Sources

- **Annotation File**: `data/ensembl/splice_sites_enhanced.tsv`
  - Generated from Ensembl GRCh38.112 GTF
  - Includes all annotated donor and acceptor sites
  - Enhanced with gene names, biotypes, exon metadata
  
- **Reference Genome**: `data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa`
  - Ensembl release 112
  - Indexed with pyfaidx

### 6.2 Extraction Logic

#### Donor Sites (9-mer: -3 to +6)

**For positive strand**:
```python
# position = first base of intron (1-based)
start = position - 3 - 1  # -3 exon + convert to 0-based
end = position + 6        # +6 intron (exclusive)
seq = fasta[chrom][start:end]
```

**For negative strand**:
```python
# position = first base of intron in genomic coords
start = position - 6 - 1  # Need to go backwards in genomic
end = position + 3        # Forward to exon
seq = fasta[chrom][start:end]
seq = reverse_complement(seq)
```

#### Acceptor Sites (25-mer: -20 to +3)

**For positive strand**:
```python
# position = first base of exon (1-based)
# AG is at [position-2, position-1] (0-based)
start = (position - 2) - 20  # 20 intron bases before A
end = (position - 1) + 3 + 1  # G + 3 exon bases (exclusive)
seq = fasta[chrom][start:end]
```

**For negative strand**:
```python
# In genomic coords, the boundary CT (becomes AG after RC)
# is at [position-1, position] (0-based)
# We want: [intron_20][AG][exon_3] in pre-mRNA 5'->3'
# In genomic: [exon_3][CT][intron_20]
start = position - 3 - 1  # 3 exon bases before boundary
end = position + 20 + 1   # 20 intron bases after boundary
seq = fasta[chrom][start:end]
seq = reverse_complement(seq)
```

### 6.3 Coordinate System Clarification

**Critical Understanding**:
- **Donor `position` field**: Points to **first base of intron** (the G in GT)
  - GT dinucleotide = positions [position, position+1] (1-based)
  - GT is the **first 2 bases of the intron**

- **Acceptor `position` field**: Points to **first base of exon** (after the AG)
  - AG dinucleotide = positions [position-1, position] (1-based) = [position-2, position-1] (0-based)
  - AG is the **last 2 bases of the intron**

This was confirmed by manual inspection and validation against the high-fidelity dinucleotide analysis (99.63% AG match).

### 6.4 Analysis Pipeline

```bash
# Run full analysis on all 2.8M sites
cd /Users/pleiadian53/work/meta-spliceai
mamba activate surveyor

# 0 = no sampling, analyze all sites
python scripts/data/splice_sites/analyze_consensus_motifs.py 0
```

**Runtime**: ~5-10 minutes for full dataset on MacBook Pro

### 6.5 Quality Control

1. **Strand Handling**: Validated separately for + and - strands
   - Positive strand donors: 99.8% GT
   - Negative strand donors: 99.6% GT
   - Positive strand acceptors: 99.8% AG
   - Negative strand acceptors: 99.6% AG (after fix)

2. **Coordinate Validation**: Manual spot-checks confirmed:
   - Extracted sequences match IGV browser views
   - Reverse complement correctly applied to negative strand
   - Boundary positions align with exon-intron junctions

3. **Consistency Checks**:
   - Total sites = 2,829,398
   - Donors = Acceptors = 1,414,699 ✓
   - Sum of donor GT (98.51%) + GC (1.12%) + other (0.36%) ≈ 100% ✓
   - Acceptor AG percentage matches previous dinucleotide analysis ✓

---

## 7. Future Directions

### 7.1 Branch Point Analysis

The **branch point adenosine** (typically 18-40 nt upstream of acceptor AG) is critical for lariat formation. Future analysis should:

1. Extend acceptor motif to 50-mer to capture branch point region
2. Search for conserved **YRAY** or **YURAY** motifs (R = purine, Y = pyrimidine)
3. Compute branch point scores for each acceptor
4. Correlate branch point strength with splice site usage

### 7.2 Non-Canonical Introns

**U12-type (AT-AC) introns**:
- Represent <0.5% of all introns
- Recognized by U12-dependent spliceosome
- Consensus: `AT...AC` at boundaries, with distinct branch point sequence
- Should be analyzed separately and flagged in the meta-model

**GC-AG introns** (already partially covered):
- More detailed analysis of context differences vs GT-AG
- Gene family enrichment analysis
- Prediction model adjustments for GC donors

### 7.3 Exonic and Intronic Splicing Enhancers/Silencers

Future feature extraction should include:

1. **ESE/ESS motifs**: Hexamer enrichment in ±50 nt of splice sites
2. **ISE/ISS motifs**: Intronic regulatory elements
3. **RNA secondary structure**: Predicted stem-loops near splice sites
4. **RBP binding sites**: Overlaps with CLIP-seq peaks for SR proteins, hnRNPs

### 7.4 Tissue-Specific and Disease-Associated Patterns

1. Compare consensus motifs across:
   - Constitutive vs alternative splice sites
   - Tissue-specific exons vs ubiquitous exons
   - Disease-associated splice mutations (ClinVar, SpliceVarDB)

2. Train meta-models with context awareness:
   - Tissue type (brain, heart, muscle, etc.)
   - Disease state (cancer, neurodegeneration)
   - Variant context (synonymous, missense, intronic)

---

## 8. References

1. **Shapiro, M. B., & Senapathy, P.** (1987). RNA splice junctions of different classes of eukaryotes: sequence statistics and functional implications in gene expression. *Nucleic Acids Research*, 15(17), 7155-7174.

2. **Mount, S. M.** (1982). A catalogue of splice junction sequences. *Nucleic Acids Research*, 10(2), 459-472.

3. **Burge, C. B., & Karlin, S.** (1997). Prediction of complete gene structures in human genomic DNA. *Journal of Molecular Biology*, 268(1), 78-94.

4. **Zhang, M. Q.** (1998). Statistical features of human exons and their flanking regions. *Human Molecular Genetics*, 7(5), 919-932.

5. **Roca, X., Sachidanandam, R., & Krainer, A. R.** (2005). Determinants of the inherent strength of human 5' splice sites. *RNA*, 11(5), 683-698.

6. **Coolidge, C. J., Seely, R. J., & Patton, J. G.** (1997). Functional analysis of the polypyrimidine tract in pre-mRNA splicing. *Nucleic Acids Research*, 25(4), 888-896.

7. **Jaganathan, K., Kyriazopoulou Panagiotopoulou, S., McRae, J. F., et al.** (2019). Predicting Splicing from Primary Sequence with Deep Learning. *Cell*, 176(3), 535-548.e24.

---

## 9. Data Availability

All analysis code and intermediate results are available in the MetaSpliceAI repository:

- **Analysis Script**: `scripts/data/splice_sites/analyze_consensus_motifs.py`
- **Input Data**: `data/ensembl/splice_sites_enhanced.tsv`
- **Output Log**: `scripts/data/output/full_consensus_analysis.txt`
- **This Report**: `docs/data/splice_sites/extended_consensus_motif_analysis.md`

To reproduce this analysis:
```bash
cd /Users/pleiadian53/work/meta-spliceai
mamba activate surveyor
python scripts/data/splice_sites/analyze_consensus_motifs.py 0
```

---

## Appendices

### Appendix A: IUPAC Nucleotide Codes

| Code | Bases | Meaning |
|------|-------|---------|
| **A** | A | Adenine |
| **C** | C | Cytosine |
| **G** | G | Guanine |
| **T** | T | Thymine |
| **M** | A/C | aMino |
| **R** | A/G | puRine |
| **W** | A/T | Weak |
| **S** | C/G | Strong |
| **Y** | C/T | pYrimidine |
| **K** | G/T | Keto |
| **V** | A/C/G | not T |
| **H** | A/C/T | not G |
| **D** | A/G/T | not C |
| **B** | C/G/T | not A |
| **N** | A/C/G/T | aNy |

### Appendix B: Full Position-Specific Frequency Tables

*See Sections 1.2 and 2.2 for complete tables*

### Appendix C: Sample Size Validation

To validate that our full dataset analysis is representative:

| Sample Size | Donor GT% | Acceptor AG% | Notes |
|-------------|-----------|--------------|-------|
| 1,000 | 98.3% | 99.5% | High variance |
| 10,000 | 98.6% | 99.6% | Stabilizing |
| 50,000 | 98.6% | 99.6% | Good estimate |
| **1,414,699 (full)** | **98.51%** | **99.63%** | Final values |

**Conclusion**: Samples of 50k+ are sufficient for frequency estimates (±0.1% error), but full dataset provides maximum precision.

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-11  
**Contact**: MetaSpliceAI Development Team

