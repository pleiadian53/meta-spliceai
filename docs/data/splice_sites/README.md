# Splice Site Analysis Documentation

This directory contains documentation and analysis results for splice site consensus motif studies in MetaSpliceAI.

---

## Documents

### Core Documentation

1. **`enhanced_splice_site_annotations.md`**
   - Specification for the enhanced splice site annotation file
   - Describes `splice_sites_enhanced.tsv` format
   - Includes additional columns: `gene_name`, `exon_id`, `exon_number`, `exon_rank`, `gene_biotype`, `transcript_biotype`
   - Statistics and quality metrics

2. **`consensus_dinucleotide_analysis.md`**
   - Initial dinucleotide (GT-AG) validation
   - Confirms 99.63% AG for acceptors, 98.51% GT for donors
   - Validates strand handling (positive and negative)
   - Cross-references with literature

3. **`extended_consensus_motif_analysis.md`** ⭐
   - **Comprehensive technical report** on extended consensus sequences
   - Donor sites: 9-mer analysis (MAG|GTRAGT)
   - Acceptor sites: 25-mer analysis (polypyrimidine tract + AG)
   - Full position-specific frequency tables
   - Biological interpretation and literature comparison
   - Methods, reproducibility, and implications for MetaSpliceAI
   - **Primary reference document for splice site features**

4. **`CONSENSUS_ANALYSIS_SUMMARY.md`**
   - Executive summary of the consensus motif analysis
   - Key results and validation steps
   - Files generated and how to reproduce
   - Quick reference for main findings

### Related Files

- **`ANSWERS_TO_QUESTIONS.md`**: Addresses specific user questions about splice site annotations
- **`../../../scripts/data/splice_sites/analyze_consensus_motifs.py`**: Production-ready analysis script

---

## Key Findings

### Donor Sites (Exon|Intron Boundary)

| Metric | Value |
|--------|-------|
| Canonical GT at +1,+2 | **98.51%** |
| Non-canonical GC at +1,+2 | 1.12% |
| Full MAG\|GTRAGT match | 4.01% |
| Observed consensus | NAG\|GTAAGN |

**Position-specific preferences**:
- Position -1: 80.4% G (exon side)
- Position -2: 64.3% A (exon side)
- Position +3: 61.3% A (intron side)
- Position +5: 77.1% G (intron side)

### Acceptor Sites (Intron|Exon Boundary)

| Metric | Value |
|--------|-------|
| AG dinucleotide at boundary | **99.63%** |
| Average polypyrimidine content | 74.9% |
| Peak polypyrimidine (position -5) | 84.5% |
| YAG motif (CAG + TAG) | 93.0% |

**YAG distribution**:
- CAG: 63.4% (dominant)
- TAG: 29.6%
- AAG: 6.1% (non-pyrimidine at -1)

---

## Critical Coordinate System

Understanding the coordinate system is essential for sequence extraction:

### Donor Sites
- **`position` field** = first base of **intron** (the G in GT)
- GT dinucleotide = `[position, position+1]` (1-based)
- Extraction pattern: `...exon]GT[intron...`

### Acceptor Sites
- **`position` field** = first base of **exon** (after the AG)
- AG dinucleotide = `[position-1, position]` (1-based) = last 2 bases of intron
- Extraction pattern: `...intron]AG[exon...`

**Rationale**: The GT-AG consensus is **within the intron**:
- GT = first 2 bases of intron (donor boundary)
- AG = last 2 bases of intron (acceptor boundary)

---

## Analysis Scripts

### Primary Script

**`scripts/data/splice_sites/analyze_consensus_motifs.py`**

Production-ready script for extended consensus motif analysis.

**Usage**:
```bash
# Quick test (50k sample, default)
python scripts/data/splice_sites/analyze_consensus_motifs.py

# Full analysis (all 2.8M sites)
python scripts/data/splice_sites/analyze_consensus_motifs.py --full

# Save to file
python scripts/data/splice_sites/analyze_consensus_motifs.py --full --output results.txt

# Help
python scripts/data/splice_sites/analyze_consensus_motifs.py --help
```

**Features**:
- Analyzes donor (9-mer) and acceptor (25-mer) consensus motifs
- Position-specific nucleotide frequency tables
- Polypyrimidine tract analysis
- YAG motif analysis
- Full CLI with sampling, filtering, and output options
- Proper error handling and validation

---

## Data Files

### Input Files

1. **`data/ensembl/splice_sites_enhanced.tsv`**
   - Enhanced splice site annotations
   - 2,829,398 sites (1,414,699 donors + 1,414,699 acceptors)
   - Includes gene names, biotypes, exon metadata

2. **`data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa`**
   - Reference genome (GRCh38, Ensembl release 112)
   - Used for sequence extraction

### Output Files

1. **`scripts/data/output/full_consensus_analysis.txt`**
   - Complete analysis output for all 2.8M sites
   - Position-specific frequency tables
   - Summary statistics

---

## How to Reproduce

```bash
# 1. Activate environment
mamba activate surveyor

# 2. Navigate to project root
cd /Users/pleiadian53/work/meta-spliceai

# 3. Run analysis
# Quick test (30 seconds)
python scripts/data/splice_sites/analyze_consensus_motifs.py --sample 50000

# Full analysis (5-10 minutes)
python scripts/data/splice_sites/analyze_consensus_motifs.py --full

# With output file
python scripts/data/splice_sites/analyze_consensus_motifs.py --full \
    --output "scripts/data/output/consensus_$(date +%Y%m%d).txt"
```

---

## Validation

All results have been validated against:
- ✅ Literature values (Shapiro & Senapathy 1987, Burge & Karlin 1997)
- ✅ Manual IGV inspection of genomic coordinates
- ✅ Separate analysis for positive and negative strands
- ✅ Consistency between sample (50k) and full dataset (2.8M)
- ✅ Cross-validation with original dinucleotide analysis

---

## Implications for MetaSpliceAI

### Feature Engineering

The meta-model can leverage:
1. **Extended donor scores** (positions -3 to +6)
2. **Polypyrimidine tract strength** (positions -20 to -3 for acceptors)
3. **YAG motif bonus** (CAG > TAG > other)
4. **Non-canonical site flags** (GC-AG introns)

### Variant Impact Prediction

Prioritize variants that:
- Create/disrupt GT or AG dinucleotides
- Strengthen/weaken extended consensus context
- Affect polypyrimidine tract (for acceptors)
- Alter position-specific conserved bases

### Base Model Recalibration

Use consensus patterns to:
- Boost predictions for strong extended context
- Penalize predictions with weak context
- Distinguish true sites from false positives
- Identify functional non-canonical sites (GC-AG)

---

## Future Work

See **`docs/FEATURE_WISHLIST.md`** for planned enhancements:

1. **Branch Point Analysis** (Priority: Medium)
   - Identify branch point adenosine upstream of acceptors
   - Score YRAY motifs at -20 to -40 positions
   - Correlate with splice site strength

2. **U12-Type Intron Analysis** (Priority: Medium)
   - Characterize AT-AC introns (<0.5%)
   - Analyze U12-specific consensus (RTATCCTT donor, TCCTTAAC branch point)
   - Flag genes with U12-type introns

3. **ESE/ISS Motif Analysis** (Priority: Medium-Low)
   - Exonic/intronic splicing enhancers and silencers
   - Hexamer enrichment near splice sites
   - Integration with CLIP-seq data

---

## References

1. **Shapiro, M. B., & Senapathy, P.** (1987). RNA splice junctions of different classes of eukaryotes: sequence statistics and functional implications in gene expression. *Nucleic Acids Research*, 15(17), 7155-7174.

2. **Burge, C. B., & Karlin, S.** (1997). Prediction of complete gene structures in human genomic DNA. *Journal of Molecular Biology*, 268(1), 78-94.

3. **Zhang, M. Q.** (1998). Statistical features of human exons and their flanking regions. *Human Molecular Genetics*, 7(5), 919-932.

4. **Roca, X., Sachidanandam, R., & Krainer, A. R.** (2005). Determinants of the inherent strength of human 5' splice sites. *RNA*, 11(5), 683-698.

5. **Jaganathan, K., et al.** (2019). Predicting Splicing from Primary Sequence with Deep Learning. *Cell*, 176(3), 535-548.

---

## Contact

For questions or issues related to splice site analysis:
- Review the comprehensive report: `extended_consensus_motif_analysis.md`
- Check the feature wish list: `docs/FEATURE_WISHLIST.md`
- See the analysis script: `scripts/data/splice_sites/analyze_consensus_motifs.py`

---

**Last Updated**: 2025-10-11  
**Maintained By**: MetaSpliceAI Development Team

