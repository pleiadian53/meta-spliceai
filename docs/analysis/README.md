# Analysis Tools Documentation

This directory contains documentation for analysis tools in the MetaSpliceAI framework.

---

## Available Tools

### 1. Universal Consensus Analysis ✅

**Script**: `scripts/data/splice_sites/analyze_consensus_motifs_universal.py`  
**Documentation**: `UNIVERSAL_CONSENSUS_ANALYSIS.md`

Analyzes splice site consensus sequences across different genomic builds:
- Extended donor motifs (MAG|GTRAGT)
- Acceptor motifs with polypyrimidine tracts
- Comparative analysis across builds
- Biological validation

**Quick Start**:
```bash
# Compare GRCh37 and GRCh38
python scripts/data/splice_sites/analyze_consensus_motifs_universal.py --compare
```

### 2. Conservation Quantification ✅

**Script**: `scripts/analysis/quantify_conservation.py`  
**Documentation**: `QUANTIFY_CONSERVATION_SUMMARY.md`  
**Status**: ✅ Production Ready (Bugs Fixed, Universal Build Support)

Quantifies conservation at splice sites:
- Position Frequency Matrices (PFM)
- Position Probability Matrices (PPM)
- Information Content (IC)
- Log-odds scoring matrices

**Quick Start**:
```bash
# GRCh37/Ensembl
python scripts/analysis/quantify_conservation.py \
  --sites data/ensembl/GRCh37/splice_sites_enhanced.tsv \
  --fasta data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa \
  --site-type both \
  --outdir grch37_conservation

# GRCh38/MANE
python scripts/analysis/quantify_conservation.py \
  --sites data/mane/GRCh38/splice_sites_enhanced.tsv \
  --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
  --site-type both \
  --outdir grch38_conservation
```

---

## Use Cases

### Data Quality Validation

Verify that extracted splice sites are biologically valid:

```bash
# Validate GRCh38/MANE data
python scripts/data/splice_sites/analyze_consensus_motifs_universal.py \
  --build GRCh38_MANE --release 1.3 --sample 10000
```

**Success Criteria**:
- GT percentage: 97-100%
- AG percentage: 98-100%
- Polypyrimidine content: 70-80%

### Cross-Build Comparison

Ensure consistency across different genomic builds:

```bash
python scripts/data/splice_sites/analyze_consensus_motifs_universal.py --compare
```

**Expected Results**:
- GT/AG percentages within 2% of each other
- Similar polypyrimidine content
- No systematic differences

### Base Model Training Data Validation

Verify that base model training data is correct:

```bash
# For SpliceAI (GRCh37/Ensembl)
python scripts/data/splice_sites/analyze_consensus_motifs_universal.py \
  --build GRCh37 --release 87 --full

# For OpenSpliceAI (GRCh38/MANE)
python scripts/data/splice_sites/analyze_consensus_motifs_universal.py \
  --build GRCh38_MANE --release 1.3 --full
```

---

## Expected Results

### Donor Sites (GT-AG Introns)

| Build | GT % | GC % |
|-------|------|------|
| GRCh37/Ensembl | 97-99% | 1-2% |
| GRCh38/MANE | 97-99% | 1-2% |

**Consensus**: MAG|GTRAGT

### Acceptor Sites

| Build | AG % | Polypyrimidine |
|-------|------|----------------|
| GRCh37/Ensembl | 99-100% | 70-80% |
| GRCh38/MANE | 99-100% | 70-80% |

**Consensus**: (Y)n YYYYYYYYYYYYYYAG|G

---

## Integration with MetaSpliceAI

### Workflow Integration

The consensus analysis tools integrate with the main workflow:

1. **Data Preparation**
   - Extract splice sites from GTF
   - Derive consensus sequences
   - Validate biological patterns

2. **Base Model Training**
   - Use validated splice sites
   - Ensure data quality
   - Compare across builds

3. **Meta-Model Training**
   - Use consensus features
   - PWM-based scoring
   - Information content features

### Feature Engineering

Consensus analysis results can be used as features:

```python
# Example: Add PWM scores as features
from meta_spliceai.analysis import load_pwm_matrices

donor_pwm = load_pwm_matrices('donor')
acceptor_pwm = load_pwm_matrices('acceptor')

# Score sequences
donor_score = score_sequence(donor_seq, donor_pwm)
acceptor_score = score_sequence(acceptor_seq, acceptor_pwm)
```

---

## Related Documentation

- **Universal Base Model Support**: `../base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md`
- **Data Validation**: `../base_models/GRCH38_MANE_VALIDATION_COMPLETE.md`
- **Genomic Resources**: `../setup/GENOMIC_RESOURCE_DOWNLOAD_GUIDE.md`

---

## Contributing

When adding new analysis tools:

1. Create the script in `scripts/analysis/` or `scripts/data/`
2. Add documentation to this directory
3. Update this README
4. Ensure universal build support via Registry
5. Add tests and validation

---

**Last Updated**: 2025-11-06


