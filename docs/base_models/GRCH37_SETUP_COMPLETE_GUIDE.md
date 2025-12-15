# Complete GRCh37 Setup Guide: Step-by-Step

## Date: 2025-10-31

## Overview

This guide walks you through downloading GRCh37 data and generating the enhanced splice site annotation file (`splice_sites_enhanced.tsv`) to match SpliceAI's training data.

## Why This Matters

**Critical Finding**: SpliceAI was trained on GRCh37, not GRCh38. Using GRCh38 causes **44% performance drop**.

**Solution**: Download GRCh37 and generate build-matched annotations.

## Step-by-Step Instructions

### Step 1: Download GRCh37 GTF and FASTA

#### Option A: Automated Script (Recommended) ⭐

```bash
# Navigate to project root
cd /Users/pleiadian53/work/meta-spliceai

# Run automated download script
bash scripts/setup/download_grch37_data.sh
```

**What this does**:
1. Downloads GRCh37 GTF (~1.5 GB)
2. Downloads GRCh37 FASTA (~3.0 GB)
3. Derives splice sites automatically
4. Verifies all files
5. Shows next steps

**Expected time**: 15-30 minutes (depends on internet speed)

#### Option B: Manual Download

```bash
# Activate conda environment
conda activate surveyor

# Download GTF and FASTA
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --species homo_sapiens \
  --build GRCh37 \
  --release 87 \
  --verbose
```

**Files created**:
```
data/ensembl/GRCh37/
├── Homo_sapiens.GRCh37.87.gtf           # ~1.5 GB
└── Homo_sapiens.GRCh37.dna.primary_assembly.fa  # ~3.0 GB
```

### Step 2: Generate Enhanced Splice Site Annotations

This is the **key step** that creates `splice_sites_enhanced.tsv` with all the columns needed for variant analysis and other downstream tasks.

#### What is `splice_sites_enhanced.tsv`?

This file contains **all splice sites** (donors and acceptors) extracted from the GTF, with:

**Columns**:
- `chrom`: Chromosome (e.g., '1', '2', 'X', 'Y')
- `start`: Start position of consensus window
- `end`: End position of consensus window
- `position`: **Exact splice site position** (1-based)
- `strand`: '+' or '-'
- `site_type`: 'donor' or 'acceptor'
- `gene_id`: Ensembl gene ID (e.g., 'ENSG00000141510')
- `transcript_id`: Ensembl transcript ID (e.g., 'ENST00000269305')

**Consensus Window**: By default, `consensus_window=2`, meaning:
- `start = position - 2`
- `end = position + 2`
- This captures the dinucleotide motif (GT for donor, AG for acceptor) plus flanking context

#### Generate the File

```bash
# Activate conda environment
conda activate surveyor

# Generate splice sites for GRCh37
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --consensus-window 2 \
  --verbose
```

**Parameters**:
- `--build GRCh37`: Use GRCh37 build
- `--splice-sites`: Derive splice sites
- `--consensus-window 2`: Include ±2 bp around splice site
- `--verbose`: Show detailed progress

**Output**:
```
data/ensembl/GRCh37/splice_sites_enhanced.tsv
```

**Expected**:
- ~500,000 - 600,000 splice sites
- ~5-10 MB file size
- Processing time: 5-10 minutes

#### Verify the File

```bash
# Check file exists
ls -lh data/ensembl/GRCh37/splice_sites_enhanced.tsv

# Check line count
wc -l data/ensembl/GRCh37/splice_sites_enhanced.tsv

# View first few lines
head -20 data/ensembl/GRCh37/splice_sites_enhanced.tsv

# Check column names
head -1 data/ensembl/GRCh37/splice_sites_enhanced.tsv
```

**Expected output**:
```
chrom	start	end	position	strand	site_type	gene_id	transcript_id
1	11868	11872	11870	+	donor	ENSG00000223972	ENST00000456328
1	12226	12230	12228	+	acceptor	ENSG00000223972	ENST00000456328
...
```

### Step 3: Verify GRCh37 Data Structure

After download and derivation, your directory structure should look like:

```
data/ensembl/
├── GRCh38/  # Existing (for modern analysis)
│   ├── Homo_sapiens.GRCh38.112.gtf
│   ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa
│   ├── splice_sites_enhanced.tsv
│   └── ...
└── GRCh37/  # New (for SpliceAI compatibility)
    ├── Homo_sapiens.GRCh37.87.gtf
    ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
    ├── splice_sites_enhanced.tsv  ← This is what we just created
    └── ... (other derived files as needed)
```

## Understanding the Splice Site Extraction Pipeline

### What Happens Under the Hood

The `derive --splice-sites` command uses the following pipeline:

1. **Load GTF**: Parse GRCh37 GTF file using `gffutils`
2. **Extract Transcripts**: Find all transcripts with ≥2 exons (splicing required)
3. **Identify Splice Sites**: For each transcript:
   - **Donors**: At exon-intron boundaries (GT dinucleotide)
   - **Acceptors**: At intron-exon boundaries (AG dinucleotide)
4. **Handle Strand**: Correctly identify sites based on strand orientation
5. **Add Consensus Window**: Include ±N bp around each site
6. **Deduplicate**: Remove duplicate sites (same position, type, gene)
7. **Save**: Write to `splice_sites_enhanced.tsv`

### Strand-Specific Logic

**Positive Strand (+)**:
```
----xxxxxx----xxxxx---------xxxxxx---------->
          D  A     D       A
```
- **Donor (D)**: `exon_end + 1` (GT at start of intron)
- **Acceptor (A)**: `exon_start - 1` (AG at end of intron)

**Negative Strand (-)**:
```
<---xxxxxx----xxxxx---------xxxxxx-----------
          A  D     A       D
```
- **Donor (D)**: `exon_start - 1` (GT at start of intron, transcription order)
- **Acceptor (A)**: `exon_end + 1` (AG at end of intron, transcription order)

### Why "Enhanced"?

The "enhanced" version includes:
1. **Exact position**: Single nucleotide precision
2. **Consensus window**: Flanking context for motif analysis
3. **Gene/Transcript IDs**: For linking to other annotations
4. **Strand information**: For correct interpretation
5. **Site type**: Donor vs acceptor classification

This is more comprehensive than basic BED files and is optimized for:
- **Variant effect prediction**: Check if variant overlaps splice site
- **Model evaluation**: Ground truth for splice site prediction
- **Feature engineering**: Extract sequence context for ML models

## Advanced: Customizing the Extraction

### Change Consensus Window

```bash
# Larger window (±5 bp)
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --consensus-window 5 \
  --verbose
```

**Use cases**:
- Larger window: More sequence context for deep learning models
- Smaller window: Focus on core dinucleotide motif

### Filter by Chromosome

```bash
# Only extract for specific chromosomes
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --chromosomes 1 2 X Y \
  --verbose
```

**Use cases**:
- Testing on specific chromosomes
- Reducing file size for development
- Matching SpliceAI's test set (chr 1, 3, 5, 7, 9)

### Force Regeneration

```bash
# Force regenerate even if file exists
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --force \
  --verbose
```

**Use cases**:
- File corrupted
- Want to change parameters
- Updated GTF file

## Using the Enhanced Splice Sites File

### In Python

```python
import polars as pl

# Load splice sites
splice_sites = pl.read_csv(
    'data/ensembl/GRCh37/splice_sites_enhanced.tsv',
    separator='\t',
    schema_overrides={'chrom': pl.Utf8}
)

print(f"Total splice sites: {len(splice_sites):,}")

# Filter by type
donors = splice_sites.filter(pl.col('site_type') == 'donor')
acceptors = splice_sites.filter(pl.col('site_type') == 'acceptor')

print(f"Donors: {len(donors):,}")
print(f"Acceptors: {len(acceptors):,}")

# Filter by gene
tp53_sites = splice_sites.filter(pl.col('gene_id') == 'ENSG00000141510')
print(f"TP53 splice sites: {len(tp53_sites)}")
```

### For Variant Analysis

```python
def check_variant_overlaps_splice_site(variant_pos, chrom, splice_sites_df):
    """Check if variant position overlaps any splice site."""
    overlaps = splice_sites_df.filter(
        (pl.col('chrom') == chrom) &
        (pl.col('position') == variant_pos)
    )
    return len(overlaps) > 0

# Example
variant_chr = '17'
variant_pos = 7577548  # TP53 splice site
overlaps = check_variant_overlaps_splice_site(variant_pos, variant_chr, splice_sites)
print(f"Variant overlaps splice site: {overlaps}")
```

### For Model Evaluation

```python
def evaluate_predictions(predictions_df, splice_sites_df, threshold=0.5):
    """Evaluate splice site predictions against ground truth."""
    # Merge predictions with ground truth
    merged = predictions_df.join(
        splice_sites_df,
        on=['chrom', 'position', 'strand', 'site_type'],
        how='left'
    )
    
    # Calculate metrics
    tp = len(merged.filter(
        (pl.col('score') >= threshold) & 
        (pl.col('gene_id').is_not_null())
    ))
    
    # ... calculate FP, FN, precision, recall, F1
    return metrics
```

## Next Steps After Setup

### 1. Re-run Evaluation on GRCh37

```bash
# Comprehensive evaluation
python scripts/testing/comprehensive_spliceai_evaluation.py \
  --build GRCh37 \
  --output predictions/evaluation_grch37.parquet
```

**Expected**:
- PR-AUC: 0.80-0.90 (vs 0.54 on GRCh38)
- Top-k Accuracy: 0.75-0.85 (vs 0.55 on GRCh38)

### 2. Re-run Adjustment Detection on GRCh37

```bash
# Detect optimal adjustments
python scripts/testing/test_score_adjustment_detection.py \
  --build GRCh37 \
  --genes 20
```

**Expected**: Adjustments may differ from zero (current GRCh38 finding)

### 3. Update Workflows to Use GRCh37

**Option A: Environment Variable**
```bash
export SS_BUILD=GRCh37
export SS_RELEASE=87
```

**Option B: Configuration File**
Edit `configs/genomic_resources.yaml`:
```yaml
default_build: GRCh37  # Changed from GRCh38
default_release: "87"  # Changed from 112
```

**Option C: Per-Command**
```bash
python scripts/training/train_meta_model.py --build GRCh37
```

### 4. Generate Additional Derived Files (Optional)

```bash
# Generate all derived datasets for GRCh37
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --all \
  --verbose
```

**This creates**:
- `gene_features.tsv`: Gene-level metadata
- `transcript_features.tsv`: Transcript-level metadata
- `exon_features.tsv`: Exon-level metadata
- `overlapping_gene_counts.tsv`: Overlapping gene analysis
- `gene_sequence_*.parquet`: Genomic sequences per chromosome

## Troubleshooting

### Issue: Download Fails

**Error**: Network timeout or connection error

**Solution**:
```bash
# Try manual download
wget https://grch37.ensembl.org/pub/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz
wget https://grch37.ensembl.org/pub/release-87/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz

# Decompress
gunzip *.gz

# Move to correct location
mkdir -p data/ensembl/GRCh37
mv Homo_sapiens.GRCh37.87.gtf data/ensembl/GRCh37/
mv Homo_sapiens.GRCh37.dna.primary_assembly.fa data/ensembl/GRCh37/
```

### Issue: Derivation Fails

**Error**: `gffutils` database error or parsing error

**Solution**:
```bash
# Remove any partial database files
rm -f data/ensembl/GRCh37/*.db

# Try again with force flag
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --force \
  --verbose
```

### Issue: Wrong Number of Splice Sites

**Expected**: ~500,000 - 600,000 splice sites

**If much lower** (<100,000):
- Check GTF file is complete
- Verify no chromosome filtering applied
- Check for parsing errors in logs

**If much higher** (>1,000,000):
- Check for duplicates
- Verify consensus_window is reasonable

### Issue: File Permissions

**Error**: Permission denied when writing files

**Solution**:
```bash
# Check directory permissions
ls -ld data/ensembl/GRCh37/

# Fix if needed
chmod 755 data/ensembl/GRCh37/
```

## Summary

### Quick Reference

**Download GRCh37**:
```bash
bash scripts/setup/download_grch37_data.sh
```

**Generate splice sites manually**:
```bash
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --consensus-window 2 \
  --verbose
```

**Verify**:
```bash
ls -lh data/ensembl/GRCh37/splice_sites_enhanced.tsv
head data/ensembl/GRCh37/splice_sites_enhanced.tsv
```

**Expected Result**:
- File: `data/ensembl/GRCh37/splice_sites_enhanced.tsv`
- Size: ~5-10 MB
- Lines: ~500,000 - 600,000
- Columns: chrom, start, end, position, strand, site_type, gene_id, transcript_id

### Key Points

1. **Enhanced file includes**:
   - Exact splice site positions
   - Consensus window for context
   - Gene/transcript IDs for linking
   - Strand-aware extraction

2. **Used for**:
   - Model evaluation (ground truth)
   - Variant effect prediction
   - Feature engineering
   - Coordinate adjustment detection

3. **Build-specific**:
   - GRCh37 for SpliceAI compatibility
   - GRCh38 for modern analysis
   - Keep both for flexibility

## References

- [Genome Build Compatibility](GENOME_BUILD_COMPATIBILITY.md)
- [GRCh37 Download Guide](GRCH37_DOWNLOAD_GUIDE.md)
- [Genomic Resources README](../../meta_spliceai/system/genomic_resources/README.md)
- [SpliceAI Documentation](../../meta_spliceai/splice_engine/base_models/docs/SPLICEAI.md)

---

**Ready to proceed**: Run the automated script or follow manual steps above! 🚀

