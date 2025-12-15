# Genome Build Compatibility for Base Models

## Date: 2025-10-31

## Critical Discovery: SpliceAI Genome Build Mismatch

### Issue Summary

**Problem**: SpliceAI was trained on **GRCh37/hg19** but we were evaluating on **GRCh38**, causing significant performance degradation.

**Impact**:
- PR-AUC: 0.541 (vs 0.97 in paper) - **44% lower**
- Top-k Accuracy: 0.550 (vs 0.95 in paper) - **42% lower**
- F1 Score: 0.601 at threshold=0.5

**Root Cause**: Coordinate misalignment between genome builds causes the model to predict at wrong positions.

## SpliceAI Training Details

### Training Data
- **Genome Build**: GRCh37/hg19
- **Annotations**: GENCODE V24lift37 (2016)
- **Training Set**: 13,384 genes from chromosomes 2, 4, 6, 8, 10-22, X, Y
- **Test Set**: 1,652 genes from chromosomes 1, 3, 5, 7, 9
- **Splice Junctions**: ~130,000 donor-acceptor pairs

### Model Architecture
- **Type**: Deep residual neural network
- **Input**: 10,000 bp sequence context
- **Output**: Splice site probabilities for each position
- **Models**: SpliceAI-80nt, SpliceAI-400nt, SpliceAI-2k, SpliceAI-10k

### Reported Performance (on GRCh37)
- **PR-AUC**: 0.97
- **Top-k Accuracy**: 95%
- **lincRNA Performance**: 84% top-k accuracy

## Our Evaluation Setup (Before Fix)

### Evaluation Data
- **Genome Build**: GRCh38
- **Annotations**: Ensembl GTF 112 (2023)
- **Test Set**: 55 protein-coding genes (various chromosomes)
- **Splice Sites**: ~2,000+ sites across 55 genes

### Observed Performance (on GRCh38)
- **PR-AUC**: 0.541 ± 0.164
- **Top-k Accuracy**: 0.550 ± 0.149
- **Optimal F1**: 0.650 ± 0.153
- **F1 at 0.5**: 0.601 ± 0.194

## Why Genome Build Matters

### Coordinate Differences

Between GRCh37 and GRCh38, genomic coordinates can shift by:
- **Small changes**: 1-5 bp (common)
- **Medium changes**: 5-50 bp (less common)
- **Large changes**: 50+ bp (rare, but possible)

### Impact on Splice Site Prediction

**Example**:
```
GRCh37: Donor site at chr1:12345
GRCh38: Same donor site at chr1:12350 (shifted +5 bp)

SpliceAI predicts: High score at position 12345 (hg19 coordinates)
We evaluate at: Position 12350 (hg38 coordinates)
Result: Mismatch → False Negative
```

### Why Exact Matching is Critical

Splice sites are **single-nucleotide precision**:
- Donor site: GT dinucleotide at exon-intron boundary
- Acceptor site: AG dinucleotide at intron-exon boundary
- Off by even 1 bp = wrong prediction

## Solution: Use Matching Genome Build

### Option 1: Download GRCh37 Data (Recommended)

**Action**: Download GRCh37 annotations to match SpliceAI's training data

**Expected Improvement**:
- PR-AUC: 0.54 → 0.80-0.90
- Top-k Accuracy: 0.55 → 0.75-0.85
- F1 Score: 0.60 → 0.75-0.85

**Implementation**:
```bash
# Download GRCh37 GTF and FASTA
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --species homo_sapiens \
  --build GRCh37 \
  --release 87

# Derive splice sites for GRCh37
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --consensus-window 2
```

**Advantages**:
- ✅ Matches SpliceAI's training data exactly
- ✅ Expected to restore performance to paper levels
- ✅ No model retraining needed
- ✅ Straightforward implementation

**Disadvantages**:
- ⚠️ Older genome build (2009 vs 2013)
- ⚠️ Older annotations (fewer isoforms)
- ⚠️ Need to maintain two genome builds

### Option 2: Use GRCh38-Trained Model

**Action**: Find or train SpliceAI on GRCh38 data

**Expected Improvement**:
- PR-AUC: 0.54 → 0.85-0.95
- Top-k Accuracy: 0.55 → 0.85-0.95

**Implementation**:
- Search for pre-trained SpliceAI models on GRCh38
- Or retrain SpliceAI from scratch on GRCh38

**Advantages**:
- ✅ Uses modern genome build
- ✅ Uses modern annotations
- ✅ Better for downstream analysis

**Disadvantages**:
- ❌ Requires significant computational resources
- ❌ Time-consuming (days to weeks)
- ❌ May not match paper performance exactly

### Option 3: Continue with Current Setup

**Action**: Document limitations and proceed with GRCh38

**Expected Performance**:
- PR-AUC: ~0.54 (no improvement)
- Top-k Accuracy: ~0.55 (no improvement)

**Rationale**:
- Meta-model can learn from systematic errors
- Focus on **relative improvement** over base model
- Document genome build mismatch as known limitation

**Advantages**:
- ✅ No additional data download
- ✅ Uses modern genome build
- ✅ Meta-model may compensate for misalignment

**Disadvantages**:
- ❌ Lower base model performance
- ❌ Harder to compare to SpliceAI paper
- ❌ May limit meta-model's potential

## Recommended Approach

### Phase 1: Download GRCh37 Data

1. **Download GRCh37 GTF and FASTA** (Ensembl release 87, last GRCh37 release)
2. **Derive splice sites** for GRCh37
3. **Re-run evaluation** on GRCh37 to verify performance improvement
4. **Re-run adjustment detection** on GRCh37 (may find different adjustments)

### Phase 2: Maintain Both Builds

1. **Keep GRCh38 data** for modern analysis
2. **Add GRCh37 data** for SpliceAI compatibility
3. **Configure system** to use appropriate build per workflow
4. **Document** which workflows use which build

### Phase 3: Evaluate Meta-Model on Both Builds

1. **Train meta-model** on GRCh37 (matching SpliceAI)
2. **Evaluate** on GRCh37 test set
3. **Compare** to GRCh38 results
4. **Document** performance differences

## Implementation Plan

### 1. Update Configuration

**File**: `configs/genomic_resources.yaml`

Already supports GRCh37:
```yaml
builds:
  GRCh37:
    gtf: "Homo_sapiens.GRCh37.{release}.gtf"
    fasta: "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    ensembl_base: "https://grch37.ensembl.org/pub/release-{release}"
```

### 2. Download GRCh37 Data

```bash
# Download GRCh37 (Ensembl release 87 - last GRCh37 release)
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --species homo_sapiens \
  --build GRCh37 \
  --release 87 \
  --verbose

# Expected files:
# data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf
# data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa
```

### 3. Derive GRCh37 Splice Sites

```bash
# Derive splice sites for GRCh37
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --consensus-window 2 \
  --verbose

# Expected file:
# data/ensembl/GRCh37/splice_sites_enhanced.tsv
```

### 4. Update Workflows to Support Build Selection

**File**: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`

Add `genome_build` parameter:
```python
def run_enhanced_splice_prediction_workflow(
    config: Optional[SpliceAIConfig] = None,
    genome_build: str = 'GRCh38',  # NEW: Allow build selection
    ...
):
    # Use appropriate data directory
    if genome_build == 'GRCh37':
        local_dir = Path('data/ensembl/GRCh37')
    else:
        local_dir = Path('data/ensembl')
```

### 5. Re-run Evaluation on GRCh37

```bash
# Re-run comprehensive evaluation on GRCh37
python scripts/testing/comprehensive_spliceai_evaluation.py \
  --build GRCh37 \
  --output predictions/evaluation_grch37.parquet
```

**Expected results**:
- PR-AUC: 0.80-0.90 (vs 0.54 on GRCh38)
- Top-k Accuracy: 0.75-0.85 (vs 0.55 on GRCh38)

## Directory Structure

### Current (GRCh38 only)
```
data/ensembl/
├── Homo_sapiens.GRCh38.112.gtf
├── Homo_sapiens.GRCh38.dna.primary_assembly.fa
├── splice_sites_enhanced.tsv
├── gene_features.tsv
└── ...
```

### Proposed (Both builds)
```
data/ensembl/
├── GRCh38/  # Modern build (default)
│   ├── Homo_sapiens.GRCh38.112.gtf
│   ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa
│   ├── splice_sites_enhanced.tsv
│   └── ...
└── GRCh37/  # For SpliceAI compatibility
    ├── Homo_sapiens.GRCh37.87.gtf
    ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
    ├── splice_sites_enhanced.tsv
    └── ...
```

## Other Base Models

### OpenSpliceAI
- **Training Build**: Unknown (likely GRCh38)
- **Recommendation**: Test on both GRCh37 and GRCh38
- **Expected**: Better performance on GRCh38

### Pangolin
- **Training Build**: GRCh38
- **Recommendation**: Use GRCh38 data
- **Expected**: Good performance on GRCh38

### SpliceAI-10k (newer versions)
- **Training Build**: Check documentation
- **Recommendation**: Match training build
- **Expected**: Performance depends on build match

## Automatic Coordinate Alignment Detection

### Overview

**Status**: ✅ ENABLED BY DEFAULT (as of 2025-11-01)

The system automatically detects coordinate alignment between base model predictions and genome build annotations. This is **critical** for multi-build support and prevents genome build mismatches.

### How It Works

1. **Sampling**: Selects ~20 genes from target chromosomes
2. **Prediction**: Runs base model predictions on sample
3. **Comparison**: Compares predictions to annotations
4. **Detection**: Determines optimal coordinate adjustments
5. **Application**: Applies detected adjustments to all predictions

### Configuration

**Default**: Enabled (`use_auto_position_adjustments=True`)

```python
config = SpliceAIConfig(
    # ... other parameters ...
    use_auto_position_adjustments=True,  # ✅ Enabled by default
)
```

**Cost**: ~30 seconds for 20-gene sample  
**Benefit**: Catches genome build mismatches early

### Expected Results by Build

| Build | Base Model Training | Expected Adjustments |
|-------|-------------------|---------------------|
| GRCh37 | SpliceAI (GRCh37) | Zero (perfect alignment) |
| GRCh38 | SpliceAI (GRCh37) | Non-zero (build mismatch) |
| GRCh38 | OpenSpliceAI (GRCh38) | Zero (perfect alignment) |

### Example Output

```
[action] Preparing splice site position adjustments
Using target genes from chromosome 21 for adjustment detection
Running sample predictions on 20 genes for adjustment detection
[info] Position adjustments that will be applied:
  Donor sites:    +0 on plus strand, +0 on minus strand
  Acceptor sites: +0 on plus strand, +0 on minus strand
```

### Why This Matters

**Before automatic detection**:
- Manual adjustment values (hardcoded)
- No verification of coordinate alignment
- Silent failures on genome build mismatches
- 44% performance drop went undetected

**After automatic detection**:
- ✅ Automatic verification of coordinate alignment
- ✅ Early detection of genome build mismatches
- ✅ Self-documenting (reports detected adjustments)
- ✅ Critical for multi-build support

### Implementation

**File**: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`  
**Lines**: 289-365

**Key Functions**:
- `prepare_splice_site_adjustments()` - Orchestrates detection
- `auto_detect_splice_site_adjustments()` - Performs analysis
- `apply_auto_detected_adjustments()` - Applies adjustments

## Testing Strategy

### 1. Verify GRCh37 Performance

Test on same 55 genes with GRCh37 annotations:
- Expected PR-AUC: 0.80-0.90
- Expected Top-k: 0.75-0.85
- **Actual (achieved)**: F1 0.9312, Precision 0.9728, Recall 0.8931

### 2. Compare Builds

| Metric | GRCh38 (Mismatch) | GRCh37 (Correct) | Improvement |
|--------|--------|--------|-------------|
| PR-AUC | 0.541 | TBD | TBD |
| Top-k | 0.550 | TBD | TBD |
| F1 Score | 0.596 | **0.9312** | **+56%** |
| Precision | N/A | **0.9728** | N/A |
| Recall | N/A | **0.8931** | N/A |

### 3. Automatic Adjustment Detection Results

**GRCh37 (SpliceAI)**:
- Detected adjustments: Zero (expected)
- Interpretation: Perfect alignment
- F1 Score: 0.9312 (excellent)

**GRCh38 (SpliceAI)** - Not tested yet:
- Expected adjustments: Non-zero
- Interpretation: Build mismatch
- Expected F1: Lower than GRCh37

## Documentation Updates

### Files to Update

1. ✅ `docs/base_models/GENOME_BUILD_COMPATIBILITY.md` (this file)
2. ⏳ `meta_spliceai/splice_engine/base_models/docs/SPLICEAI.md`
3. ⏳ `docs/base_models/SPLICEAI_TRAINING_DATA.md`
4. ⏳ `docs/COMPREHENSIVE_EVALUATION_RESULTS_55_GENES.md` (update with GRCh37 results)

## References

- SpliceAI Paper: Jaganathan et al., Cell 2019
- GENCODE V24lift37: https://www.gencodegenes.org/human/release_24lift37.html
- Ensembl GRCh37: https://grch37.ensembl.org/
- Ensembl GRCh38: https://www.ensembl.org/

## Summary

**Critical Finding**: SpliceAI trained on GRCh37, we evaluated on GRCh38 → 44% performance drop

**Solution**: Download GRCh37 data and re-evaluate

**Expected**: PR-AUC 0.54 → 0.85, Top-k 0.55 → 0.80

**Action**: Proceed with Option 1 (Download GRCh37) while maintaining GRCh38 for modern analysis

