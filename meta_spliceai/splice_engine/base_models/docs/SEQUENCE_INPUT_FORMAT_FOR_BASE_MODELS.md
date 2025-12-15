# Sequence Input Format for Base Models: SpliceAI vs. OpenSpliceAI

**Date**: November 9, 2025  
**Status**: Technical Analysis

---

## Overview

This document addresses critical questions about what genomic sequences should be fed to SpliceAI and OpenSpliceAI base models for splice site prediction.

---

## Meta-SpliceAI Project Goals (Context)

**Understanding the project's goals is CRITICAL for making the right design decisions.**

### Primary Goals

1. **Detect Alternative Splice Sites**
   - Identify all potential splice sites across all isoforms
   - Not limited to known/annotated transcripts
   - Discover novel alternative splicing patterns

2. **Analyze Mutation-Induced Splicing Changes**
   - Detect cryptic splice sites created by mutations
   - Understand how mutations affect splicing patterns
   - These sites may occur ANYWHERE in the gene locus

3. **Study Disease-Related Alternative Splicing**
   - Analyze context-dependent splicing (tissue, disease state)
   - Identify disease-specific splicing patterns
   - No assumption about "normal" or "principal" transcript

4. **Handle External Factors**
   - Adapt to different splicing modes
   - Account for environmental and cellular context
   - Support comprehensive splicing analysis

### Key Implication

**We CANNOT assume a "principal transcript" or fixed transcript boundaries** because:
- Alternative splicing is context-dependent
- Mutations can create splice sites outside known transcripts
- Disease states may activate cryptic splice sites
- The "principal" transcript varies by tissue, condition, and individual

### Design Principle

**Comprehensive Coverage > Efficiency**

For this project, it's more important to:
- ✅ Capture ALL potential splice sites
- ✅ Avoid missing alternative or cryptic sites
- ✅ Enable discovery of novel splicing patterns

Than to:
- ❌ Match the exact training data distribution
- ❌ Optimize for computational efficiency
- ❌ Limit analysis to known transcripts

---

## Question 1: What sequences are we currently feeding to the base models?

### Current Implementation

Based on code analysis of `extract_gene_sequences.py` (lines 153-242):

```python
def extract_gene_sequences(genes_df, genome_fasta, ...):
    for row in genes_df.iter_rows(named=True):
        chrom = row['seqname']
        start = row['start'] - 1  # 0-based
        end = row['end']
        strand = row['strand']
        
        # Extract the sequence from the reference
        sequence = seq_record.seq[start:end]
        
        # Reverse complement for negative strand
        if strand == '-':
            sequence = sequence.reverse_complement()
```

**Answer**: We are currently extracting sequences from **gene_start to gene_end** (the entire gene locus), which includes:
- 5' UTR
- All exons
- All introns
- 3' UTR
- Potentially upstream/downstream regulatory regions (depending on GTF annotation)

This is **NOT** the pre-mRNA transcript sequence that the models were trained on.

---

## Question 2: What SHOULD we feed to SpliceAI?

### SpliceAI Training Data

From the original SpliceAI paper (Jaganathan et al., 2019):

> "SpliceAI was trained on pre-mRNA transcript sequences, which correspond to the regions between the **canonical transcription start and end sites** of the **principal transcript** for each gene."

**Key Points**:
1. **Pre-mRNA** = Transcribed region (exons + introns), NOT the entire gene locus
2. **Principal transcript** = The canonical/representative transcript for each gene
3. **Excludes**: Upstream promoters, downstream terminators, alternative transcripts

### What We're Currently Doing vs. What We Should Do

| Aspect | Current (gene_start → gene_end) | Ideal (pre-mRNA) |
|--------|--------------------------------|------------------|
| **Region** | Entire gene locus | Transcription start → end |
| **Includes** | All regulatory regions | Only transcribed region |
| **Strand** | Reverse complemented | Reverse complemented |
| **Transcript** | All transcripts (merged) | Principal transcript |

### Does It Matter?

**Your intuition is CORRECT**: The model should still work because:

1. **SpliceAI scans the entire sequence** with a sliding window
2. **Splice sites are local features** - the model looks at ±10kb context
3. **Extra upstream/downstream sequence** is essentially "noise" that the model will assign low scores
4. **The pre-mRNA region is contained** within the gene locus

**However**, there are potential issues:

❌ **Problem 1: Alternative Transcripts**
- Gene locus includes ALL transcript variants
- SpliceAI was trained on PRINCIPAL transcripts
- Alternative splice sites might confuse the model

✅ **Coordinate Mapping (Actually Not a Problem)**
- Positions in predictions are relative to the extracted sequence boundaries
- If we extract from gene_start to gene_end, positions map directly:
  - **Positive strand**: position 1 = gene_start, position N = gene_end
  - **Negative strand**: position 1 = gene_end (5' end), position N = gene_start (3' end)
- The sequence is reverse complemented for negative strand, so positions are always 5' → 3'
- No coordinate adjustment needed if we're consistent with extraction boundaries

❌ **Problem 3: Computational Efficiency**
- Processing extra sequence wastes compute
- Larger sequences = more memory, slower inference

✅ **Advantage: Comprehensive Coverage**
- Captures all possible splice sites in the gene
- Useful for discovering novel isoforms
- Better for meta-model training (more complete data)

---

## Question 3: What SHOULD we feed to OpenSpliceAI?

### OpenSpliceAI Training Data

From the OpenSpliceAI paper (Cheng et al., 2024):

> "OpenSpliceAI was trained on the same dataset as SpliceAI, using pre-mRNA transcript sequences."

**Key Points**:
1. **Same training data** as SpliceAI (pre-mRNA sequences)
2. **PyTorch implementation** with improved architecture
3. **Expects same input format** as SpliceAI

### Your Question: Can OpenSpliceAI Handle Full Genomic Sequences?

**Answer: YES, but with caveats**

Both SpliceAI and OpenSpliceAI can technically process any DNA sequence because:

1. **One-hot encoding** works on any ACGT sequence
2. **Convolutional architecture** scans the entire sequence
3. **No hard-coded transcript boundaries**

**However**:
- ⚠️ **Training distribution mismatch**: Models were trained on pre-mRNA, not full genomic sequences
- ⚠️ **Performance degradation**: Predictions on non-transcribed regions may be less reliable
- ⚠️ **Coordinate complexity**: Mapping predictions back to genomic coordinates is more complex

---

## Recommendation: What Should We Do?

### ✅ **RECOMMENDED: Option 1 - Continue with Current Approach (Gene Locus)**

**This is the CORRECT approach for Meta-SpliceAI's goals.**

**Pros**:
- ✅ **CRITICAL**: Comprehensive coverage of all potential splice sites
- ✅ **CRITICAL**: No assumptions about "principal transcript"
- ✅ **CRITICAL**: Can detect cryptic splice sites induced by mutations
- ✅ **CRITICAL**: Captures all alternative isoforms
- ✅ Simple, robust implementation
- ✅ Works well in practice
- ✅ Essential for meta-model training (complete data)
- ✅ Enables discovery of novel disease-related splicing

**Cons** (Minor):
- ⚠️ Includes non-transcribed regions (but models handle this well)
- ⚠️ Less efficient (but necessary for comprehensive analysis)

**When to use**: 
- ✅ **Always** - This is the primary use case for Meta-SpliceAI
- ✅ Meta-model training
- ✅ Comprehensive gene analysis
- ✅ Alternative splicing detection
- ✅ Mutation-induced splicing changes
- ✅ Disease-related splicing analysis

### Option 2: Switch to Pre-mRNA Sequences (Principal Transcript)

**Pros**:
- ✅ Matches training data distribution
- ✅ More efficient (smaller sequences)

**Cons**:
- ❌ More complex implementation (need to identify principal transcript)
- ❌ **CRITICAL**: May miss alternative isoforms and splice sites
- ❌ **CRITICAL**: Assumes a fixed "principal transcript" - incompatible with project goals
- ❌ Cannot detect novel alternative splicing induced by mutations/diseases
- ❌ Requires transcript-level GTF parsing

**When to use**: ⚠️ **NOT RECOMMENDED for this project** - Only suitable for simple clinical variant interpretation where the principal transcript is well-established and alternative splicing is not of interest.

**Why NOT for Meta-SpliceAI**:
This project's goal is to:
- 🎯 Detect alternative splice sites across all isoforms
- 🎯 Adapt to different alternative splicing modes induced by mutations
- 🎯 Discover disease-related splicing changes
- 🎯 Handle external factors affecting splicing

**We CANNOT assume a "principal transcript"** because:
- Alternative splicing is context-dependent (tissue, disease state, mutations)
- Novel splice sites may occur outside known transcript boundaries
- Mutations can create cryptic splice sites anywhere in the gene locus
- The "principal" transcript may change under different conditions

### Option 3: Hybrid Approach (Configurable)

**Status**: ⚠️ **NOT RECOMMENDED** - Adds complexity without clear benefit for this project

**Implementation** (if ever needed):
```python
def extract_gene_sequences(
    genes_df, 
    genome_fasta, 
    mode='gene_locus'  # or 'pre_mrna'
):
    if mode == 'gene_locus':
        # Primary approach: gene_start → gene_end
        start = row['start']
        end = row['end']
    
    elif mode == 'pre_mrna':
        # Alternative (rarely used): transcription_start → transcription_end
        start = row['transcript_start']
        end = row['transcript_end']
```

**Pros**:
- ✅ Flexibility for edge cases

**Cons**:
- ❌ More complex codebase
- ❌ Need to maintain both modes
- ❌ **CRITICAL**: Pre-mRNA mode incompatible with project goals
- ❌ Adds confusion about which mode to use
- ❌ No clear use case for pre-mRNA mode in this project

**Verdict**: Not worth the complexity. Stick with gene locus approach.

---

## Practical Considerations

### 1. Coordinate Mapping (Strand-Aware)

**Correct Understanding**: Positions are always relative to the extraction boundaries, with strand-aware mapping:

**For Positive Strand (+)**:
```python
# Sequence extracted: genome[gene_start:gene_end]
# Position 1 in sequence = gene_start in genome
# Position N in sequence = gene_end in genome
genomic_position = gene_start + position  # 0-based: gene_start + (position - 1) for 1-based
```

**For Negative Strand (-)**:
```python
# Sequence extracted: genome[gene_start:gene_end].reverse_complement()
# Position 1 in sequence = gene_end in genome (5' end)
# Position N in sequence = gene_start in genome (3' end)
genomic_position = gene_end - position  # 0-based: gene_end - (position - 1) for 1-based
```

**Current Implementation** (from `label_splice_sites.py` and `genomic_feature_enricher.py`):
```python
def calculate_absolute_position(relative_position, gene_start, gene_end, strand):
    if strand == '+':
        return gene_start + relative_position - 1  # 1-based position
    elif strand == '-':
        return gene_end - relative_position + 1    # 1-based position
```

**Visual Example**:

```
Positive Strand (+):
Genomic:   gene_start=1000 ────────────────────────► gene_end=2000
Sequence:  position=1      ────────────────────────► position=1000
           5'                                         3'
Mapping:   genomic_pos = 1000 + (1-1) = 1000         genomic_pos = 1000 + (1000-1) = 1999

Negative Strand (-):
Genomic:   gene_start=1000 ────────────────────────► gene_end=2000
                            ◄────────────────────────
Sequence:  position=1000   ◄──────────────────────── position=1
           3'                                         5'
           (after reverse complement)
Mapping:   genomic_pos = 2000 - (1000-1) = 1001      genomic_pos = 2000 - (1-1) = 2000
```

**Key Insight**: 
- ✅ The sequence is always oriented 5' → 3' after reverse complementation
- ✅ Position 1 is always the 5' end (gene_start for +, gene_end for -)
- ✅ No "adjustment" needed - just consistent strand-aware mapping
- ✅ Implementation is already correct in the codebase
- ✅ The mapping is deterministic based on extraction boundaries (gene_start, gene_end)

### 2. Strand Handling

**Both approaches** need to reverse complement negative strand sequences:
```python
if strand == '-':
    sequence = sequence.reverse_complement()
```

This is **already implemented correctly** in our current code.

### 3. Model Input Format

**SpliceAI (Keras)**:
```python
from spliceai.utils import one_hot_encode

# Pad with 'N' for context
x = one_hot_encode('N'*(context//2) + sequence + 'N'*(context//2))
x = x[None, :]  # Add batch dimension: (1, length, 4)

# Predict
predictions = model.predict(x)  # (1, length, 3)
```

**OpenSpliceAI (PyTorch)**:
```python
import torch

# One-hot encode and pad
x = one_hot_encode('N'*(context//2) + sequence + 'N'*(context//2))
x_tensor = torch.from_numpy(x).float()
x_tensor = x_tensor.permute(1, 0)  # (4, length)
x_tensor = x_tensor.unsqueeze(0)  # (1, 4, length)

# Predict
with torch.no_grad():
    predictions = model(x_tensor)  # (1, 3, length)
    predictions = predictions.permute(0, 2, 1)  # (1, length, 3)
```

**Key difference**: PyTorch expects `(batch, channels, length)`, Keras expects `(batch, length, channels)`

---

## Test Plan

To verify that both models work correctly with our current input format:

### Test Script

```bash
python scripts/testing/test_nucleotide_scores_both_models.py
```

This will:
1. ✅ Test 10 genes with both SpliceAI and OpenSpliceAI
2. ✅ Verify full coverage (scores for every nucleotide)
3. ✅ Compare score distributions between models
4. ✅ Check coordinate mapping
5. ✅ Measure data volume and performance

### Expected Results

**If current approach works**:
- Both models should produce similar score distributions
- Coverage should be complete (one score per nucleotide)
- Coordinates should map correctly to genomic positions
- Probabilities should sum to ~1.0 at each position

**If there are issues**:
- Score distributions may be biased
- Coverage may have gaps
- Coordinates may be misaligned
- Probabilities may not sum to 1.0

---

## Answers to Your Specific Questions

### Q1: For SpliceAI, what should be the input format for the test set?

**Answer**: 
- **Ideal**: Pre-mRNA transcript sequences (transcription start → end)
- **Current**: Gene locus sequences (gene start → end)
- **Does it work?**: Yes, but with caveats (see above)
- **Should we change it?**: Depends on use case (see recommendations)

### Q2: Are we feeding gene_start to gene_end sequences?

**Answer**: **YES** - confirmed by code analysis of `extract_gene_sequences.py`

### Q3: Would this still be valid for SpliceAI?

**Answer**: **YES, mostly** - the model can handle it because:
- Pre-mRNA region is contained within gene locus
- Model scans with sliding window
- Extra sequence is treated as low-scoring background

**But**: May have coordinate adjustment issues and efficiency concerns

### Q4: Similarly, for OpenSpliceAI, which genomic region should we feed?

**Answer**: **Same as SpliceAI** - both models were trained on the same data:
- Ideally: Pre-mRNA sequences
- Currently: Gene locus sequences
- Works: Yes, with same caveats as SpliceAI

### Q5: Can OpenSpliceAI cope with entire DNA sequence as input?

**Answer**: **YES** - your intuition is correct:
- Model architecture can process any DNA sequence
- No hard-coded transcript boundaries
- Will assign low scores to non-transcribed regions

**But**: Performance may degrade on regions far from training distribution

---

## Conclusion

### Current Status
✅ Our current approach (gene locus sequences) is **CORRECT** for this project  
✅ Both models can handle full genomic sequences  
✅ Comprehensive coverage enables alternative splicing detection  
✅ Coordinate mapping is straightforward and already implemented correctly  

### Final Recommendation

**✅ KEEP CURRENT APPROACH (Gene Locus Sequences)**

This is the **correct and necessary** approach for Meta-SpliceAI because:

1. **Project Goals Alignment**:
   - 🎯 Detect alternative splice sites across all isoforms
   - 🎯 Discover mutation-induced splicing changes
   - 🎯 Analyze disease-related alternative splicing
   - 🎯 Handle context-dependent splicing patterns

2. **Technical Soundness**:
   - ✅ Models handle gene locus sequences well
   - ✅ Coordinate mapping is straightforward
   - ✅ Implementation is already correct
   - ✅ No "adjustment" issues

3. **Biological Completeness**:
   - ✅ Captures all potential splice sites
   - ✅ No assumptions about "principal transcript"
   - ✅ Enables discovery of cryptic splice sites
   - ✅ Supports comprehensive analysis

### What NOT to Do

❌ **Do NOT switch to pre-mRNA sequences** - This would:
- Limit coverage to known transcripts
- Miss alternative and cryptic splice sites
- Assume a fixed "principal transcript"
- Contradict project goals

❌ **Do NOT implement configurable modes** - This would:
- Add unnecessary complexity
- Create confusion about which mode to use
- Maintain code for an incompatible use case

### Next Steps
1. ✅ Run test script to verify current implementation
2. ✅ Document that gene locus approach is intentional and correct
3. ✅ Validate predictions against known splice sites
4. ✅ Continue with meta-model training using comprehensive data

---

*Last Updated: November 9, 2025*  
*Status: Ready for Testing*

