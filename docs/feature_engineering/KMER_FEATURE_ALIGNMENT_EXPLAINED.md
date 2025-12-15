# K-mer Feature Alignment: Why Missing K-mers Are Normal

## The Key Insight

**User's Correct Understanding:**
> "If missing features are k-mers, that's probably normal because the motifs in the test data (new genes) may not have all k-mers in their sequences. The k-mers in the training data are likely more comprehensive. If the test dataset (unseen positions or unseen genes) do not have certain k-mers, then their counts are 0."

## Visual Example

### Training Data (1000 Genes)

```
Gene 1:  AAACCCGGGTTTAAACCCGGGTTT...  (10,000 bp)
Gene 2:  AAAGGGCCCTTTAAAGGGCCCTTT...  (8,500 bp)
...
Gene 1000: CCCGGGAAATTTCCCGGGAAATTT...  (12,000 bp)

K-mer extraction across all genes:
┌─────────────────────────────────────┐
│ All Possible 3-mers Found:          │
│                                     │
│ AAA: 15,234 occurrences             │
│ AAC: 12,456 occurrences             │
│ AAG: 11,789 occurrences             │
│ AAT: 13,567 occurrences             │
│ ACA: 10,234 occurrences             │
│ ...                                 │
│ TTT: 14,890 occurrences             │
│                                     │
│ Total: 64 unique 3-mers (4³)        │
│ Coverage: 100% (all possible)       │
└─────────────────────────────────────┘

Model trained with: 121 features
  - 64 k-mer features (AAA, AAC, ..., TTT)
  - 57 other features (scores, context, genomic)
```

### Test Data (1 Gene)

```
Test Gene: AAACCCGGGTTTAAACCCGGGTTT  (3,500 bp)

K-mer extraction from single gene:
┌─────────────────────────────────────┐
│ K-mers Found in Test Gene:          │
│                                     │
│ AAA: 234 occurrences  ✅            │
│ AAC: 156 occurrences  ✅            │
│ AAG: 0 occurrences    ❌ (not in sequence!) │
│ AAT: 0 occurrences    ❌ (not in sequence!) │
│ ACA: 89 occurrences   ✅            │
│ ...                                 │
│ TTT: 198 occurrences  ✅            │
│                                     │
│ Total: 45 unique 3-mers found       │
│ Coverage: 70% (45 of 64)            │
│ Missing: 19 k-mers (AAG, AAT, ...)  │
└─────────────────────────────────────┘

Inference needs: 121 features
  - 45 k-mer features (present in sequence)
  - 19 k-mer features (NOT in sequence → fill with 0)
  - 57 other features (scores, context, genomic)
```

## The Three Cases

### Case 1: Extra K-mers (Rare in Training)

```
Training k-mers:  [AAA, AAC, AAG, ..., TTG, TTT]  (64 k-mers)
Test gene k-mers: [AAA, AAC, AAG, ..., TTG, TTT, CNN, NNN]  (66 k-mers)
                                                   ↑    ↑
                                        Rare k-mers with N (ambiguous base)

Action: Drop CNN, NNN (model never trained on them)
Result: ✅ Use 64 k-mers
```

**Why?** Model was never trained on these rare k-mers, so they provide no information.

### Case 2: Missing K-mers (Not in Test Sequence) ✅ **NORMAL**

```
Training k-mers:  [AAA, AAC, AAG, AAT, ..., TTT]  (64 k-mers)
Test gene k-mers: [AAA, AAC, ..., TTT]  (45 k-mers)
                       ↑ Missing: AAG, AAT, AAN, CAN, ... (19 k-mers)

Action: Fill missing k-mers with count = 0
Result: ✅ Feature matrix has all 64 k-mers (45 with counts, 19 with 0)
```

**Why?** Test gene sequence simply doesn't contain these k-mer motifs. Count = 0 is correct!

**Example:**
```
Test sequence: AAACCCGGGTTT
               ↓↓↓ ↓↓↓ ↓↓↓ ↓↓↓
K-mers:        AAA AAC ACC CCC CCG CGG GGG GGT GTT TTT

K-mers present: AAA, AAC, ACC, CCC, CCG, CGG, GGG, GGT, GTT, TTT
K-mers absent:  AAG, AAT, ACA, ACG, ACT, ... (54 others)

For absent k-mers: count = 0 (correct!)
```

### Case 3: Missing Non-K-mer Features 🚨 **ERROR**

```
Training features:  [donor_score, acceptor_score, tx_start, AAA, AAC, ...]  (121)
Test gene features: [donor_score, acceptor_score, AAA, AAC, ...]  (118)
                                                    ↑ Missing: tx_start, tx_end, num_overlaps

Action: RAISE ERROR
Result: ❌ Incomplete feature generation (bug in inference pipeline)
```

**Why?** Non-k-mer features should ALWAYS be present. Missing them indicates a bug.

## Implementation Logic

### Distinguishing K-mers from Other Features

```python
def is_kmer_feature(feature_name: str) -> bool:
    """
    Check if a feature is a k-mer.
    
    K-mers are:
    - 3 or 4 characters long
    - All uppercase
    - Only contain A, C, G, T
    
    Examples:
      'AAA' → True  (3-mer)
      'ACGT' → True  (4-mer)
      'donor_score' → False  (not a k-mer)
      'tx_start' → False  (not a k-mer)
    """
    kmer_re = re.compile(r'^[ACGT]{3,4}$')
    return bool(kmer_re.match(feature_name))
```

### Alignment Algorithm

```python
def _align_features_with_model(features, model):
    """Align inference features with model expectations."""
    
    # Get expected features from model
    expected_features = model.feature_names_in_
    
    # Compare
    missing = set(expected_features) - set(features.columns)
    extra = set(features.columns) - set(expected_features)
    
    # Separate k-mers from non-k-mers
    missing_kmers = [f for f in missing if is_kmer_feature(f)]
    missing_non_kmers = [f for f in missing if not is_kmer_feature(f)]
    
    # CRITICAL: Non-k-mer features must be present
    if missing_non_kmers:
        raise ValueError(
            f"CRITICAL: Missing {len(missing_non_kmers)} non-k-mer features. "
            f"This indicates incomplete feature generation."
        )
    
    # NON-CRITICAL: K-mers can be missing (fill with 0)
    if missing_kmers:
        logger.info(f"Missing {len(missing_kmers)} k-mers (not in test sequence)")
        for kmer in missing_kmers:
            features[kmer] = 0  # Count = 0 for absent k-mers
    
    # NON-CRITICAL: Extra features can be dropped
    if extra:
        logger.info(f"Dropping {len(extra)} extra features")
    
    # Reindex to match model's expectations
    # - Adds missing k-mers with 0
    # - Drops extra features
    # - Ensures correct column order
    features = features.reindex(columns=expected_features, fill_value=0)
    
    return features
```

## Expected Behavior

### Success Case 1: Perfect Match

```
Feature alignment:
  Model expects: 121 features
  Inference has: 121 features
  Common: 121 features
✅ Features aligned: 121 columns
```

### Success Case 2: Missing K-mers (Normal)

```
Feature alignment:
  Model expects: 121 features (64 k-mers + 57 other)
  Inference has: 102 features (45 k-mers + 57 other)
  Common: 102 features
  ℹ️  Missing 19 k-mers (not in test sequence, will fill with 0)
     K-mers: ['AAG', 'AAT', 'AAN', 'CAN', 'CNN', ...]
✅ Features aligned: 121 columns (19 k-mers filled with 0)
```

### Success Case 3: Extra K-mers (Rare)

```
Feature alignment:
  Model expects: 121 features (64 k-mers + 57 other)
  Inference has: 125 features (68 k-mers + 57 other)
  Common: 121 features
  ℹ️  Extra 4 k-mers (not in training, will drop)
     K-mers: ['CNN', 'NNN', 'AAN', 'GAN']
✅ Features aligned: 121 columns (4 rare k-mers dropped)
```

### Error Case: Missing Non-K-mer Features

```
Feature alignment:
  Model expects: 121 features (64 k-mers + 57 other)
  Inference has: 118 features (64 k-mers + 54 other)
  Common: 118 features
  ❌ Missing 3 CRITICAL features
     Features: ['tx_start', 'tx_end', 'num_overlaps']
❌ CRITICAL: Inference is missing 3 non-k-mer features.
   This indicates incomplete feature generation.
```

## Why This Design Is Correct

### Biological Reality

1. **K-mer Distribution Is Uneven**
   - Some k-mers are common (e.g., AAA, TTT, CCC, GGG)
   - Some k-mers are rare (e.g., AAAA, TTTT in 3-mer context)
   - Some k-mers may not appear in a single gene

2. **Training Data Has More Coverage**
   - 1000 genes → ~10 million bp → all possible 3-mers likely present
   - 1 gene → ~10,000 bp → only subset of 3-mers present

3. **Count = 0 Is Meaningful**
   - "This k-mer does not appear in this gene's sequence"
   - Model can learn from this (e.g., absence of certain motifs)

### Statistical Correctness

```python
# Training
gene_1_kmers = {'AAA': 100, 'AAC': 50, 'AAG': 30, ...}  # 64 k-mers
gene_2_kmers = {'AAA': 80, 'AAC': 60, 'AAG': 0, ...}   # 64 k-mers (AAG absent)
...

# Model learns:
# - AAA is common (high counts)
# - AAG varies (present in some genes, absent in others)
# - Count = 0 for AAG is a valid data point

# Inference
test_gene_kmers = {'AAA': 120, 'AAC': 70, 'AAG': 0, ...}  # AAG absent

# Prediction uses:
# - AAA: 120 (present)
# - AAC: 70 (present)
# - AAG: 0 (absent, but this is informative!)
```

## Summary

| Feature Type | Missing in Inference | Action | Reason |
|--------------|---------------------|--------|--------|
| **K-mers** | ⚠️ Non-critical | Fill with 0 | Not all k-mers appear in every gene sequence |
| **Non-k-mers** | 🚨 **CRITICAL** | Raise error | Should always be computable (scores, genomic features) |
| **Extra k-mers** | ⚠️ Non-critical | Drop | Rare k-mers not in training data |
| **Extra non-k-mers** | ⚠️ Warning | Drop | Unexpected but not fatal |

### Key Principle

**K-mers represent sequence composition:**
- If a k-mer is absent from a sequence, its count is naturally 0
- This is **data**, not an error

**Non-k-mers represent computed features:**
- These should always be computable from the input data
- If missing, it indicates a bug in the feature generation pipeline

---

**Status:** ✅ Implemented correctly in `_align_features_with_model()`

**User's insight was 100% correct!** 🎯

