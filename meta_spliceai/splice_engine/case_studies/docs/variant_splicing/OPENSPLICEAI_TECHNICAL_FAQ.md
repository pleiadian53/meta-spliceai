# OpenSpliceAI Technical FAQ: Delta Score Implementation

## Overview

This document provides comprehensive technical answers to key questions about OpenSpliceAI's delta score calculation and variant analysis implementation. These answers are based on direct code analysis of the OpenSpliceAI variant analysis pipeline in `meta_spliceai/openspliceai/variant/utils.py`.

---

## ❓ **Question 1: Is Delta Score a vector of the same length/dimension as the queried sequence?**

### ✅ **Answer: YES, but with important implementation details**

**Technical Implementation:**
```python
# From get_delta_scores() in openspliceai/variant/utils.py

# 1. Sequence window preparation
wid = 2 * flanking_size + cov  # Typically ~10,000 bp window
seq = ann.ref_fasta[chrom][record.pos - wid // 2 - 1 : record.pos + wid // 2].seq

# 2. Model predictions are vectors for each position
y_ref = model.predict(x_ref)  # Shape: [1, sequence_length, 3]
y_alt = model.predict(x_alt)  # Shape: [1, sequence_length, 3]

# Where the 3 dimensions represent:
# [:, :, 0] = Background (no splice site)
# [:, :, 1] = Acceptor site probability  
# [:, :, 2] = Donor site probability
```

**Key Points:**
- **Input sequences** are padded to fixed window size (`wid`)
- **Model outputs** are probability vectors for every position in the window
- **Delta scores** exist for every position: `delta[pos] = y_alt[pos] - y_ref[pos]`
- **Final output** reports only the **maximum delta scores** within the search region

**Evidence from Code:**
```python
# Lines 512-515: Finding maximum delta scores
idx_pa = (y[1, :, 1] - y[0, :, 1]).argmax()  # Max acceptor gain position
idx_na = (y[0, :, 1] - y[1, :, 1]).argmax()  # Max acceptor loss position
idx_pd = (y[1, :, 2] - y[0, :, 2]).argmax()  # Max donor gain position
idx_nd = (y[0, :, 2] - y[1, :, 2]).argmax()  # Max donor loss position
```

---

## ❓ **Question 2: Does the formula assume Alternative and Reference sequences are the same length?**

### ✅ **Answer: NO, OpenSpliceAI handles different lengths intelligently**

**Technical Implementation:**

OpenSpliceAI has sophisticated indel handling logic:

```python
# Lines 494-506: Indel handling in get_delta_scores()

# Handle DELETIONS (ref longer than alt)
if ref_len > 1 and alt_len == 1:
    y_alt = np.concatenate([
        y_alt[:, : cov // 2 + alt_len],
        np.zeros((1, del_len, 3)),  # Insert zeros for deleted positions
        y_alt[:, cov // 2 + alt_len:]
    ], axis=1)

# Handle INSERTIONS (alt longer than ref)  
elif ref_len == 1 and alt_len > 1:
    y_alt = np.concatenate([
        y_alt[:, : cov // 2],
        np.max(y_alt[:, cov // 2 : cov // 2 + alt_len], axis=1)[:, None, :],  # Take max over inserted region
        y_alt[:, cov // 2 + alt_len:]
    ], axis=1)
```

**Sequence Construction:**
```python
# Lines 431-432: Reference and alternative sequence construction
x_ref = 'N' * pad_size[0] + seq[pad_size[0]: wid - pad_size[1]] + 'N' * pad_size[1]
x_alt = x_ref[: wid // 2] + str(record.alts[j]) + x_ref[wid // 2 + ref_len:]
```

**Supported Variant Types:**
- ✅ **SNVs** (Single Nucleotide Variants): Same length sequences
- ✅ **Indels** (Insertions/Deletions): Different length sequences with special processing
- ⚠️ **Complex variants**: Limited support (multi-nucleotide variants return placeholder scores)

**Evidence from Code:**
```python
# Lines 418-421: Multi-nucleotide variant handling
if len(record.ref) > 1 and len(record.alts[j]) > 1:
    delta_scores.append("{}|{}|.|.|.|.|.|.|.|.".format(record.alts[j], genes[i]))
    continue  # Skip complex variants
```

---

## ❓ **Question 3: Does every position/nucleotide have a delta score?**

### ✅ **Answer: YES, but only maxima are reported in final output**

**Technical Implementation:**

```python
# Every position gets a prediction and delta score
for position in range(sequence_length):
    delta_acceptor[position] = y_alt[0, position, 1] - y_ref[0, position, 1]
    delta_donor[position] = y_alt[0, position, 2] - y_ref[0, position, 2]

# But only maximum values are reported
DS_AG = max(delta_acceptor)  # Maximum acceptor gain
DS_AL = max(-delta_acceptor) # Maximum acceptor loss  
DS_DG = max(delta_donor)     # Maximum donor gain
DS_DL = max(-delta_donor)    # Maximum donor loss
```

**Position Reporting:**
```python
# Lines 535-538: Position calculation relative to variant
idx_pa - cov // 2,  # DP_AG: Distance to acceptor gain
idx_na - cov // 2,  # DP_AL: Distance to acceptor loss
idx_pd - cov // 2,  # DP_DG: Distance to donor gain  
idx_nd - cov // 2   # DP_DL: Distance to donor loss
```

**Key Insights:**
- **Full vector** of delta scores is calculated internally
- **Search window** is limited by `dist_var` parameter (default 50bp)
- **Position distances** are relative to the variant position
- **Only significant changes** (maxima) are reported to reduce noise

---

## ❓ **Question 4: Do paired donor/acceptor sites represent predicted alternative splicing patterns?**

### ⚠️ **Answer: PARTIALLY - Individual site changes are reported, but splicing pattern inference requires additional logic**

**Current OpenSpliceAI Output:**
```python
# Lines 528-539: Final delta score format
format_str = "{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}"
delta_scores.append(format_str.format(
    record.alts[j],        # ALT: Alternative allele
    genes[i],              # SYMBOL: Gene symbol
    DS_AG,                 # Acceptor gain score
    DS_AL,                 # Acceptor loss score  
    DS_DG,                 # Donor gain score
    DS_DL,                 # Donor loss score
    DP_AG,                 # Distance to acceptor gain
    DP_AL,                 # Distance to acceptor loss
    DP_DG,                 # Distance to donor gain
    DP_DL                  # Distance to donor loss
))
```

**Limitations for Alternative Splicing Analysis:**
- **Individual sites**: Each donor/acceptor change is reported independently
- **No pairing logic**: OpenSpliceAI doesn't determine which donors pair with which acceptors
- **Pattern inference needed**: Additional logic required to infer:
  - Exon skipping (donor loss + downstream acceptor loss)
  - Intron retention (donor loss + acceptor loss at same intron)
  - Alternative 5'/3' splice sites (nearby donor/acceptor gains)
  - Mutually exclusive exons (complex donor-acceptor combinations)

**Recommended Approach for Splicing Pattern Analysis:**
```python
def infer_splicing_patterns(delta_scores, transcript_annotation):
    """
    Infer alternative splicing patterns from OpenSpliceAI delta scores.
    Requires additional transcript structure analysis.
    """
    patterns = []
    
    # Example: Intron retention detection
    if DS_DL > threshold and DS_AL > threshold:
        if same_intron(DP_DL, DP_AL, transcript_annotation):
            patterns.append("intron_retention")
    
    # Example: Exon skipping detection  
    if DS_DL > threshold and DS_AL > threshold:
        if flanking_same_exon(DP_DL, DP_AL, transcript_annotation):
            patterns.append("exon_skipping")
            
    return patterns
```

---

## ❓ **Question 5: What happens with different length sequences (like intron retention)?**

### ✅ **Answer: OpenSpliceAI handles sequence length differences, but complex splicing modes need additional interpretation**

**Sequence Length Handling:**

```python
# Variable length support in sequence construction
ref_len = len(record.ref)
alt_len = len(record.alts[j])
del_len = max(ref_len - alt_len, 0)

# Dynamic sequence adjustment based on variant type
x_alt = x_ref[: wid // 2] + str(record.alts[j]) + x_ref[wid // 2 + ref_len:]
```

**Intron Retention Analysis:**

Intron retention requires detecting **lack of splicing** rather than just site changes:

```python
def detect_intron_retention(delta_scores, transcript_structure):
    """
    Detect intron retention from OpenSpliceAI scores.
    
    Intron retention indicators:
    1. Simultaneous donor loss (DS_DL > threshold) 
    2. Simultaneous acceptor loss (DS_AL > threshold)
    3. Lost sites flank the same intron boundary
    """
    
    # Check for simultaneous splice site losses
    donor_loss = delta_scores['DS_DL'] > 0.2
    acceptor_loss = delta_scores['DS_AL'] > 0.2
    
    if donor_loss and acceptor_loss:
        # Verify positions correspond to intron boundaries
        donor_pos = variant_pos + delta_scores['DP_DL']
        acceptor_pos = variant_pos + delta_scores['DP_AL']
        
        # Check if positions match known intron boundaries
        if is_intron_boundary(donor_pos, acceptor_pos, transcript_structure):
            return "intron_retention"
    
    return None
```

**Complex Variant Limitations:**

```python
# Lines 404-406: Reference length restrictions
if len(record.ref) > 2 * dist_var:
    logging.warning('Skipping record (ref too long): {}'.format(record))
    return delta_scores
```

**Supported Analysis Types:**
- ✅ **Point mutations** affecting splice sites
- ✅ **Small indels** with length adjustments
- ✅ **Cryptic site activation** (new donor/acceptor gains)
- ⚠️ **Large structural variants** (limited by window size)
- ⚠️ **Complex rearrangements** (require specialized analysis)

---

## 🔬 **Practical Applications**

### **1. Variant Prioritization**
```python
# High-impact variants for further analysis
high_impact = (abs(DS_AG) > 0.5) | (abs(DS_AL) > 0.5) | 
              (abs(DS_DG) > 0.5) | (abs(DS_DL) > 0.5)
```

### **2. Cryptic Site Detection**
```python
# New splice sites created by variants
cryptic_donors = (DS_DG > 0.2) & (DP_DG != 0)  # Gain away from canonical site
cryptic_acceptors = (DS_AG > 0.2) & (DP_AG != 0)  # Gain away from canonical site
```

### **3. Splice Site Disruption**
```python
# Canonical splice sites disrupted
canonical_disruption = (DS_DL > 0.2) & (abs(DP_DL) < 3) |  # Donor at splice site
                      (DS_AL > 0.2) & (abs(DP_AL) < 3)    # Acceptor at splice site
```

---

## 📚 **References**

- **Source Code**: `meta_spliceai/openspliceai/variant/utils.py`
- **Key Function**: `get_delta_scores()` (lines 352-540)
- **Model Architecture**: SpliceAI neural network with 3-class output (background, acceptor, donor)
- **Original Paper**: Jaganathan et al. (2019) "Predicting Splicing from Primary Sequence with Deep Learning"

---

## 🔗 **Related Documentation**

- [VARIANT_SPLICING_BIOLOGY_Q10_Q12.md](../VARIANT_SPLICING_BIOLOGY_Q10_Q12.md) - Biological interpretation
- [OpenSpliceAI Integration Guide](../../../../tests/dev/OPENSPLICEAI_INTEGRATION_DEV_GUIDE.md) - System integration
- [Schema Adapter Framework](../../meta_models/openspliceai_adapter/docs/SCHEMA_ADAPTER_FRAMEWORK.md) - Data format handling
