# 📏 Context Window Strategy for Splice Variant Analysis

## 🎯 **The Two-Context Approach**

You're absolutely correct! The optimal splice variant analysis uses **two different context sizes** for different purposes:

### **🔍 The Strategy**

| **Stage** | **Context Size** | **Purpose** | **Rationale** |
|-----------|------------------|-------------|---------------|
| **Sequence Construction** | **±5000bp** | Get reliable basewise scores | Large context provides full gene structure |
| **Impact Summarization** | **±50bp** | Localize splice effects | Small window reduces false positives |

---

## 🧬 **Why This Two-Stage Approach Works**

### **Stage 1: Large Context (±5000bp) for Sequence Construction**

```python
# For OpenSpliceAI model predictions
constructor = SequenceConstructor(
    reference_fasta="GRCh38.fa",
    context_size=5000  # Large context for reliable basewise scores
)

# Creates sequences like:
# [----5000bp----][variant][----5000bp----]
# Total: 10,001bp sequences
```

**Why 5000bp context?**
- ✅ **Model accuracy**: OpenSpliceAI trained on large contexts
- ✅ **Gene structure**: Captures multiple exons/introns
- ✅ **Splice site interactions**: Distant regulatory elements
- ✅ **Reliable predictions**: Reduces edge effects

### **Stage 2: Small Window (±50bp) for Impact Analysis**

```python
# OpenSpliceAI internally uses small coverage window
coverage_window = 101bp  # ±50bp around variant
dist_var = 50           # Coverage radius

# Reports delta scores in:
# [--50bp--][variant][--50bp--]
# Only 101bp window for impact assessment
```

**Why ±50bp window?**
- ✅ **Localized effects**: Focuses on nearby splice sites
- ✅ **Reduces false positives**: Ignores distant fluctuations
- ✅ **Specific attribution**: Links effect to nearby cryptic/canonical sites
- ✅ **OpenSpliceAI standard**: Matches tool's default behavior

---

## 📊 **Current Implementation Status**

### **✅ Universal VCF Parser** (Correctly Implemented)
```python
# In VCFParsingConfig:
sequence_context: int = 50  # ✅ Small context for variant-focused analysis
```

### **✅ Complete ClinVar Pipeline** (Correctly Implemented)
```python
# In CompletePipelineConfig:
sequence_context: int = 50  # ✅ Small context for delta score preparation
```

### **✅ Tutorial** (Now Corrected)
```python
# Updated tutorial guidance:
constructor = SequenceConstructor(
    context_size=5000  # ✅ Large context for OpenSpliceAI basewise scores
)

# With explanation of dual-context approach
```

---

## 🔬 **Technical Details**

### **OpenSpliceAI's Internal Architecture**

```python
# OpenSpliceAI variant analysis pipeline:

# Step 1: Large context sequence construction
total_window = 2 * flanking_size + coverage_window
total_window = 2 * 5000 + 101 = 10,101bp

# Step 2: Full sequence prediction
wt_scores = model.predict(wt_sequence)    # 10,101 positions
alt_scores = model.predict(alt_sequence)  # 10,101 positions

# Step 3: Delta calculation (full window)
delta_scores = alt_scores - wt_scores     # 10,101 delta values

# Step 4: Impact summarization (small window)
coverage_start = flanking_size - dist_var  # Position 4950
coverage_end = flanking_size + dist_var    # Position 5050
                                          # Only 101bp coverage window

# Step 5: Report max effects in coverage window
DS_AG = max(delta_acceptor[coverage_start:coverage_end])
DS_DL = max(delta_donor[coverage_start:coverage_end])
```

### **Why This Architecture is Optimal**

1. **🎯 Accurate Predictions**: Large context gives model full gene structure
2. **🔍 Localized Effects**: Small window focuses on variant-proximal impacts
3. **📉 Noise Reduction**: Ignores distant, unrelated splice site changes
4. **⚡ Computational Efficiency**: Reports only relevant effects
5. **🔬 Biological Relevance**: Matches splice site influence distances

---

## 🧪 **Practical Examples**

### **Example 1: Canonical Splice Site Disruption**
```python
# Variant: chr17:43045751 G>A (BRCA1 splice donor)

# Large context construction (±5000bp):
wt_sequence = "ATCG...GTAAGT...GCTA"  # 10,001bp with GT at center
alt_sequence = "ATCG...ATAAGT...GCTA"  # Same length, G>A change

# OpenSpliceAI prediction:
# - Full 10kb context used for accurate basewise scores
# - ±50bp window shows: DS_DL = -0.95 at DP = +1 (donor loss)
# - Effect localized to immediate splice site
```

### **Example 2: Cryptic Site Activation**
```python
# Variant: chr1:12345678 C>T (creates new donor)

# Large context construction (±5000bp):
wt_sequence = "GCTA...CTAAGT...ATCG"  # 10,001bp with CT at center
alt_sequence = "GCTA...TTAAGT...ATCG"  # C>T creates TT (weak donor)

# OpenSpliceAI prediction:
# - Full 10kb context captures gene structure
# - ±50bp window shows: DS_DG = +0.23 at DP = -2 (cryptic gain upstream)
# - Distant effects (>50bp) ignored as likely noise
```

---

## 📋 **Best Practices Summary**

### **✅ DO: Use Large Context for Sequence Construction**
```python
# For model input preparation
context_size = 5000  # OpenSpliceAI standard

# Benefits:
# - Reliable basewise predictions
# - Captures gene structure
# - Reduces model edge effects
```

### **✅ DO: Use Small Window for Impact Assessment**
```python
# For delta score interpretation
coverage_window = 101  # ±50bp (OpenSpliceAI default)

# Benefits:
# - Localizes effects to specific sites
# - Reduces false positive noise
# - Matches biological relevance
```

### **❌ DON'T: Mix Up Context Purposes**
```python
# WRONG: Small context for sequence construction
context_size = 50  # Too small for reliable predictions

# WRONG: Large window for impact assessment  
coverage_window = 10001  # Too large, includes noise
```

---

## 🎯 **Implementation Verification**

Our current MetaSpliceAI implementations correctly follow this strategy:

### **✅ Universal VCF Parser**
- Uses 50bp context for variant-focused sequence extraction
- Appropriate for delta score preparation

### **✅ Complete Pipeline**  
- Uses 50bp context for WT/ALT sequence construction
- Ready for OpenSpliceAI analysis

### **✅ Documentation**
- Tutorial now explains dual-context approach
- Clarifies the purpose of each context size

**The two-context strategy is now properly implemented and documented throughout MetaSpliceAI! 🚀**

---

## 📚 **References**

- OpenSpliceAI Variant Analysis Guide: Uses 5000bp flanking + 50bp coverage
- Delta Score Implementation Guide: Explains window architecture
- MetaSpliceAI Universal VCF Parser: Implements 50bp context for variants
- Complete ClinVar Pipeline: Uses appropriate context sizes for each stage

**Key Insight**: The magic is in using the **right context for the right purpose** - large for predictions, small for impact localization! 🎯
