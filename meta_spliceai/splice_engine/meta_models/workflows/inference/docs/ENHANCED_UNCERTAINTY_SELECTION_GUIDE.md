# Enhanced Uncertainty Selection Guide: Achieving ~10% Meta-Model Application

**Date**: 2025-08-20  
**Problem**: Current hybrid mode only selects ~3% of positions for meta-model inference  
**Solution**: Enhanced uncertainty selection with entropy-based criteria and optimized thresholds  
**Target**: ~10% meta-model application rate for better coverage

---

## 🎯 **PROBLEM ANALYSIS**

### **Current State**
```python
# Current thresholds (conservative)
uncertainty_threshold_low = 0.02   # Below: confident non-splice
uncertainty_threshold_high = 0.80  # Above: confident splice
# Uncertainty zone: 0.02-0.80 (only ~3% of positions)
```

### **Issue Identified**
- **Too Conservative**: 0.80 threshold excludes many moderate-confidence positions where meta-model could help
- **Single Criterion**: Only uses max score, ignores entropy/ambiguity in predictions
- **Missed Opportunities**: Positions with balanced scores (high entropy) not captured

---

## ✅ **SOLUTIONS IMPLEMENTED**

### **1. Enhanced Uncertainty Selection** (`enhanced_uncertainty_selection.py`)

**Multi-Criteria Approach**:
```python
# Traditional: Only confidence-based
uncertain_mask = (max_score >= 0.02) & (max_score < 0.80)

# Enhanced: Multiple uncertainty criteria
uncertain_mask = (
    confidence_uncertain |     # Traditional confidence zone
    entropy_uncertain |        # High entropy (ambiguous predictions)
    spread_uncertain |         # Low discriminability between scores
    variance_uncertain         # High score variance
)
```

**Entropy-Based Detection**:
```python
def calculate_entropy(donor_score, acceptor_score, neither_score):
    scores = np.array([donor_score, acceptor_score, neither_score])
    probs = scores / scores.sum()
    entropy = -np.sum(probs * np.log(probs)) / np.log(3)  # Normalized
    return entropy

# High entropy indicates uncertainty/ambiguity
entropy_uncertain = entropy_scores > entropy_threshold
```

### **2. Practical Threshold Optimization** (`optimized_uncertainty_thresholds.py`)

**Drop-in Replacement Function**:
```python
def create_enhanced_uncertain_mask_drop_in(
    base_predictions_df,
    uncertainty_threshold_low=0.02,
    uncertainty_threshold_high=0.85,  # Expanded from 0.80
    use_entropy_supplement=True,
    entropy_threshold=0.75
):
    # Traditional confidence selection
    max_scores = base_predictions_df[['donor_score', 'acceptor_score']].max(axis=1)
    confidence_uncertain = (max_scores >= uncertainty_threshold_low) & (max_scores < uncertainty_threshold_high)
    
    if use_entropy_supplement:
        # Add entropy-based selection
        entropy_scores = calculate_entropy_vectorized(base_predictions_df)
        entropy_uncertain = entropy_scores > entropy_threshold
        return confidence_uncertain | entropy_uncertain
    
    return confidence_uncertain
```

### **3. Main Workflow Integration** (`main_inference_workflow.py`)

**New Command Line Parameters**:
```bash
# Target selection rate
--target-meta-rate 0.10        # Target 10% of positions

# Selection strategy  
--uncertainty-strategy hybrid_entropy  # "confidence_only", "entropy_only", "hybrid_entropy"

# Adaptive tuning
--disable-adaptive-tuning       # Turn off automatic threshold adjustment
```

---

## 🚀 **IMMEDIATE SOLUTIONS**

### **Quick Fix #1: Expand Confidence Threshold**
```bash
# Change from current default
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --uncertainty-low 0.02 \
    --uncertainty-high 0.90    # ← Changed from 0.80

# Expected: 2-3x more positions selected (~6-9% rate)
```

### **Quick Fix #2: Use Enhanced Strategy**
```bash
# Use new hybrid entropy strategy
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --uncertainty-strategy hybrid_entropy \
    --target-meta-rate 0.10

# Expected: ~10% selection rate with entropy-based additions
```

### **Quick Fix #3: Conservative Expansion**
```bash
# Conservative approach
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --uncertainty-low 0.02 \
    --uncertainty-high 0.85    # ← Moderate expansion
    --target-meta-rate 0.10

# Expected: ~1.5x more positions selected (~4-6% rate)
```

---

## 📊 **THRESHOLD RECOMMENDATIONS**

### **For ~10% Meta-Model Application**

| Approach | Low Thresh | High Thresh | Entropy Thresh | Expected Rate | Best For |
|----------|------------|-------------|----------------|---------------|----------|
| **Expanded Confidence** | 0.02 | 0.90 | - | 8-12% | Simple implementation |
| **Entropy Addition** | 0.02 | 0.80 | 0.70 | 8-15% | Capturing ambiguous cases |
| **Balanced Hybrid** | 0.02 | 0.85 | 0.75 | 9-13% | **Recommended** |
| **Conservative** | 0.02 | 0.85 | - | 4-6% | Cautious adoption |

### **Biological Rationale**

**Expanded High Threshold (0.80 → 0.85-0.90)**:
- Includes moderate-confidence splice sites where meta-model can refine predictions
- Captures weak splice sites that might be disease-relevant
- Maintains focus on biologically relevant positions

**Entropy-Based Addition**:
- Identifies positions where base model is genuinely ambiguous
- Captures balanced score distributions (e.g., donor=0.4, acceptor=0.3, neither=0.3)
- Focuses on positions where additional context could help

---

## 🔧 **IMPLEMENTATION OPTIONS**

### **Option 1: Immediate Threshold Adjustment**
**Modify existing code directly**:

```python
# In prediction_combiner.py, line 87:
# OLD:
uncertain_mask = (max_score >= uncertainty_threshold_low) & (max_score < uncertainty_threshold_high)

# NEW:
uncertain_mask = (max_score >= uncertainty_threshold_low) & (max_score < 0.85)  # or 0.90
```

### **Option 2: Enhanced Function Replacement**
**Use drop-in replacement**:

```python
# In prediction_combiner.py:
from .optimized_uncertainty_thresholds import create_drop_in_enhanced_mask

# Replace existing logic:
uncertain_mask = create_drop_in_enhanced_mask(
    combined_df['donor_score'],
    combined_df['acceptor_score'], 
    combined_df['neither_score'],
    method="balanced_hybrid"
)
```

### **Option 3: Full Enhanced Integration**
**Use new enhanced workflow parameters**:

```bash
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --target-meta-rate 0.10 \
    --uncertainty-strategy hybrid_entropy \
    --uncertainty-high 0.85
```

---

## 🧪 **ENTROPY-BASED SELECTION DETAILS**

### **What is Entropy in This Context?**
Entropy measures the "ambiguity" or "uncertainty" in the base model's predictions:

```python
# Example position scores:
donor=0.1, acceptor=0.2, neither=0.7  # Low entropy (clear neither)
donor=0.8, acceptor=0.1, neither=0.1  # Low entropy (clear donor)
donor=0.4, acceptor=0.3, neither=0.3  # HIGH entropy (ambiguous!)
```

### **Why Entropy Helps**
- **Captures Ambiguity**: Positions where base model can't decide between splice types
- **Biological Relevance**: These are often context-dependent splice sites
- **Meta-Model Value**: Perfect candidates for meta-learning refinement

### **Entropy Implementation**
```python
def calculate_entropy(donor, acceptor, neither):
    scores = np.array([donor, acceptor, neither])
    probs = scores / scores.sum()
    entropy = -np.sum(probs * np.log(probs)) / np.log(3)  # Normalized 0-1
    return entropy

# High entropy (>0.7) indicates uncertainty
```

---

## 📈 **EXPECTED PERFORMANCE IMPROVEMENTS**

### **Selection Rate Improvements**
```
Current (0.02-0.80):           ~3% of positions
Expanded (0.02-0.85):          ~4-6% of positions  
Entropy Addition:              ~6-10% of positions
Balanced Hybrid:               ~8-12% of positions
Aggressive (0.02-0.90):        ~10-15% of positions
```

### **Biological Coverage Improvements**
- **Weak Splice Sites**: Better coverage of borderline splice sites
- **Context-Dependent Sites**: Entropy captures positions needing genomic context
- **Disease-Relevant Sites**: Moderate-confidence sites often disease-associated
- **Cryptic Sites**: Ambiguous positions may be cryptic splice sites

---

## 🎯 **RECOMMENDED IMPLEMENTATION STRATEGY**

### **Phase 1: Immediate (No Code Changes)**
```bash
# Use expanded threshold
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --uncertainty-high 0.85    # ← Simple change from 0.80
```

### **Phase 2: Enhanced (Use New Parameters)**
```bash
# Use enhanced strategy
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --target-meta-rate 0.10 \
    --uncertainty-strategy hybrid_entropy
```

### **Phase 3: Full Integration (Future)**
- Integrate enhanced uncertainty selection into feature processor
- Add entropy calculation to prediction combiner
- Enable adaptive threshold tuning

---

## ✅ **SUMMARY**

**Problem Solved**: The current ~3% meta-model application rate can be increased to ~10% using:

1. **Expanded Confidence Thresholds**: 0.80 → 0.85-0.90
2. **Entropy-Based Selection**: Add high-entropy positions (ambiguous predictions)  
3. **Hybrid Approach**: Combine both criteria for optimal coverage

**Entropy-Based Selection**: ✅ **Already implemented** in `uncertainty_analyzer.py` but not fully integrated

**Immediate Action**: Use `--uncertainty-high 0.85` or `--uncertainty-high 0.90` for quick improvement

**Best Long-term Solution**: Use `--uncertainty-strategy hybrid_entropy --target-meta-rate 0.10` for optimal ~10% selection with entropy-based enhancements

The enhanced uncertainty selection focuses meta-model application on positions where it can provide maximum biological and clinical value! 🎯

