# Uncertainty Threshold Analysis: High-Uncertainty, Low-Confidence Position Identification

**Date**: 2025-08-20  
**Analysis**: Hybrid Inference Mode Uncertainty Rules  
**Source**: Main Inference Workflow (`main_inference_workflow.py`)

---

## 🎯 **Core Question Answered**

**How does the inference workflow identify high-uncertainty, low-confidence positions for meta-model application in hybrid mode?**

The workflow uses a **dual-threshold uncertainty detection system** based on the maximum splice site prediction score from the base model (SpliceAI).

---

## 🧮 **Uncertainty Detection Algorithm**

### **Step 1: Calculate Maximum Base Model Score**

```python
# For each position, compute the maximum score across splice site types
max_score = combined_df[['donor_score', 'acceptor_score']].max(axis=1)
```

### **Step 2: Apply Dual-Threshold Logic**

```python
# Identify uncertain positions using dual thresholds
uncertain_mask = (
    (max_score >= uncertainty_threshold_low) & 
    (max_score < uncertainty_threshold_high)
)
```

### **Default Threshold Values**

```python
uncertainty_threshold_low: float = 0.02   # Below this: confident non-splice
uncertainty_threshold_high: float = 0.80  # Above this: confident splice
```

---

## 📊 **Position Classification System**

The dual-threshold system creates **three confidence categories**:

| Max Score Range | Classification | Meta-Model Applied? | Rationale |
|-----------------|----------------|-------------------|-----------|
| **< 0.02** | **Confident Non-Splice** | ❌ No | Base model is confident this is NOT a splice site |
| **0.02 - 0.80** | **Uncertain** | ✅ **YES** | Base model is uncertain - meta-model can help |
| **≥ 0.80** | **Confident Splice** | ❌ No | Base model is confident this IS a splice site |

### **Key Insight: "Uncertainty Zone"**

The meta-model is applied **only to the middle uncertainty zone** where the base model lacks confidence. This is where meta-learning can provide the most value.

---

## 🔍 **Implementation Details**

### **Hybrid Mode Logic** (Default)

```python
if inference_mode == "hybrid":
    # Initialize meta columns with base values first
    combined_df['donor_meta'] = combined_df['donor_score']
    combined_df['acceptor_meta'] = combined_df['acceptor_score'] 
    combined_df['neither_meta'] = combined_df['neither_score']
    
    # Identify uncertain positions based on confidence thresholds
    max_score = combined_df[['donor_score', 'acceptor_score']].max(axis=1)
    uncertain_mask = (
        (max_score >= uncertainty_threshold_low) & 
        (max_score < uncertainty_threshold_high)
    )
```

### **Position Selection for Meta-Model**

```python
# In feature_processor.py
if config.inference_mode == "hybrid":
    # Identify uncertain positions based on thresholds
    uncertain_mask = (
        (max_scores >= config.uncertainty_threshold_low) & 
        (max_scores < config.uncertainty_threshold_high)
    )

uncertain_positions = complete_base_pd[uncertain_mask].copy()

print(f"🎯 Identified {len(uncertain_positions)} uncertain positions for meta-model inference")
print(f"📊 Uncertainty range: [{config.uncertainty_threshold_low:.3f}, {config.uncertainty_threshold_high:.3f})")
```

---

## 🎚️ **Threshold Rationale**

### **Lower Threshold (0.02): Confident Non-Splice**
- **Purpose**: Exclude positions where base model is confident there's no splice site
- **Rationale**: Positions with very low scores (< 0.02) are clearly non-splice sites
- **Meta-Model Value**: Minimal - base model is already confident
- **Efficiency**: Saves computational resources on obvious negatives

### **Upper Threshold (0.80): Confident Splice** 
- **Purpose**: Exclude positions where base model is confident there is a splice site
- **Rationale**: Positions with high scores (≥ 0.80) are clearly splice sites
- **Meta-Model Value**: Minimal - base model is already confident
- **Efficiency**: Saves computational resources on obvious positives

### **Uncertainty Zone (0.02 - 0.80): Where Meta-Learning Shines**
- **Purpose**: Focus meta-model on positions where base model is uncertain
- **Rationale**: This is where meta-learning can provide the most value
- **Meta-Model Value**: **Maximum** - base model needs help with classification
- **Examples**: 
  - Weak splice sites (score ~0.1-0.3)
  - Cryptic splice sites (score ~0.2-0.5)
  - Context-dependent sites (score ~0.3-0.7)

---

## 📈 **Performance Impact**

### **Selective Efficiency**

Based on the documentation, this selective approach typically processes:

```
📊 Complete coverage: 2,151 positions
🤖 Meta-model recalibrated: 65 (3.0%)
🔄 Base model reused: 2,086 (97.0%)
```

**Key Benefits:**
- **97% efficiency**: Only 3% of positions need meta-model processing
- **Complete coverage**: All positions get predictions
- **Targeted improvement**: Meta-model focuses where it can help most

### **Computational Savings**

```python
# Without selective approach: Process ALL positions through meta-model
# With selective approach: Process only ~3% through meta-model
# Performance improvement: ~33x faster while maintaining accuracy
```

---

## 🔬 **Comparison Across Inference Modes**

### **Base Only Mode**
```python
if inference_mode == "base_only":
    uncertain_mask = pd.Series([False] * len(combined_df))  # No positions marked as uncertain
```
- **Meta-model applied**: Never
- **Uncertain positions**: 0

### **Hybrid Mode (Default)**
```python
uncertain_mask = (max_score >= uncertainty_threshold_low) & (max_score < uncertainty_threshold_high)
```
- **Meta-model applied**: Only to uncertainty zone (0.02-0.80)
- **Uncertain positions**: ~3% typically

### **Meta Only Mode**
```python
if inference_mode == "meta_only":
    uncertain_mask = pd.Series([True] * len(combined_df))  # All positions marked as uncertain
```
- **Meta-model applied**: To all positions
- **Uncertain positions**: 100%

---

## 🧪 **Validation and Verification**

### **Threshold Validation**

```python
# Validate thresholds during config initialization
if not (0 <= self.uncertainty_threshold_low <= 1):
    raise ValueError("uncertainty_threshold_low must be between 0 and 1")
if not (0 <= self.uncertainty_threshold_high <= 1):
    raise ValueError("uncertainty_threshold_high must be between 0 and 1")
if self.uncertainty_threshold_low >= self.uncertainty_threshold_high:
    raise ValueError("uncertainty_threshold_low must be less than uncertainty_threshold_high")
```

### **Position Count Verification**

```python
# Verify expected uncertain position counts
if config.inference_mode == "hybrid":
    expected_uncertain_mask = (
        (max_scores >= config.uncertainty_threshold_low) & 
        (max_scores < config.uncertainty_threshold_high)
    )
    expected_uncertain_count = np.sum(expected_uncertain_mask)

if len(uncertain_positions) != expected_uncertain_count:
    print(f"❌ ERROR: Uncertain position count mismatch")
    print(f"   Expected: {expected_uncertain_count} (mode: {config.inference_mode})")
    print(f"   Actual: {len(uncertain_positions)}")
```

---

## 🎯 **Key Design Principles**

### **1. Confidence-Based Selection**
- Focus meta-model where base model lacks confidence
- Avoid wasting computation on confident predictions

### **2. Biological Relevance**
- 0.02-0.80 range captures biologically interesting "gray zone"
- Includes weak splice sites, cryptic sites, context-dependent sites

### **3. Computational Efficiency**
- Process only ~3% of positions through expensive meta-model
- Maintain complete coverage with 97% base model reuse

### **4. Tunable Parameters**
- Thresholds can be adjusted based on specific use cases
- Lower threshold → more conservative (fewer meta-model applications)
- Higher threshold → more aggressive (more meta-model applications)

---

## 🔧 **Usage Examples**

### **Default Hybrid Mode**
```bash
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --inference-mode hybrid \
    --uncertainty-low 0.02 \
    --uncertainty-high 0.80
```

### **Custom Uncertainty Thresholds**
```bash
# More conservative (fewer meta-model applications)
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --inference-mode hybrid \
    --uncertainty-low 0.05 \
    --uncertainty-high 0.70

# More aggressive (more meta-model applications)
python main_inference_workflow.py \
    --model model.pkl \
    --genes ENSG00000123456 \
    --inference-mode hybrid \
    --uncertainty-low 0.01 \
    --uncertainty-high 0.90
```

---

## 🎉 **Summary**

The inference workflow identifies high-uncertainty, low-confidence positions using a **dual-threshold system** on the maximum base model splice site score:

1. **Positions with max score < 0.02**: Confident non-splice sites (skip meta-model)
2. **Positions with max score 0.02-0.80**: **Uncertain positions** (apply meta-model) 
3. **Positions with max score ≥ 0.80**: Confident splice sites (skip meta-model)

This elegant approach achieves:
- ✅ **Selective efficiency**: Only ~3% of positions processed through meta-model
- ✅ **Complete coverage**: All positions receive predictions  
- ✅ **Targeted improvement**: Meta-model focuses where it provides most value
- ✅ **Biological relevance**: Captures the "gray zone" where splice site classification is most challenging

The 0.02-0.80 uncertainty zone represents the **sweet spot** where meta-learning can provide maximum benefit over the base SpliceAI model.

