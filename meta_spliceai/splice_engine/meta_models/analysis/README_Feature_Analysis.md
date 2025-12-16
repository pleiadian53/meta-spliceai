# Probability Feature Analysis Tools

**📊 Comprehensive documentation and visualization tools for understanding SpliceAI meta-model features**

## Overview

This collection of tools helps you understand, visualize, and analyze the probability-based features derived from SpliceAI predictions. These features are essential for meta-model training and splice site error correction.

## 🎯 What Are These Features?

The SpliceAI meta-model uses sophisticated **signal processing** and **probability-based features** derived from:

- **Raw scores**: Direct SpliceAI outputs (donor_score, acceptor_score, neither_score)
- **Context scores**: Probabilities at nearby positions (±1, ±2 nucleotides)
- **Derived features**: Mathematical transformations capturing signal characteristics

### Key Feature Categories

| Category | Purpose | Example Features |
|----------|---------|------------------|
| **Signal Processing** | Detect peak characteristics | `peak_height_ratio`, `second_derivative` |
| **Cross-Type Comparison** | Distinguish donor vs acceptor | `type_signal_difference`, `donor_acceptor_peak_ratio` |
| **Context Analysis** | Analyze surrounding patterns | `context_asymmetry`, `signal_strength` |
| **Probability Ratios** | Normalized confidence measures | `splice_probability`, `relative_donor_probability` |

## 🛠️ Available Tools

### 1. **Documentation**
- **`Probability_Feature_Documentation.md`** - Comprehensive reference manual
  - Mathematical formulas for all features
  - Biological interpretations
  - Usage guidelines and thresholds

### 2. **Conceptual Diagrams**
- **`create_feature_diagrams.py`** - Educational visualizations
  - No data required - pure conceptual diagrams
  - Explains signal processing concepts visually

### 3. **Data Analysis Reports**
- **`generate_feature_report.py`** - Statistical analysis of your data
  - Feature distributions and correlations
  - Performance by prediction type
  - Interpretation guidance

### 4. **Interactive Visualizations**
- **`visualize_probability_features.py`** - Comprehensive data visualizations
  - Real examples from your data
  - Feature distributions and correlations
  - Context pattern analysis

### 5. **Example Workflow**
- **`example_feature_analysis.py`** - Demonstration script
  - Shows how to use all tools together
  - Provides interpretation guidance

## 🚀 Quick Start

### Step 1: Create Conceptual Diagrams (No Data Required)

```bash
# Create educational diagrams explaining the concepts
python -m meta_spliceai.splice_engine.meta_models.analysis.create_feature_diagrams --output-dir results/probability_feature_analysis/diagrams/
```

**Output**: Visual diagrams explaining peak detection, curvature analysis, and type discrimination.

### Step 2: Generate Sample Data (If Needed)

```bash
# Generate enhanced positions data with features
python -m meta_spliceai.splice_engine.meta_models.analysis.run_fn_rescue_pipeline --top-genes 5 --output-dir results/fn_analysis/
# OR
python -m meta_spliceai.splice_engine.meta_models.analysis.run_fp_reduction_pipeline --top-genes 5 --output-dir results/fp_analysis/
```

### Step 3: Analyze Your Data

```bash
# Generate comprehensive feature analysis report
python -m meta_spliceai.splice_engine.meta_models.analysis.generate_feature_report \
  --data-file results/fn_analysis/2_enhanced_workflow/positions_enhanced_aggregated.tsv \
  --output-dir results/probability_feature_analysis/reports/

# Create data-driven visualizations
python -m meta_spliceai.splice_engine.meta_models.analysis.visualize_probability_features \
  --data-file results/fn_analysis/2_enhanced_workflow/positions_enhanced_aggregated.tsv \
  --output-dir results/probability_feature_analysis/visualizations/ \
  --sample-size 5000
```

### Step 4: Run Complete Demonstration

```bash
# Automated workflow demonstration
python -m meta_spliceai.splice_engine.meta_models.analysis.example_feature_analysis

# Quick demo (diagrams only)
python -m meta_spliceai.splice_engine.meta_models.analysis.example_feature_analysis --quick
```

## 📊 Understanding Key Features

### 🔍 **Signal Processing Features**

These features apply digital signal processing concepts to identify true splice sites:

#### **Peak Height Ratio**
```python
donor_peak_height_ratio = donor_score / neighbor_mean
```
- **> 2.0**: Sharp, isolated peak → **True splice site**
- **< 1.5**: Broad, weak signal → **False positive**

#### **Second Derivative (Curvature)**
```python
second_derivative = (score_center - score_left) - (score_right - score_center)
```
- **Positive**: Sharp peak (concave up) → **True splice site**
- **Negative**: Broad signal (concave down) → **False positive**

#### **Signal Strength**
```python
signal_strength = score_center - neighbor_mean
```
- **> 0.2**: Strong signal above background
- **< 0.05**: Weak signal, likely noise

### 🔄 **Cross-Type Features**

These features distinguish donor from acceptor splice sites:

#### **Type Signal Difference**
```python
type_signal_difference = donor_signal_strength - acceptor_signal_strength
```
- **> +0.1**: **Donor preferred**
- **< -0.1**: **Acceptor preferred**
- **≈ 0**: Ambiguous type

#### **Peak Ratio**
```python
donor_acceptor_peak_ratio = donor_peak_height_ratio / acceptor_peak_height_ratio
```
- **> 2.0**: Strong donor
- **< 0.5**: Strong acceptor

### 📍 **Context Features**

These features analyze the surrounding probability landscape:

#### **Context Asymmetry**
```python
context_asymmetry = (upstream_scores) - (downstream_scores)
```
- **Positive**: Upstream bias
- **Negative**: Downstream bias

#### **Context Maximum**
```python
context_max = max(context_score_m2, context_score_m1, context_score_p1, context_score_p2)
```
- High values indicate nearby competing sites

## 🎯 Practical Applications

### **False Positive Reduction**
Focus on these features to identify likely false positives:

```python
# High FP risk indicators
- donor_peak_height_ratio < 1.5    # Weak peak
- donor_second_derivative < 0       # Broad signal
- splice_probability < 0.3          # Low confidence
- signal_strength < 0.05            # Weak signal
```

### **False Negative Rescue**
Look for these patterns in missed splice sites:

```python
# Potential FN rescue indicators
- 0.3 < splice_probability < 0.5    # Below threshold but reasonable
- second_derivative > 0             # Good peak shape
- is_local_peak == True             # Local maximum
- |type_signal_difference| > 0.1    # Clear type preference
```

### **Type Classification**
Use these features to distinguish donor from acceptor:

```python
# Strong donor indicators
- type_signal_difference > 0.1
- donor_acceptor_peak_ratio > 2.0
- relative_donor_probability > 0.8

# Strong acceptor indicators  
- type_signal_difference < -0.1
- donor_acceptor_peak_ratio < 0.5
- relative_donor_probability < 0.2
```

## 📁 File Structure

```
meta_spliceai/splice_engine/meta_models/analysis/
├── README_Feature_Analysis.md              # This file
├── Probability_Feature_Documentation.md    # Comprehensive reference  
├── create_feature_diagrams.py             # Conceptual diagrams
├── generate_feature_report.py             # Statistical analysis
├── visualize_probability_features.py      # Data visualizations
├── example_feature_analysis.py            # Demonstration workflow
├── run_fn_rescue_pipeline.py              # Generate FN analysis data
├── run_fp_reduction_pipeline.py           # Generate FP analysis data
└── results/                                # Organized output directory
    └── probability_feature_analysis/      # All feature analysis outputs
        ├── diagrams/                       # Conceptual diagrams (no data needed)
        ├── reports/                        # Statistical analysis reports
        ├── visualizations/                 # Data-driven plots and charts
        └── example/                        # Example workflow outputs
            ├── diagrams/
            ├── reports/
            └── visualizations/
```

## 🔧 Dependencies

Required Python packages:
```bash
pip install pandas polars numpy matplotlib seaborn argparse pathlib
```

## 💡 Tips for Effective Analysis

### **1. Start with Conceptual Understanding**
- Read `Probability_Feature_Documentation.md` first
- Create diagrams to understand the concepts
- Then analyze your specific data

### **2. Use Representative Data**
- Sample 1000-5000 positions for visualization
- Include all prediction types (TP, FP, FN, TN)
- Focus on genes with known splice site issues

### **3. Interpret Results Carefully**
- Compare feature distributions across prediction types
- Look for clear separation between TP/FP or FN/TN
- Consider biological context, not just statistical significance

### **4. Validate Findings**
- Use SHAP analysis to confirm feature importance
- Test on independent datasets
- Validate against known splice site properties

## 🆘 Troubleshooting

### **Common Issues**

1. **"No features found in data"**
   - Ensure you're using enhanced positions data
   - Check that the workflow included `add_derived_features=True`

2. **"Missing context columns"**
   - Verify the enhanced workflow was run completely
   - Check that context scores were computed

3. **"Empty visualizations"**
   - Increase sample size
   - Check data filtering criteria
   - Verify prediction types are present

### **Getting Help**

1. Check the comprehensive documentation
2. Run the example workflow
3. Examine the generated reports for insights
4. Compare with known good examples

## 📚 Additional Resources

- **SHAP Analysis Documentation**: For post-hoc feature importance
- **FN/FP Pipeline Scripts**: For generating analysis data
- **Meta-Model Training Documentation**: For using features in practice

## 🎓 Learning Path

1. **Beginner**: Start with `python -m meta_spliceai.splice_engine.meta_models.analysis.example_feature_analysis --quick`
2. **Intermediate**: Read documentation and create diagrams
3. **Advanced**: Analyze your own data and interpret results
4. **Expert**: Integrate insights into meta-model training

---

**🔬 Remember**: These features represent sophisticated signal processing applied to biological data. Understanding them deeply will help you build better meta-models and interpret their decisions effectively. 