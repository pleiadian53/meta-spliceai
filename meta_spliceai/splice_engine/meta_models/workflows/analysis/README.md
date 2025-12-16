# Position Count Analysis Package

This package provides comprehensive analysis tools for understanding position count behavior in SpliceAI inference workflows.

## 🎯 Quick Start

### **Option 1: Interactive Mode (Recommended)**
```bash
cd meta_spliceai/splice_engine/meta_models/workflows/analysis
python run_analysis.py
# Follow the interactive menu
```

### **Option 2: Direct Analysis**
```bash
# Quick analysis of specific genes
python analyze_position_counts.py --genes ENSG00000142748 ENSG00000000003

# Comprehensive analysis with all tools
python analyze_position_counts.py --genes ENSG00000142748 --comprehensive

# Cross-mode validation
python analyze_position_counts.py --genes ENSG00000142748 --validate-modes \
    --model results/gene_cv_pc_1000_3mers_run_4 \
    --training-dataset train_pc_1000_3mers
```

## 📁 Package Structure

### **Driver Scripts (User Interface)**
- **`run_analysis.py`** - Main entry point with interactive mode
- **`analyze_position_counts.py`** - Command-line analysis driver

### **Core Analysis Modules**
- **`position_counts.py`** - Core position count analysis framework
- **`inference_validation.py`** - Cross-mode consistency validation
- **`boundary_effects.py`** - Boundary position investigation
- **`pipeline_tracing.py`** - Evaluation pipeline analysis
- **`detailed_analysis.py`** - Comprehensive question answering

### **Utility Modules**
- **`debug_position_counts.py`** - Focused debugging explanations
- **`test_position_analysis.py`** - Testing and validation suite
- **`deep_discrepancy_analysis.py`** - In-depth discrepancy investigation

## 🔍 What This Package Analyzes

### **Position Count Discrepancies**
- **11,443 vs 5,716**: Raw donor+acceptor vs final consolidated positions
- **+1 discrepancies**: Boundary enhancement positions
- **Donor/acceptor asymmetries**: Small (0.1-0.3%) expected asymmetries

### **Inference Mode Consistency**
- **Cross-mode validation**: Ensures all modes produce identical position counts
- **Performance comparison**: Processing time and resource usage across modes
- **Quality assurance**: Validates expected behavior patterns

### **Biological and Technical Factors**
- **Splice site biology**: Why donor and acceptor sites behave differently
- **Sequence processing**: Block processing and boundary effects
- **Coordinate systems**: Position calculation and transformation effects

## 📊 Key Findings Documented

### **Normal Behavior (Not Bugs)**
1. **11,443 → 5,716 consolidation**: Expected donor/acceptor merging
2. **+1 position discrepancy**: Boundary enhancement at 3' end
3. **0.1-0.3% donor/acceptor asymmetry**: Biologically expected
4. **Identical counts across modes**: Evaluation-dependent, not inference-dependent

### **When to Investigate**
- Position count differences >±3 positions
- Inconsistent counts across inference modes
- Zero position counts (indicates failure)
- Extremely large asymmetries (>1%)

## 🧪 Example Usage

### **Quick Gene Analysis**
```python
from meta_spliceai.splice_engine.meta_models.workflows.analysis import analyze_position_counts

# Analyze specific genes
results = analyze_position_counts(['ENSG00000142748', 'ENSG00000000003'])
print(f"Analysis complete for {len(results)} genes")
```

### **Cross-Mode Validation**
```python
from meta_spliceai.splice_engine.meta_models.workflows.analysis import validate_inference_consistency

# Validate consistency across inference modes
validation = validate_inference_consistency(
    gene_ids=['ENSG00000142748'],
    model_path='results/gene_cv_pc_1000_3mers_run_4',
    training_dataset='train_pc_1000_3mers'
)

if validation['validation_passed']:
    print("✅ All modes show consistent behavior")
else:
    print("❌ Found inconsistencies requiring investigation")
```

## 🔬 Technical Details

### **Position Count Pipeline**
```
1. SpliceAI Prediction
   ↓ Generates exactly gene_length positions
2. Donor/Acceptor Processing  
   ↓ Separate predictions for each splice type
3. Evaluation Pipeline
   ↓ Adds boundary positions for completeness
4. Final Consolidation
   ↓ One prediction per genomic position
```

### **Expected Patterns**
- **Raw predictions**: ~2x gene length (donor + acceptor)
- **Final positions**: ~gene length + 1 (consolidated + boundary)
- **Cross-mode consistency**: Identical position counts
- **Small asymmetries**: Normal due to biological differences

## 📚 Related Documentation

- [`FINAL_POSITION_ANALYSIS_REPORT.md`](../inference/FINAL_POSITION_ANALYSIS_REPORT.md) - Complete technical analysis
- [`INFERENCE_WORKFLOW_TROUBLESHOOTING.md`](../inference/docs/INFERENCE_WORKFLOW_TROUBLESHOOTING.md) - Troubleshooting guide
- [`META_ONLY_MODE_LESSONS_LEARNED.md`](../inference/docs/META_ONLY_MODE_LESSONS_LEARNED.md) - Meta-only mode insights

---

**This package eliminates confusion about normal position count behavior and provides comprehensive tools for investigating actual issues when they occur.**

