# Position Count Analysis - Usage Guide

## 🚀 Quick Start (2 Simple Options)

### **Option 1: Interactive Mode (Easiest)**
```bash
cd meta_spliceai/splice_engine/meta_models/workflows/analysis
python main_driver.py
# Follow the menu prompts
```

### **Option 2: Direct Command (Fastest)**
```bash
cd meta_spliceai/splice_engine/meta_models/workflows/analysis
python debug_position_counts.py --case-study --asymmetry-analysis
```

## 📋 Common Use Cases

### **Understand Position Count Behavior**
```bash
# Quick explanation of 11,443 vs 5,716 position counts
python debug_position_counts.py --case-study

# Detailed analysis of specific genes
python detailed_analysis.py
```

### **Validate Inference Mode Consistency**
```bash
# Test that all inference modes produce identical position counts
python -c "
import subprocess
gene = 'ENSG00000142748'
model = 'results/gene_cv_pc_1000_3mers_run_4'
dataset = 'train_pc_1000_3mers'

for mode in ['base_only', 'meta_only']:
    print(f'Testing {mode} mode...')
    result = subprocess.run([
        'python', '-m', 
        'meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow',
        '--model', model, '--training-dataset', dataset,
        '--genes', gene, '--inference-mode', mode,
        '--output-dir', f'temp_{mode}', '--verbose'
    ], capture_output=True, text=True)
    
    for line in result.stdout.split('\n'):
        if '📊 Total positions:' in line:
            print(f'  {mode}: {line.strip()}')
            break
"
```

### **Investigate Boundary Effects**
```bash
# Analyze where +1 positions are added
python boundary_effects.py

# Trace evaluation pipeline
python pipeline_tracing.py
```

## 🎯 Key Insights (Summary)

### **Normal Behavior (Not Bugs)**
1. **11,443 → 5,716**: Normal donor/acceptor consolidation
2. **+1 discrepancy**: Boundary enhancement (expected)
3. **0.1-0.3% asymmetry**: Biologically normal
4. **Identical across modes**: All inference modes show same counts

### **When to Investigate**
- Position counts differ across inference modes
- Discrepancies >±3 positions
- Zero position counts (indicates failure)
- Asymmetries >1%

## 📊 Package Contents

```
workflows/analysis/
├── README.md                    # Package overview
├── USAGE_GUIDE.md              # This file
├── main_driver.py              # Interactive interface
├── analyze_position_counts.py  # Command-line driver
├── position_counts.py          # Core analysis framework
├── inference_validation.py     # Cross-mode validation
├── boundary_effects.py         # Boundary investigation
├── pipeline_tracing.py         # Pipeline analysis
├── detailed_analysis.py        # Question answering
├── debug_position_counts.py    # Quick debugging
├── test_position_analysis.py   # Testing suite
└── deep_discrepancy_analysis.py # In-depth investigation
```

## 🔬 Technical Background

The position count analysis package was created to address common confusion about:

1. **Why two different position counts?** (11,443 vs 5,716)
   - Answer: Raw donor+acceptor vs consolidated unique positions

2. **Why donor/acceptor asymmetry?** 
   - Answer: Different splice site recognition mechanisms

3. **Why +1 position discrepancy?**
   - Answer: Boundary enhancement for complete coverage

4. **Why consistent across inference modes?**
   - Answer: Position counts determined in evaluation, not inference

## 💡 Pro Tips

1. **Start with `debug_position_counts.py --case-study`** for quick understanding
2. **Use `main_driver.py`** for guided analysis
3. **Test multiple genes** to see consistent patterns
4. **Validate across inference modes** to confirm system health
5. **Focus on relative patterns** rather than absolute numbers

---

**This package turns position count "mysteries" into well-understood system behavior!** 🎯

