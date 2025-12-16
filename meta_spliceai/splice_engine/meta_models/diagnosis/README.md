# Meta-Model Training Diagnosis Tools

This package provides diagnostic tools for assessing system resources and potential issues before running meta-model training experiments.

## Memory Checker

The `memory_checker` module provides comprehensive memory usage assessment to prevent OOM (Out of Memory) errors during training.

### Command Line Usage

```bash
# Basic assessment
python -m meta_spliceai.splice_engine.meta_models.diagnosis.memory_checker train_pc_1000_3mers/master

# With specific parameters
python -m meta_spliceai.splice_engine.meta_models.diagnosis.memory_checker train_pc_1000_3mers/master --cv-folds 5 --n-estimators 500

# Quiet mode (only show risk assessment)
python -m meta_spliceai.splice_engine.meta_models.diagnosis.memory_checker train_pc_1000_3mers/master --quiet
```

### Python API Usage

```python
from meta_spliceai.splice_engine.meta_models.diagnosis import assess_oom_risk

# Quick assessment
assessment = assess_oom_risk('train_pc_1000_3mers/master', cv_folds=5, n_estimators=500)
print(f"Risk Level: {assessment['risk_level']}")
print(f"Recommendation: {assessment['recommendation']}")

# Detailed assessment
from meta_spliceai.splice_engine.meta_models.diagnosis import MemoryChecker

checker = MemoryChecker('train_pc_1000_3mers/master')
checker.print_assessment(cv_folds=5, n_estimators=500)
```

### Integration into Training Scripts

```python
from meta_spliceai.splice_engine.meta_models.diagnosis.integration import check_memory_before_training

def main(args):
    # Check memory before starting training
    assessment = check_memory_before_training(
        args.dataset,
        cv_folds=args.n_folds,
        n_estimators=args.n_estimators,
        verbose=True
    )
    
    # Auto-adjust row cap if needed
    if assessment.get('risk_level') == 'HIGH' and args.row_cap == 0:
        if 'suggested_row_cap' in assessment:
            print(f"Auto-adjusting row cap to {assessment['suggested_row_cap']}")
            args.row_cap = assessment['suggested_row_cap']
        else:
            print("High OOM risk detected. Consider using --row-cap parameter.")
            return
    
    # Continue with training...
```

## Risk Levels

- **✅ SAFE**: Estimated memory usage is well below available memory. Safe to use `--row-cap 0`.
- **⚠️ MODERATE**: Memory usage is close to available memory. Try `--row-cap 0` but monitor usage.
- **❌ HIGH**: Memory usage exceeds available memory. Use suggested `--row-cap` value or smaller dataset.

## Memory Estimation

The tool estimates memory usage based on:

1. **Dataset Size**: Parquet files are ~3x larger in memory than on disk
2. **XGBoost Training**: ~3x dataset size per model during training
3. **Cross-Validation**: Multiple models in memory simultaneously
4. **Safety Buffer**: 1.5x multiplier for unexpected memory spikes
5. **Estimator Impact**: Higher n_estimators increases memory usage

## Exit Codes

When used as a command-line tool:
- `0`: Safe to proceed
- `1`: High risk (should not proceed without row cap)
- `2`: Moderate risk (proceed with caution)
- `3`: Error occurred

## Examples

### Safe Scenario
```
✅ OOM Risk Assessment: SAFE
Your system has 35.4 GB available, estimated peak usage is 25.9 GB (73.2% of available).
Recommendation: Use --row-cap 0 for full dataset
```

### High Risk Scenario
```
❌ OOM Risk Assessment: HIGH
Your system has 8.0 GB available, estimated peak usage is 25.9 GB (324% of available).
Recommendation: Use --row-cap 150000 or smaller dataset
```

## Dependencies

- `psutil` (optional): For accurate system memory detection
- `pathlib`: For path handling
- Standard library modules

If `psutil` is not available, the tool falls back to `/proc/meminfo` on Linux systems.
