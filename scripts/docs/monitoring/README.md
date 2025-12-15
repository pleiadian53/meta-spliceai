# Monitoring Tools Documentation

This directory contains documentation for training monitoring and debugging tools.

## Available Documentation

### Training Monitoring
- **[Training Monitor Guide](TRAINING_MONITOR_GUIDE.md)** - Universal monitoring for all training modes
  - Single-instance training monitoring
  - Multi-instance ensemble training monitoring  
  - Batch ensemble training monitoring
  - Error detection and progress tracking
  - Output validation and completeness checking

## Related Tools

### Monitoring Scripts
- **`scripts/monitoring/monitor_training_universal.py`** - Universal training monitor
  - Auto-detects training modes
  - Comprehensive progress tracking
  - Resource usage monitoring
  - Output validation

## Quick Start

### Monitor Active Training
```bash
# Auto-detect and monitor active runs
python scripts/monitoring/monitor_training_universal.py --auto-detect

# Monitor specific run
python scripts/monitoring/monitor_training_universal.py --run-name <run_name>

# Continuous monitoring
python scripts/monitoring/monitor_training_universal.py --run-name <run_name> --watch
```

### List All Runs
```bash
python scripts/monitoring/monitor_training_universal.py --list-runs
```

## Integration with Training Pipeline

The monitoring tools are designed to work seamlessly with:
- **Single Model Training**: `run_gene_cv_sigmoid.py` (small/medium datasets)
- **Multi-Instance Ensemble**: Large datasets with `--train-all-genes` 
- **Batch Ensemble Training**: Alternative large dataset approach

## Features

✅ **Universal Compatibility**: Works with all training modes  
✅ **Real-time Monitoring**: Live progress tracking  
✅ **Error Detection**: Early issue identification  
✅ **Resource Tracking**: Memory and CPU monitoring  
✅ **Output Validation**: Complete file structure verification  
✅ **SHAP Analysis Tracking**: Enhanced vs standard SHAP detection  
