# Troubleshooting – SHAP Analysis Step

This note captures recurring errors observed while computing SHAP feature
importance after external-memory meta-model training, together with the applied
fixes.  All referenced code lives under
`splice_engine/meta_models/training/` and `splice_engine/meta_models/evaluation/`.

**Last Updated:** July 10, 2025

---

## Variant 1 – `ColumnNotFoundError` on sparse k-mer features

**Symptom**
```text
polars.exceptions.ColumnNotFoundError: did not find column 6mer_GGATCN, consider passing `missing_columns='insert'`
```
*Raised inside `classifier_utils.shap_importance()` during the initial Polars
`collect()` call.*

**Why it happens**
MetaSpliceAI constructs a **union** feature set across all chromosome shards.
A particular 6-mer (e.g. `6mer_GGATCN`) may be completely absent in one
shard.  When Polars lazily scans the directory and then executes
`lf.select(feature_names)` it aborts on every file that lacks the requested
column.

**Fix (commit XXXX)**
1. The scan now opts into Polars' native fallback:
   ```python
   pl.scan_parquet("*.parquet", missing_columns="insert")
   ```
   which auto-adds **null** columns for any missing features.
2. Kept the previous `try/except ColumnNotFoundError` *fallback* – if the
   environment’s Polars build predates the `missing_columns` option the code
   falls back to file-by-file loading and zero-fills absent columns.

**Verification**
`demo_train_meta_model_multiclass.py` now runs the SHAP step without raising
`ColumnNotFoundError`, regardless of heterogeneous feature schemas.

---

## Variant 2 – `ValueError: Per-column arrays must each be 1-dimensional`

**Symptom**
```text
ValueError: Per-column arrays must each be 1-dimensional
```
*Thrown from `explainers.shap_feature_importance()` when building the Pandas
DataFrame of importances.*

**Root cause**
Recent versions of **XGBoost (≥2.0)** return SHAP values as a **3-D array**
with shape `(n_samples, n_features, n_classes)` instead of the older list-of-2-D
format. Pandas rejects the 3-D array when we attempt to construct a DataFrame
column-wise.

**Fix (commit YYYY)**
`explainers.shap_feature_importance()` now detects both new and old formats:
```python
if isinstance(shap_values, list):
    # classic multiclass, list length == n_classes
    shap_values_arr = np.stack([np.abs(v).mean(axis=0) for v in shap_values]).mean(axis=0)
else:
    if shap_values.ndim == 3:   # (samples, features, classes)
        shap_values_arr = np.abs(shap_values).mean(axis=0).mean(axis=-1)
    else:
        shap_values_arr = np.abs(shap_values).mean(axis=0)
```
The resulting `shap_values_arr` is always **1-D** (length = number of
features), so `pd.DataFrame({"feature": names, "importance": shap_values_arr})`
works for all versions.

**Impact**
No functional change to the importance scores; only the dimensional reduction
is new.  Downstream plots and CSV outputs remain identical.

---

## Variant 3 – TorchVision Import Failures (July 2025)

**Symptom A - Missing NMS Operator**
```text
RuntimeError: Failed to import transformers.modeling_utils because of the following error:
operator torchvision::nms does not exist
```

**Symptom B - Circular Import Error**
```text
RuntimeError: Failed to import transformers.modeling_utils because of the following error:
partially initialized module 'torchvision' has no attribute 'extension' (most likely due to a circular import)
```

*Both errors occur during `shap.TreeExplainer()` initialization when SHAP attempts to check if the model is a transformers model.*

**Why it happens**
The error occurs through this import chain:
```
shap.TreeExplainer() 
→ shap.utils.transformers.is_transformers_lm()
→ transformers.modeling_utils 
→ transformers.image_utils
→ torchvision.transforms.InterpolationMode
→ torchvision.__init__
→ torchvision._meta_registrations  # Fails here
```

**Root Causes**
1. **PyTorch/TorchVision Version Mismatch**: TorchVision operators not compatible with installed PyTorch
2. **Corrupted Installation**: TorchVision partially installed or corrupted
3. **Mixed Installation Sources**: Conflicting pip/conda installations
4. **CUDA Compatibility**: TorchVision compiled for different CUDA version

**Fix A - Reinstall Compatible PyTorch/TorchVision**
```bash
# Remove all PyTorch components
conda remove pytorch torchvision torchaudio --force
pip uninstall torch torchvision torchaudio -y

# Clean cache
conda clean --all

# Reinstall with compatible versions
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Fix B - Remove TorchVision (If Not Needed)**
```bash
conda remove torchvision --force
pip uninstall torchvision -y
```

**Fix C - Enhanced Module-Level Patching**
Added comprehensive transformers patching at module import time in `shap_incremental.py`:
```python
# Set environment isolation before importing SHAP
os.environ.update({
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "CUDA_VISIBLE_DEVICES": "",
})

# Patch transformers check immediately after SHAP import
import shap
def dummy_is_transformers_lm(model):
    return False
    
from shap.utils import transformers as shap_transformers
shap_transformers.is_transformers_lm = dummy_is_transformers_lm
```

**Diagnostic Tool**
Created comprehensive diagnostic script `debug_shap_issue.py`:
```python
def test_pytorch_torchvision_compatibility():
    """Test PyTorch/TorchVision compatibility including NMS operator"""
    from torchvision.ops import nms
    # Test with dummy data
    
def test_transformers_import():
    """Test transformers import without triggering TorchVision issues"""
    
def test_shap_patches():
    """Verify SHAP patches are correctly applied"""
    
def test_simple_shap_operation():
    """End-to-end SHAP functionality test"""
```

**Verification**
Run the diagnostic script to confirm fix:
```bash
python debug_shap_issue.py
```

Expected output after successful fix:
```
🎉 ALL TESTS PASSED! SHAP should work correctly.
```

**Impact**
TorchVision fixes restore full SHAP analysis capability across different computing environments. The diagnostic approach enables rapid identification and resolution of compatibility issues.

---

**Last updated:** July 10, 2025
