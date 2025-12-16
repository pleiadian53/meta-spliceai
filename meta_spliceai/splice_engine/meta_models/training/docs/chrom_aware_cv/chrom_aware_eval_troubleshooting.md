# Chromosome-Aware Evaluation – Troubleshooting

This page collects common pitfalls encountered when running the **leave-one-chromosome-out cross-validation (LOCO-CV)** and other chromosome-aware evaluation workflows for the 3-class meta-model.

---

## Variant 1 – `xgboost.core.XGBoostError` when using `gpu_hist`

**Symptom**
```text
xgboost.core.XGBoostError: Exception in gpu_hist: ... Check failed: ctx_->Ordinal() >= 0 (-1 vs. 0) : Must have at least one device
```
Often preceded by warnings such as:
```text
WARNING: The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` parameter to CUDA instead.
WARNING: No visible GPU is found, setting device to CPU.
WARNING: XGBoost is not compiled with CUDA support.
```

**Root cause**
`tree_method="gpu_hist"` requires XGBoost to be compiled with CUDA *and* a visible GPU.  In containers or headless CI environments, there may be **no CUDA-capable GPU** or the XGBoost build is CPU-only, causing the internal check to fail.

**Fix (commit YYYY)**
1. **CPU fallback** – simply switch to CPU histogram:
   ```bash
   --tree-method hist   # drop the "gpu_" prefix
   ```
2. **Modern GPU syntax (if you *do* have CUDA)** – keep histogram but add `device=cuda`:
   ```bash
   --tree-method hist --xgb-param device=cuda
   ```
   or programmatically:
   ```python
   XGBClassifier(tree_method="hist", device="cuda", ...)
   ```
3. **Re-install XGBoost with GPU support** if the host has CUDA but the package was compiled without it (`conda install -c nvidia -c conda-forge xgboost`).

**Impact**
Switching to CPU (`hist`) is usually fast enough for ≤100 k rows datasets and always works.  GPU provides marginal benefit only for millions of rows / deep trees.

---

## Variant 2 – *Mismatched Python vs. native* XGBoost versions

**Symptom**
```text
ValueError: Mismatched version between the Python package and the native shared object.  Python package version: 3.0.2. Shared object version: 2.1.1.
```

**Root cause**  
Two separate installs (e.g. *conda* **and** *pip*) leave conflicting files in `site-packages/xgboost/`.  The Python wheel is updated (3.x) but the `libxgboost.so` binary is still the old 2.x build, so the ABI check fails at import time.

**Fix (commit YYYY)**
1. Remove **all** existing copies:
   ```bash
   pip uninstall -y xgboost
   conda remove -y xgboost
   ```
2. Re-install a *single*, GPU-enabled build, e.g.:
   ```bash
   conda install -c nvidia -c conda-forge xgboost==2.1.1
   # or latest stable GPU build available
   ```
   Verify with:
   ```bash
   python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
   ```
   and ensure only **one** location is shown.

**Impact**  
After version alignment, XGBoost loads cleanly and GPU training (`device=cuda`) works.

---

## Variant 3 – `ImportError: Numba needs NumPy 2.0 or less`

**Symptom**
```text
ImportError: Numba needs NumPy 2.0 or less. Got NumPy 2.2.
```
(Typically raised when *SHAP* imports `numba` which then checks NumPy’s major version.)

**Root cause**  
`numba 0.60` only supports **NumPy ≤ 2.0**.  Installing or upgrading other
packages (e.g. XGBoost wheel) may silently pull in *NumPy 2.2+*, breaking
Numba and any downstream library (SHAP, RAPIDS, …) that depends on it.

**Fix (commit YYYY)**
Downgrade NumPy to a compatible release (1.26.x or 2.0.x):
```bash
conda install -n surveyor -y -c conda-forge "numpy<2.1"
# or pick an explicit version, e.g.
# conda install -n surveyor -y -c conda-forge numpy==2.0.2
```
Conda will resolve a consistent set (NumPy ≤ 2.0, numba 0.60, shap 0.46, …).

**Impact**  
Restores compatibility; LOCO-CV and SHAP pipelines import fine.  There is **no
runtime penalty**—NumPy 2.0 and 1.26 have very similar CPU performance.

---

*Last updated: 2025-06-21*
