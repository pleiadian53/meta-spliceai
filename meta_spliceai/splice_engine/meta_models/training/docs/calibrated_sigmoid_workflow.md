# Calibrated Sigmoid Ensemble Workflow

This document describes the **calibration option** added to
`run_gene_cv_sigmoid.py` that produces probabilities on the familiar
SpliceAI scale (confident splice sites ≈ 0.9).

## TL;DR

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
  --dataset train_pc_1000/master \
  --out-dir runs/gene_cv_sigmoid_cal \
  --annotations data/ensembl/splice_sites.tsv \
  --calibrate 
```

* `--calibrate` fits **isotonic regression** on the splice-site
  probability  `s = p_d + p_a` using the validation splits across all
  CV folds.
* The saved model (`model_multiclass.pkl`) is a
  `CalibratedSigmoidEnsemble` wrapper.  During inference:
  1. Three binary XGBoost models predict independent class sigmoids.
  2. The splice-site score *s* is passed through the isotonic
     calibrator → `s_cal`.
  3. Donor/acceptor probabilities are rescaled so that their sum equals
     `s_cal`; `p_neither = 1 − s_cal`.
* After calibration, confident sites now reach ~0.8–1.0; a threshold
  of 0.9 is once again intuitive.

## Implementation details

1. **Validation collection**  
   For each fold we store `(s_val, y_val_bin)` where `y_val_bin = 1`
   for donor/acceptor rows, `0` for neither.  All folds are
   concatenated.
2. **IsotonicRegression**  
   Chosen for its flexibility and monotonicity.
   Out-of-bounds values are clipped.
3. **Wrapper**  
   `CalibratedSigmoidEnsemble` inherits from `SigmoidEnsemble` and
   rescales probabilities at prediction time.  It is pickle-safe.
4. **Diagnostics**  
   Existing evaluation scripts detect the wrapper automatically because
   it still exposes `predict_proba(X)`.

## When to use

* Use `--calibrate` if human-interpretability of raw probability values
  (e.g. threshold 0.9) is important, or when you want to align scales
  with the original SpliceAI conventions.
* Skip it if you only care about relative ranking / ROC-AUC; the
  uncalibrated logits are equally fine for model selection and usually
  train a few seconds faster.
