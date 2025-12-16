# Meta-Model Evaluation & Diagnostics Guide

A structured checklist for validating **multiclass splice-site meta-models** (and the simpler binary variants).  Follow each pillar in order—the later ones assume the earlier leakage checks have passed.

---

## 1  Rigorous Leakage Checks  🧯

### 1.1  Group-aware CV
* **Gene-wise** split (`--group-col gene_id`) is mandatory for scientific claims.
* Complement with **chromosome-wise** and random CV to spot brittleness.
* Minimum rows per fold: tune `--min-rows-test` so every fold has ≥1 k rows.

### 1.2  Feature Leakage
* Exclude **coordinates**, IDs, hashes, or any encoding of the label.
* Red-flag columns:
  * Raw SpliceAI probabilities (donor / acceptor / neither)
  * Derived *diff / ratio / entropy* features if they use the **true** label.
* Use `classifier_utils.leakage_probe`:
  * Correlation heat-map of *feature ↔ label*.
  * Quickly trains a linear probe; >95 % F1 ⇒ investigate!

### 1.3  Shuffle-label Sanity
* `shuffle_label_sanity_{binary,multiclass}.py` expects **random-guess** F1 (≈0.5 binary, ≈0.33 macro-F1 multiclass).
* Repeat with 3–5 different random seeds.

---

## 2  Alternative Validation Splits  🔀

| Strategy | What it guards against | How-to |
|----------|------------------------|--------|
| **Chromosome hold-out** | Over-fitting to regional composition biases | `--group-col chrom` and optionally `--holdout-chroms` |
| **Difficult-gene set** | Performance on genes with rare motifs, high alt-splicing | Manually curate gene list → filter test subset |
| **External genome / tech** | Species or platform generalisation | Convert dataset to Parquet → reuse the same CLI |

A genuine model should stay >90 % macro-F1 across these.

---

## 3  Performance Metrics  📊

* **Per-class** precision, recall, PR-AUC, and ROC-AUC (macro + micro).
* **Gene-level** accuracy & kappa via `gene_score_delta_multiclass`.
* **Calibration curves** & Brier score (see `classifier_utils.richer_metrics`).
* Track **confusion matrix drift** between folds and external tests.
* Confusion matrix orientation: **rows = actual class (neither, donor, acceptor); columns = predicted**.  Example from a chromosome-1 fold:

  ```text
               predicted
             neither donor acceptor
   actual 0   12921      0        0
   actual 1       0    654        0
   actual 2       0      2      628
  ```
  Only two acceptor sites were mis-classified as donor; all other predictions were correct.

* **Top-k accuracy**: recall-at-k over all splice sites in a test split.  Given a fold with *k* true splice sites (donor + acceptor), rank positions by `P(donor)+P(acceptor)` and ask how many of those *k* sites appear in the top-*k* rows.  Formally

$$\text{top
\_k
\_accuracy}=\frac{|\text{true splice sites} \cap \text{top-}k|}{k}$$

  • Reported as `top_k_accuracy` in `metrics_fold*.json` and the fold summary CSV.
  • The accompanying `top_k` field records the value of *k* for transparency (e.g. *top_k = 1812* for fold 4 of the 1 000-gene dataset).  This matches the "Top-k" definition in the original SpliceAI paper.

Example excerpt (fold 4, train_pc_1000):

```json
{
  "fold": 4,
  "test_rows": 19740,
  "top_k": 1812,
  "top_k_accuracy": 1.0
}
```

---

## 4  Interpretability  🔍

### 4.1  Global Importance
* `classifier_utils.shap_importance` → SHAP bar plot, shapley density.
* Cross-check with XGBoost gain / cover to catch SHAP quirks.

### 4.2  Local Explanations
* Use `bar_chart_local_feature_importance` – focus on *misclassified* samples.
* Verify that top features make **biological sense** (splice motifs, GC, etc.).

### 4.3  Ablation Study
```
for block in [prob_features, context, kmers]:
    drop columns → retrain → record macro-F1 Δ
```
Large drop (>5 pp) from a single block = dependency / risk of leakage.

---

## 5  Threshold & Calibration Analysis  🎚️

* Is the model just **re-thresholding** SpliceAI?  Compare:
  * Raw SpliceAI → logistic calibration
  * Meta-model predictions
  * Plot reliability curves.
* If curves overlap, the gain is mostly calibration; consider a lighter solution.

---

## 6  Qualitative Error Review  🔬

1. Sample 50 corrected FP/FN sites per class.
2. Visualise donor/acceptor signal tracks (IGV, sashimi, etc.).
3. Tag causes: weak motif, nearby pseudo-site, noisy region, etc.
4. Feedback into **feature engineering** loop.

---

## 7  Recommended Workflow  🛠️

1. `run_loco_cv_multiclass` with `gene_id` groups + `--leakage-probe`.
2. `shuffle_label_sanity_multiclass` to rule out trivial leaks.
3. `run_loco_cv_multiclass_extmem` without row cap ⇒ full dataset stress-test.
4. External hold-out evaluation using `classifier_utils.richer_metrics`.
5. SHAP & ablation for insight; refine features accordingly.
6. Manual review of hardest errors; iterate.

---

## 8  Automation Cheatsheet  ⌨️

| Script | Purpose |
|--------|---------|
| `run_loco_cv_multiclass.py` | Gene / chrom CV in-RAM |
| `run_loco_cv_multiclass_extmem.py` | External-memory CV |
| `leakage_probe.py` | Fast leakage detection |
| `shuffle_label_sanity_multiclass.py` | Random-label baseline |
| `classifier_utils.richer_metrics()` | On-disk model evaluation |
| `gene_score_delta_multiclass.py` | Gene-level deltas |

---

## 9  Interpreting “Perfect” Scores  🧐

Even after all checks, >0.995 macro-F1 can be real **if**:
* Base predictor already encodes near-complete information.
* Meta-model simply calibrates / smooths decision boundaries.
* Dataset class imbalance inflates macro-F1 (verify per-class recall!).

If perfect scores coincide with **low feature diversity**, revisit leakage gates.

---

## 10  Example CLI Recipes  🧾

Below are common training & evaluation commands using the bundled scripts.  Paths assume the project root and training dataset `train_pc_1000/master`.

### 10.1  Gene-wise LOCO-CV (CPU)
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass \
    --dataset train_pc_1000/master \
    --out-dir runs/loco_cv_gene_cpu \
    --tree-method hist --max-bin 256 \
    --n-estimators 1200 \
    --group-col gene_id \
    --min-rows-test 2000 \
    --diag-sample 40000 \
    --leakage-probe
```

### 10.2  Chromosome LOCO-CV (GPU)
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass \
    --dataset train_pc_1000/master \
    --out-dir runs/loco_cv_chr_gpu \
    --tree-method hist --device cuda \
    --max-bin 256 --n-estimators 1200 \
    --diag-sample 40000
```

### 10.3  Convert Parquet → LibSVM for external-memory
```bash
conda run -n surveyor python - <<'PY'
from meta_spliceai.splice_engine.meta_models.training import external_memory_utils as emu
emu.convert_dataset_to_libsvm(
    dataset_dir='train_pc_1000/master',
    out_path='data/train_pc_1000.libsvm',
    chunk_rows=250_000,
)
PY
```

### 10.4  External-memory LOCO-CV (CPU)
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_extmem \
    --dataset train_pc_1000/master \
    --libsvm data/train_pc_1000.libsvm \
    --out-dir runs/loco_cv_extmem \
    --tree-method hist --max-bin 256 \
    --n-estimators 1200 \
    --diag-sample 40000
```

### 10.5  Re-run diagnostics for an existing run
```bash
conda run -n surveyor python - <<'PY'
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as cu
cu.richer_metrics('train_pc_1000/master', 'runs/loco_cv_gene_cpu', sample=0)  # 0 = full dataset
PY
```

**Common flags now available**

* `--heldout-chroms` – comma-separated list of chromosomes forming a *single* test set (e.g. `1,3,5,7,9`). If given, the script trains on all remaining chromosomes and reports metrics for this SpliceAI-style split.

* `--diag-sample` – subsample rows for diagnostics (0 = full set).
* `--device` – explicit device selection (`cuda`, `cpu`, or `auto`).
* `--min-rows-test` – bucket small groups so each test fold meets the minimum row threshold.

### 10.6  Ablation study across feature blocks
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
    --dataset train_pc_1000/master \
    --out-dir runs/ablate_full \
    --modes full,no_spliceai,no_probs,no_kmer,only_kmer \
    --row-cap 200000 \
    --n-estimators 400 --tree-method hist --device cpu
```
This will create a sub-directory per *mode* and write an `ablation_summary.tsv` aggregating the macro-F1 drop for each feature block removal.

### 10.7  SpliceAI-style fixed chromosome evaluation
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass \
    --dataset train_pc_1000/master \
    --out-dir runs/chrom_holdout_13579 \
    --heldout-chroms 1,3,5,7,9 \
    --tree-method hist --max-bin 256 \
    --n-estimators 1200
```
For an external-memory run:
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_extmem \
    --dataset train_pc_1000/master \
    --libsvm data/train_pc_1000.libsvm \
    --out-dir runs/chrom_holdout_13579_ext \
    --heldout-chroms 1,3,5,7,9 \
    --tree-method hist --max-bin 256 \
    --n-estimators 1200
```
These commands replicate the evaluation protocol used in the original SpliceAI paper: chromosomes 1, 3, 5, 7, 9 are excluded from training and used solely for testing.

---

---

## 11  Neighbour-Window Probability Diagnostic  🔎

A *qualitative* check that the meta-model produces sharp probability peaks at true splice sites and low scores elsewhere.

### 11.1  Stand-alone usage

```bash
python -m meta_spliceai.splice_engine.meta_models.analysis.neighbour_window_diagnostics \
    --run-dir runs/gene_cv_pc1000/fold0 \   # directory with trained model
    --annotations data/ensembl/splice_sites.tsv \
    --genes ENSG00000123456,ENSG00000987654 \ # or --n-sample 5
    --window 10                                # ±10 nt window
```

`--dataset` is optional; if omitted the script automatically resolves the default
`data/ensembl/spliceai_eval/meta_models/full_splice_positions_enhanced.parquet` file via `MetaModelDataHandler`.

The tool prints a compact table and writes `neighbour_window_meta_scores.tsv` for plotting:

```
[diag] ENSG… donor @ pos 12345 (✓)
rel_pos :  -2   -1    0   +1   +2
 donor  : 0.01 0.02 0.97 0.05 0.01
 accept : 0.00 0.01 0.02 0.03 0.84
 neither: 0.98 0.96 0.01 0.92 0.15
```

### 11.2  Integrated into CV pipelines

All CV drivers (`run_gene_cv_multiclass*.py`, `run_loco_cv_multiclass*.py`) now accept:

* `--neigh-sample N` – number of genes to diagnose (0 = skip, default)
* `--neigh-window W` – window size for the diagnostic (default 10)

When `N > 0` the driver automatically calls the diagnostic after standard metrics:

```bash
python run_gene_cv_multiclass.py \
  --dataset train_pc_1000/master \
  --out-dir runs/gene_cv_pc1000 \
  --annotations data/ensembl/splice_sites.tsv \
  --neigh-sample 3 --neigh-window 12
```

The TSV is saved alongside other artefacts inside the fold directory.

### 11.3  What to look for

* **True splice-site centre (TP/FN rows)** – `meta_donor` or `meta_acceptor` should show a *sharp maximum* at `rel_pos 0`, decaying rapidly in the flanks.

  ```text
  rel_pos  -2  -1   0  +1  +2
  donor    0.01 0.03 0.98 0.04 0.01   ✓ peak at centre
  ```

* **True negative centre (TN rows)** – centre remains low; if a real site is nearby you will see a peak offset from 0.

  ```text
  rel_pos  -2  -1   0  +1  +2  +3
  donor    0.02 0.04 0.05 0.06 0.10 0.93   ← peak +3 nt away
  neither  0.90 0.85 0.80 0.75 0.60 0.05
  ```

These patterns confirm that the meta-model (i) amplifies true sites while (ii) not hallucinating peaks at negative positions.

---

## 12  Run-Directory Artefacts & Reporting Helpers  📁

After running the example command:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_multiclass \
  --dataset train_pc_1000/master \
  --out-dir runs/gene_cv_pc1000 \
  --annotations data/ensembl/splice_sites.tsv \
  --tree-method hist --max-bin 256 --diag-sample 8000 \
  --neigh-sample 3 --neigh-window 12 \
  --leakage-probe
```

you should find roughly the following files inside `runs/gene_cv_pc1000/`:

| File | Meaning / how to read |
|------|-----------------------|
| `gene_cv_metrics.csv` | One row per CV fold with accuracy, macro-F1, splice metrics. |
| `metrics_fold<i>.json` | Same metrics as above but JSON per fold. |
| `metrics_aggregate.json` | Mean across folds – headline score to quote. |
| `feature_manifest.csv` | Names of the features used at training time (audit/check reproducibility). |
| `model_multiclass.pkl` (+ `model_multiclass.json` for ext-mem) | Trained XGBoost model; load with `pickle` or `xgboost.Booster.load_model`. |
| `compare_base_meta.json` | TP/FP/FN & accuracy comparison between raw SpliceAI and the meta-model. |
| `probability_diagnostics.png` | Histogram & calibration curves for donor/acceptor probabilities. |
| `threshold_suggestion.txt` | Two-line TSV with `best_threshold` and expected F1 – handy for deployment cut-off. |
| `gene_deltas.csv` | Per-gene changes in top-1 hit rate, macro-F1, mean probabilities (raw→meta). |
| `full_splice_performance_meta.tsv` | Donor/acceptor site-level precision/recall/F1 produced by the meta-model. |
| `shap_importance.csv` | Global feature importance (SHAP or gain fallback). |
| `neighbour_window_meta_scores.tsv` | Long table of ±W window probabilities for each inspected splice site. |
| `neigh_<gene>_<pos>.png` | Mini line-plots of donor / acceptor / neither curves (when `--neigh-sample` & `--plot`). |

Legacy artefacts (e.g. `pr_*csv`, `roc_curve.png`) are no longer generated – they are superseded by the probability diagnostics above.

### 12.1  Site-level evaluation (donor / acceptor)

The TSV above is generated automatically by all four CV drivers via a call to
`classifier_utils.meta_splice_performance`, which wraps the standalone
`eval_meta_splice.py` tool.  The evaluation:

1. Converts meta-model probabilities to *peak* calls (local maxima ≥ threshold).  
2. Matches peaks to truth within ±`consensus_window` nt.  
3. Outputs per-gene TP/FP/FN and precision / recall / F1.

Run it by hand if needed:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.eval_meta_splice \
  --dataset train_pc_1000/master \
  --run-dir runs/gene_cv_pc1000 \
  --annotations data/ensembl/splice_sites.tsv \
  --threshold 0.90 --consensus-window 2 --sample 200000
```

### 12.2  Quick summaries in the terminal

Two tiny helpers turn those artefacts into a human-readable report:

| Script | Typical run | What it prints |
|--------|-------------|---------------|
| `gene_report.py` | `python -m meta_spliceai.splice_engine.meta_models.training.gene_report --run-dir runs/gene_cv_pc1000` | Fold table, aggregate metrics, hardest genes, suggested threshold, top SHAP features, neighbour-window accuracy. |
| `loco_report.py` | `python -m meta_spliceai.splice_engine.meta_models.training.loco_report --run-dir runs/loco_cv_pc1000` | Same, but expects `loco_metrics.csv` and highlights the weakest chromosome. |

Both scripts are **read-only**; feel free to run them repeatedly as the directory evolves.

---

**Last updated:** 2025-06-22
