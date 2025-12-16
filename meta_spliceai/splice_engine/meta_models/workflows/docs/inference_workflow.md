# Enhanced Splice-Inference Workflow

File: `meta_spliceai/splice_engine/meta_models/workflows/splice_inference_workflow.py`

## Purpose
Produce a *small, information-rich* feature matrix for a **trained meta-model**
so it can re-score borderline SpliceAI predictions and thus:
* demote false-positives (FPs)
* rescue false-negatives (FNs)
while avoiding new errors.

The workflow supports calibrated models, neighborhood analysis, and diagnostic sampling to provide comprehensive insights into model performance.

## Entry-Point
```python
run_enhanced_splice_inference_workflow(
    # Basic parameters
    covered_pos: Optional[Dict[str, Set[int]]] = None,
    t_low: float = 0.02,
    t_high: float = 0.80,
    target_genes: Optional[List[str]] = None,
    # Workflow control
    do_prepare_sequences: bool = False,
    do_prepare_position_tables: bool = False,
    do_prepare_feature_matrices: bool = False,
    # House-keeping
    cleanup: bool = True,
    cleanup_features: bool = False,  # Whether to remove feature matrices after prediction
    verbosity: int = 1,
    max_analysis_rows: int = 500_000,
    max_positions_per_gene: int = 0,
    # Model and prediction options
    model_path: Optional[Path] = None,
    use_calibration: bool = True,
    # Analysis options
    neigh_sample: int = 0,       # Number of positions for neighborhood analysis
    neigh_window: int = 50,      # Window size for neighborhood analysis
    diag_sample: int = 0,        # Number of positions for diagnostic sampling
    # Directory options
    output_dir: Optional[str | Path] = None,
    feature_dir: Optional[str | Path] = None,
    ...)
```

## Pipeline
1. **Set up directories**
   * Creates output and feature directories with proper organization
   * Generates tracking files for feature origin and model information

2. **Delegate to prediction workflow**
   * Calls `run_enhanced_splice_prediction_workflow` with
     `no_final_aggregate=True` → chunk TSVs only, very low RAM.

3. **Streaming load of `analysis_sequences_*`**
   * Reads *only essential columns* (`gene_id`, `position`, scores).
   * Applies filters & ranking *on the fly* (see below).

4. **Filter & prioritise unseen positions**
   1. *Ambiguous zone*: `t_low ≤ max(donor, acceptor) < t_high`.
   2. *Training-set exclusion*: drop `(gene_id, position)` in `covered_pos`.
   3. *Uncertainty ranking* (per gene)
      * Shannon entropy over donor / acceptor / neither.
      * Tie-breaker: distance to nearest decision threshold.
      * Keep `max_positions_per_gene` rows.
   4. *Global cap*: stop early at `max_analysis_rows` and head() again after
      concatenation.

5. **Write inference TSV**
   `analysis_sequences_inference.tsv` contains only the retained positions.

6. **Build feature matrix**
   Calls `incremental_build_training_dataset(...)` which produces an Arrow
   dataset in the specified feature directory.

7. **Load model for prediction** (if model_path provided)
   * Supports both standard XGBoost models (.json) and calibrated models (.pkl)
   * Uses calibration when available and enabled

8. **Perform neighborhood analysis** (if neigh_sample > 0)
   * Samples positions and analyzes predictions in their genomic neighborhood
   * Generates visualizations of prediction patterns

9. **Perform diagnostic sampling** (if diag_sample > 0)
   * Creates detailed diagnostic reports on sampled positions
   * Saves diagnostic information in CSV format

10. **Optional cleanup**
    * Standard cleanup: remove bulky per-position chunk TSVs
    * Feature cleanup (if cleanup_features=True): remove feature matrix files after prediction

## Memory Safety
* Streaming ensures contextual-sequence columns are never loaded.
* Per-gene buffers hold ≤ `max_positions_per_gene` rows.
* Global row cap enforced twice.

## Selecting the “least-confident” positions
| Step | Rationale |
|------|-----------|
| Ambiguous zone | Focus on border-line calls; confident TN/TPs excluded. |
| Entropy (highest) | Captures overall class uncertainty. |
| Distance to threshold (closest) | Prefers scores *just* below or above decision boundary. |
| Per-gene cap | Guarantees even coverage; prevents any one gene dominating. |

Result: a balanced, memory-friendly test set where the meta-model is most
likely to improve error metrics.

---

### Algorithmic details of ranking

**1. Entropy (overall uncertainty)**

Treat the three SpliceAI class probabilities as a categorical distribution
$p = (p_{\text{donor}},\, p_{\text{acceptor}},\, p_{\text{neither}})$.  The
Shannon entropy in *bits* is

```math
H = -\frac{1}{\log 2} \sum_{c \in \{D, A, N\}} p_c \; \log p_c
```

where each $p_c$ is clipped to a small positive constant (\texttt{eps = 1e-9})
to avoid the $\log 0$ singularity.  Higher $H$ → model is less confident.

**2. Distance to decision threshold ($\Delta$)**

```python
max_score = max(p_donor, p_acceptor)
Δ = min(abs(max_score - t_low), abs(max_score - t_high))
```

This measures how close the highest splice-site probability is to one of the
pre-defined decision boundaries (`t_low`, `t_high`).  A smaller $\Delta$ means
the call lies just inside the ambiguous zone and is therefore more
interesting for re-scoring.

**3. Sorting & selection**

For each gene we sort rows by:

1.   Highest uncertainty → descending entropy (implemented as ascending
     `-H` for efficiency in Polars).
2.   Tie-breaker → ascending $\Delta$ (closer to the boundary first).

After sorting we keep only the first `max_positions_per_gene` rows per gene.
This prevents large genes with many candidate sites from drowning out
smaller genes.

The resulting subset is small, balanced, and contains the positions where the
meta-model is most likely to correct base-model mistakes.

---

## Typical Usage

### Basic Usage
```python
run_enhanced_splice_inference_workflow(
    target_genes=my_genes,
    max_positions_per_gene=10,
    max_analysis_rows=200_000,
    verbosity=2)
```

### With Model Prediction and Neighborhood Analysis
```python
run_enhanced_splice_inference_workflow(
    target_genes=my_genes,
    model_path=Path("models/meta_model.json"),
    use_calibration=True,
    neigh_sample=100,
    neigh_window=50,
    diag_sample=50,
    output_dir="inference_results",
    verbosity=2)
```

### With Custom Feature Directory and Cleanup
```python
run_enhanced_splice_inference_workflow(
    target_genes=my_genes,
    model_path=Path("models/meta_model.pkl"),
    feature_dir=Path("data/feature_matrices"),
    cleanup_features=True,  # Clean up feature files after prediction
    verbosity=2)
```

---

## Relationship to Prediction Workflow
The inference workflow *re-uses* all heavy lifting (SpliceAI scoring) from the
prediction workflow but short-circuits the aggregation step and adds a smart
sampling layer before feature extraction.

---

## Artefact Layout & Paths
By default, outputs from the enhanced splice inference workflow are organized as follows:

### Directory Structure

```
<OUTPUT_DIR>/                   # Main output directory (with timestamp if auto-generated)
  ├── features/                 # Feature matrices (either in output_dir or custom feature_dir)
  │   ├── feature_info.txt      # Metadata about feature creation (timestamp, model, etc.)
  │   ├── master/part-*.parquet # Sharded Arrow dataset with feature matrix
  │   ├── meta/columns.json     # Ordered list of feature columns
  │   ├── meta/schema.parquet   # Arrow schema snapshot for tooling
  │   ├── diagnostics/          # (Optional) Diagnostic sampling results
  │   └── neighborhood/         # (Optional) Neighborhood analysis results
  │
  ├── analysis_sequences_*.tsv.gz    # Raw chunk TSVs from base SpliceAI run
  ├── analysis_sequences_inference.tsv # Filtered positions for rescoring
  ├── splice_positions_enhanced_*.tsv  # Enhanced prediction results
  ├── logs/                     # Timing and diagnostic CSVs (if verbosity >= 2)
  └── model_info.json           # Information about the model used
```

### Custom Feature Directory
When specifying a custom `feature_dir`, the feature matrices are stored separately:

```
<FEATURE_DIR>/
  ├── <model_name>_<timestamp>/   # Model-specific feature directory
  │   ├── feature_info.txt        # Metadata about feature creation
  │   ├── master/part-*.parquet   # Feature matrices
  │   └── meta/...                # Feature metadata
```

### Directory Management
- `output_dir` can be specified explicitly or auto-generated with timestamp
- `feature_dir` can be a custom path for storing feature matrices
- Model name is embedded in feature directory names for traceability
- `feature_info.txt` contains metadata to track the origin of feature matrices

### Optional Directories
- `diagnostics/`: Present only when `diag_sample > 0`
- `neighborhood/`: Present only when `neigh_sample > 0`

These artefacts are designed to be **traceable** and **configurable** for downstream prediction.

To verify row counts:

```bash
# Number of positions selected for re-scoring
wc -l <INF>/analysis_sequences_inference.tsv   # minus 1 for header

# Number of feature-matrix rows (should match)
polars python -c "import polars as pl, glob, sys; print(pl.scan_parquet(sys.argv[1]).select(pl.count()).collect().item())" '<INF>/features/master/*.parquet'
```

