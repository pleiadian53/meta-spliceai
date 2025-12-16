# Enhanced Splice-Prediction Workflow

File: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`

## Purpose
Run the base **SpliceAI** model over a (sub-)genome and produce the per-nucleotide
artifacts that later feed the meta-model training & inference pipelines.

## Entry-Point
```python
run_enhanced_splice_prediction_workflow(
    config: Optional[SpliceAIConfig] = None,
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None,
    verbosity: int = 1,
    no_final_aggregate: bool = False,
    ...)
```

## High-Level Flow
1. **Preparation (optional)**
   * Verify / generate annotations, FASTA chunks and overlap masks.
2. **Chunk loop** (memory-friendly)
   * For each genomic chunk:
     * Run the base splice predictor (e.g., SpliceAI) to get raw `donor`, `acceptor`, and `neither` probabilities for every position.
     * **Generate Context-Aware Features**: The `enhanced_workflow.py` and `enhanced_evaluation.py` modules perform a detailed analysis on these raw scores (see section below).
     * Write two TSVs containing the raw scores and the newly generated context features:
       * `splice_positions_enhanced_<chr>_chunk_<i>.tsv`
       * `analysis_sequences_<chr>_chunk_<i>.tsv`
3. **Aggregation** *(skippable)*
   * If `no_final_aggregate=False`, concatenate all chunk-level TSVs into
     `full_splice_positions_enhanced.tsv` and
     `full_analysis_sequences.tsv` (may use tens of GiB of RAM).
4. **Return value** – a dict with success flag, (possibly empty)
   `analysis_sequences` DataFrame, and key paths.

## Detailed Feature Generation Flow

A key innovation of this workflow is the generation of **Context-Aware Probability Features**, which are essential for training the downstream meta-model. This process moves beyond single-point predictions to analyze the local signal landscape.

1.  **Windowed Analysis**: For every position predicted by the base model, the workflow analyzes a **5-point sliding window** consisting of the position itself (center), its two upstream neighbors (`-1`, `-2`), and its two downstream neighbors (`+1`, `+2`).

2.  **Signal Shape Characterization**: Using the probability scores from all 5 positions in the window, the `enhanced_evaluation.py` module calculates a rich set of features that describe the shape and quality of the prediction signal. This includes features that answer questions like:
    *   **Is the signal a distinct peak?** (`is_local_peak`)
    *   **How sharp and strong is the signal?** (`signal_strength`, `peak_height_ratio`)
    *   **What is the signal's curvature?** (`second_derivative`)
    *   **Is the signal symmetric?** (`context_asymmetry`)

3.  **Enriched Output**: The final output TSV (`splice_positions_enhanced`) contains not only the raw donor/acceptor/neither scores for each position but also this full suite of context-aware features. This provides a much richer dataset for the meta-model to learn from, enabling it to effectively distinguish true splice sites from noisy predictions.

## Key Toggles & Flags
| Argument | Effect |
|----------|--------|
| `verbosity` | 0 silent → 2 very chatty. |
| `no_final_aggregate` | **True** skips memory-heavy final concat. |
| `target_genes / target_chromosomes` | Restrict run to subsets for quick tests. |
| `cleanup` | Remove heavy intermediates after success. |

## Typical Usage (building training data)
```python
run_enhanced_splice_prediction_workflow(
    config=my_cfg,
    target_genes=gene_list,
    no_final_aggregate=False,
    verbosity=2)
```

## Artefacts Produced
All files are written to the run-specific *output directory* (reported in the
return dict).  Filenames follow the pattern `<kind>_<chrom>_chunk_<start>_<end>.tsv`.

### 1. Analysis-sequence artefacts  `analysis_sequences_*`
Per-nucleotide context and raw SpliceAI probabilities.
Examples
```
analysis_sequences_10_chunk_1001_1500.tsv
analysis_sequences_11_chunk_1501_2000.tsv
analysis_sequences_12_chunk_1_500.tsv
```

### 2. Error-analysis artefacts  `splice_errors_*`
Diagnostics comparing predicted splice sites with the reference annotation.
Examples
```
splice_errors_10_chunk_1001_1500.tsv
splice_errors_11_chunk_1501_2000.tsv
splice_errors_12_chunk_1_500.tsv
```

### 3. Enhanced splice-position artefacts  `splice_positions_enhanced_*`
One row per candidate splice position enriched with additional metadata.
Examples
```
splice_positions_enhanced_10_chunk_1001_1500.tsv
splice_positions_enhanced_11_chunk_1501_2000.tsv
splice_positions_enhanced_12_chunk_1_500.tsv
```

### 4. Aggregated artefacts  `full_*`
If `no_final_aggregate=False` the workflow concatenates each chunk category
into genome-wide files:
```
full_analysis_sequences.tsv
full_splice_errors.tsv
full_splice_positions_enhanced.tsv
```

---

---

## Internal Safety Considerations
* Uses streaming for the chunk loop so peak RAM ~ chunk size.
* `no_final_aggregate=True` avoids concatenating millions of rows in memory.
* Verbose diagnostics (`verbosity>=2`) print row / column stats per chunk.
