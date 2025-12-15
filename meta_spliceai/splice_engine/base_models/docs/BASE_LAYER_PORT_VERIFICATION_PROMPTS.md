# Base Layer Port Verification Prompts

**Last Updated**: December 12, 2025  
**Purpose**: Verification prompts for AI agents to validate base layer port completeness  
**Target System**: agentic-spliceai (ported from meta-spliceai)

---

## Status Update: Base Model Pass Complete ✅

The OpenSpliceAI full genome production pass has **successfully completed**:
- **Status**: All chromosomes (1-22, X, Y) processed
- **Final Chromosome**: Chr Y completed
- **Artifacts Location**: `data/mane/GRCh38/openspliceai_eval/meta_models/`
- **Chunk Files**: `analysis_sequences_*_chunk_*.tsv`, `analysis_positions_*_chunk_*.tsv`
- **Process**: Completed with automatic checkpoint recovery

These artifacts serve as **reference outputs** for verifying the ported base layer.

---

## Overview

This document provides **systematic verification prompts** to ensure the ported base layer in `agentic-spliceai` can replicate all functionality from `meta-spliceai`.

### Key Requirements

The ported base layer must support:

1. ✅ **Regular Prediction Mode**: Per-splice-site predictions (donor/acceptor identification)
2. ✅ **Full Coverage Mode**: Per-nucleotide scores for all positions (for meta-model training)
3. ✅ **Artifact Generation**: TSV/Parquet outputs compatible with meta-learning pipeline
4. ✅ **Both Base Models**: SpliceAI (GRCh37) and OpenSpliceAI (GRCh38)
5. ✅ **Evaluation**: TP/FP/FN/TN classification against reference annotations

---

## Main Entry Points Reference

### Meta-SpliceAI (Source System)

**Primary Entry Points**:
1. **Python API**: `meta_spliceai/run_base_model.py`
   - Function: `run_base_model_predictions()`
   - Returns: Dictionary with positions, nucleotide_scores, gene_manifest, etc.

2. **CLI**: `meta_spliceai/cli/run_base_model_cli.py`
   - Command: `meta-spliceai-run`
   - Arguments: `--genes`, `--chromosomes`, `--base-model`, `--mode`, `--coverage`

3. **Shell Script**: `scripts/training/process_chromosomes_sequential_smart.sh`
   - Purpose: Sequential chromosome processing with checkpoint recovery
   - Used for: Full genome production passes

**Core Workflow**: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`
- Function: `run_enhanced_splice_prediction_workflow()`
- This is the **main orchestration function** that all entry points call

---

### Agentic-SpliceAI (Target System)

**Ported Structure**:
```
agentic-spliceai/src/agentic_spliceai/splice_engine/base_layer/
├── models/
│   ├── config.py        # BaseModelConfig, SpliceAIConfig, OpenSpliceAIConfig
│   └── runner.py        # BaseModelRunner (currently shows TODO)
├── prediction/
│   ├── core.py          # predict_splice_sites_for_genes() - PARTIALLY PORTED
│   └── evaluation.py    # Evaluation logic (needs verification)
├── data/
│   ├── types.py         # Data types
│   ├── sequence_extraction.py
│   └── genomic_extraction.py
└── io/
    └── (handlers for artifact I/O)
```

**Current Status**: 
- ✅ Configuration classes ported
- ✅ Core prediction function ported (`predict_splice_sites_for_genes()`)
- ⚠️  Runner shows "TODO: Call actual prediction workflow when ported"
- ❓ Full workflow orchestration (needs verification)
- ❓ Evaluation and artifact generation (needs verification)

---

## Verification Prompt 1: Configuration System Validation

### Prompt

```markdown
Verify that the BaseModelConfig in agentic-spliceai correctly replicates meta-spliceai's configuration system.

**Test 1: Configuration Creation**
Create configs for both SpliceAI and OpenSpliceAI:

```python
from agentic_spliceai.splice_engine.base_layer import (
    BaseModelConfig, 
    SpliceAIConfig, 
    OpenSpliceAIConfig,
    create_config
)

# Test SpliceAI config
spliceai_config = SpliceAIConfig(
    mode='test',
    coverage='gene_subset',
    threshold=0.5,
    save_nucleotide_scores=True
)

# Test OpenSpliceAI config  
openspliceai_config = OpenSpliceAIConfig(
    mode='production',
    coverage='full_genome',
    save_nucleotide_scores=False
)

# Test factory function
config_via_factory = create_config(
    base_model='openspliceai',
    mode='test'
)
```

**Verify**:
1. ✅ Configs instantiate without errors
2. ✅ Paths auto-resolve based on base_model:
   - SpliceAI → `data/ensembl/GRCh37/`
   - OpenSpliceAI → `data/mane/GRCh38/`
3. ✅ `gtf_file`, `genome_fasta`, `eval_dir` auto-populated
4. ✅ `genomic_build` property returns correct build ('GRCh37' or 'GRCh38')
5. ✅ `annotation_source` property returns correct source ('ensembl' or 'mane')

**Expected Output**:
```
✓ SpliceAI config created
  - Build: GRCh37
  - GTF: data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf
  - FASTA: data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa
  - Eval dir: data/ensembl/GRCh37/spliceai_eval

✓ OpenSpliceAI config created  
  - Build: GRCh38
  - GTF: data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff
  - FASTA: data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna
  - Eval dir: data/mane/GRCh38/openspliceai_eval
```

**Comparison**:
Compare the outputs with the equivalent code in meta-spliceai:

```python
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.run_base_model import BaseModelConfig

# Create equivalent configs
meta_config = SpliceAIConfig(base_model='openspliceai', mode='test')
```

Verify that paths match exactly.
```

---

## Verification Prompt 2: Core Prediction Function Validation

### Prompt

```markdown
Verify that the ported `predict_splice_sites_for_genes()` function produces identical output to meta-spliceai.

**Test Setup**:
Use a small test gene (e.g., 'BRCA1') for rapid iteration.

**Test 2.1: Single Gene Prediction**

```python
import polars as pl
from agentic_spliceai.splice_engine.base_layer.prediction.core import (
    predict_splice_sites_for_genes,
    load_spliceai_models
)

# Load models
models = load_spliceai_models(model_type='openspliceai', verbosity=2)

# Load gene sequence (you'll need to implement sequence loading)
gene_df = pl.DataFrame({
    'gene_id': ['gene-BRCA1'],
    'gene_name': ['BRCA1'],
    'seqname': ['17'],
    'start': [43044295],
    'end': [43125364],
    'strand': ['-'],
    'sequence': ['ACGT...']  # Extract actual sequence
})

# Run prediction
predictions = predict_splice_sites_for_genes(
    gene_df=gene_df,
    models=models,
    context=10000,
    output_format='dict',
    verbosity=2
)

print(predictions)
```

**Verify**:
1. ✅ Models load successfully (5 models for ensemble)
2. ✅ Predictions return dictionary with structure:
   ```python
   {
       'gene-BRCA1': {
           'seqname': '17',
           'gene_name': 'BRCA1',
           'strand': '-',
           'gene_start': 43044295,
           'gene_end': 43125364,
           'positions': [43044295, 43044296, ..., 43125364],
           'donor_prob': [0.001, 0.002, ..., 0.999],
           'acceptor_prob': [0.001, 0.003, ..., 0.998],
           'neither_prob': [0.998, 0.995, ..., 0.003]
       }
   }
   ```
3. ✅ Number of positions matches gene length: `len(positions) == (gene_end - gene_start + 1)`
4. ✅ Probabilities sum to ~1.0: `donor_prob + acceptor_prob + neither_prob ≈ 1.0`
5. ✅ Strand handling correct (negative strand sequences reverse-complemented)

**Comparison with Meta-SpliceAI**:

Run identical code in meta-spliceai:

```python
from meta_spliceai.splice_engine.run_spliceai_workflow import (
    predict_splice_sites_for_genes as meta_predict
)
from meta_spliceai.splice_engine.meta_models.utils.model_utils import (
    load_base_model_ensemble
)

# Load models
meta_models, _ = load_base_model_ensemble('openspliceai', context=10000)

# Run prediction with same gene_df
meta_predictions = meta_predict(
    gene_df=gene_df,
    models=meta_models,
    context=10000,
    output_format='dict',
    verbosity=2
)
```

**Validate**:
- Compare `predictions['gene-BRCA1']['positions']` with `meta_predictions['gene-BRCA1']['positions']`
- Compare probabilities (should match within floating-point tolerance, e.g., < 1e-6)

```python
import numpy as np

# Check position alignment
assert predictions['gene-BRCA1']['positions'] == meta_predictions['gene-BRCA1']['positions']

# Check probability alignment
donor_diff = np.abs(
    np.array(predictions['gene-BRCA1']['donor_prob']) - 
    np.array(meta_predictions['gene-BRCA1']['donor_prob'])
)
assert np.max(donor_diff) < 1e-6, f"Max donor prob diff: {np.max(donor_diff)}"
```
```

---

## Verification Prompt 3: Full Workflow Orchestration

### Prompt

```markdown
Verify that the complete workflow can be executed end-to-end, replicating meta-spliceai's `run_enhanced_splice_prediction_workflow()`.

**Test 3: Full Workflow Execution**

The full workflow must:
1. Load configuration
2. Prepare data (annotations, splice sites, sequences)
3. Load models
4. Run predictions (with mini-batching)
5. Evaluate against reference annotations (TP/FP/FN/TN)
6. Extract analysis sequences (±250nt windows)
7. Generate artifacts (TSV files)
8. Create gene manifest

**Current Issue**: 
The `BaseModelRunner.run_single_model()` in agentic-spliceai shows:
```python
# TODO: Call actual prediction workflow when ported
return BaseModelResult(
    model_name=model_name,
    success=False,
    error="Prediction workflow not yet ported"
)
```

**Implementation Needed**:

In `agentic-spliceai/src/agentic_spliceai/splice_engine/base_layer/models/runner.py`, replace the TODO with:

```python
def run_single_model(self, model_name: str, target_genes: List[str], ...):
    # Import the ported workflow
    from agentic_spliceai.splice_engine.base_layer.workflows import (
        run_enhanced_prediction_workflow  # This needs to be ported!
    )
    
    # Create configuration
    config = BaseModelConfig(
        base_model=model_name,
        mode=mode,
        coverage=coverage,
        test_name=f"{test_name}_{model_name}",
        ...
    )
    
    # Run workflow
    results = run_enhanced_prediction_workflow(
        config=config,
        target_genes=target_genes,
        verbosity=verbosity
    )
    
    # Extract results
    return BaseModelResult(
        model_name=model_name,
        success=results['success'],
        runtime_seconds=runtime,
        positions=results['positions'],
        nucleotide_scores=results.get('nucleotide_scores'),
        gene_manifest=results.get('gene_manifest'),
        ...
    )
```

**Missing Module**: `agentic_spliceai.splice_engine.base_layer.workflows`

This module needs to be created by porting:
- `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`

**Test After Implementation**:

```python
from agentic_spliceai.splice_engine.base_layer import BaseModelRunner

runner = BaseModelRunner()

result = runner.run_single_model(
    model_name='openspliceai',
    target_genes=['BRCA1', 'TP53'],
    test_name='validation_test',
    mode='test',
    coverage='gene_subset',
    save_nucleotide_scores=True,
    no_tn_sampling=True,
    verbosity=2
)

# Verify success
assert result.success, f"Run failed: {result.error}"
assert result.positions.height > 0, "No positions generated"
assert len(result.processed_genes) == 2, "Not all genes processed"
assert result.nucleotide_scores is not None, "Nucleotide scores not saved"

print(f"✓ Processed {len(result.processed_genes)} genes")
print(f"✓ Generated {result.positions.height} positions")
print(f"✓ Nucleotide scores: {result.nucleotide_scores.height} rows")
```

**Comparison with Meta-SpliceAI**:

```python
from meta_spliceai import run_base_model_predictions

meta_results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53'],
    config=None,  # Use defaults
    verbosity=2,
    save_nucleotide_scores=True
)

# Compare outputs
assert result.positions.height == meta_results['positions'].height
# ... additional comparisons
```
```

---

## Verification Prompt 4: Artifact Generation Validation

### Prompt

```markdown
Verify that artifact files generated by agentic-spliceai match meta-spliceai's format and content.

**Test 4: Artifact File Comparison**

After running a test prediction, verify that output files match expected schemas.

**Expected Artifacts** (from meta-spliceai):

1. **analysis_positions_{chr}_chunk_{start}_{end}.tsv**
   - Columns: `gene_id`, `gene_name`, `seqname`, `position`, `strand`, `pred_type`, `donor_score`, `acceptor_score`, `neither_score`, `splice_type`, `distance_to_nearest`, ...
   - Pred types: TP, FP, FN, TN (with optional TN sampling)

2. **analysis_sequences_{chr}_chunk_{start}_{end}.tsv**
   - Columns: `gene_id`, `position`, `sequence`, `label`, `pred_type`, ...
   - Sequences: ±250nt windows around splice sites

3. **error_analysis_{chr}_chunk_{start}_{end}.tsv**
   - False positives and false negatives with error context

4. **gene_manifest.tsv**
   - Columns: `gene_id`, `gene_name`, `seqname`, `status`, `error_message`, `n_positions`, `n_tp`, `n_fp`, `n_fn`, `n_tn`

5. **nucleotide_scores_{chr}_chunk_{start}_{end}.tsv** (if enabled)
   - Columns: `gene_id`, `position`, `donor_prob`, `acceptor_prob`, `neither_prob`

**Test Code**:

```python
import polars as pl
from pathlib import Path

# Run test prediction
# ... (use code from Prompt 3)

# Check artifacts exist
output_dir = Path(result.paths['output_dir'])
assert output_dir.exists(), f"Output directory not found: {output_dir}"

# Find chunk files
positions_files = list(output_dir.glob('analysis_positions_*_chunk_*.tsv'))
sequences_files = list(output_dir.glob('analysis_sequences_*_chunk_*.tsv'))
manifest_file = output_dir / 'gene_manifest.tsv'

assert len(positions_files) > 0, "No position files generated"
assert len(sequences_files) > 0, "No sequence files generated"
assert manifest_file.exists(), "Gene manifest not generated"

# Verify positions file schema
positions_df = pl.read_csv(positions_files[0], separator='\t')
required_columns = [
    'gene_id', 'gene_name', 'seqname', 'position', 'strand',
    'pred_type', 'donor_score', 'acceptor_score', 'splice_type'
]
for col in required_columns:
    assert col in positions_df.columns, f"Missing column: {col}"

# Verify pred_type values
pred_types = positions_df['pred_type'].unique().to_list()
assert set(pred_types).issubset({'TP', 'FP', 'FN', 'TN'}), f"Invalid pred_types: {pred_types}"

# Verify sequences file schema
sequences_df = pl.read_csv(sequences_files[0], separator='\t')
required_seq_columns = ['gene_id', 'position', 'sequence', 'label', 'pred_type']
for col in required_seq_columns:
    assert col in sequences_df.columns, f"Missing column: {col}"

# Verify sequence lengths (should be ~500nt: ±250 window)
seq_lens = sequences_df['sequence'].str.lengths().to_list()
assert all(400 <= l <= 600 for l in seq_lens), "Unexpected sequence lengths"

# Verify manifest
manifest_df = pl.read_csv(manifest_file, separator='\t')
assert 'gene_id' in manifest_df.columns
assert 'status' in manifest_df.columns
assert manifest_df.filter(pl.col('status') == 'processed').height > 0

print("✓ All artifact files generated with correct schema")
```

**Comparison with Meta-SpliceAI Artifacts**:

Load reference artifacts from meta-spliceai's completed run:

```python
# Load meta-spliceai reference (OpenSpliceAI full genome pass)
meta_ref_dir = Path('data/mane/GRCh38/openspliceai_eval/meta_models')
meta_positions = pl.read_csv(
    meta_ref_dir / 'analysis_positions_21_chunk_1_500.tsv',
    separator='\t'
)

# Compare schemas
print("Meta-SpliceAI columns:", sorted(meta_positions.columns))
print("Agentic-SpliceAI columns:", sorted(positions_df.columns))

# Verify column compatibility
assert set(required_columns).issubset(set(positions_df.columns))
assert set(required_columns).issubset(set(meta_positions.columns))
```
```

---

## Verification Prompt 5: Full Coverage Mode (Nucleotide Scores)

### Prompt

```markdown
Verify that full coverage mode generates per-nucleotide scores for all positions in a gene.

**Test 5: Full Coverage Nucleotide Scores**

Full coverage mode is **critical** for meta-model training, as it provides:
- Donor scores for every nucleotide
- Acceptor scores for every nucleotide  
- Neither scores for every nucleotide
- **No sampling** (all positions retained)

**Test Code**:

```python
from agentic_spliceai.splice_engine.base_layer import BaseModelRunner

runner = BaseModelRunner()

# Run with nucleotide score saving enabled
result = runner.run_single_model(
    model_name='openspliceai',
    target_genes=['TP53'],  # Small gene for testing (~20kb)
    test_name='full_coverage_test',
    mode='test',
    coverage='gene_subset',
    save_nucleotide_scores=True,  # Enable full coverage
    no_tn_sampling=True,          # Disable TN sampling
    verbosity=2
)

# Verify nucleotide scores saved
assert result.nucleotide_scores is not None, "Nucleotide scores not generated"
nucleotide_df = result.nucleotide_scores

# Check schema
required_cols = ['gene_id', 'position', 'donor_prob', 'acceptor_prob', 'neither_prob']
for col in required_cols:
    assert col in nucleotide_df.columns, f"Missing column: {col}"

# Verify full coverage (every position has scores)
gene_positions = result.positions.filter(pl.col('gene_id') == 'gene-TP53')
nucleotide_positions = nucleotide_df.filter(pl.col('gene_id') == 'gene-TP53')

# Should have same number of positions
assert gene_positions.height == nucleotide_positions.height, \
    f"Position count mismatch: {gene_positions.height} vs {nucleotide_positions.height}"

# Verify all positions covered
all_positions = set(gene_positions['position'].to_list())
nucleotide_positions_set = set(nucleotide_positions['position'].to_list())
assert all_positions == nucleotide_positions_set, "Position coverage mismatch"

# Verify probability sums
prob_sums = (
    nucleotide_df['donor_prob'] + 
    nucleotide_df['acceptor_prob'] + 
    nucleotide_df['neither_prob']
)
assert all(0.99 <= s <= 1.01 for s in prob_sums.to_list()), \
    "Probabilities don't sum to 1.0"

print(f"✓ Full coverage verified: {nucleotide_df.height} positions")
print(f"✓ Probability sums valid")
```

**Output File Verification**:

```python
# Check that nucleotide scores file was saved
output_dir = Path(result.paths['output_dir'])
nuc_score_files = list(output_dir.glob('nucleotide_scores_*_chunk_*.tsv'))

assert len(nuc_score_files) > 0, "Nucleotide score files not saved"

# Load and verify
nuc_df_from_file = pl.read_csv(nuc_score_files[0], separator='\t')
assert nuc_df_from_file.height > 0
print(f"✓ Nucleotide scores saved to disk: {nuc_score_files[0].name}")
```

**Critical Requirement**:
The nucleotide scores **must be generated for every position** in the gene, not just predicted splice sites. This is what distinguishes "full coverage" from "regular prediction" mode.
```

---

## Verification Prompt 6: Multi-Chromosome Production Run

### Prompt

```markdown
Verify that multi-chromosome production runs work with checkpoint recovery.

**Test 6: Chromosome-Level Processing**

**Test Scenario**: Run predictions on Chr21 and Chr22 (small chromosomes, ~500-800 genes each)

```python
from agentic_spliceai.splice_engine.base_layer import BaseModelRunner

runner = BaseModelRunner()

# Run production mode on multiple chromosomes
# Note: This requires the workflow to support target_chromosomes parameter

# Option 1: Via runner (if implemented)
result = runner.run_single_model(
    model_name='openspliceai',
    target_chromosomes=['21', '22'],  # Small test chromosomes
    test_name='chr21_22_production',
    mode='production',
    coverage='full_genome',
    save_nucleotide_scores=False,  # Disable for speed
    no_tn_sampling=False,          # Enable TN sampling
    verbosity=1
)

# Option 2: Via direct workflow call (if Option 1 not implemented)
from agentic_spliceai.splice_engine.base_layer.workflows import (
    run_enhanced_prediction_workflow
)
from agentic_spliceai.splice_engine.base_layer import BaseModelConfig

config = BaseModelConfig(
    base_model='openspliceai',
    mode='production',
    coverage='full_genome',
    chromosomes=['21', '22'],
    save_nucleotide_scores=False,
    verbosity=1
)

result = run_enhanced_prediction_workflow(
    config=config,
    target_chromosomes=['21', '22'],
    verbosity=1
)
```

**Verify**:
1. ✅ Both chromosomes processed
2. ✅ Chunk files created for each chromosome:
   - `analysis_positions_21_chunk_1_500.tsv`
   - `analysis_positions_21_chunk_501_1000.tsv` (if Chr21 has >500 genes)
   - `analysis_positions_22_chunk_1_500.tsv`
   - ...
3. ✅ Gene manifest includes genes from both chromosomes
4. ✅ No errors or missing genes (check manifest for `status='processed'`)

**Checkpoint Recovery Test**:

To test checkpoint recovery:

1. Start run on Chr21
2. Interrupt after first chunk completes
3. Restart with same parameters
4. Verify: Second run skips completed chunk and continues from chunk 2

```python
# First run (will be interrupted manually)
result1 = runner.run_single_model(
    model_name='openspliceai',
    target_chromosomes=['21'],
    test_name='checkpoint_test',
    mode='production',
    verbosity=1
)
# Interrupt after ~1 minute (should complete 1-2 chunks)

# Check which chunks exist
output_dir = Path('data/mane/GRCh38/openspliceai_eval/meta_models')
completed_chunks = list(output_dir.glob('analysis_positions_21_chunk_*.tsv'))
print(f"Completed chunks before interrupt: {len(completed_chunks)}")

# Second run (should resume from last completed chunk)
result2 = runner.run_single_model(
    model_name='openspliceai',
    target_chromosomes=['21'],
    test_name='checkpoint_test',
    mode='production',
    verbosity=1
)

# Verify: More chunks completed
completed_chunks_after = list(output_dir.glob('analysis_positions_21_chunk_*.tsv'))
print(f"Completed chunks after resume: {len(completed_chunks_after)}")

assert len(completed_chunks_after) > len(completed_chunks), \
    "Checkpoint recovery failed - no new chunks completed"
```
```

---

## Verification Prompt 7: Cross-Model Consistency

### Prompt

```markdown
Verify that both SpliceAI and OpenSpliceAI can be run on the same genes (with appropriate genome build mapping).

**Test 7: SpliceAI vs OpenSpliceAI Comparison**

**Challenge**: SpliceAI uses GRCh37, OpenSpliceAI uses GRCh38. Need to:
1. Map gene IDs between builds
2. Run both models
3. Compare outputs (qualitatively, not expecting exact matches due to build differences)

**Test Code**:

```python
from agentic_spliceai.splice_engine.base_layer import BaseModelRunner
from agentic_spliceai.splice_engine.resources import get_gene_mapper

# Map genes between builds
mapper = get_gene_mapper()
test_genes = ['BRCA1', 'TP53', 'UNC13A']

# Get gene IDs for each build
grch37_genes = [mapper.get_gene_id(g, build='GRCh37') for g in test_genes]
grch38_genes = [mapper.get_gene_id(g, build='GRCh38') for g in test_genes]

print(f"GRCh37 genes: {grch37_genes}")
print(f"GRCh38 genes: {grch38_genes}")

# Run SpliceAI (GRCh37)
runner = BaseModelRunner()
spliceai_result = runner.run_single_model(
    model_name='spliceai',
    target_genes=grch37_genes,
    test_name='cross_model_test',
    mode='test',
    verbosity=1
)

# Run OpenSpliceAI (GRCh38)
openspliceai_result = runner.run_single_model(
    model_name='openspliceai',
    target_genes=grch38_genes,
    test_name='cross_model_test',
    mode='test',
    verbosity=1
)

# Compare results
print("\nSpliceAI Results:")
print(f"  Genes processed: {len(spliceai_result.processed_genes)}")
print(f"  Positions: {spliceai_result.positions.height}")
print(f"  Metrics: {spliceai_result.metrics}")

print("\nOpenSpliceAI Results:")
print(f"  Genes processed: {len(openspliceai_result.processed_genes)}")
print(f"  Positions: {openspliceai_result.positions.height}")
print(f"  Metrics: {openspliceai_result.metrics}")

# Verify both succeeded
assert spliceai_result.success, "SpliceAI run failed"
assert openspliceai_result.success, "OpenSpliceAI run failed"
assert len(spliceai_result.processed_genes) == len(test_genes)
assert len(openspliceai_result.processed_genes) == len(test_genes)

print("\n✓ Both models ran successfully on corresponding genes")
```

**Qualitative Comparison**:
- Position counts may differ (GRCh37 vs GRCh38 annotations differ)
- Splice site calls should be similar (not identical) for orthologous regions
- Performance metrics (F1, precision, recall) should be comparable

**Expected Outcome**:
Both models should complete without errors, even though they use different genome builds and annotations.
```

---

## Summary: Verification Checklist

After running all verification prompts, you should be able to confirm:

- [ ] **Prompt 1**: Configuration system works for both SpliceAI and OpenSpliceAI
- [ ] **Prompt 2**: Core prediction function generates correct per-nucleotide scores
- [ ] **Prompt 3**: Full workflow orchestration executes end-to-end
- [ ] **Prompt 4**: Artifact files generated with correct schemas
- [ ] **Prompt 5**: Full coverage mode (nucleotide scores) works
- [ ] **Prompt 6**: Multi-chromosome runs work with checkpoint recovery
- [ ] **Prompt 7**: Both base models can be run and compared

### Critical Gaps Identified

Based on examination of agentic-spliceai code:

1. **Missing Workflow Module**: `agentic_spliceai/splice_engine/base_layer/workflows/`
   - Needs to port: `run_enhanced_splice_prediction_workflow()` from meta-spliceai
   - This is the **main orchestration function**

2. **Runner Implementation**: `BaseModelRunner.run_single_model()` shows TODO
   - Needs to call the workflow function (once ported)

3. **Evaluation Module**: `agentic_spliceai/splice_engine/base_layer/prediction/evaluation.py`
   - Needs verification that it includes TP/FP/FN/TN classification
   - Needs to generate error_analysis artifacts

4. **Artifact I/O**: `agentic_spliceai/splice_engine/base_layer/io/`
   - Needs handlers for saving chunk-level artifacts
   - Needs gene manifest generation

5. **Data Preparation**: Missing utilities for:
   - Loading and filtering GTF/GFF3 annotations
   - Extracting splice sites from annotations
   - Extracting gene sequences from FASTA
   - Detecting overlapping genes

---

## Recommended Porting Strategy

**Phase 1**: Core Workflow (Highest Priority)
1. Port `splice_prediction_workflow.py` → create `workflows/` module
2. Port `data_preparation.py` functions
3. Port `enhanced_workflow.py` (evaluation with all scores)

**Phase 2**: Artifact Generation
4. Port artifact handlers (`MetaModelDataHandler`, `ModelEvaluationFileHandler`)
5. Port chunk-level checkpointing logic
6. Port gene manifest generation

**Phase 3**: Integration & Testing
7. Wire up `BaseModelRunner.run_single_model()` to call workflow
8. Run Verification Prompts 1-7
9. Compare outputs with meta-spliceai reference artifacts

**Phase 4**: Documentation
10. Update agentic-spliceai docs with entry points
11. Create usage examples
12. Document differences from meta-spliceai (if any)

---

## Reference Artifacts for Validation

Use these meta-spliceai artifacts as ground truth:

**Location**: `meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/`

**Files**:
- `analysis_positions_{chr}_chunk_{start}_{end}.tsv` (18 chromosomes × multiple chunks)
- `analysis_sequences_{chr}_chunk_{start}_{end}.tsv`
- `gene_manifest.tsv` (when complete)

**Test Cases**:
1. Load reference Chr21 chunk: `analysis_positions_21_chunk_1_500.tsv`
2. Run agentic-spliceai on same genes
3. Compare schemas and position counts
4. Verify pred_type distributions match (TP/FP/FN/TN ratios)

---

## Contact & Support

For questions about verification:
- Check: `docs/base_models/AI_AGENT_PORTING_GUIDE.md` - Stage-by-stage porting instructions
- Check: `docs/base_models/BASE_LAYER_INTEGRATION_GUIDE.md` - Integration patterns
- Check: `docs/data/DATA_LAYOUT_MASTER_GUIDE.md` - Data layout conventions

---

*Last Updated: December 12, 2025*  
*Status: Base model pass complete, verification prompts ready*

