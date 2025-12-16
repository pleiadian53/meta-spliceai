# Inference Workflow Scenarios, Outputs, and Examples

This document provides comprehensive examples for running the splice site prediction inference workflow with different types of genes and scenarios.

## Overview

The inference workflow supports two primary scenarios based on whether target genes were included in the training data:

- **Scenario 1**: Genes from training data (focusing on unseen positions)
- **Scenario 2A**: Non-training genes (with artifacts)
- **Scenario 2B**: Non-training genes (no artifacts; first-time processing)

In all scenarios, complete coverage ensures predictions for every nucleotide position. Selective meta-model inference recalibrates only uncertain positions, reusing base model predictions elsewhere.

## Gene Classification

Terminology note:
- "Non-training genes" are genes that were not included in the meta-model training set. They may still appear in precomputed analysis artifacts and are not necessarily biologically novel.

### Finding Genes for Each Scenario

You can determine which scenario applies to your target genes by consulting the training dataset's gene manifest and the authoritative gene universe from `gene_features.tsv`:

```bash
# 1) Training genes (from the training dataset manifest)
#    Assumes column 2 is gene_id in train_pc_1000_3mers/master/gene_manifest.csv
cut -d',' -f2 train_pc_1000_3mers/master/gene_manifest.csv | tail -n +2 | sort -u > training_genes.txt

# 2) Authoritative gene universe (from Ensembl gene features)
#    Robustly extract the gene_id column by header name
awk -F '\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="gene_id") c=i; next} {print $c}' \
  data/ensembl/spliceai_analysis/gene_features.tsv | tail -n +2 | sort -u > all_genes.txt

# 3) Genes present in artifacts (analysis_sequences_*.tsv)
#    Extract gene_id from all chunk files, then unique-sort
tmp_art=artifact_genes.tmp
> "$tmp_art"
for f in data/ensembl/spliceai_eval/meta_models/analysis_sequences_*.tsv; do \
  awk -F '\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="gene_id") c=i; next} {print $c}' "$f" | tail -n +2 >> "$tmp_art"; \
done
sort -u "$tmp_art" > artifact_genes.txt && rm -f "$tmp_art"

# 4) Useful sets
#    a) Genes not in training (overall)
comm -23 all_genes.txt training_genes.txt > non_training_overall.txt

#    b) Non-training genes that are present in artifacts (preferred for Scenario 2A)
comm -23 artifact_genes.txt training_genes.txt > non_training_in_artifacts.txt
```

### Convenience script: find_test_genes.sh

You can also use the bundled utility to automatically identify representative genes for all scenarios and print ready-to-use export commands and example invocations:

```bash
# Activate environment
mamba activate surveyor

# Run the convenience script
./meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh

# It prints exports like:
# export TEST_SCENARIO1_GENES="ENSG000001...,ENSG000002..."
# export TEST_SCENARIO2A_GENES="..."
# export TEST_SCENARIO2B_GENES="..."

# Example: run Scenario 2B quickly with the exported genes
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes $TEST_SCENARIO2B_GENES \
    --output-dir test_pc_1000_3mers/predictions/scenario2b \
    --inference-mode hybrid \
    --verbose
```

This script is documented in the docs index and generates `test_genes.json` with the selected genes.

### Statistics on an example training dataset: train_pc_1000_3mers
- **Training genes**: 1,002 genes with comprehensive training coverage
- **Non-training genes (with artifacts)**: 83 genes with artifacts available but not in training data
- **Total artifacts**: Available for 1,085+ genes across all chromosomes

## Scenario 1: Genes from Training Data (Unseen Positions)

### Description
Focus on positions within known genes that weren't covered during training. The workflow leverages existing training coverage and applies selective meta-model inference only to uncertain positions.

### Key Characteristics
- **Artifact Utilization**: Reuses pre-computed base model predictions where available
- **Selective Processing**: Only generates features for uncertain positions (low confidence)
- **Efficient Coverage**: Combines existing predictions with meta-model recalibration
- **Training Familiarity**: Gene structure and characteristics are familiar to the model

### Example: CROCC Gene (ENSG00000058453)

```bash
# Activate environment
mamba activate surveyor

# Run inference workflow
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000058453 \
    --output-dir results/scenario_1_training_gene \
    --inference-mode hybrid \
    --verbose
```

### Expected Results
```
📚 STEP 1: Loading training coverage
   ✅ Loaded 391 training positions across 1 genes

🔬 STEP 2: Running selective inference for uncertain positions
   ✅ Step 2 completed in 0.3 seconds

📊 STEP 3: Loading complete base model predictions
   📊 Loaded 412 complete base model predictions

🤖 STEP 4: Loading meta-model predictions for uncertain positions
   🎯 Identified 15 uncertain positions for meta-model inference
   ✅ Generated 124 features for 15 positions
   ✅ Generated meta-model predictions for 15 positions

🔗 STEP 5: Combining predictions for complete coverage
   📊 Complete coverage: 412 positions
   🤖 Meta-model recalibrated: 15 (3.6%)
   🔄 Base model reused: 397 (96.4%)

⏱️  Processing time: 1.0s
```

### Key Observations
- **Efficiency**: Fast processing (~1 second) due to reuse of existing predictions
- **Selective Enhancement**: Only 3.6% of positions required meta-model recalibration
- **Complete Coverage**: All 412 positions covered with hybrid predictions
- **Training Benefit**: Leverages 391 existing training positions

## Scenario 2A: Non-training Genes (With Artifacts)

### Description
Process genes that were not included in the training dataset but have pre-computed artifacts available. Requires full base model pass followed by selective meta-model inference.

### Key Characteristics
- **Full Base Model Pass**: Generates complete base model predictions for all positions
- **Artifact Reuse**: Leverages existing `analysis_sequences_*` and `splice_positions_enhanced_*` artifacts
- **Novel Gene Processing**: Handles genes not seen during training
- **Comprehensive Coverage**: Provides predictions for entire gene sequence

### Example: ENSG00000000460 (non-training gene with artifacts)

```bash
# Activate environment
mamba activate surveyor

# Run inference workflow for a non-training gene with artifacts
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000000460 \
    --output-dir results/scenario_2_novel_gene \
    --inference-mode hybrid \
    --verbose
```

### Expected Results
```

## Scenario 2B: Non-training Genes (No Artifacts)

### Description
Non-training genes without prior artifacts go through a full base model pass to create `analysis_sequences_*` and enriched position files, followed by selective meta-model recalibration for uncertain positions.

### Key Characteristics
- **Full Base Model Pass**: Always generates complete base predictions
- **No Pre-existing Artifacts**: Artifacts are created during the run and can be preserved
- **Selective Meta-Modeling**: Only uncertain positions are recalibrated

### Example
```bash
# Activate environment
mamba activate surveyor

# Run inference workflow for two non-training genes without artifacts
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000284616,ENSG00000115705 \
    --complete-coverage \
    --output-dir test_pc_1000_3mers/predictions/scenario2b \
    --inference-mode hybrid \
    --verbose
``` 

### Expected Results
- Complete coverage predictions for each gene
- Selective recalibration rate typically ~0.1–0.5% of positions
- Preserved artifacts (optional; see next section)

## Output Locations and Interpretation

### Where outputs are saved

- Per-run hybrid predictions directory:
  - `test_<train_name>/predictions/<run_id>/`
  - Files:
    - `complete_coverage_predictions.parquet`: Hybrid predictions for all positions
    - `meta_model_predictions.parquet`: Meta-model predictions for uncertain positions
    - `base_model_predictions.parquet`: Base model predictions for reference

- Aggregated workflow outputs per experiment:
  - `test_<train_name>/predictions/<scenario>/`
  - Contents:
    - `genes/<GENE_ID>/<GENE_ID>_predictions.parquet`: Per-gene hybrid predictions
    - `<GENE_ID>_statistics.json`: Per-gene summary statistics
    - `gene_manifest.json`: Process tracking for the run
    - `performance_report.txt`: Timing and throughput metrics

- Preserved artifacts and test dataset (optional; if `--keep-artifacts-dir` or programmatic equivalent is used):
  - `test_<train_name>/`
  - Contents:
    - `complete_coverage_output/meta_models/`: Raw TSVs from base model pass
      - `analysis_sequences_*\.tsv`: Per-position base scores and context
      - `splice_positions_enhanced_*\.tsv`: Enriched position-level summaries
      - `splice_errors_*\.tsv`: FP/FN diagnostics
    - `master/`: Assembled test dataset batches
      - `batch_00001.parquet`, `gene_manifest.csv`
    - `metadata/inference_metadata.json`: Run metadata (model path, thresholds, mode, counts)

### How to interpret the hybrid predictions

- `donor_score`, `acceptor_score`, `neither_score`: Base model probabilities
- `donor_meta`, `acceptor_meta`, `neither_meta`: Recalibrated probabilities for uncertain positions; equals base scores for confident positions
- `prediction_source`: `meta_model` if recalibration applied, otherwise `base_model`
- `is_uncertain`: Boolean flag indicating selective treatment
- `confidence_category`: `confident_splice`, `confident_non_splice`, or `uncertain` (from thresholds)

### Sanity checks and verification

- Selective efficiency: proportion of rows with `is_uncertain == True` is typically small (<1%)
- No label leakage: training-only labels (e.g., `true_position`, `splice_type`, etc.) are not present in the model feature matrix
- Feature harmonization: the meta-model feature matrix matches the training `feature_manifest.csv` exactly in names and order

## Quick helper to obtain scenario test genes

```bash
# Generate scenario candidates (writes JSON + convenient exports)
python meta_spliceai/splice_engine/meta_models/workflows/inference/identify_test_genes.py --verbose --output meta_spliceai/splice_engine/meta_models/workflows/inference/test_genes.json

# Prepare small two-gene lists per scenario and exports under temp/
python temp/prepare_scenarios.py
source temp/export_scenarios.sh

# Run a scenario with two genes and complete coverage + artifact preservation
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
  --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
  --training-dataset train_pc_1000_3mers \
  --genes "$TEST_SCENARIO2A_GENES" \
  --complete-coverage \
  --keep-artifacts-dir test_pc_1000_3mers \
  --output-dir test_pc_1000_3mers/predictions/scenario2a \
  --verbose
```

📚 STEP 1: Loading training coverage
   ✅ Loaded 0 training positions across 0 genes

🔬 STEP 2: Running selective inference for uncertain positions
   ✅ Step 2 completed in 0.4 seconds

📊 STEP 3: Loading complete base model predictions
   📊 Loaded 419 complete base model predictions

🤖 STEP 4: Loading meta-model predictions for uncertain positions
   🎯 Identified 29 uncertain positions for meta-model inference
   ✅ Generated 124 features for 29 positions
   ✅ Generated meta-model predictions for 29 positions

🔗 STEP 5: Combining predictions for complete coverage
   📊 Complete coverage: 419 positions
   🤖 Meta-model recalibrated: 29 (6.9%)
   🔄 Base model reused: 390 (93.1%)

⏱️  Processing time: 1.0s
```

### Key Observations
- **No Training Coverage**: 0 existing training positions (completely novel)
- **Higher Uncertainty**: 6.9% of positions required meta-model recalibration (vs 3.6% for training genes)
- **Complete Processing**: All 419 positions covered with full workflow
- **Efficient Performance**: Still fast (~1 second) due to optimized feature enrichment

## Scenario 2B: Completely Unprocessed Genes (No Artifacts) ⚠️

### Description
Process genes that have no pre-computed artifacts and require full sequence extraction and base model processing from scratch. This represents the most comprehensive workflow scenario.

### Current Status: **Partially Implemented**

#### ✅ **Working Components**
- **Artifact Detection**: Correctly identifies missing artifacts
- **Pipeline Triggering**: Successfully initiates base model workflow  
- **Resource Discovery**: Proper GTF/FASTA file location via systematic manager
- **Configuration Setup**: Dynamic workflow configuration for unprocessed genes

#### ⚠️ **Current Limitations**
- **Sequence Processing Integration**: Workflow has complexity in sequence extraction and artifact coordination
- **Dual-Path Synchronization**: The optimized and original workflow paths need better integration
- **Format Compatibility**: Generated artifacts may not match expected formats for downstream processing

#### 🛠️ **Infrastructure Requirements**
For full Scenario 2B support, additional setup is needed:
- **Genomic Sequence Infrastructure**: Proper per-chromosome sequence processing
- **Artifact Pipeline Integration**: Seamless coordination between generation and consumption paths
- **Enhanced Error Handling**: Better recovery mechanisms for complex processing chains

### Example: ENSG00000142611 (PRDM16) - Unprocessed Gene

```bash
# This example demonstrates the current partial implementation
# Note: May require additional genomic infrastructure setup

python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000142611 \
    --output-dir results/scenario_2b_unprocessed \
    --inference-mode hybrid \
    --verbose
```

#### Expected Behavior (Current State)
```
🔬 STEP 2: Running selective inference for uncertain positions
   [artifact-generation] 🧬 Generating artifacts for genes...
   [artifact-generation] 🚀 Running full SpliceAI pipeline...
   
   ⚠️ Workflow integration limitations may cause incomplete processing
   📋 Logs provide detailed information about processing attempts
```

#### 📋 **Production Readiness**
- **Scenario 1 & 2A**: ✅ **Production Ready** - Well-tested and reliable
- **Scenario 2B**: ⚠️ **Development Phase** - Core logic implemented but needs integration refinements

#### 🎯 **Recommended Approach**
For immediate production use:
1. **Use Scenario 1** for genes in training data
2. **Use Scenario 2A** for novel genes with available artifacts
3. **Contact development team** for Scenario 2B requirements or contribute to completing the integration

## Comparison Summary

| Aspect | Scenario 1 (Training) | Scenario 2A (Novel w/ Artifacts) | Scenario 2B (Unprocessed) |
|--------|----------------------|----------------------------------|---------------------------|
| **Gene Familiarity** | Known from training | Completely novel | Completely novel |
| **Artifacts Available** | ✅ Full coverage | ✅ Pre-computed | ❌ None (generates) |
| **Training Positions** | 391 positions | 0 positions | 0 positions |
| **Total Coverage** | 412 positions | 419 positions | Varies by gene |
| **Meta-model Usage** | 15 positions (3.6%) | 29 positions (6.9%) | TBD (partial) |
| **Processing Time** | ~1.0 seconds | ~1.0 seconds | Varies (generation) |
| **Workflow Status** | ✅ Production Ready | ✅ Production Ready | ⚠️ Development Phase |
| **Infrastructure Needs** | Minimal | Minimal | Additional setup |

## Additional Examples

### Multiple Training Genes
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000058453,ENSG00000083067 \
    --output-dir results/multiple_training_genes \
    --inference-mode hybrid \
    --verbose
```

### Multiple Novel Genes
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000000460,ENSG00000012048 \
    --output-dir results/multiple_novel_genes \
    --inference-mode hybrid \
    --verbose
```

### Using Gene Files
```bash
# Create gene list file
echo "ENSG00000058453" > training_genes.txt
echo "ENSG00000083067" >> training_genes.txt

python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes-file training_genes.txt \
    --output-dir results/from_gene_file \
    --inference-mode hybrid \
    --verbose
```

## Best Practices

### Model Selection
Always use the most recent trained model by checking run numbers:
```bash
# Find the most recent model
ls -la results/gene_cv_pc_1000_3mers_run_*/model_multiclass.pkl | tail -1
```

### Performance Optimization
- Use `--inference-mode hybrid` for optimal balance of accuracy and efficiency
- Process multiple genes in a single run when possible
- Monitor meta-model usage percentage to understand uncertainty patterns

### Output Analysis
Results are saved in organized directory structure:
```
results/scenario_name/
├── selective_inference/
│   ├── predictions/
│   │   └── selective_inference_TIMESTAMP/
│   │       ├── complete_coverage_predictions.parquet
│   │       ├── meta_model_predictions.parquet
│   │       └── base_model_predictions.parquet
│   └── cache/
│       └── gene_manifests/
│           └── selective_inference_manifest.csv
```

## Troubleshooting

### Missing Artifacts
If you encounter "No analysis_sequences files available", the gene may not have pre-computed artifacts. Check:
```bash
# Verify gene has artifacts
grep "ENSG00000XXXXXX" data/ensembl/spliceai_eval/meta_models/analysis_sequences_*.tsv
```

### Feature Harmonization Warnings
Warnings about missing features are normal and handled automatically:
```
⚠️ Missing 10 non-kmer features: ['gene_start', 'transcript_length', ...]
```
The system fills missing features with appropriate defaults.

### Performance Issues
If processing is slow, consider:
- Using fewer genes per run
- Checking disk space for temporary files
- Monitoring memory usage during feature enrichment

## Validation

Both scenarios successfully demonstrate:
- ✅ **Enhanced Feature Matrix Generation**: Complete 124-feature matrices with all probability scores
- ✅ **Selective Meta-Model Inference**: Efficient computational usage (3.6-6.9% positions)
- ✅ **Mixed Predictions**: Seamless combination of base and meta-model predictions
- ✅ **Production Readiness**: Fast, reliable processing with comprehensive error handling