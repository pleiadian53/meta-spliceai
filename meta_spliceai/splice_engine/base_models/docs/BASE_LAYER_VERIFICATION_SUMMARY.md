# Base Layer Port Verification - Quick Summary

**Date**: December 12, 2025  
**Status**: Base model pass ✅ COMPLETE, Verification prompts ready

---

## Base Model Pass Status ✅

The OpenSpliceAI full genome production pass has **successfully completed**:

- **Status**: All chromosomes processed (1-22, X, Y)
- **Final Chromosome**: Chr Y completed  
- **Artifacts**: Located in `data/mane/GRCh38/openspliceai_eval/meta_models/`
- **Files Generated**: ~60+ chunk files + manifest
- **Process**: Completed with automatic checkpoint recovery

---

## What Was Created

I've created a comprehensive verification document:

**📄 `BASE_LAYER_PORT_VERIFICATION_PROMPTS.md`**

This document provides 7 systematic verification prompts to validate that the ported base layer in `agentic-spliceai` can fully replicate `meta-spliceai`'s functionality.

---

## Main Entry Points Identified

### Meta-SpliceAI (Source)

1. **Python API**: `meta_spliceai/run_base_model.py`
   - Function: `run_base_model_predictions()`
   - Use case: Programmatic access

2. **CLI**: `meta_spliceai/cli/run_base_model_cli.py`  
   - Command: `meta-spliceai-run`
   - Use case: Command-line interface

3. **Shell Script**: `scripts/training/process_chromosomes_sequential_smart.sh`
   - Use case: Sequential chromosome processing with checkpointing

**Core Workflow**: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`
- Function: `run_enhanced_splice_prediction_workflow()` ⭐ **Main orchestrator**

### Agentic-SpliceAI (Target)

Current structure:
```
agentic-spliceai/src/agentic_spliceai/splice_engine/base_layer/
├── models/
│   ├── config.py ✅       # Config classes ported
│   └── runner.py ⚠️       # Shows TODO - needs workflow integration
├── prediction/
│   ├── core.py ✅         # Core prediction ported
│   └── evaluation.py ❓   # Needs verification
├── data/ ✅               # Data types ported
└── io/ ❓                 # Needs verification
```

---

## 7 Verification Prompts Overview

### 1. Configuration System Validation ✅
**Tests**: Config creation for SpliceAI and OpenSpliceAI  
**Verifies**: Path auto-resolution, build detection, annotation source mapping

### 2. Core Prediction Function Validation ⚠️
**Tests**: `predict_splice_sites_for_genes()` output  
**Verifies**: Model loading, prediction generation, probability calculations  
**Compares**: Against meta-spliceai reference output

### 3. Full Workflow Orchestration ❌
**Tests**: End-to-end workflow execution  
**Current Gap**: Workflow module not yet ported  
**Critical**: This is the **main missing piece**

### 4. Artifact Generation Validation ⚠️
**Tests**: Output file schemas (positions, sequences, manifest, errors)  
**Verifies**: TSV files match expected format  
**Reference**: Use completed base model pass artifacts

### 5. Full Coverage Mode (Nucleotide Scores) ❌
**Tests**: Per-nucleotide score generation (for meta-model training)  
**Critical**: Required for training data generation  
**Verifies**: All positions have donor/acceptor/neither scores

### 6. Multi-Chromosome Production Run ❌
**Tests**: Chromosome-level processing with checkpoint recovery  
**Verifies**: Chunk files, manifest, resume functionality

### 7. Cross-Model Consistency ⚠️
**Tests**: Running both SpliceAI and OpenSpliceAI  
**Verifies**: Gene ID mapping between builds, both models execute

**Legend**:
- ✅ Likely working (code ported)
- ⚠️ Partially working (needs testing/verification)
- ❌ Not working (missing implementation)

---

## Critical Gaps Identified

### 1. Missing Workflow Module (Highest Priority) 🔴

**What's Missing**:
```python
agentic_spliceai/splice_engine/base_layer/workflows/
└── prediction_workflow.py  # Needs to be created
    └── run_enhanced_prediction_workflow()
```

**Source to Port**:
```
meta_spliceai/splice_engine/meta_models/workflows/
├── splice_prediction_workflow.py  ⭐ Main workflow (~1200 lines)
├── data_preparation.py            ⭐ Data loading/prep (~800 lines)
└── sequence_data_utils.py         ⭐ Sequence extraction (~600 lines)
```

**Why Critical**: This is the **orchestrator** that:
- Loads annotations and sequences
- Runs predictions in mini-batches (memory efficiency)
- Evaluates predictions (TP/FP/FN/TN)
- Generates artifacts (TSV files)
- Handles chunk-level checkpointing

### 2. Runner Integration 🟡

**Current State**:
```python
# BaseModelRunner.run_single_model() shows:
# TODO: Call actual prediction workflow when ported
return BaseModelResult(success=False, error="Prediction workflow not yet ported")
```

**Needs**: Integration with workflow (once ported)

### 3. Artifact I/O Handlers 🟡

**Likely Needed**:
- `MetaModelDataHandler` (save chunk-level TSV files)
- `ModelEvaluationFileHandler` (save error analysis)
- Gene manifest generation

### 4. Evaluation Module 🟡

**Location**: `agentic_spliceai/splice_engine/base_layer/prediction/evaluation.py`

**Needs Verification**:
- TP/FP/FN/TN classification logic
- Error analysis (distance to nearest splice site)
- Consensus calling

### 5. Data Preparation Utilities 🟡

**Likely Needed**:
- Load and filter GTF/GFF3 annotations
- Extract splice sites from annotations
- Extract gene sequences from FASTA
- Detect overlapping genes
- Schema standardization (Ensembl vs RefSeq column names)

---

## Recommended Porting Strategy

### Phase 1: Core Workflow (2-4 hours)
1. Port `splice_prediction_workflow.py` → create `workflows/prediction_workflow.py`
2. Port `data_preparation.py` → create `workflows/data_preparation.py`
3. Port `enhanced_workflow.py` → create `workflows/enhanced_evaluation.py`

**Critical Functions**:
- `run_enhanced_splice_prediction_workflow()` - main orchestrator
- `prepare_gene_annotations()` - load GTF/GFF3
- `prepare_splice_site_annotations()` - extract splice sites
- `prepare_genomic_sequences()` - extract gene sequences
- `enhanced_process_predictions_with_all_scores()` - evaluation

### Phase 2: Artifact Generation (1-2 hours)
4. Port `io/handlers.py` → artifact I/O
5. Implement chunk-level checkpointing
6. Implement gene manifest generation

### Phase 3: Integration (1 hour)
7. Wire up `BaseModelRunner.run_single_model()` to workflow
8. Test end-to-end with small gene set

### Phase 4: Validation (2-3 hours)
9. Run all 7 verification prompts
10. Compare outputs with reference artifacts
11. Fix discrepancies

### Phase 5: Documentation (1 hour)
12. Update agentic-spliceai README
13. Create usage examples
14. Document API

**Total Estimated Time**: 7-11 hours

---

## How to Use the Verification Prompts

### For AI Agents

Copy each prompt from `BASE_LAYER_PORT_VERIFICATION_PROMPTS.md` and paste into your AI assistant with this context:

```
I'm porting the base layer from meta-spliceai to agentic-spliceai.
Please help me verify [Prompt X: Description].

Context:
- Source system: /Users/pleiadian53/work/meta-spliceai
- Target system: /Users/pleiadian53/work/agentic-spliceai
- Reference artifacts: meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/

[Paste verification prompt here]
```

### For Developers

1. **Start with Prompt 1** (Configuration) - should work immediately
2. **Run Prompt 2** (Core Prediction) - tests basic prediction function
3. **Identify gaps** from Prompt 3 (Workflow) - this will show what needs porting
4. **Port missing modules** using porting guide
5. **Run Prompts 4-7** after implementation
6. **Compare outputs** with reference artifacts

---

## Reference Artifacts for Comparison

Use the completed base model pass as ground truth:

**Location**: `meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/`

**Key Files**:
- `analysis_positions_21_chunk_1_500.tsv` - Small chromosome for testing
- `analysis_sequences_21_chunk_1_500.tsv` - Sequence windows
- `gene_manifest.tsv` - Processing metadata (when complete)

**Usage**:
```python
import polars as pl

# Load reference
ref_positions = pl.read_csv(
    'meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/analysis_positions_21_chunk_1_500.tsv',
    separator='\t'
)

# Check schema
print("Columns:", ref_positions.columns)
print("Shape:", ref_positions.shape)
print("Pred types:", ref_positions['pred_type'].value_counts())
```

---

## Key Modes to Support

### 1. Regular Prediction Mode
- **Purpose**: Identify splice sites (donors and acceptors)
- **Output**: TP/FP/FN splice site positions
- **TN Sampling**: Enabled (reduces output size)
- **Nucleotide Scores**: Optional (usually disabled)

### 2. Full Coverage Mode
- **Purpose**: Generate training data for meta-models
- **Output**: Per-nucleotide scores for ALL positions
- **TN Sampling**: Disabled (keep all positions)
- **Nucleotide Scores**: **Required** (this is the key artifact)
- **Critical**: Must generate donor/acceptor/neither scores for every nucleotide

---

## Next Steps

1. **Read**: `BASE_LAYER_PORT_VERIFICATION_PROMPTS.md` (full prompts)
2. **Check**: Current agentic-spliceai implementation status
3. **Identify**: Which prompts pass vs fail
4. **Port**: Missing modules (follow porting guide: `AI_AGENT_PORTING_GUIDE.md`)
5. **Verify**: Run all 7 prompts
6. **Document**: Update agentic-spliceai docs

---

## Related Documentation

### For Porting
- **AI_AGENT_PORTING_GUIDE.md** - Stage-by-stage porting instructions (995 lines)
- **BASE_LAYER_INTEGRATION_GUIDE.md** - Integration patterns and examples (870 lines)

### For Understanding
- **DATA_LAYOUT_MASTER_GUIDE.md** - Data organization conventions
- **BASE_MODEL_DATA_MAPPING.md** - Model-to-build mapping
- **UNIVERSAL_BASE_MODEL_SUPPORT.md** - Extensibility design

### For Context
- **SESSION_COMPLETE_20251201_CONFIG_AND_DOCS.md** - Latest session summary
- **CONFIG_REFACTORING_COMPLETE.md** - Config system design

---

## Questions to Answer

After running verification prompts, you should be able to answer:

1. ✅ Can agentic-spliceai create configs for both SpliceAI and OpenSpliceAI?
2. ❓ Can it run predictions on a single gene (e.g., BRCA1)?
3. ❓ Can it evaluate predictions (TP/FP/FN/TN)?
4. ❓ Can it generate artifact files matching meta-spliceai's schema?
5. ❓ Can it generate full coverage nucleotide scores (for meta-learning)?
6. ❓ Can it process multiple chromosomes with checkpoint recovery?
7. ❓ Can it handle both SpliceAI (GRCh37) and OpenSpliceAI (GRCh38)?

**Target**: All questions answered with ✅

---

*Created: December 12, 2025*  
*For: Verification of agentic-spliceai base layer port from meta-spliceai*

