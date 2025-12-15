# Inference Workflow to Variant Analysis Bridge

**Created:** 2025-10-28  
**Purpose:** Connect the new inference workflow with existing case studies infrastructure  
**Status:** 🔗 Integration Guide

## Overview

This document bridges **two major components** of MetaSpliceAI:

1. **New Inference Workflow** (`meta_spliceai/splice_engine/meta_models/workflows/inference/`)
   - Enhanced selective inference with 3 operational modes
   - Metadata-aware predictions
   - Production-ready as of 2025-10-28

2. **Existing Case Studies Infrastructure** (`meta_spliceai/splice_engine/case_studies/`)
   - Comprehensive variant analysis pipelines
   - Delta score computation via OpenSpliceAI bridge
   - VCF processing and ClinVar integration
   - Built August-September 2025

## Key Insight

**The existing case studies infrastructure already has:**
- ✅ Delta score computation (`delta_score_workflow.py`)
- ✅ VCF processing (`universal_vcf_parser.py`, `vcf_preprocessing.py`)
- ✅ OpenSpliceAI integration (`openspliceai_delta_bridge.py`)
- ✅ ClinVar pipeline (`complete_clinvar_pipeline.py`)
- ✅ Alternative splice site detection (`alternative_splicing_pipeline.py`)
- ✅ Cryptic site detection (`cryptic_site_detector.py`)

**What's new in the inference workflow:**
- ✅ Meta-model integration (3 modes: base, hybrid, meta-only)
- ✅ Metadata preservation (uncertainty, confidence, etc.)
- ✅ Production-ready inference on arbitrary genes
- ✅ Memory-efficient incremental processing

## Integration Strategy

### Option 1: Use Existing Case Studies for Variants (RECOMMENDED)

**When to use:** For variant analysis, VCF processing, delta scores

**Workflow:**
```python
# 1. Use case studies for variant processing
from meta_spliceai.splice_engine.case_studies.workflows.delta_score_workflow import (
    DeltaScoreWorkflow
)

workflow = DeltaScoreWorkflow(
    vcf_path='clinvar.vcf.gz',
    reference_genome='GRCh38.fa'
)

# Compute delta scores using existing infrastructure
delta_results = workflow.compute_delta_scores()
```

**Benefits:**
- Mature, tested infrastructure
- OpenSpliceAI integration already working
- VCF normalization and preprocessing built-in
- ClinVar compatibility verified

### Option 2: Extend Inference Workflow for Variants (FUTURE)

**When to use:** For meta-model enhanced variant analysis

**Proposed workflow:**
```python
# 1. Use inference workflow for base predictions
from meta_spliceai.splice_engine.meta_models.workflows.inference import (
    EnhancedSelectiveInferenceWorkflow
)

ref_workflow = EnhancedSelectiveInferenceWorkflow(config_ref)
ref_predictions = ref_workflow.run_incremental()

# 2. Apply variant to sequence (NEW - needs implementation)
variant_sequence = apply_variant_to_sequence(ref_sequence, variant)

# 3. Use inference workflow for variant predictions
var_workflow = EnhancedSelectiveInferenceWorkflow(config_var)
var_predictions = var_workflow.predict_custom_sequence(variant_sequence)

# 4. Compute metadata-aware delta scores (NEW)
from meta_spliceai.splice_engine.meta_models.workflows.inference.delta_scores import (
    MetaModelDeltaScores
)

delta_result = MetaModelDeltaScores.compute(
    ref_predictions=ref_predictions,
    var_predictions=var_predictions,
    preserve_metadata=True  # NEW: confidence-weighted deltas
)
```

**Benefits:**
- Meta-model calibration in variant analysis
- Metadata-aware delta scores (uncertainty, confidence)
- Selective meta-model application to variants
- Consistent with inference workflow architecture

## Architecture Comparison

### Case Studies Delta Scores (Existing)

**Flow:**
```
VCF → Variant Standardization → OpenSpliceAI Prediction 
→ Delta Score Computation → Alternative Splice Sites
```

**Key Components:**
- `universal_vcf_parser.py` - VCF parsing and normalization
- `openspliceai_delta_bridge.py` - Bridge to OpenSpliceAI
- `delta_score_workflow.py` - End-to-end pipeline
- `alternative_splicing_pipeline.py` - Splice site extraction

**Strengths:**
- ✅ Production-ready VCF processing
- ✅ Tested on ClinVar variants
- ✅ Robust error handling
- ✅ Comprehensive documentation

### Inference Workflow (New)

**Flow:**
```
Gene ID → Base Model Prediction → Meta-Model Calibration 
→ Metadata-Enriched Predictions
```

**Key Components:**
- `enhanced_selective_inference.py` - Main inference driver
- `genomic_feature_enricher.py` - Feature enrichment
- `uncertainty_detector.py` - Uncertainty quantification
- `meta_model_applicator.py` - Selective meta-model

**Strengths:**
- ✅ Meta-model integration (3 modes)
- ✅ Metadata preservation
- ✅ Memory-efficient
- ✅ Gene-centric workflow

## Integration Points

### Point 1: Sequence-Based Prediction

**Bridge:** The inference workflow needs to support arbitrary sequences (not just gene IDs)

**Current State:** Case studies already have this via `sequence_predictor.py`

**Action:** Extend inference workflow to accept custom sequences

**Implementation:**
```python
# In enhanced_selective_inference.py
def predict_custom_sequence(
    self,
    sequence: str,
    coordinates: dict,  # {'chr': '17', 'start': 123, 'end': 456}
    gene_context: Optional[dict] = None
) -> pl.DataFrame:
    """
    Predict splice sites for arbitrary sequence
    
    Enables:
    - Variant analysis (WT vs ALT sequences)
    - Synthetic sequence testing
    - Custom region analysis
    """
    # Use existing base model prediction logic
    # Add meta-model application
    # Preserve metadata
```

### Point 2: Delta Score Computation

**Bridge:** Connect meta-model predictions with delta score logic

**Current State:**
- Case studies: OpenSpliceAI delta scores (base model only)
- Inference: Meta-model predictions with metadata

**Opportunity:** **Metadata-aware delta scores**

**Implementation:**
```python
class MetaModelDeltaScores:
    """
    Compute delta scores with meta-model calibration
    and uncertainty quantification
    """
    
    @staticmethod
    def compute(
        ref_predictions: pl.DataFrame,
        var_predictions: pl.DataFrame,
        preserve_metadata: bool = True
    ) -> DeltaScoreResult:
        """
        Compute delta scores with optional metadata
        
        Returns:
        - delta_donor, delta_acceptor (standard)
        - confidence_weighted_delta (NEW)
        - uncertainty_flags (NEW)
        - meta_model_applied (NEW)
        """
        
        # Standard delta
        delta_df = var_predictions.join(ref_predictions, on='position', suffix='_ref')
        delta_df = delta_df.with_columns([
            (pl.col('donor_score') - pl.col('donor_score_ref')).alias('delta_donor'),
            (pl.col('acceptor_score') - pl.col('acceptor_score_ref')).alias('delta_acceptor')
        ])
        
        if preserve_metadata:
            # Confidence-weighted delta
            delta_df = delta_df.with_columns([
                (pl.col('delta_donor') * pl.col('max_confidence')).alias('weighted_delta_donor'),
                (pl.col('delta_acceptor') * pl.col('max_confidence')).alias('weighted_delta_acceptor')
            ])
            
            # Uncertainty flags
            delta_df = delta_df.with_columns([
                (pl.col('is_uncertain') | pl.col('is_uncertain_ref')).alias('uncertain_position')
            ])
        
        return DeltaScoreResult(delta_df)
```

### Point 3: VCF Integration

**Bridge:** Process VCF variants through inference workflow

**Current State:** Case studies handle VCF → Delta Scores

**Action:** Add VCF input support to inference workflow

**Implementation:**
```python
class VariantInferenceWorkflow:
    """
    Extends EnhancedSelectiveInferenceWorkflow for variant analysis
    """
    
    def __init__(self, config: EnhancedSelectiveInferenceConfig):
        self.base_workflow = EnhancedSelectiveInferenceWorkflow(config)
        self.vcf_parser = UniversalVCFParser()  # From case studies
        
    def process_vcf(
        self,
        vcf_path: str,
        gene_filter: Optional[List[str]] = None
    ) -> List[DeltaScoreResult]:
        """
        Process VCF variants with meta-model
        
        Workflow:
        1. Parse VCF (reuse case studies)
        2. For each variant:
           a. Get reference predictions (inference workflow)
           b. Apply variant to sequence
           c. Get variant predictions (inference workflow)
           d. Compute metadata-aware delta scores
        """
        
        # Parse VCF using existing infrastructure
        variants = self.vcf_parser.parse(vcf_path, gene_filter=gene_filter)
        
        results = []
        for variant in variants:
            # Get gene region
            gene_id = variant['gene_id']
            
            # Reference predictions
            ref_pred = self.base_workflow.run_incremental(gene_ids=[gene_id])
            
            # Apply variant
            var_seq = self._apply_variant(variant)
            
            # Variant predictions
            var_pred = self.base_workflow.predict_custom_sequence(
                sequence=var_seq,
                coordinates=variant['coordinates']
            )
            
            # Compute delta with metadata
            delta = MetaModelDeltaScores.compute(ref_pred, var_pred)
            
            results.append(delta)
        
        return results
```

## Recommended Actions

### Phase 1: Documentation Consolidation (Immediate)

1. ✅ **Create this bridge document** - Connect two systems
2. **Update case studies README** - Point to new inference workflow
3. **Update inference docs** - Reference case studies for variants
4. **Cross-reference** - Add links between documentation sets

### Phase 2: Light Integration (1-2 weeks)

1. **Extend inference for custom sequences**
   - Add `predict_custom_sequence()` method
   - Support arbitrary DNA sequences
   - Maintain metadata throughout

2. **Create simple delta score utility**
   - Reuse case studies for VCF processing
   - Add metadata-aware delta computation
   - Connect outputs to case studies pipelines

### Phase 3: Full Integration (4-6 weeks)

1. **Unified variant analysis workflow**
   - Single entry point for variant analysis
   - Choose base vs meta-model mode
   - Preserve metadata through entire pipeline

2. **Enhanced clinical interpretation**
   - Confidence-weighted pathogenicity scores
   - Uncertainty-flagged variants for manual review
   - Meta-model improvement quantification

### Phase 4: Production Deployment (8-10 weeks)

1. **Complete VCF pipeline with meta-model**
   - Batch processing of VCF files
   - ClinVar integration
   - Clinical report generation

2. **Performance optimization**
   - Caching strategies
   - Parallel processing
   - Resource management

## File Organization Recommendations

### Current Structure (Keep)
```
meta_spliceai/
├── splice_engine/
│   ├── case_studies/              # Variant analysis (existing)
│   │   ├── workflows/
│   │   │   ├── delta_score_workflow.py
│   │   │   ├── openspliceai_delta_bridge.py
│   │   │   ├── complete_clinvar_pipeline.py
│   │   │   └── ...
│   │   └── docs/
│   │       ├── DELTA_SCORE_BRIDGE_IMPLEMENTATION.md
│   │       ├── VCF_VARIANT_ANALYSIS_WORKFLOW.md
│   │       └── ...
│   └── meta_models/
│       └── workflows/
│           └── inference/         # Gene inference (new)
│               ├── enhanced_selective_inference.py
│               ├── genomic_feature_enricher.py
│               └── ...
```

### Proposed Additions
```
meta_spliceai/splice_engine/meta_models/workflows/
├── inference/
│   ├── enhanced_selective_inference.py  # Existing
│   ├── variant_inference.py              # NEW: Variant extension
│   ├── delta_scores.py                   # NEW: Meta-aware deltas
│   └── sequence_predictor.py             # NEW: Custom sequences
```

## Document Updates Needed

### 1. Update `docs/VARIANT_ANALYSIS_USE_CASES.md`

**Changes:**
- Add section: "Existing Implementation in Case Studies"
- Reference `case_studies/docs/DELTA_SCORE_BRIDGE_IMPLEMENTATION.md`
- Clarify: "Use case studies for VCF processing"
- Add: "Future: Meta-model enhanced variant analysis"

### 2. Update `docs/DELTA_SCORE_QUICK_START.md`

**Changes:**
- Add: "Quick Start: Using Existing Infrastructure"
```python
from meta_spliceai.splice_engine.case_studies.workflows import DeltaScoreWorkflow

# Use existing, production-ready pipeline
workflow = DeltaScoreWorkflow(vcf='clinvar.vcf.gz')
results = workflow.compute_delta_scores()
```

### 3. Update `case_studies/docs/README.md`

**Changes:**
- Add section: "Integration with Meta-Model Inference"
- Reference new inference workflow
- Clarify relationship between components

## Summary

### What Exists ✅
- **Case Studies:** Complete VCF → Delta Scores pipeline
- **Inference:** Complete Gene → Meta-Model predictions

### What's Missing 🔧
- Bridge between inference and variant analysis
- Custom sequence support in inference workflow
- Metadata-aware delta scores

### Recommended Approach 🎯

**SHORT TERM (Use existing):**
```python
# For variant analysis, use case studies
from meta_spliceai.splice_engine.case_studies.workflows import DeltaScoreWorkflow
results = DeltaScoreWorkflow(vcf='variants.vcf.gz').run()
```

**LONG TERM (Extend inference):**
```python
# For meta-model enhanced variants
from meta_spliceai.splice_engine.meta_models.workflows.inference import (
    VariantInferenceWorkflow
)
results = VariantInferenceWorkflow(config).process_vcf('variants.vcf.gz')
```

---

**Next Steps:**
1. ✅ This bridge document created
2. Update `VARIANT_ANALYSIS_USE_CASES.md` to reference case studies
3. Update `DELTA_SCORE_QUICK_START.md` with existing workflow
4. Test integration between components
5. Implement `variant_inference.py` extension

**Status:** Documentation bridge complete, code integration pending


