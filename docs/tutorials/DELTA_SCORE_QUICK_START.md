# Delta Score Analysis: Quick Start Guide

**Created:** 2025-10-28  
**Purpose:** Practical guide for implementing variant analysis with delta scores

## What Are Delta Scores?

**Delta scores (Δ-scores)** quantify how a genetic variant changes splice site predictions:

```
Δ_donor[position] = donor_score_variant[position] - donor_score_reference[position]
```

### Interpretation Table

| Δ-score | Interpretation | Example |
|---------|----------------|---------|
| **> +0.2** | Creates new splice site | Cryptic donor gain |
| **+0.05 to +0.2** | Strengthens splice site | Enhanced acceptor |
| **-0.05 to +0.05** | Neutral | No effect |
| **-0.2 to -0.05** | Weakens splice site | Reduced efficiency |
| **< -0.2** | Destroys splice site | Canonical site loss |

---

## Quick Example

### Input
```
Gene: TP53 (ENSG00000141510)
Variant: chr17:7,579,312 C>T
```

### Steps

1. **Get reference predictions:**
```python
from meta_spliceai.splice_engine.meta_models.workflows.inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)

# Predict reference sequence
config_ref = EnhancedSelectiveInferenceConfig(
    target_genes=['ENSG00000141510'],
    inference_mode='hybrid',
    model_path='results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'
)

workflow = EnhancedSelectiveInferenceWorkflow(config_ref)
ref_results = workflow.run_incremental()
```

2. **Apply variant (in silico mutagenesis):**
```python
# Load reference sequence
ref_seq = load_gene_sequence('ENSG00000141510', start=7579300, end=7579400)

# Apply substitution
variant_seq = ref_seq[:11] + 'T' + ref_seq[12:]  # C>T at position 12
```

3. **Get variant predictions:**
```python
# Predict variant sequence (same config, different sequence input)
var_results = workflow.predict_custom_sequence(
    sequence=variant_seq,
    coordinates={'chr': '17', 'start': 7579300, 'end': 7579400}
)
```

4. **Compute delta scores:**
```python
import polars as pl

# Load predictions
ref_pred = pl.read_parquet(ref_results.predictions_path)
var_pred = pl.read_parquet(var_results.predictions_path)

# Compute deltas
delta_df = ref_pred.join(var_pred, on='position', suffix='_var')
delta_df = delta_df.with_columns([
    (pl.col('donor_score_var') - pl.col('donor_score')).alias('delta_donor'),
    (pl.col('acceptor_score_var') - pl.col('acceptor_score')).alias('delta_acceptor')
])

# Find high-impact positions
high_impact = delta_df.filter(
    (pl.col('delta_donor').abs() > 0.2) |
    (pl.col('delta_acceptor').abs() > 0.2)
)

print(f"Found {len(high_impact)} positions with high impact")
```

---

## Use Case: Clinical Variant Interpretation

### Scenario
Patient has undiagnosed disease. Candidate variant found in disease gene.

### Question
**Does this variant disrupt splicing?**

### Answer via Delta Scores

#### Example 1: Canonical Site Loss
```
Variant: chr17:7,579,312 C>T (in TP53 intron 7 donor site)

Delta scores:
  Position 7,579,312:
    Δ_donor = -0.85  ← MAJOR LOSS
    Δ_acceptor = +0.02  ← No change
    
Interpretation: 
  - Loss of canonical donor site
  - Likely causes exon skipping
  - Classification: Pathogenic
```

#### Example 2: Cryptic Site Gain
```
Variant: chr17:7,577,548 G>A (in TP53 exon 8)

Delta scores:
  Position 7,577,550:
    Δ_donor = +0.65  ← NEW DONOR SITE
    Δ_acceptor = -0.01  ← No change
    
Interpretation:
  - Creates cryptic donor in exon
  - May cause partial exon inclusion
  - Classification: Likely pathogenic
```

#### Example 3: Neutral Variant
```
Variant: chr17:7,579,800 A>G (in TP53 intron, far from splice sites)

Delta scores:
  All positions:
    |Δ_donor| < 0.05
    |Δ_acceptor| < 0.05
    
Interpretation:
  - No splice site disruption
  - Classification: Likely benign
```

---

## Metadata Features for Variant Analysis

### Why Metadata Matters

The 9 metadata features help identify:
1. **Positions prone to cryptic splicing** (`is_uncertain`, `is_low_discriminability`)
2. **Confidence in delta score** (`max_confidence`, `confidence_category`)
3. **Alternative splicing potential** (`score_entropy`, `is_high_entropy`)

### Example: Uncertainty-Weighted Delta Scores

```python
# Compute delta scores with uncertainty weighting
delta_df = delta_df.with_columns([
    # Weighted delta (down-weight uncertain predictions)
    (pl.col('delta_donor') * pl.col('max_confidence')).alias('weighted_delta_donor'),
    
    # Flag high-risk positions (high delta + low confidence)
    (
        (pl.col('delta_donor').abs() > 0.2) &
        (pl.col('max_confidence') < 0.7)
    ).alias('high_risk_position')
])

# Prioritize high-confidence, high-impact variants
priority_variants = delta_df.filter(
    (pl.col('delta_donor').abs() > 0.2) &
    (pl.col('max_confidence') > 0.8) &
    (pl.col('confidence_category') == 'high')
)
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Implement `VariantSplicePredictor` class
- [ ] Add in silico mutagenesis for SNVs
- [ ] Implement delta score computation
- [ ] Create visualization functions

### Phase 2: VCF Integration (Week 3-4)
- [ ] VCF file parser
- [ ] Batch variant processing
- [ ] Indel support (insertions/deletions)
- [ ] Multi-sample support

### Phase 3: Clinical Features (Week 5-6)
- [ ] ACMG classification integration
- [ ] ClinVar annotation
- [ ] Pathogenicity prediction
- [ ] HTML/PDF report generation

### Phase 4: Advanced Analysis (Week 7-8)
- [ ] Meta-model delta scores (hybrid/meta-only modes)
- [ ] Uncertainty propagation
- [ ] Alternative isoform prediction
- [ ] Interactive visualization dashboard

---

## Key Design Decisions

### 1. Preserve Existing Workflow
```python
# Reference predictions (unchanged)
ref_workflow = EnhancedSelectiveInferenceWorkflow(config)
ref_results = ref_workflow.run_incremental()

# Variant predictions (new method)
var_results = ref_workflow.predict_with_variant(
    variant={'chr': '17', 'pos': 7579312, 'ref': 'C', 'alt': 'T'}
)

# Delta computation (new utility)
delta_result = compute_delta_scores(ref_results, var_results)
```

### 2. Metadata Throughout Pipeline
```python
class DeltaScoreResult:
    reference_pred: pl.DataFrame  # With metadata
    variant_pred: pl.DataFrame    # With metadata
    delta_scores: pl.DataFrame    # With inherited metadata
    
    # Metadata enables:
    # - Confidence-weighted deltas
    # - Uncertainty-aware interpretation
    # - Meta-model calibration tracking
```

### 3. Modular Components
```
┌─────────────────────────────┐
│  VariantSplicePredictor     │
└──────────┬──────────────────┘
           │
           ├─> InSilicoMutator (sequence manipulation)
           ├─> EnhancedSelectiveInferenceWorkflow (predictions)
           ├─> DeltaScoreComputer (score computation)
           └─> ClinicalAnnotator (pathogenicity)
```

---

## Next Steps

1. **Run current test:** Verify metadata preservation across diverse genes ✅ (in progress)
2. **Prototype delta scorer:** Implement basic SNV delta score computation
3. **Validate on known pathogenic variants:** Test on ClinVar splice variants
4. **Extend to indels:** Support insertions/deletions
5. **Build VCF pipeline:** Batch processing for clinical use

---

## Expected Timeline

| Milestone | Duration | Deliverable |
|-----------|----------|-------------|
| **Basic Delta Scores** | 1-2 weeks | SNV delta score computation |
| **VCF Integration** | 2-3 weeks | Batch variant processing |
| **Clinical Annotation** | 2-3 weeks | Pathogenicity classification |
| **Production Ready** | 3-4 weeks | Full pipeline + documentation |

**Total: ~8-10 weeks to production-ready variant analysis**

---

## Summary

✅ **Current Inference Workflow Provides:**
- Position-wise splice site predictions (3 modes)
- Uncertainty quantification (6/9 metadata features)
- Meta-model calibration

🚀 **Delta Score Extension Enables:**
- Quantitative variant impact assessment
- Clinical pathogenicity prediction
- Batch VCF processing
- Uncertainty-aware variant interpretation

📊 **Key Innovation:**
**Metadata-aware delta scores** = More confident clinical decisions

---

**Status:** Design complete, awaiting implementation  
**Priority:** High (clinical utility)  
**Complexity:** Medium (builds on existing workflow)

