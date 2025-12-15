# Variant Analysis Use Cases & Delta Score Strategy

**Created:** 2025-10-28  
**Status:** 🎯 Design Document for Next Phase  
**⚠️ IMPORTANT:** See existing implementation in `meta_spliceai/splice_engine/case_studies/`

## ⚡ Quick Start: Use Existing Infrastructure

**For VCF variant analysis and delta scores, use the existing case studies infrastructure:**

```python
from meta_spliceai.splice_engine.case_studies.workflows import DeltaScoreWorkflow

# Production-ready delta score computation
workflow = DeltaScoreWorkflow(
    vcf_path='clinvar.vcf.gz',
    reference_genome='GRCh38.fa'
)
results = workflow.compute_delta_scores()
```

**See:** `meta_spliceai/splice_engine/case_studies/docs/DELTA_SCORE_BRIDGE_IMPLEMENTATION.md`

**This document** describes how to extend the new meta-model inference workflow for variant analysis with metadata-aware delta scores.

---

## Overview

This document outlines how the current inference workflow can be extended to support **variant analysis** through **delta scores** (Δ-scores), enabling prediction of alternative splice sites caused by genetic variants.

**Key Integration:** This builds on the existing case studies infrastructure (August-September 2025) by adding meta-model calibration and metadata awareness to variant analysis.

## Table of Contents

1. [Core Concept: Delta Scores](#core-concept-delta-scores)
2. [Use Cases for Current Predictions](#use-cases-for-current-predictions)
3. [Variant Analysis Workflow](#variant-analysis-workflow)
4. [Delta Score Implementation](#delta-score-implementation)
5. [Clinical Applications](#clinical-applications)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Core Concept: Delta Scores

### Definition

**Delta Score (Δ-score)** = Score<sub>variant</sub> - Score<sub>reference</sub>

Where:
- **Score<sub>reference</sub>**: Splice site probability at position *i* using reference genome sequence
- **Score<sub>variant</sub>**: Splice site probability at position *i* with variant allele substituted

### Interpretation

| Δ-score Range | Interpretation | Clinical Impact |
|---------------|----------------|-----------------|
| **Δ > +0.2** | Gained splice site | Creates cryptic splice site |
| **+0.05 < Δ ≤ +0.2** | Enhanced splice site | Strengthens existing site |
| **-0.05 ≤ Δ ≤ +0.05** | Neutral | No significant effect |
| **-0.2 ≤ Δ < -0.05** | Weakened splice site | Reduces splice efficiency |
| **Δ < -0.2** | Lost splice site | Disrupts canonical splicing |

### Why Delta Scores?

1. **Quantifies variant impact** - Measures how much a variant perturbs splicing
2. **Position-specific** - Tracks impact at every nucleotide in the region
3. **Direction-aware** - Distinguishes gain vs. loss of splice function
4. **Interpretable** - Directly relates to splice probability change

---

## Use Cases for Current Predictions

### Use Case 1: Baseline Splice Site Mapping

**Scenario:** Identify all potential splice sites in a gene region

**Current Capability:**
```python
# Run base-only mode to get canonical splice sites
predictions = run_inference(
    gene_id='ENSG00000134202',
    mode='base_only'
)

# Filter high-confidence splice sites
canonical_donors = predictions.filter(
    (pl.col('donor_score') > 0.5) &
    (pl.col('splice_type') == 'donor')
)

canonical_acceptors = predictions.filter(
    (pl.col('acceptor_score') > 0.5) &
    (pl.col('splice_type') == 'acceptor')
)
```

**Output:**
- Complete splice site landscape for gene
- Confidence scores for each site
- Metadata for uncertainty quantification

**Applications:**
- Reference map for variant analysis
- Baseline for delta score computation
- Quality control for annotations

---

### Use Case 2: Uncertainty-Aware Predictions

**Scenario:** Identify positions where model is uncertain (potential cryptic sites)

**Current Capability:**
```python
# Run hybrid mode to identify uncertain positions
predictions = run_inference(
    gene_id='ENSG00000134202',
    mode='hybrid'
)

# Find uncertain positions
uncertain_positions = predictions.filter(
    pl.col('is_uncertain') == True
)

# Analyze uncertainty characteristics
uncertain_analysis = uncertain_positions.group_by('confidence_category').agg([
    pl.count().alias('count'),
    pl.col('max_confidence').mean().alias('avg_confidence'),
    pl.col('score_entropy').mean().alias('avg_entropy'),
    pl.col('is_adjusted').mean().alias('meta_model_correction_rate')
])
```

**Output:**
- Positions with ambiguous splicing signals
- Positions where meta-model improves prediction
- Regions prone to alternative splicing

**Applications:**
- Target regions for variant analysis
- Cryptic splice site candidates
- Alternative splicing hotspots

---

### Use Case 3: Meta-Model Calibration Assessment

**Scenario:** Evaluate where meta-model improves base model predictions

**Current Capability:**
```python
# Run all 3 modes and compare
base_pred = run_inference(gene_id='ENSG00000134202', mode='base_only')
hybrid_pred = run_inference(gene_id='ENSG00000134202', mode='hybrid')
meta_pred = run_inference(gene_id='ENSG00000134202', mode='meta_only')

# Compare predictions
comparison = base_pred.join(hybrid_pred, on=['gene_id', 'position'], suffix='_hybrid')
comparison = comparison.join(meta_pred, on=['gene_id', 'position'], suffix='_meta')

# Identify positions where meta-model changed prediction
corrections = comparison.filter(
    (pl.col('splice_type') != pl.col('splice_type_hybrid')) |
    (pl.col('splice_type') != pl.col('splice_type_meta'))
)
```

**Output:**
- Positions where meta-model disagrees with base model
- Magnitude of score adjustments
- Success rate of meta-model corrections

**Applications:**
- Model quality assessment
- Understanding meta-model behavior
- Identifying regions needing manual review

---

## Variant Analysis Workflow

### Phase 1: Reference Genome Predictions (Current)

```
Input: Gene region (e.g., chr17:7,571,720-7,590,868 for TP53)
       ↓
Run Inference (Base/Hybrid/Meta-only)
       ↓
Output: Reference predictions with metadata
        - Position-wise scores (donor, acceptor, neither)
        - Uncertainty flags
        - Confidence metrics
```

### Phase 2: Variant Genome Predictions (New)

```
Input: Gene region + Variant (e.g., chr17:7,579,312 C>T)
       ↓
1. Load reference sequence
2. Apply variant to sequence (in silico mutagenesis)
3. Run inference on variant sequence
       ↓
Output: Variant predictions with metadata
```

### Phase 3: Delta Score Computation (New)

```
Input: Reference predictions + Variant predictions
       ↓
For each position i:
    Δ_donor[i] = donor_variant[i] - donor_ref[i]
    Δ_acceptor[i] = acceptor_variant[i] - acceptor_ref[i]
    Δ_neither[i] = neither_variant[i] - neither_ref[i]
       ↓
Output: Delta score profile
        - Position-wise Δ-scores
        - Affected splice site annotations
        - Clinical impact classification
```

---

## Delta Score Implementation

### Design Principles

1. **Minimal Modifications** - Reuse existing inference workflow
2. **Modular Architecture** - Separate variant handling from prediction
3. **Efficient Computation** - Only predict affected region (variant ± context window)
4. **Metadata Preservation** - Track uncertainty for both ref and variant

### Proposed Architecture

```python
class VariantSplicePredictor:
    """
    Extends EnhancedSelectiveInferenceWorkflow for variant analysis
    """
    
    def __init__(self, config: EnhancedSelectiveInferenceConfig):
        self.base_workflow = EnhancedSelectiveInferenceWorkflow(config)
        
    def predict_with_variant(
        self,
        gene_id: str,
        variant: Variant,  # chr, pos, ref, alt
        context_window: int = 5000
    ) -> DeltaScoreResult:
        """
        Predict delta scores for a variant
        
        Workflow:
        1. Get reference predictions for region
        2. Apply in silico mutagenesis
        3. Get variant predictions for region
        4. Compute delta scores
        5. Annotate affected splice sites
        """
        
        # Define prediction region (variant ± context_window)
        region_start = max(variant.pos - context_window, 1)
        region_end = variant.pos + context_window
        
        # 1. Reference predictions
        ref_predictions = self.base_workflow.run_incremental()
        ref_region = self._extract_region(ref_predictions, region_start, region_end)
        
        # 2. Apply variant
        variant_sequence = self._apply_variant_to_sequence(
            gene_id, variant, region_start, region_end
        )
        
        # 3. Variant predictions
        var_predictions = self._predict_with_custom_sequence(
            gene_id, variant_sequence, region_start, region_end
        )
        
        # 4. Compute delta scores
        delta_scores = self._compute_delta_scores(ref_region, var_predictions)
        
        # 5. Annotate impact
        impact = self._annotate_variant_impact(delta_scores, variant)
        
        return DeltaScoreResult(
            variant=variant,
            reference_predictions=ref_region,
            variant_predictions=var_predictions,
            delta_scores=delta_scores,
            impact=impact
        )
```

### Key Methods

#### 1. In Silico Mutagenesis

```python
def _apply_variant_to_sequence(
    self,
    gene_id: str,
    variant: Variant,
    region_start: int,
    region_end: int
) -> str:
    """
    Apply variant to reference sequence
    
    Steps:
    1. Load reference sequence for region
    2. Convert genomic position to sequence index
    3. Substitute ref allele with alt allele
    4. Validate substitution
    """
    
    # Load reference sequence
    ref_seq = self._load_gene_sequence(gene_id, region_start, region_end)
    
    # Convert position (handle strand orientation)
    gene_info = self._get_gene_info(gene_id)
    
    if gene_info['strand'] == '+':
        seq_idx = variant.pos - region_start
        alt_allele = variant.alt
    else:
        # Reverse complement for minus strand
        seq_idx = region_end - variant.pos
        alt_allele = self._reverse_complement(variant.alt)
    
    # Validate reference allele matches
    if ref_seq[seq_idx:seq_idx+len(variant.ref)] != variant.ref:
        raise ValueError(
            f"Reference mismatch at {variant.pos}: "
            f"expected {variant.ref}, got {ref_seq[seq_idx:seq_idx+len(variant.ref)]}"
        )
    
    # Apply substitution
    var_seq = (
        ref_seq[:seq_idx] +
        alt_allele +
        ref_seq[seq_idx + len(variant.ref):]
    )
    
    return var_seq
```

#### 2. Delta Score Computation

```python
def _compute_delta_scores(
    self,
    ref_predictions: pl.DataFrame,
    var_predictions: pl.DataFrame
) -> pl.DataFrame:
    """
    Compute delta scores: variant - reference
    
    Returns DataFrame with:
    - position
    - delta_donor, delta_acceptor, delta_neither
    - impact_category (gained/lost/neutral)
    - max_abs_delta (maximum absolute change)
    """
    
    # Join on position
    delta_df = ref_predictions.join(
        var_predictions,
        on='position',
        suffix='_var'
    )
    
    # Compute deltas
    delta_df = delta_df.with_columns([
        (pl.col('donor_score_var') - pl.col('donor_score')).alias('delta_donor'),
        (pl.col('acceptor_score_var') - pl.col('acceptor_score')).alias('delta_acceptor'),
        (pl.col('neither_score_var') - pl.col('neither_score')).alias('delta_neither'),
    ])
    
    # Compute max absolute delta
    delta_df = delta_df.with_columns([
        pl.max_horizontal([
            pl.col('delta_donor').abs(),
            pl.col('delta_acceptor').abs()
        ]).alias('max_abs_delta')
    ])
    
    # Classify impact
    delta_df = delta_df.with_columns([
        pl.when(pl.col('max_abs_delta') > 0.2)
          .then(pl.lit('high'))
          .when(pl.col('max_abs_delta') > 0.05)
          .then(pl.lit('moderate'))
          .otherwise(pl.lit('low'))
          .alias('impact_level')
    ])
    
    # Identify affected splice type
    delta_df = delta_df.with_columns([
        pl.when(pl.col('delta_donor').abs() > pl.col('delta_acceptor').abs())
          .then(pl.lit('donor'))
          .otherwise(pl.lit('acceptor'))
          .alias('affected_splice_type')
    ])
    
    return delta_df
```

#### 3. Variant Impact Annotation

```python
def _annotate_variant_impact(
    self,
    delta_scores: pl.DataFrame,
    variant: Variant
) -> VariantImpact:
    """
    Annotate clinical impact of variant
    
    Returns:
    - impact_summary: High-level classification
    - affected_sites: List of splice sites with significant changes
    - predicted_consequences: Molecular consequences
    """
    
    # Find high-impact positions
    high_impact = delta_scores.filter(
        pl.col('max_abs_delta') > 0.2
    )
    
    # Classify variant effect
    if len(high_impact) == 0:
        impact_type = 'neutral'
        severity = 'benign'
    else:
        # Check if canonical site is affected
        canonical_affected = self._is_canonical_site_affected(high_impact, variant)
        
        if canonical_affected:
            # Loss or gain of canonical site
            if high_impact[0, 'delta_donor'] < -0.2 or high_impact[0, 'delta_acceptor'] < -0.2:
                impact_type = 'canonical_site_loss'
                severity = 'pathogenic'
            else:
                impact_type = 'canonical_site_gain'
                severity = 'uncertain'
        else:
            # Cryptic site gain
            impact_type = 'cryptic_site_gain'
            severity = 'likely_pathogenic'
    
    # Predict molecular consequences
    consequences = self._predict_molecular_consequences(delta_scores, variant)
    
    return VariantImpact(
        variant=variant,
        impact_type=impact_type,
        severity=severity,
        affected_positions=high_impact,
        predicted_consequences=consequences
    )
```

---

## Clinical Applications

### Application 1: Rare Disease Diagnosis

**Clinical Scenario:**
- Patient with undiagnosed rare disease
- Whole exome/genome sequencing performed
- Candidate variants identified in disease-relevant genes

**Workflow:**
```python
# For each candidate variant in gene
for variant in candidate_variants:
    # Predict delta scores
    result = variant_predictor.predict_with_variant(
        gene_id=gene_id,
        variant=variant
    )
    
    # Check if pathogenic impact
    if result.impact.severity in ['pathogenic', 'likely_pathogenic']:
        # Report as candidate causative variant
        print(f"Variant {variant} predicted to disrupt splicing:")
        print(f"  Impact: {result.impact.impact_type}")
        print(f"  Affected sites: {len(result.impact.affected_positions)}")
        print(f"  Consequences: {result.impact.predicted_consequences}")
```

**Output:**
- Ranked list of variants by predicted splice impact
- Detailed splice site disruption profiles
- Evidence for functional validation

---

### Application 2: Cancer Somatic Variant Analysis

**Clinical Scenario:**
- Tumor exome sequencing reveals somatic mutations
- Need to identify splice-altering mutations in tumor suppressors/oncogenes

**Workflow:**
```python
# Analyze all somatic variants in cancer gene panel
cancer_genes = ['TP53', 'BRCA1', 'BRCA2', 'ATM', ...]

for gene in cancer_genes:
    somatic_variants = load_somatic_variants(tumor_sample, gene)
    
    for variant in somatic_variants:
        delta_result = variant_predictor.predict_with_variant(
            gene_id=gene_id_map[gene],
            variant=variant
        )
        
        # Check for LOF via splice disruption
        if delta_result.impact.impact_type == 'canonical_site_loss':
            # Potential loss-of-function via aberrant splicing
            report_candidate_driver(variant, delta_result)
```

**Output:**
- Splice-disrupting somatic mutations
- Alternative splicing isoforms in tumor
- Therapeutic targets

---

### Application 3: Pharmacogenomics

**Clinical Scenario:**
- Patient germline variants affect drug metabolism genes
- Need to predict impact on alternative splicing of CYP enzymes

**Workflow:**
```python
# Analyze CYP2D6 variants
cyp2d6_variants = get_patient_variants('CYP2D6')

for variant in cyp2d6_variants:
    delta_result = variant_predictor.predict_with_variant(
        gene_id='ENSG00000100197',  # CYP2D6
        variant=variant
    )
    
    # Predict functional impact
    if delta_result.impact.severity != 'benign':
        # Variant may affect enzyme expression/function
        dosage_adjustment = predict_dosage_adjustment(delta_result)
        print(f"Variant {variant} may require dosage adjustment: {dosage_adjustment}")
```

**Output:**
- Variants affecting splice site usage
- Predicted functional consequences
- Dosing recommendations

---

### Application 4: Population Genetics & Evolution

**Research Scenario:**
- Study splice site evolution across populations
- Identify population-specific splice variants

**Workflow:**
```python
# Compare allele-specific splicing across populations
populations = ['AFR', 'EUR', 'EAS', 'SAS', 'AMR']

for gene in genes_of_interest:
    for pop in populations:
        common_variants = get_common_variants(gene, pop, maf_threshold=0.05)
        
        for variant in common_variants:
            delta_result = variant_predictor.predict_with_variant(
                gene_id=gene,
                variant=variant
            )
            
            # Track population-specific splice effects
            if delta_result.impact.max_abs_delta > 0.1:
                record_population_splice_variant(variant, pop, delta_result)
```

**Output:**
- Population-specific splice-QTLs
- Evolutionary constraints on splice sites
- Adaptive splicing variants

---

## Implementation Roadmap

### Phase 1: Core Delta Score Infrastructure (Weeks 1-2)

**Tasks:**
1. Implement `VariantSplicePredictor` class
2. Add in silico mutagenesis (`_apply_variant_to_sequence`)
3. Implement delta score computation (`_compute_delta_scores`)
4. Add basic variant impact annotation

**Deliverables:**
- Working delta score computation for SNVs
- Unit tests for sequence manipulation
- Example notebooks demonstrating usage

---

### Phase 2: VCF Integration (Weeks 3-4)

**Tasks:**
1. Add VCF file parser
2. Implement batch variant processing
3. Add strand-aware coordinate conversion
4. Support indels (insertions/deletions)

**Deliverables:**
- VCF input support
- Batch processing for multiple variants
- Indel handling

---

### Phase 3: Clinical Annotation (Weeks 5-6)

**Tasks:**
1. Integrate ClinVar annotations
2. Add ACMG classification logic
3. Implement consequence prediction
4. Add report generation

**Deliverables:**
- Clinical impact classification
- Pathogenicity predictions
- PDF/HTML reports

---

### Phase 4: Advanced Features (Weeks 7-8)

**Tasks:**
1. Meta-model delta scores (use hybrid/meta-only modes)
2. Uncertainty quantification for delta scores
3. Comparative analysis (base vs meta delta scores)
4. Visualization tools (delta score heatmaps)

**Deliverables:**
- Meta-model enhanced variant analysis
- Uncertainty-aware predictions
- Interactive visualization dashboard

---

## Generalization Strategy

### Key Design Decisions

#### 1. Sequence-Centric API

```python
# Current: Gene-centric
workflow.predict(gene_id='ENSG00000134202')

# Generalized: Sequence-centric
workflow.predict(
    sequence=custom_sequence,
    coordinates={'chr': '17', 'start': 7579300, 'end': 7579400},
    gene_context={'gene_id': 'ENSG00000134202', 'strand': '+'}
)
```

**Benefits:**
- Works with any sequence (ref, variant, synthetic)
- Supports custom coordinate systems
- Enables in silico experiments

#### 2. Modular Prediction Pipeline

```
┌─────────────────────────────────────┐
│   Sequence Input Layer              │
│  (FASTA, VCF, Custom)               │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   Feature Extraction                │
│  (K-mers, Genomic, Context)         │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   Base Model Prediction             │
│  (SpliceAI Ensemble)                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   Meta-Model Calibration            │
│  (Selective Application)            │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   Delta Score Computation           │
│  (Ref vs Variant)                   │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   Output Layer                      │
│  (Predictions, Delta, Metadata)     │
└─────────────────────────────────────┘
```

#### 3. Metadata Tracking Throughout

```python
class PredictionWithMetadata:
    """
    Unified prediction object with metadata
    """
    predictions: pl.DataFrame  # Scores for each position
    metadata: dict             # Uncertainty, confidence, etc.
    provenance: dict           # Model version, config, etc.
    
    def compute_delta(self, other: 'PredictionWithMetadata') -> 'DeltaPrediction':
        """Compute delta scores while preserving metadata"""
        pass
```

---

## Summary

### Current Capabilities ✅
1. Position-wise splice site predictions (3 modes)
2. Uncertainty quantification (6/9 metadata features)
3. Meta-model calibration
4. Confidence-based filtering

### Next Phase: Variant Analysis 🚀
1. **Delta score computation** - Quantify variant impact
2. **In silico mutagenesis** - Apply variants to sequences
3. **Clinical annotation** - Pathogenicity classification
4. **Batch processing** - VCF file support

### Long-term Vision 🌟
1. **Uncertainty-aware delta scores** - Propagate confidence through variant analysis
2. **Alternative splicing prediction** - Predict full splicing isoform changes
3. **Allele-specific expression** - Integrate with RNA-seq data
4. **Therapeutic target identification** - Splice-modulating drug screening

---

**Next Steps:**
1. Run comprehensive test on diverse genes ✅ (in progress)
2. Implement `VariantSplicePredictor` base class
3. Add in silico mutagenesis
4. Create example notebooks for common use cases
5. Integrate with existing VCF processing pipeline

**Estimated Timeline:** 8-10 weeks for full variant analysis capability


