# VCF Variant Analysis Workflow Documentation

## Overview

This document provides comprehensive documentation for the MetaSpliceAI VCF variant analysis workflow, covering the complete pipeline from raw VCF files to splice site impact assessment using OpenSpliceAI and meta-models.

## Table of Contents

1. [Workflow Architecture](#workflow-architecture)
2. [VCF Preprocessing Pipeline](#vcf-preprocessing-pipeline)
3. [Variant Parsing and Standardization](#variant-parsing-and-standardization)
4. [WT/ALT Sequence Construction](#wtalt-sequence-construction)
5. [OpenSpliceAI Integration](#openspliceai-integration)
6. [Meta-Model Enhancement](#meta-model-enhancement)
7. [Alternative Splice Site Prediction](#alternative-splice-site-prediction)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)

## Workflow Architecture

The MetaSpliceAI variant analysis workflow consists of several interconnected components:

```
Raw VCF → VCF Normalization → Variant Parsing → Sequence Construction → 
OpenSpliceAI Scoring → Meta-Model Analysis → Alternative Splice Prediction
```

### Key Components

1. **VCF Preprocessing** (`workflows/vcf_preprocessing.py`)
   - bcftools-based normalization
   - Multiallelic splitting
   - Left-alignment and trimming
   - Indexing and validation

2. **Resource Management** (`data_sources/resource_manager.py`)
   - Dynamic path resolution
   - Reference genome management
   - Case study data organization

3. **Variant Standardization** (`formats/variant_standardizer.py`)
   - Coordinate normalization
   - Allele standardization
   - Variant type classification

4. **Sequence Construction** (integrated in tutorials)
   - Reference sequence extraction
   - Variant application
   - Context window management

## VCF Preprocessing Pipeline

### 1. VCF Normalization with bcftools

The preprocessing pipeline uses `bcftools norm` to ensure consistent variant representation:

```python
from meta_spliceai.splice_engine.case_studies.workflows.vcf_preprocessing import VCFPreprocessor

preprocessor = VCFPreprocessor()
normalized_vcf = preprocessor.normalize_vcf(
    input_vcf="clinvar.vcf.gz",
    reference_fasta="GRCh38.fa",
    output_vcf="clinvar_normalized.vcf.gz"
)
```

#### Normalization Steps

1. **Multiallelic Splitting** (`-m -both`)
   - Splits multiallelic variants into separate records
   - Handles both SNVs and indels
   - Preserves original INFO fields

2. **Left-Alignment** (`-f reference.fa`)
   - Aligns variants to leftmost genomic position
   - Strand-independent positioning
   - Critical for consistent representation in repetitive regions

3. **Trimming**
   - Removes redundant nucleotides from REF/ALT alleles
   - Reduces variants to minimal representation
   - Maintains biological equivalence

4. **Indexing and Validation**
   - Creates tabix index for efficient querying
   - Validates VCF format compliance
   - Generates processing statistics

### 2. Quality Control and Filtering

```python
# Apply ClinVar-specific filtering
filtered_variants = preprocessor.filter_splice_variants(
    normalized_vcf,
    clinical_significance=["Pathogenic", "Likely_pathogenic"],
    molecular_consequence=["splice_acceptor_variant", "splice_donor_variant"]
)
```

## Variant Parsing and Standardization

### 1. VCF Record Parsing

The workflow parses VCF records into standardized variant objects:

```python
from meta_spliceai.splice_engine.case_studies.formats.variant_standardizer import VariantStandardizer

standardizer = VariantStandardizer()

# Parse VCF record
variant = standardizer.standardize_from_vcf(
    chrom="chr1",
    pos=12345,
    ref="A",
    alt="G"
)
```

### 2. Coordinate System Handling

The standardizer handles multiple coordinate systems:

- **VCF coordinates**: 1-based, inclusive
- **BED coordinates**: 0-based, half-open
- **Internal coordinates**: Configurable based on context

### 3. Variant Classification

Variants are classified by type and impact:

```python
variant_types = {
    "SNV": "Single nucleotide variant",
    "insertion": "Insertion variant", 
    "deletion": "Deletion variant",
    "complex": "Complex structural variant"
}
```

## WT/ALT Sequence Construction

### Overview

The WT/ALT sequence construction is the core of variant impact analysis. It involves extracting reference sequences and applying variants to create alternate sequences for comparison.

### 1. Reference Sequence Extraction

```python
import pysam

# Load reference genome
fasta = pysam.FastaFile("GRCh38.fa")

# Extract sequence with context
def extract_reference_sequence(chrom, start, end, context_size=5000):
    """
    Extract reference sequence with flanking context.
    
    Args:
        chrom: Chromosome name (e.g., "chr1")
        start: Start position (1-based)
        end: End position (1-based, inclusive)
        context_size: Flanking sequence length on each side
    
    Returns:
        tuple: (sequence, adjusted_start, adjusted_end)
    """
    # Add flanking context
    seq_start = max(1, start - context_size)
    seq_end = end + context_size
    
    # Extract sequence
    sequence = fasta.fetch(chrom, seq_start - 1, seq_end)  # pysam uses 0-based
    
    return sequence.upper(), seq_start, seq_end
```

### 2. Variant Application Logic

The variant application process depends on variant type:

#### SNV (Single Nucleotide Variant)

```python
def apply_snv(reference_seq, variant_pos, ref_allele, alt_allele, seq_start):
    """
    Apply SNV to reference sequence.
    
    Args:
        reference_seq: Reference sequence string
        variant_pos: Variant position (1-based genomic coordinate)
        ref_allele: Reference allele
        alt_allele: Alternate allele
        seq_start: Start position of reference sequence
    
    Returns:
        str: Alternate sequence with SNV applied
    """
    # Convert to 0-based sequence coordinate
    seq_pos = variant_pos - seq_start
    
    # Validate reference allele matches
    if reference_seq[seq_pos] != ref_allele:
        raise ValueError(f"Reference mismatch at position {variant_pos}")
    
    # Apply substitution
    alt_seq = (reference_seq[:seq_pos] + 
               alt_allele + 
               reference_seq[seq_pos + 1:])
    
    return alt_seq
```

#### Insertion

```python
def apply_insertion(reference_seq, variant_pos, ref_allele, alt_allele, seq_start):
    """
    Apply insertion to reference sequence.
    
    For insertions, REF is the base before insertion site,
    ALT is REF + inserted sequence.
    """
    seq_pos = variant_pos - seq_start
    
    # Validate reference base
    if reference_seq[seq_pos] != ref_allele:
        raise ValueError(f"Reference mismatch at position {variant_pos}")
    
    # Extract inserted sequence (ALT minus REF)
    inserted_seq = alt_allele[len(ref_allele):]
    
    # Apply insertion after reference base
    alt_seq = (reference_seq[:seq_pos + 1] + 
               inserted_seq + 
               reference_seq[seq_pos + 1:])
    
    return alt_seq
```

#### Deletion

```python
def apply_deletion(reference_seq, variant_pos, ref_allele, alt_allele, seq_start):
    """
    Apply deletion to reference sequence.
    
    For deletions, REF includes deleted sequence,
    ALT is the remaining base(s).
    """
    seq_pos = variant_pos - seq_start
    
    # Validate reference sequence matches
    ref_in_seq = reference_seq[seq_pos:seq_pos + len(ref_allele)]
    if ref_in_seq != ref_allele:
        raise ValueError(f"Reference mismatch at position {variant_pos}")
    
    # Apply deletion
    alt_seq = (reference_seq[:seq_pos] + 
               alt_allele + 
               reference_seq[seq_pos + len(ref_allele):])
    
    return alt_seq
```

### 3. Context Window Management

For splice site analysis, proper context windows are critical:

```python
def create_splice_context(sequence, variant_pos, seq_start, context_size=5000):
    """
    Create splice analysis context around variant.
    
    Args:
        sequence: Full sequence (WT or ALT)
        variant_pos: Variant position (1-based genomic)
        seq_start: Sequence start position
        context_size: Context size for analysis
    
    Returns:
        dict: Context information for splice analysis
    """
    # Calculate relative position in sequence
    rel_pos = variant_pos - seq_start
    
    # Define analysis window
    window_start = max(0, rel_pos - context_size)
    window_end = min(len(sequence), rel_pos + context_size)
    
    # Extract context sequence
    context_seq = sequence[window_start:window_end]
    
    return {
        'sequence': context_seq,
        'variant_pos_in_context': rel_pos - window_start,
        'genomic_start': seq_start + window_start,
        'genomic_end': seq_start + window_end
    }
```

## OpenSpliceAI Integration

### 1. Delta Score Computation

The workflow computes delta scores between WT and ALT sequences:

```python
def compute_openspliceai_delta_scores(wt_context, alt_context):
    """
    Compute OpenSpliceAI delta scores for variant impact.
    
    Args:
        wt_context: Wild-type sequence context
        alt_context: Alternate sequence context
    
    Returns:
        dict: Delta scores for donor and acceptor sites
    """
    # Get OpenSpliceAI predictions for both sequences
    wt_scores = openspliceai_predict(wt_context['sequence'])
    alt_scores = openspliceai_predict(alt_context['sequence'])
    
    # Compute delta scores
    delta_scores = {
        'donor_delta': alt_scores['donor'] - wt_scores['donor'],
        'acceptor_delta': alt_scores['acceptor'] - wt_scores['acceptor'],
        'variant_pos': wt_context['variant_pos_in_context']
    }
    
    return delta_scores
```

### 2. Splice Site Impact Classification

```python
def classify_splice_impact(delta_scores, thresholds=None):
    """
    Classify splice site impact based on delta scores.
    
    Args:
        delta_scores: Delta scores from OpenSpliceAI
        thresholds: Impact classification thresholds
    
    Returns:
        dict: Impact classification and confidence
    """
    if thresholds is None:
        thresholds = {
            'high_impact': 0.5,
            'moderate_impact': 0.2,
            'low_impact': 0.1
        }
    
    max_delta = max(abs(delta_scores['donor_delta']), 
                   abs(delta_scores['acceptor_delta']))
    
    if max_delta >= thresholds['high_impact']:
        impact = 'high'
    elif max_delta >= thresholds['moderate_impact']:
        impact = 'moderate'
    elif max_delta >= thresholds['low_impact']:
        impact = 'low'
    else:
        impact = 'minimal'
    
    return {
        'impact_level': impact,
        'max_delta_score': max_delta,
        'confidence': min(max_delta * 2, 1.0)  # Simple confidence metric
    }
```

## Meta-Model Enhancement

### 1. Feature Engineering for Meta-Models

The workflow extracts additional features for meta-model analysis:

```python
def extract_meta_features(variant, wt_context, alt_context, delta_scores):
    """
    Extract features for meta-model analysis.
    
    Returns:
        dict: Comprehensive feature set for meta-learning
    """
    features = {
        # Variant features
        'variant_type': variant.variant_type,
        'ref_length': len(variant.ref),
        'alt_length': len(variant.alt),
        'length_change': len(variant.alt) - len(variant.ref),
        
        # Sequence features
        'gc_content_wt': calculate_gc_content(wt_context['sequence']),
        'gc_content_alt': calculate_gc_content(alt_context['sequence']),
        
        # Delta score features
        'donor_delta': delta_scores['donor_delta'],
        'acceptor_delta': delta_scores['acceptor_delta'],
        'max_abs_delta': max(abs(delta_scores['donor_delta']), 
                           abs(delta_scores['acceptor_delta'])),
        
        # Positional features
        'distance_to_exon_boundary': calculate_exon_distance(variant),
        'splice_region_overlap': check_splice_region_overlap(variant),
        
        # Conservation features (if available)
        'phylop_score': get_conservation_score(variant),
        'phastcons_score': get_phastcons_score(variant)
    }
    
    return features
```

### 2. Meta-Model Prediction Pipeline

```python
def predict_with_meta_model(features, base_scores, meta_model):
    """
    Generate enhanced predictions using meta-model.
    
    Args:
        features: Extracted variant features
        base_scores: OpenSpliceAI base model scores
        meta_model: Trained meta-learning model
    
    Returns:
        dict: Enhanced predictions with confidence intervals
    """
    # Combine base scores with meta features
    input_features = {**features, **base_scores}
    
    # Generate meta-model predictions
    meta_prediction = meta_model.predict(input_features)
    
    # Calculate prediction confidence
    confidence = meta_model.predict_confidence(input_features)
    
    return {
        'base_prediction': base_scores,
        'meta_prediction': meta_prediction,
        'confidence': confidence,
        'improvement_over_base': meta_prediction - base_scores['max_score']
    }
```

## Alternative Splice Site Prediction

### 1. Cryptic Splice Site Detection

The workflow can identify potential cryptic splice sites activated by variants:

```python
def detect_cryptic_splice_sites(alt_context, delta_scores, threshold=0.3):
    """
    Detect potential cryptic splice sites in alternate sequence.
    
    Args:
        alt_context: Alternate sequence context
        delta_scores: Delta scores from variant analysis
        threshold: Minimum score threshold for cryptic sites
    
    Returns:
        list: Detected cryptic splice sites with positions and scores
    """
    # Get full splice predictions for alternate sequence
    alt_predictions = openspliceai_predict_full(alt_context['sequence'])
    
    # Find peaks above threshold
    cryptic_sites = []
    
    # Donor sites
    donor_peaks = find_peaks(alt_predictions['donor'], height=threshold)
    for peak_pos in donor_peaks:
        cryptic_sites.append({
            'type': 'donor',
            'position': alt_context['genomic_start'] + peak_pos,
            'score': alt_predictions['donor'][peak_pos],
            'sequence_motif': extract_motif(alt_context['sequence'], peak_pos, 'donor')
        })
    
    # Acceptor sites
    acceptor_peaks = find_peaks(alt_predictions['acceptor'], height=threshold)
    for peak_pos in acceptor_peaks:
        cryptic_sites.append({
            'type': 'acceptor', 
            'position': alt_context['genomic_start'] + peak_pos,
            'score': alt_predictions['acceptor'][peak_pos],
            'sequence_motif': extract_motif(alt_context['sequence'], peak_pos, 'acceptor')
        })
    
    return cryptic_sites
```

### 2. Alternative Splicing Pattern Analysis

```python
def analyze_alternative_splicing(variant, cryptic_sites, known_sites):
    """
    Analyze potential alternative splicing patterns.
    
    Args:
        variant: Standardized variant object
        cryptic_sites: Detected cryptic splice sites
        known_sites: Known canonical splice sites
    
    Returns:
        dict: Alternative splicing analysis results
    """
    analysis = {
        'canonical_sites_affected': [],
        'cryptic_sites_activated': [],
        'potential_exon_skipping': False,
        'potential_intron_retention': False,
        'alternative_splicing_score': 0.0
    }
    
    # Analyze impact on canonical sites
    for site in known_sites:
        if is_site_affected_by_variant(site, variant):
            analysis['canonical_sites_affected'].append(site)
    
    # Analyze cryptic site activation
    for site in cryptic_sites:
        if site['score'] > 0.5:  # High confidence cryptic sites
            analysis['cryptic_sites_activated'].append(site)
    
    # Predict splicing outcomes
    if len(analysis['canonical_sites_affected']) > 0:
        if len(analysis['cryptic_sites_activated']) > 0:
            analysis['potential_exon_skipping'] = True
        else:
            analysis['potential_intron_retention'] = True
    
    # Calculate alternative splicing score
    analysis['alternative_splicing_score'] = calculate_alt_splicing_score(
        analysis['canonical_sites_affected'],
        analysis['cryptic_sites_activated']
    )
    
    return analysis
```

## Usage Examples

### 1. Complete Workflow Example

```python
#!/usr/bin/env python3
"""
Complete variant analysis workflow example.
"""

from meta_spliceai.splice_engine.case_studies.workflows.vcf_preprocessing import VCFPreprocessor
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import CaseStudyResourceManager
from meta_spliceai.splice_engine.case_studies.formats.variant_standardizer import VariantStandardizer

def run_complete_variant_analysis(vcf_file, output_dir):
    """Run complete variant analysis workflow."""
    
    # Initialize components
    preprocessor = VCFPreprocessor()
    resource_manager = CaseStudyResourceManager()
    standardizer = VariantStandardizer()
    
    # Step 1: Normalize VCF
    reference_fasta = resource_manager.get_reference_fasta()
    normalized_vcf = preprocessor.normalize_vcf(
        vcf_file, reference_fasta, f"{output_dir}/normalized.vcf.gz"
    )
    
    # Step 2: Parse and filter variants
    variants = parse_vcf_variants(normalized_vcf)
    splice_variants = filter_splice_variants(variants)
    
    # Step 3: Analyze each variant
    results = []
    for variant in splice_variants:
        # Extract sequences
        wt_context = extract_reference_sequence(
            variant.chrom, variant.pos, variant.pos + len(variant.ref) - 1
        )
        alt_context = apply_variant_to_sequence(wt_context, variant)
        
        # Compute delta scores
        delta_scores = compute_openspliceai_delta_scores(wt_context, alt_context)
        
        # Extract meta features
        meta_features = extract_meta_features(variant, wt_context, alt_context, delta_scores)
        
        # Detect cryptic sites
        cryptic_sites = detect_cryptic_splice_sites(alt_context, delta_scores)
        
        # Analyze alternative splicing
        alt_splicing = analyze_alternative_splicing(variant, cryptic_sites, [])
        
        results.append({
            'variant': variant,
            'delta_scores': delta_scores,
            'meta_features': meta_features,
            'cryptic_sites': cryptic_sites,
            'alternative_splicing': alt_splicing
        })
    
    return results

# Run analysis
if __name__ == "__main__":
    results = run_complete_variant_analysis(
        "clinvar_20250831.vcf.gz",
        "results/"
    )
    
    # Save results
    save_analysis_results(results, "results/variant_analysis.json")
```

### 2. Batch Processing Example

```python
def process_variants_batch(vcf_file, batch_size=1000):
    """Process variants in batches for memory efficiency."""
    
    batch_results = []
    variant_count = 0
    
    with pysam.VariantFile(vcf_file) as vcf:
        batch = []
        
        for record in vcf:
            batch.append(record)
            
            if len(batch) >= batch_size:
                # Process batch
                batch_result = analyze_variant_batch(batch)
                batch_results.extend(batch_result)
                
                # Reset batch
                batch = []
                variant_count += batch_size
                
                print(f"Processed {variant_count} variants...")
        
        # Process remaining variants
        if batch:
            batch_result = analyze_variant_batch(batch)
            batch_results.extend(batch_result)
    
    return batch_results
```

## Troubleshooting

### Common Issues and Solutions

1. **Reference sequence mismatch**
   ```
   Error: Reference mismatch at position 12345
   Solution: Ensure VCF and reference genome are from same build (GRCh37/38)
   ```

2. **Memory issues with large VCFs**
   ```
   Solution: Use batch processing or filter VCF before analysis
   ```

3. **bcftools not found**
   ```
   Solution: Install bcftools via conda: mamba install bcftools
   ```

4. **Coordinate system confusion**
   ```
   Solution: Always specify coordinate system (0-based vs 1-based)
   ```

### Performance Optimization

1. **Use indexed VCF files** for region-specific queries
2. **Batch process variants** to reduce memory usage
3. **Cache reference sequences** for repeated access
4. **Parallelize analysis** across chromosomes or regions

### Validation Checks

1. **Verify reference allele matches** extracted sequence
2. **Check variant coordinates** are within chromosome bounds
3. **Validate sequence lengths** after variant application
4. **Confirm splice site positions** are biologically reasonable

## Future Enhancements

1. **Structural variant support** for complex rearrangements
2. **Tissue-specific splice site models** for context-aware analysis
3. **Population frequency integration** for variant prioritization
4. **Clinical annotation** with ClinVar and HGMD data
5. **Interactive visualization** of splice site changes

---

This documentation provides a comprehensive guide to the MetaSpliceAI VCF variant analysis workflow. For specific implementation details, refer to the individual module documentation and example scripts.
