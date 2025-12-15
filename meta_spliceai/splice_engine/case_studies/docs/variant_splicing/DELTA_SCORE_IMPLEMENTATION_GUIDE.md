# Delta Score Implementation Guide

**Last Updated**: 2025-09-15  
**Status**: ✅ **OPERATIONAL - SpliceAI Delta Score Computation Working**

## Overview

This guide provides practical implementation examples for working with delta scores in variant analysis workflows. It covers both the working SpliceAI implementation and the OpenSpliceAI integration patterns. The guide complements the [Technical FAQ](./OPENSPLICEAI_TECHNICAL_FAQ.md) with concrete code examples and use cases.

---

## 🔧 **Core Implementation Patterns**

### **1. Working SpliceAI Delta Score Calculation** ✅

```python
from meta_spliceai.splice_engine.meta_models.utils.sequence_predictor import SequencePredictor

def analyze_variant_splicing_impact_spliceai(variants_df):
    """
    Analyze splicing impact of variants using SpliceAI delta scores.
    This is the WORKING implementation currently in use.
    """
    # Initialize SpliceAI predictor
    predictor = SequencePredictor(model_type="spliceai", verbose=True)
    
    results = []
    
    # Process variants from ClinVar pipeline output
    for _, variant in variants_df.iterrows():
        # Compute delta scores
        delta_result = predictor.compute_variant_delta_scores(
            wt_sequence=variant['ref_sequence'],
            alt_sequence=variant['alt_sequence'],
            variant_position=variant['variant_position_in_sequence']
        )
        
        # Add variant metadata
        delta_result.update({
            'chrom': variant['chrom'],
            'pos': variant['pos'],
            'ref': variant['ref'],
            'alt': variant['alt'],
            'gene': variant.get('gene_symbol', 'UNKNOWN'),
            'clinical_significance': variant.get('clinical_significance', 'Unknown')
        })
        
        results.append(delta_result)
    
    return results
```

### **2. Using the Delta Score Workflow** ✅

```python
# Command-line usage (TESTED & WORKING)
python meta_spliceai/splice_engine/case_studies/workflows/delta_score_workflow.py \
    --input results/clinvar_pipeline_full/clinvar_wt_alt_ready.parquet \
    --output results/delta_scores_spliceai/ \
    --model-type spliceai \
    --max-variants 1000 \
    --verbose

# Programmatic usage
from meta_spliceai.splice_engine.case_studies.workflows.delta_score_workflow import DeltaScoreWorkflow

workflow = DeltaScoreWorkflow()
workflow.run(
    input_path="results/clinvar_pipeline_full/clinvar_wt_alt_ready.parquet",
    output_dir="results/delta_scores/",
    model_type="spliceai",
    max_variants=1000
)
```

### **3. OpenSpliceAI Delta Score Calculation** (Integration Pattern)

```python
from meta_spliceai.openspliceai.variant.utils import get_delta_scores, Annotator
import pysam

def analyze_variant_splicing_impact_openspliceai(vcf_file, ref_fasta, annotations):
    """
    Analyze splicing impact of variants using OpenSpliceAI delta scores.
    Note: This requires OpenSpliceAI models to be integrated.
    """
    # Initialize annotator with reference genome and gene annotations
    ann = Annotator(
        ref_fasta=ref_fasta,
        annotations=annotations,  # 'grch37' or 'grch38' or custom file
        model_path='SpliceAI',    # Use pre-trained models
        model_type='keras'
    )
    
    results = []
    
    # Process VCF records
    with pysam.VariantFile(vcf_file) as vcf:
        for record in vcf:
            # Calculate delta scores for this variant
            delta_scores = get_delta_scores(
                record=record,
                ann=ann,
                dist_var=50,        # Coverage parameter (NOT search limit!)
                mask=0,             # No masking
                flanking_size=5000  # 10kb window around variant
            )
            
            # Parse results
            for score_line in delta_scores:
                fields = score_line.split('|')
                result = {
                    'chrom': record.chrom,
                    'pos': record.pos,
                    'ref': record.ref,
                    'alt': fields[0],
                    'gene': fields[1],
                    'DS_AG': float(fields[2]) if fields[2] != '.' else None,
                    'DS_AL': float(fields[3]) if fields[3] != '.' else None,
                    'DS_DG': float(fields[4]) if fields[4] != '.' else None,
                    'DS_DL': float(fields[5]) if fields[5] != '.' else None,
                    'DP_AG': int(fields[6]) if fields[6] != '.' else None,
                    'DP_AL': int(fields[7]) if fields[7] != '.' else None,
                    'DP_DG': int(fields[8]) if fields[8] != '.' else None,
                    'DP_DL': int(fields[9]) if fields[9] != '.' else None,
                }
                results.append(result)
    
    return results
```

### **⚠️ CRITICAL: Understanding DP Range and Search Parameters**

**IMPORTANT CORRECTION**: The DP (Delta Position) range is **NOT limited to ±50bp** as commonly assumed!

```python
# OpenSpliceAI parameters from get_delta_scores():
dist_var = 50        # Used for coverage calculation, NOT search limit
flanking_size = 5000 # Actual analysis window size

# Window calculations:
cov = 2 * dist_var + 1        # Coverage = 101bp 
wid = 2 * flanking_size + cov  # Total window = 10,101bp

# ACTUAL DP range:
# DP can range from -(flanking_size + cov//2) to +(flanking_size + cov//2)
# With default parameters: -5050 to +5050 bp!
```

**What Each Parameter Does:**

| Parameter | Default | Purpose | Impact on DP |
|-----------|---------|---------|-------------|
| `dist_var` | 50 | Coverage calculation, masking logic | Affects `cov` but NOT DP search limit |
| `flanking_size` | 5000 | Analysis window size | **Determines actual DP range** (±5050bp) |
| `mask` | 0 | Mask certain splice site types | Can zero out some delta scores |

**Key Insight**: Delta scores are calculated for the **entire 10kb window**, and DP values report where the maximum occurs within that full window!

```python
# The argmax() functions search the ENTIRE window:
idx_pa = (y[1, :, 1] - y[0, :, 1]).argmax()  # Searches all 10,101 positions
DP_AG = idx_pa - cov // 2                    # Can be anywhere in ±5050bp range
```

### **2. Sequence Length Handling for Indels**

```python
def analyze_indel_splicing_effects(record, ann):
    """
    Demonstrate how OpenSpliceAI handles different sequence lengths.
    """
    ref_len = len(record.ref)
    alt_len = len(record.alts[0])
    
    print(f"Variant: {record.ref} -> {record.alts[0]}")
    print(f"Reference length: {ref_len}")
    print(f"Alternative length: {alt_len}")
    
    if ref_len == alt_len:
        print("→ SNV: Same length sequences, standard processing")
    elif ref_len > alt_len:
        print(f"→ Deletion: {ref_len - alt_len} bp deleted")
        print("  Implementation: Zeros inserted at deletion site in y_alt")
    elif alt_len > ref_len:
        print(f"→ Insertion: {alt_len - ref_len} bp inserted")
        print("  Implementation: Maximum taken over insertion region in y_alt")
    
    # Calculate delta scores
    delta_scores = get_delta_scores(record, ann, dist_var=50, mask=0)
    
    return delta_scores
```

### **3. Vector-Based Delta Score Analysis**

```python
def extract_full_delta_vectors(x_ref, x_alt, models):
    """
    Extract complete delta score vectors for every position.
    This demonstrates the underlying vector nature of delta scores.
    """
    import numpy as np
    
    # Get predictions for reference and alternative sequences
    y_ref = np.mean([model.predict(x_ref) for model in models], axis=0)
    y_alt = np.mean([model.predict(x_alt) for model in models], axis=0)
    
    # Calculate delta vectors for every position
    delta_vectors = {
        'acceptor_deltas': y_alt[0, :, 1] - y_ref[0, :, 1],  # All acceptor deltas
        'donor_deltas': y_alt[0, :, 2] - y_ref[0, :, 2],     # All donor deltas
        'positions': np.arange(len(y_ref[0, :, 1]))           # Position indices
    }
    
    # Find maximum delta positions (what OpenSpliceAI reports)
    max_positions = {
        'acceptor_gain_pos': np.argmax(delta_vectors['acceptor_deltas']),
        'acceptor_loss_pos': np.argmax(-delta_vectors['acceptor_deltas']),
        'donor_gain_pos': np.argmax(delta_vectors['donor_deltas']),
        'donor_loss_pos': np.argmax(-delta_vectors['donor_deltas'])
    }
    
    # Maximum delta values (what OpenSpliceAI reports)
    max_deltas = {
        'DS_AG': np.max(delta_vectors['acceptor_deltas']),
        'DS_AL': np.max(-delta_vectors['acceptor_deltas']),
        'DS_DG': np.max(delta_vectors['donor_deltas']),
        'DS_DL': np.max(-delta_vectors['donor_deltas'])
    }
    
    return delta_vectors, max_positions, max_deltas
```

---

## 🧬 **Alternative Splicing Pattern Detection**

### **4. Intron Retention Detection**

```python
def detect_intron_retention(delta_results, transcript_annotation, threshold=0.2):
    """
    Detect potential intron retention events from delta scores.
    """
    intron_retention_candidates = []
    
    for result in delta_results:
        # Check for simultaneous donor and acceptor losses
        donor_loss = result.get('DS_DL', 0) > threshold
        acceptor_loss = result.get('DS_AL', 0) > threshold
        
        if donor_loss and acceptor_loss:
            # Calculate absolute positions of lost splice sites
            variant_pos = result['pos']
            donor_pos = variant_pos + result.get('DP_DL', 0)
            acceptor_pos = variant_pos + result.get('DP_AL', 0)
            
            # Check if these positions correspond to intron boundaries
            # (This requires transcript structure annotation)
            if is_intron_boundary_pair(donor_pos, acceptor_pos, 
                                     result['gene'], transcript_annotation):
                intron_retention_candidates.append({
                    'variant': f"{result['chrom']}:{result['pos']} {result['ref']}>{result['alt']}",
                    'gene': result['gene'],
                    'donor_loss_score': result['DS_DL'],
                    'acceptor_loss_score': result['DS_AL'],
                    'donor_position': donor_pos,
                    'acceptor_position': acceptor_pos,
                    'confidence': min(result['DS_DL'], result['DS_AL'])
                })
    
    return intron_retention_candidates

def is_intron_boundary_pair(donor_pos, acceptor_pos, gene, annotation):
    """
    Check if positions correspond to donor-acceptor pair of same intron.
    """
    # Implementation depends on your transcript annotation format
    # This is a placeholder for the logic
    gene_structure = annotation.get_gene_structure(gene)
    
    for intron in gene_structure.introns:
        if (abs(donor_pos - intron.start) < 3 and 
            abs(acceptor_pos - intron.end) < 3):
            return True
    
    return False
```

### **5. Cryptic Splice Site Detection**

```python
def detect_cryptic_splice_sites(delta_results, distance_threshold=10):
    """
    Detect cryptic (non-canonical) splice sites created by variants.
    """
    cryptic_sites = []
    
    for result in delta_results:
        # Cryptic donor sites (gains away from canonical positions)
        if (result.get('DS_DG', 0) > 0.2 and 
            abs(result.get('DP_DG', 0)) > distance_threshold):
            cryptic_sites.append({
                'type': 'cryptic_donor',
                'variant': f"{result['chrom']}:{result['pos']} {result['ref']}>{result['alt']}",
                'gene': result['gene'],
                'score': result['DS_DG'],
                'distance_from_variant': result['DP_DG'],
                'absolute_position': result['pos'] + result['DP_DG']
            })
        
        # Cryptic acceptor sites (gains away from canonical positions)
        if (result.get('DS_AG', 0) > 0.2 and 
            abs(result.get('DP_AG', 0)) > distance_threshold):
            cryptic_sites.append({
                'type': 'cryptic_acceptor',
                'variant': f"{result['chrom']}:{result['pos']} {result['ref']}>{result['alt']}",
                'gene': result['gene'],
                'score': result['DS_AG'],
                'distance_from_variant': result['DP_AG'],
                'absolute_position': result['pos'] + result['DP_AG']
            })
    
    return cryptic_sites
```

### **6. Exon Skipping Detection**

```python
def detect_exon_skipping(delta_results, transcript_annotation):
    """
    Detect potential exon skipping events.
    Requires donor loss upstream and acceptor loss downstream of same exon.
    """
    exon_skipping_candidates = []
    
    # Group results by gene
    by_gene = {}
    for result in delta_results:
        gene = result['gene']
        if gene not in by_gene:
            by_gene[gene] = []
        by_gene[gene].append(result)
    
    for gene, gene_results in by_gene.items():
        # Look for patterns: donor loss + downstream acceptor loss
        for i, result1 in enumerate(gene_results):
            for j, result2 in enumerate(gene_results[i+1:], i+1):
                
                # Check for donor loss in first result, acceptor loss in second
                donor_loss1 = result1.get('DS_DL', 0) > 0.2
                acceptor_loss2 = result2.get('DS_AL', 0) > 0.2
                
                if donor_loss1 and acceptor_loss2:
                    # Calculate positions
                    donor_pos = result1['pos'] + result1.get('DP_DL', 0)
                    acceptor_pos = result2['pos'] + result2.get('DP_AL', 0)
                    
                    # Check if positions flank the same exon
                    if flanks_same_exon(donor_pos, acceptor_pos, gene, transcript_annotation):
                        exon_skipping_candidates.append({
                            'gene': gene,
                            'upstream_variant': f"{result1['chrom']}:{result1['pos']}",
                            'downstream_variant': f"{result2['chrom']}:{result2['pos']}",
                            'donor_loss_score': result1['DS_DL'],
                            'acceptor_loss_score': result2['DS_AL'],
                            'skipped_exon_boundaries': (donor_pos, acceptor_pos)
                        })
    
    return exon_skipping_candidates

def flanks_same_exon(donor_pos, acceptor_pos, gene, annotation):
    """
    Check if donor and acceptor positions flank the same exon.
    """
    # Implementation depends on transcript annotation format
    gene_structure = annotation.get_gene_structure(gene)
    
    for exon in gene_structure.exons:
        if (abs(donor_pos - exon.end) < 3 and 
            abs(acceptor_pos - exon.start) < 3):
            return True
    
    return False
```

---

## 📊 **Data Analysis and Visualization**

### **7. Delta Score Distribution Analysis**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_delta_score_distributions(delta_results):
    """
    Analyze and visualize delta score distributions.
    """
    df = pd.DataFrame(delta_results)
    
    # Remove null values
    score_columns = ['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL']
    df_clean = df.dropna(subset=score_columns)
    
    # Summary statistics
    print("Delta Score Summary Statistics:")
    print(df_clean[score_columns].describe())
    
    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(score_columns):
        sns.histplot(df_clean[col], bins=50, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')
        axes[i].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # High-impact variants
    high_impact = df_clean[
        (abs(df_clean['DS_AG']) > 0.5) | 
        (abs(df_clean['DS_AL']) > 0.5) |
        (abs(df_clean['DS_DG']) > 0.5) | 
        (abs(df_clean['DS_DL']) > 0.5)
    ]
    
    print(f"\nHigh-impact variants (|delta| > 0.5): {len(high_impact)}")
    return df_clean, high_impact
```

### **8. Position Analysis**

```python
def analyze_position_patterns(delta_results):
    """
    Analyze patterns in delta position distributions.
    """
    df = pd.DataFrame(delta_results)
    position_columns = ['DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']
    
    # Position distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(position_columns):
        data = df[col].dropna()
        sns.histplot(data, bins=50, ax=axes[i])
        axes[i].set_title(f'{col} Position Distribution')
        axes[i].axvline(0, color='red', linestyle='--', alpha=0.7, label='Variant position')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Distance from variant analysis
    print("Distance from variant statistics:")
    for col in position_columns:
        distances = abs(df[col].dropna())
        print(f"{col}: mean={distances.mean():.1f}bp, median={distances.median():.1f}bp")
```

---

## 🔬 **Integration with MetaSpliceAI Framework**

### **9. Schema Adapter Integration**

```python
from meta_spliceai.splice_engine.meta_models.core.schema_adapters import create_schema_adapter

def integrate_with_metaspliceai(delta_results):
    """
    Convert OpenSpliceAI delta scores to MetaSpliceAI format.
    """
    # Create schema adapter for OpenSpliceAI format
    adapter = create_schema_adapter('openspliceai')
    
    # Convert delta results to standard format
    standardized_results = []
    
    for result in delta_results:
        # Convert to MetaSpliceAI splice site format
        if result.get('DS_DG', 0) > 0.2:  # Donor gain
            standardized_results.append({
                'chromosome': result['chrom'],
                'position': result['pos'] + result.get('DP_DG', 0),
                'strand': '+',  # Infer from gene annotation
                'splice_type': 'donor',
                'confidence_score': result['DS_DG'],
                'source': 'openspliceai_variant_analysis',
                'gene_symbol': result['gene'],
                'variant_context': f"{result['ref']}>{result['alt']}"
            })
        
        if result.get('DS_AG', 0) > 0.2:  # Acceptor gain
            standardized_results.append({
                'chromosome': result['chrom'],
                'position': result['pos'] + result.get('DP_AG', 0),
                'strand': '+',  # Infer from gene annotation
                'splice_type': 'acceptor',
                'confidence_score': result['DS_AG'],
                'source': 'openspliceai_variant_analysis',
                'gene_symbol': result['gene'],
                'variant_context': f"{result['ref']}>{result['alt']}"
            })
    
    # Apply schema adapter
    adapted_results = adapter.adapt_splice_annotations(standardized_results)
    
    return adapted_results
```

---

## 🚀 **Performance Optimization**

### **10. Batch Processing**

```python
def batch_process_variants(vcf_file, ref_fasta, annotations, batch_size=1000):
    """
    Process variants in batches for better performance.
    """
    ann = Annotator(ref_fasta, annotations, model_path='SpliceAI')
    
    all_results = []
    batch = []
    
    with pysam.VariantFile(vcf_file) as vcf:
        for i, record in enumerate(vcf):
            batch.append(record)
            
            if len(batch) >= batch_size:
                # Process batch
                batch_results = process_variant_batch(batch, ann)
                all_results.extend(batch_results)
                
                print(f"Processed {i+1} variants...")
                batch = []
        
        # Process remaining variants
        if batch:
            batch_results = process_variant_batch(batch, ann)
            all_results.extend(batch_results)
    
    return all_results

def process_variant_batch(variant_batch, annotator):
    """
    Process a batch of variants efficiently.
    """
    results = []
    
    for record in variant_batch:
        try:
            delta_scores = get_delta_scores(record, annotator, dist_var=50, mask=0)
            # Parse and store results
            for score_line in delta_scores:
                # Parse score_line as in previous examples
                pass
        except Exception as e:
            print(f"Error processing variant {record.chrom}:{record.pos}: {e}")
            continue
    
    return results
```

---

## 📝 **Usage Examples**

### **Complete Analysis Pipeline**

```python
def complete_variant_splicing_analysis(vcf_file, output_dir):
    """
    Complete pipeline for variant splicing analysis.
    """
    # 1. Basic delta score calculation
    delta_results = analyze_variant_splicing_impact(
        vcf_file=vcf_file,
        ref_fasta="path/to/reference.fa",
        annotations="grch38"
    )
    
    # 2. Alternative splicing pattern detection
    intron_retention = detect_intron_retention(delta_results)
    cryptic_sites = detect_cryptic_splice_sites(delta_results)
    exon_skipping = detect_exon_skipping(delta_results, transcript_annotation)
    
    # 3. Statistical analysis
    df_clean, high_impact = analyze_delta_score_distributions(delta_results)
    
    # 4. Integration with MetaSpliceAI
    standardized_results = integrate_with_metaspliceai(delta_results)
    
    # 5. Save results
    results_summary = {
        'total_variants': len(delta_results),
        'high_impact_variants': len(high_impact),
        'intron_retention_candidates': len(intron_retention),
        'cryptic_sites': len(cryptic_sites),
        'exon_skipping_candidates': len(exon_skipping)
    }
    
    # Export results
    pd.DataFrame(delta_results).to_csv(f"{output_dir}/delta_scores.csv", index=False)
    pd.DataFrame(high_impact).to_csv(f"{output_dir}/high_impact_variants.csv", index=False)
    
    with open(f"{output_dir}/analysis_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return results_summary
```

---

## 🔗 **Related Documentation**

- [OPENSPLICEAI_TECHNICAL_FAQ.md](./OPENSPLICEAI_TECHNICAL_FAQ.md) - Technical implementation details
- [VARIANT_SPLICING_BIOLOGY_Q10_Q12.md](../VARIANT_SPLICING_BIOLOGY_Q10_Q12.md) - Biological interpretation
- [Schema Adapter Framework](../../meta_models/openspliceai_adapter/docs/SCHEMA_ADAPTER_FRAMEWORK.md) - Data integration
