# Alternative Splicing Pattern Construction from OpenSpliceAI Delta Scores

## 🎯 Overview

This guide addresses how to construct alternative splicing patterns from OpenSpliceAI delta score predictions and identifies gold standard datasets for validation.

## 🧬 From Delta Scores to Alternative Splicing Patterns

### The Challenge: Individual Events → Coordinated Patterns

OpenSpliceAI reports **individual splice site changes** (DS_AG, DS_AL, DS_DG, DS_DL), but alternative splicing involves **coordinated patterns** of multiple sites working together.

```python
# OpenSpliceAI Output (Individual Events):
DS_AG = 0.7  # Acceptor gain at position +25
DS_AL = 0.0  # No acceptor loss
DS_DG = 0.0  # No donor gain  
DS_DL = 0.9  # Donor loss at position +11

# Question: How do these combine into splicing patterns?
```

### 🔍 Pattern Construction Methodology

#### **Step 1: Spatial Clustering of Events**

```python
def cluster_splice_events(delta_scores, distance_threshold=50):
    """
    Group splice site changes by genomic proximity
    to identify potential coordinated events
    """
    events = []
    
    # Extract significant events (|DS| > threshold)
    if abs(delta_scores['DS_AG']) > 0.2:
        events.append(('AG', delta_scores['DP_AG'], delta_scores['DS_AG']))
    if abs(delta_scores['DS_AL']) > 0.2:
        events.append(('AL', delta_scores['DP_AL'], delta_scores['DS_AL']))
    if abs(delta_scores['DS_DG']) > 0.2:
        events.append(('DG', delta_scores['DP_DG'], delta_scores['DS_DG']))
    if abs(delta_scores['DS_DL']) > 0.2:
        events.append(('DL', delta_scores['DP_DL'], delta_scores['DS_DL']))
    
    # Cluster by distance
    clusters = spatial_clustering(events, distance_threshold)
    return clusters
```

#### **Step 2: Pattern Classification**

```python
def classify_splicing_pattern(clustered_events, gene_annotation):
    """
    Classify clusters into known alternative splicing patterns
    """
    patterns = []
    
    for cluster in clustered_events:
        pattern_type = None
        
        # Exon Skipping: DL upstream + AL downstream of same exon
        if has_donor_loss_upstream(cluster) and has_acceptor_loss_downstream(cluster):
            if spans_single_exon(cluster, gene_annotation):
                pattern_type = "exon_skipping"
        
        # Intron Retention: DL + AL at same intron boundaries
        elif has_donor_loss(cluster) and has_acceptor_loss(cluster):
            if at_intron_boundaries(cluster, gene_annotation):
                pattern_type = "intron_retention"
        
        # Cryptic Site Activation: AG or DG away from canonical sites
        elif has_cryptic_gain(cluster, gene_annotation):
            pattern_type = "cryptic_site_activation"
        
        # Alternative 5'/3' Splice Sites: Multiple donors/acceptors nearby
        elif has_multiple_sites_same_type(cluster):
            pattern_type = "alternative_splice_sites"
        
        patterns.append({
            'type': pattern_type,
            'events': cluster,
            'confidence': calculate_pattern_confidence(cluster)
        })
    
    return patterns
```

### 🎯 **Specific Pattern Detection Algorithms**

#### **1. Exon Skipping Detection**

```python
def detect_exon_skipping(delta_scores, gene_model):
    """
    Detect exon skipping patterns:
    - Donor loss at exon end (upstream)
    - Acceptor loss at next exon start (downstream)
    """
    exon_skipping_events = []
    
    # Check each exon in the gene
    for exon in gene_model.exons:
        # Look for donor loss at exon end
        dl_at_exon_end = check_donor_loss_at_position(
            delta_scores, exon.end_position
        )
        
        # Look for acceptor loss at next exon start
        next_exon = gene_model.get_next_exon(exon)
        if next_exon:
            al_at_next_start = check_acceptor_loss_at_position(
                delta_scores, next_exon.start_position
            )
            
            if dl_at_exon_end and al_at_next_start:
                exon_skipping_events.append({
                    'type': 'exon_skipping',
                    'skipped_exon': exon,
                    'dl_score': dl_at_exon_end,
                    'al_score': al_at_next_start,
                    'confidence': min(abs(dl_at_exon_end), abs(al_at_next_start))
                })
    
    return exon_skipping_events
```

#### **2. Intron Retention Detection**

```python
def detect_intron_retention(delta_scores, gene_model):
    """
    Detect intron retention patterns:
    - Donor loss at intron start
    - Acceptor loss at intron end
    """
    retention_events = []
    
    for intron in gene_model.introns:
        # Check for donor loss at intron start (exon-intron boundary)
        dl_at_start = check_donor_loss_at_position(
            delta_scores, intron.start_position
        )
        
        # Check for acceptor loss at intron end (intron-exon boundary)
        al_at_end = check_acceptor_loss_at_position(
            delta_scores, intron.end_position
        )
        
        if dl_at_start and al_at_end:
            retention_events.append({
                'type': 'intron_retention',
                'retained_intron': intron,
                'dl_score': dl_at_start,
                'al_score': al_at_end,
                'confidence': min(abs(dl_at_start), abs(al_at_end))
            })
    
    return retention_events
```

#### **3. Cryptic Site Activation Detection**

```python
def detect_cryptic_sites(delta_scores, gene_model, distance_threshold=50):
    """
    Detect cryptic splice site activation:
    - Acceptor/donor gains away from canonical sites
    """
    cryptic_events = []
    
    # Check for acceptor gains
    if abs(delta_scores['DS_AG']) > 0.2:
        distance_to_canonical = min_distance_to_canonical_acceptor(
            delta_scores['DP_AG'], gene_model
        )
        
        if distance_to_canonical > distance_threshold:
            cryptic_events.append({
                'type': 'cryptic_acceptor',
                'position': delta_scores['DP_AG'],
                'score': delta_scores['DS_AG'],
                'distance_to_canonical': distance_to_canonical
            })
    
    # Check for donor gains
    if abs(delta_scores['DS_DG']) > 0.2:
        distance_to_canonical = min_distance_to_canonical_donor(
            delta_scores['DP_DG'], gene_model
        )
        
        if distance_to_canonical > distance_threshold:
            cryptic_events.append({
                'type': 'cryptic_donor',
                'position': delta_scores['DP_DG'],
                'score': delta_scores['DS_DG'],
                'distance_to_canonical': distance_to_canonical
            })
    
    return cryptic_events
```

### 🔬 **Integration with Gene Annotation**

```python
class SplicingPatternAnalyzer:
    def __init__(self, gene_annotation):
        self.gene_annotation = gene_annotation
        
    def analyze_variant_splicing(self, variant, delta_scores):
        """
        Comprehensive analysis combining delta scores with gene structure
        """
        # Get gene model for variant
        gene_model = self.gene_annotation.get_gene_model(variant.gene)
        
        # Detect different pattern types
        patterns = {
            'exon_skipping': self.detect_exon_skipping(delta_scores, gene_model),
            'intron_retention': self.detect_intron_retention(delta_scores, gene_model),
            'cryptic_sites': self.detect_cryptic_sites(delta_scores, gene_model),
            'alternative_sites': self.detect_alternative_sites(delta_scores, gene_model)
        }
        
        # Prioritize patterns by confidence and clinical significance
        prioritized_patterns = self.prioritize_patterns(patterns)
        
        return prioritized_patterns
```

## 🏆 Gold Standard Datasets for Alternative Splicing Validation

### **1. Experimental Validation Databases**

#### **SpliceVarDB** (Most Comprehensive)
```python
# Recently published comprehensive database
database_info = {
    'name': 'SpliceVarDB',
    'description': 'Comprehensive database of experimentally validated splice variants',
    'validation_methods': [
        'RT-PCR',
        'RNA-seq',
        'Minigene assays',
        'Functional studies'
    ],
    'coverage': 'Thousands of validated splice variants',
    'access': 'Public database',
    'url': 'https://splicevardb.org/'  # Check for actual URL
}
```

#### **LOVD (Leiden Open Variation Database)**
```python
database_info = {
    'name': 'LOVD',
    'description': 'Gene-specific databases with splice variant annotations',
    'validation_methods': [
        'Literature curation',
        'Functional studies',
        'Clinical evidence'
    ],
    'coverage': 'Gene-specific collections',
    'access': 'Public, gene-specific databases',
    'url': 'https://www.lovd.nl/'
}
```

### **2. ClinVar Limitations and Opportunities**

#### **What ClinVar Provides:**
```python
clinvar_splicing_info = {
    'splice_site_variants': 'Yes - variants at canonical ±1,2 positions',
    'deep_intronic_variants': 'Limited - some with functional evidence',
    'alternative_splicing_patterns': 'No - individual variants only',
    'experimental_validation': 'Variable - depends on submitter',
    'clinical_significance': 'Yes - pathogenic/benign classifications'
}
```

#### **What ClinVar Lacks:**
```python
clinvar_limitations = {
    'pattern_annotation': 'No coordinated splicing pattern information',
    'tissue_specificity': 'Limited tissue-specific splicing data',
    'quantitative_effects': 'No quantitative splicing measurements',
    'experimental_details': 'Limited experimental methodology details'
}
```

### **3. Recommended Gold Standard Sources**

#### **Tier 1: High-Confidence Experimental Data**
```python
tier1_sources = {
    'SpliceVarDB': {
        'validation': 'Direct experimental evidence',
        'methods': 'RT-PCR, RNA-seq, minigene assays',
        'confidence': 'Very High',
        'use_case': 'Primary validation dataset'
    },
    'LOVD_curated': {
        'validation': 'Literature-curated functional evidence',
        'methods': 'Published functional studies',
        'confidence': 'High',
        'use_case': 'Gene-specific validation'
    }
}
```

#### **Tier 2: Clinical Evidence**
```python
tier2_sources = {
    'ClinVar_pathogenic': {
        'validation': 'Clinical pathogenicity evidence',
        'methods': 'Clinical interpretation',
        'confidence': 'Moderate',
        'use_case': 'Clinical relevance validation'
    },
    'HGMD_splice': {
        'validation': 'Disease mutation database',
        'methods': 'Literature curation',
        'confidence': 'Moderate',
        'use_case': 'Disease association validation'
    }
}
```

#### **Tier 3: Computational Validation**
```python
tier3_sources = {
    'GTEx_RNA_seq': {
        'validation': 'Population-level RNA-seq evidence',
        'methods': 'Tissue-specific expression analysis',
        'confidence': 'Moderate',
        'use_case': 'Population frequency validation'
    },
    'TCGA_RNA_seq': {
        'validation': 'Cancer-specific RNA-seq evidence',
        'methods': 'Tumor vs normal comparison',
        'confidence': 'Moderate',
        'use_case': 'Cancer-specific validation'
    }
}
```

### **4. Validation Strategy Framework**

```python
def create_validation_framework():
    """
    Multi-tier validation approach for alternative splicing patterns
    """
    validation_framework = {
        'experimental_validation': {
            'primary': 'SpliceVarDB experimentally validated variants',
            'secondary': 'LOVD functional studies',
            'methods': ['RT-PCR', 'RNA-seq', 'minigene assays']
        },
        
        'clinical_validation': {
            'primary': 'ClinVar pathogenic splice variants',
            'secondary': 'HGMD disease-associated variants',
            'criteria': ['Clinical significance', 'Evidence level']
        },
        
        'population_validation': {
            'primary': 'GTEx tissue-specific splicing',
            'secondary': 'TCGA cancer-specific patterns',
            'metrics': ['Splicing frequency', 'Tissue specificity']
        },
        
        'computational_validation': {
            'primary': 'Cross-algorithm consensus',
            'secondary': 'Evolutionary conservation',
            'methods': ['SpliceAI', 'MMSplice', 'SQUIRLS']
        }
    }
    
    return validation_framework
```

## 🎯 **Implementation Recommendations**

### **For MetaSpliceAI Integration:**

1. **Pattern Detection Pipeline:**
   ```python
   # Implement pattern detection algorithms
   # Integrate with existing gene annotation
   # Add confidence scoring for patterns
   ```

2. **Validation Dataset Integration:**
   ```python
   # Download and curate SpliceVarDB data
   # Extract ClinVar splice variants
   # Create validation test suites
   ```

3. **Meta-Learning Enhancement:**
   ```python
   # Use pattern-level features for meta-learning
   # Train on validated alternative splicing patterns
   # Incorporate tissue-specific splicing context
   ```

### **Next Steps:**

1. **Immediate:** Implement basic pattern detection algorithms
2. **Short-term:** Integrate SpliceVarDB validation data
3. **Medium-term:** Develop tissue-specific pattern models
4. **Long-term:** Create comprehensive splicing pattern database

---

*This guide provides the foundation for constructing alternative splicing patterns from OpenSpliceAI delta scores and establishing robust validation frameworks using available gold standard datasets.*
