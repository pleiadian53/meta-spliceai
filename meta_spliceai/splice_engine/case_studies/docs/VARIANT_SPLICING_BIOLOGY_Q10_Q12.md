# Variant Impact Analysis on RNA Splicing: Biological Principles (Q10-Q12)

**Document Version**: 1.0  
**Date**: 2025-07-28  
**Purpose**: Explain the biological foundations of variant impact analysis on RNA splicing mechanisms

---

## 🎯 **EXECUTIVE SUMMARY**

This document addresses three fundamental questions about the biological principles underlying variant impact analysis on RNA splicing:

- **Q10**: Principles of variant impact analysis and delta score interpretation
- **Q11**: Mechanisms by which genetic variants alter splicing patterns  
- **Q12**: Concrete pathogenic variant example with splicing disruption

**Key Insight**: Genetic variants can disrupt the precise molecular recognition signals that guide RNA splicing, leading to aberrant transcript isoforms that often result in disease through loss of protein function, gain of toxic function, or altered protein expression levels.

---

## 🧬 **Q10: PRINCIPLES OF VARIANT IMPACT ANALYSIS**

### **What are Delta Scores?**

**Delta scores** represent the **predicted change in splice site strength** caused by a genetic variant, measured as the difference between reference and alternative sequence predictions.

#### **Mathematical Definition**
```
Delta Score = Prediction(Alternative Sequence) - Prediction(Reference Sequence)

Where:
- Prediction ranges from 0.0 (no splicing) to 1.0 (strong splicing)
- Delta scores range from -1.0 to +1.0
- Positive values indicate splice site strengthening
- Negative values indicate splice site weakening
```

#### **Technical Implementation Details**

##### **Sequence Processing**
```python
# SpliceAI processes sequences in fixed windows
window_size = 2 * flanking_size + coverage  # Typically ~10,000 bp
sequence_window = reference_genome[pos - window_size//2 : pos + window_size//2]

# Model predictions are vectors for each position
y_ref = model.predict(reference_sequence)  # Shape: [seq_length, 3]
y_alt = model.predict(alternative_sequence) # Shape: [seq_length, 3]

# Where dimension meanings are:
# [:, 0] = Background (no splice site)
# [:, 1] = Acceptor site probability  
# [:, 2] = Donor site probability
```

##### **Delta Score Calculation**
```python
# Calculate delta scores for every position
delta_acceptor = y_alt[:, 1] - y_ref[:, 1]  # Vector of acceptor deltas
delta_donor = y_alt[:, 2] - y_ref[:, 2]     # Vector of donor deltas

# Report maximum delta scores within search window
DS_AG = max(delta_acceptor)  # Maximum acceptor gain
DS_AL = max(-delta_acceptor) # Maximum acceptor loss  
DS_DG = max(delta_donor)     # Maximum donor gain
DS_DL = max(-delta_donor)    # Maximum donor loss

# Corresponding positions where max deltas occur
DP_AG = argmax(delta_acceptor) - variant_position
DP_AL = argmax(-delta_acceptor) - variant_position
DP_DG = argmax(delta_donor) - variant_position  
DP_DL = argmax(-delta_donor) - variant_position
```

##### **Handling Different Sequence Lengths**
```python
# SpliceAI intelligently handles indels
if len(ref) > len(alt):  # Deletion
    # Insert zeros for deleted positions
    y_alt = insert_zeros_at_deletion_site(y_alt)
    
elif len(alt) > len(ref):  # Insertion
    # Take maximum over inserted region
    y_alt = collapse_insertion_to_max(y_alt)
    
# This allows analysis of:
# - SNVs (same length)
# - Indels (different lengths)
# - Complex variants (with limitations)
```

#### **OpenSpliceAI Delta Score Components**
```python
# Four delta score types from OpenSpliceAI:
delta_scores = {
    'AG': acceptor_gain,      # New acceptor site created (+)
    'AL': acceptor_loss,      # Existing acceptor site lost (-)
    'DG': donor_gain,         # New donor site created (+)
    'DL': donor_loss          # Existing donor site lost (-)
}

# Example interpretation:
# AG = +0.85 → High confidence (85%) that a new acceptor site is created
# DL = -0.91 → High confidence (91%) that an existing donor site is lost
# Note: The magnitude represents probability/confidence, NOT distance in base pairs
```

### **Biological Interpretation of Delta Scores**

#### **Score Magnitude Interpretation**
```python
def interpret_delta_score_magnitude(score: float) -> str:
    """Biological interpretation of delta score magnitude"""
    
    abs_score = abs(score)
    
    if abs_score >= 0.8:
        return "High confidence - likely functional impact"
    elif abs_score >= 0.5:
        return "Moderate confidence - probable functional impact"  
    elif abs_score >= 0.2:
        return "Low-moderate confidence - possible functional impact"
    else:
        return "Low confidence - minimal predicted impact"

# Clinical significance thresholds (commonly used):
# - Pathogenic prediction: |delta_score| >= 0.2
# - High confidence pathogenic: |delta_score| >= 0.5
```

#### **Distance Interpretation**
```python
def interpret_delta_position(position: int, variant_pos: int) -> str:
    """Biological interpretation of delta score position"""
    
    distance = abs(position)
    
    if distance <= 2:
        return "Immediate splice site - direct motif disruption"
    elif distance <= 10:
        return "Proximal region - affects splice site recognition"
    elif distance <= 50:
        return "Local region - may create competing splice sites"
    else:
        return "Distant region - context-dependent effects"
```

### **Biological Foundation: Splice Site Recognition**

#### **Normal Splice Site Recognition Mechanism**
```
5' Splice Site (Donor):    GT|AAGT (consensus)
3' Splice Site (Acceptor): (Py)nNCAG|G (consensus)

Recognition Process:
1. U1 snRNP recognizes 5' splice site
2. U2AF recognizes polypyrimidine tract and 3' splice site
3. U2 snRNP binds to branch point
4. U4/U6•U5 tri-snRNP completes spliceosome assembly
5. Two-step transesterification removes intron
```

#### **How Variants Disrupt Recognition**
```python
class SpliceDisruptionMechanism:
    """Mechanisms by which variants disrupt splice site recognition"""
    
    DIRECT_MOTIF_DISRUPTION = "Variant changes GT/AG dinucleotides"
    CONSENSUS_WEAKENING = "Variant reduces splice site score"
    COMPETING_SITE_CREATION = "Variant creates stronger alternative site"
    REGULATORY_ELEMENT_DISRUPTION = "Variant affects ESE/ESS elements"
    BRANCH_POINT_DISRUPTION = "Variant affects branch point adenosine"
    POLYPYRIMIDINE_TRACT_DISRUPTION = "Variant disrupts Py tract"
```

---

## 🔄 **Q11: HOW VARIANTS ALTER SPLICING PATTERNS**

### **Major Splicing Alteration Mechanisms**

#### **1. Cryptic Splice Site Activation**
```
Normal Splicing:
Exon 1 ----GT........AG---- Exon 2

Variant Creates Cryptic Site:
Exon 1 ----GT....*GT...AG---- Exon 2
                  ↑
            New cryptic donor
            (stronger than normal)

Result: Alternative splicing using cryptic site
```

**Biological Consequence**: 
- Partial exon inclusion/exclusion
- Frame-shift if cryptic site is out-of-frame
- Altered protein domain structure

#### **2. Canonical Splice Site Loss**
```
Normal Splicing:
Exon 1 ----GT........AG---- Exon 2

Variant Disrupts Canonical Site:
Exon 1 ----AT........AG---- Exon 2
           ↑
    GT→AT disrupts donor

Result: Exon skipping or cryptic site usage
```

**Biological Consequence**:
- Complete exon skipping (if no cryptic sites available)
- Use of weaker cryptic sites
- Often leads to nonsense-mediated decay

#### **3. Exon Skipping**
```
Normal Splicing:
Exon 1 ----GT---- Exon 2 ----GT---- Exon 3

Variant Weakens Exon 2 Splice Sites:
Exon 1 ----GT---- Exon 2 ----AT---- Exon 3
                           ↑
                    Weakened donor

Result: Exon 1 directly spliced to Exon 3
```

**Biological Consequence**:
- Loss of exon 2 coding sequence
- Potential frame-shift
- Loss of protein functional domains

#### **4. Intron Retention**
```
Normal Splicing:
Exon 1 ----GT........AG---- Exon 2
       (intron removed)

Variant Weakens Both Splice Sites:
Exon 1 ----AT........GG---- Exon 2
           ↑         ↑
       Weak donor  Weak acceptor

Result: Intron remains in mature transcript
```

**Biological Consequence**:
- Premature stop codons in retained intron
- Nonsense-mediated decay
- Reduced protein expression

#### **5. Pseudoexon Activation**
```
Normal Splicing:
Exon 1 ----GT................AG---- Exon 2
           (long intron)

Variant Creates Internal Splice Sites:
Exon 1 ----GT....AG--GT....AG---- Exon 2
                  ↑    ↑
            New pseudoexon

Result: Inclusion of intronic sequence as exon
```

**Biological Consequence**:
- Insertion of intronic sequence
- Frame-shift and premature termination
- Gain of cryptic protein domains

### **Quantitative Impact Assessment**

#### **Splice Site Strength Calculation**
```python
def calculate_splice_site_strength(sequence: str, site_type: str) -> float:
    """
    Calculate splice site strength using position weight matrices
    
    Based on:
    - Nucleotide frequencies at each position
    - Evolutionary conservation
    - Functional validation data
    """
    
    if site_type == 'donor':
        # 5' splice site: positions -3 to +6 relative to GT
        pwm = DONOR_PWM
        motif_start = sequence.find('GT')
    else:  # acceptor
        # 3' splice site: positions -20 to +3 relative to AG
        pwm = ACCEPTOR_PWM  
        motif_start = sequence.find('AG')
    
    score = 0.0
    for i, nucleotide in enumerate(sequence[motif_start-3:motif_start+6]):
        score += pwm[i][nucleotide]
    
    return 1.0 / (1.0 + exp(-score))  # Sigmoid transformation
```

#### **Delta Score Calculation Process**
```python
def calculate_delta_scores(reference_seq: str, 
                          alternative_seq: str,
                          variant_position: int) -> dict:
    """
    Calculate delta scores for all potential splice sites
    
    This is the core of variant impact analysis!
    """
    
    # Extract flanking sequence (typically ±5000bp)
    ref_context = extract_sequence_context(reference_seq, variant_position)
    alt_context = extract_sequence_context(alternative_seq, variant_position)
    
    # Predict splice sites in both sequences
    ref_predictions = predict_splice_sites(ref_context)
    alt_predictions = predict_splice_sites(alt_context)
    
    # Calculate delta scores for each position
    delta_scores = {}
    for position in range(-50, 51):  # Within 50bp of variant
        ref_donor = ref_predictions.get(position, {}).get('donor', 0.0)
        alt_donor = alt_predictions.get(position, {}).get('donor', 0.0)
        ref_acceptor = ref_predictions.get(position, {}).get('acceptor', 0.0)
        alt_acceptor = alt_predictions.get(position, {}).get('acceptor', 0.0)
        
        delta_scores[position] = {
            'DG': alt_donor - ref_donor,      # Donor gain
            'DL': ref_donor - alt_donor,      # Donor loss  
            'AG': alt_acceptor - ref_acceptor, # Acceptor gain
            'AL': ref_acceptor - alt_acceptor  # Acceptor loss
        }
    
    # Return highest magnitude scores
    return extract_significant_deltas(delta_scores)
```

---

## 🔬 **Q12: CONCRETE PATHOGENIC VARIANT EXAMPLE**

### **Case Study: CFTR c.1521_1523delCTT (ΔF508)**

Let's walk through a detailed example of how the most common cystic fibrosis mutation affects splicing.

#### **Variant Details**
```
Gene: CFTR (Cystic Fibrosis Transmembrane Conductance Regulator)
Location: chr7:117559590-117559592
Variant: c.1521_1523delCTT (deletion of CTT)
Protein: p.Phe508del (ΔF508)
Disease: Cystic Fibrosis
Frequency: ~70% of CF alleles worldwide
```

#### **Normal Splicing Context**
```
CFTR Exon 10 (containing F508):
5' splice site: ...CTTTGGTGT|gtaagt...
                      ↑
               Normal GT donor

3' splice site: ...ttttag|GAAATAT...
                       ↑
               Normal AG acceptor

Normal Transcript:
Exon 9 -- Exon 10 (with F508) -- Exon 11
```

#### **Impact of ΔF508 Deletion**
```
Reference Sequence:
...AATCGATCTTGGTGTTTCCTATGATGAATATAGATACAGAAGCGTCATCAAAGCATGCCAACTAGAAGAG...
                ↑
         F508 codon (TTT)

Alternative Sequence (ΔF508):
...AATCGATCTTGGTGT---CCTATGATGAATATAGATACAGAAGCGTCATCAAAGCATGCCAACTAGAAGAG...
                ↑
         F508 deleted

Delta Score Analysis:
Position: +15bp from deletion
DG (Donor Gain): +0.23 (weak cryptic donor created)
AL (Acceptor Loss): +0.15 (downstream acceptor weakened)
```

#### **Splicing Consequences**

**Primary Effect: Protein Folding Defect**
```python
# The ΔF508 primarily affects protein folding, not splicing
# But creates secondary splicing effects:

splicing_effects = {
    'primary_transcript': 'Normal splicing maintained',
    'cryptic_donor_activation': {
        'position': '+15bp downstream of deletion',
        'strength': 'Weak (delta_score = +0.23)',
        'frequency': '~5% of transcripts'
    },
    'nonsense_mediated_decay': {
        'trigger': 'Cryptic splicing creates premature stop',
        'effect': 'Reduced transcript levels'
    }
}
```

**Biological Mechanism**:
1. **Primary Effect**: ΔF508 deletion removes phenylalanine from NBD1 domain
2. **Protein Folding**: Misfolded protein retained in ER, degraded
3. **Secondary Splicing**: Deletion creates weak cryptic splice sites
4. **Transcript Instability**: Some aberrant transcripts undergo NMD

#### **Clinical Consequences**
```python
clinical_impact = {
    'protein_function': {
        'normal_cftr': 'Chloride channel at cell surface',
        'delta_f508_cftr': 'Misfolded, retained in ER, degraded'
    },
    'cellular_phenotype': {
        'chloride_transport': 'Severely reduced (~1-5% of normal)',
        'cell_surface_expression': 'Minimal (<1% of normal)'
    },
    'clinical_phenotype': {
        'lung_disease': 'Progressive bronchiectasis, infections',
        'pancreatic_insufficiency': 'Malabsorption, diabetes',
        'other_manifestations': 'Sinusitis, male infertility'
    }
}
```

### **Alternative Example: Canonical Splice Site Disruption**

#### **Case Study: BRCA1 c.5266dupC (5382insC)**
```
Gene: BRCA1 (Breast Cancer 1)
Location: chr17:43070927
Variant: c.5266dupC (insertion of C)
Effect: Disrupts exon 20 donor splice site
Disease: Hereditary Breast and Ovarian Cancer
```

#### **Splicing Analysis**
```
Normal Exon 20 Donor Site:
...GAAGATACTG|gtgagt...
           ↑
    Strong GT donor (score: 0.89)

Variant Effect (c.5266dupC):
...GAAGATACTCG|gtgagt...
            ↑
    Inserted C disrupts donor

Delta Score Analysis:
Position: 0 (at splice site)
DL (Donor Loss): +0.87 (near-complete loss of donor)
AG (Acceptor Gain): +0.34 (cryptic acceptor 12bp upstream)
```

#### **Splicing Consequences**
```python
splicing_outcomes = {
    'exon_skipping': {
        'frequency': '~60% of transcripts',
        'mechanism': 'Exon 20 skipped due to donor loss',
        'consequence': 'Frameshift, premature stop codon'
    },
    'cryptic_site_usage': {
        'frequency': '~30% of transcripts', 
        'mechanism': 'Weak cryptic acceptor 12bp upstream',
        'consequence': '4 amino acid deletion, partial function loss'
    },
    'intron_retention': {
        'frequency': '~10% of transcripts',
        'mechanism': 'Failure to remove intron 20',
        'consequence': 'Premature stop, nonsense-mediated decay'
    }
}
```

#### **Delta Score Interpretation**
```python
def interpret_brca1_variant():
    """Biological interpretation of BRCA1 c.5266dupC delta scores"""
    
    interpretation = {
        'DL_score': 0.87,
        'biological_meaning': 'Near-complete loss of normal donor site',
        'predicted_outcome': 'Severe splicing disruption',
        'clinical_significance': 'Pathogenic',
        
        'AG_score': 0.34,
        'biological_meaning': 'Moderate strength cryptic acceptor created',
        'predicted_outcome': 'Partial rescue through alternative splicing',
        'clinical_significance': 'Reduces but does not eliminate pathogenicity'
    }
    
    return interpretation
```

---

## 🎯 **INTEGRATION WITH COMPUTATIONAL WORKFLOW**

### **How Biological Principles Guide Computational Analysis**

#### **1. Delta Score Thresholds Based on Biology**
```python
# Biologically-informed thresholds
PATHOGENIC_THRESHOLDS = {
    'high_confidence': 0.8,    # Near-complete splice site loss/gain
    'moderate_confidence': 0.5, # Significant alteration in site strength  
    'low_confidence': 0.2,     # Detectable but uncertain functional impact
    'minimal_impact': 0.1      # Below noise threshold
}
```

#### **2. Distance-Dependent Effects**
```python
def weight_delta_score_by_distance(delta_score: float, distance: int) -> float:
    """Weight delta scores based on biological distance effects"""
    
    if distance == 0:
        return delta_score * 1.0    # Direct splice site impact
    elif abs(distance) <= 2:
        return delta_score * 0.9    # Immediate vicinity
    elif abs(distance) <= 10:
        return delta_score * 0.7    # Local regulatory region
    elif abs(distance) <= 50:
        return delta_score * 0.5    # Extended regulatory region
    else:
        return delta_score * 0.2    # Distant effects (context-dependent)
```

#### **3. Multi-Modal Impact Assessment**
```python
def assess_variant_pathogenicity(delta_scores: dict, 
                               variant_context: dict) -> dict:
    """Comprehensive pathogenicity assessment based on biological principles"""
    
    assessment = {
        'splice_disruption_score': calculate_splice_disruption(delta_scores),
        'protein_impact_prediction': predict_protein_consequences(delta_scores),
        'transcript_stability': assess_nmd_likelihood(delta_scores),
        'clinical_significance': integrate_clinical_evidence(variant_context),
        'confidence_level': calculate_prediction_confidence(delta_scores)
    }
    
    return assessment
```

---

## 🧬 **BIOLOGICAL VALIDATION OF COMPUTATIONAL PREDICTIONS**

### **Experimental Validation Methods**

#### **1. Minigene Assays**
```python
# Experimental validation of splice predictions
minigene_validation = {
    'method': 'Transfect cells with genomic construct containing variant',
    'readout': 'RT-PCR analysis of splicing products',
    'validation_metrics': {
        'splice_site_usage': 'Quantify normal vs. aberrant splicing',
        'isoform_ratios': 'Measure relative abundance of splice variants',
        'functional_impact': 'Assess protein expression and activity'
    }
}
```

#### **2. Patient RNA Analysis**
```python
# Clinical validation using patient samples
patient_rna_analysis = {
    'sample_types': ['Blood', 'Fibroblasts', 'Tissue biopsies'],
    'analysis_methods': {
        'rt_pcr': 'Targeted analysis of specific splice junctions',
        'rna_seq': 'Genome-wide splicing analysis',
        'long_read_sequencing': 'Full-length isoform characterization'
    },
    'validation_outcomes': {
        'confirms_prediction': 'Delta scores match experimental results',
        'refines_prediction': 'Additional complexity revealed',
        'contradicts_prediction': 'Requires model refinement'
    }
}
```

---

## 📊 **SUMMARY: BIOLOGICAL PRINCIPLES → COMPUTATIONAL IMPLEMENTATION**

### **Key Biological Insights**
1. **Splice Site Recognition**: Precise molecular interactions guide normal splicing
2. **Variant Disruption**: Multiple mechanisms can alter splicing patterns
3. **Quantitative Effects**: Splice site strength changes are measurable and predictable
4. **Clinical Consequences**: Splicing alterations often lead to disease through protein dysfunction

### **Computational Implementation**
1. **Delta Scores**: Quantify predicted changes in splice site strength
2. **Position-Dependent Effects**: Weight predictions based on distance from variant
3. **Multi-Modal Assessment**: Integrate multiple types of evidence
4. **Clinical Translation**: Convert predictions to actionable clinical insights

### **Integration with Meta-Model Training**
```python
# How biological principles inform training data
training_data_principles = {
    'sequence_context': 'Include sufficient flanking sequence for splice recognition',
    'label_generation': 'Multi-task labels reflecting biological mechanisms',
    'feature_engineering': 'Capture splice site strength, conservation, regulatory elements',
    'validation_strategy': 'Test against experimentally validated variants'
}
```

**This biological foundation ensures that computational predictions are grounded in mechanistic understanding and clinically relevant for disease diagnosis and treatment!** 🧬🎯🔬
