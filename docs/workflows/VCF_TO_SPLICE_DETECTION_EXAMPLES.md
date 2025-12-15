# VCF to Splice Detection: Practical Examples

## Complete Workflow: VCF → Delta Scores → Alternative Splicing Detection

### Example 1: Exon Skipping in BRCA1

#### Input VCF
```vcf
#CHROM  POS      ID  REF  ALT  QUAL  FILTER  INFO
17      43094077  .   G    A    99    PASS    GENE=BRCA1;EXON=11
```

#### Step 1: Generate Delta Scores with SpliceAI/OpenSpliceAI
```python
from meta_spliceai.openspliceai import OpenSpliceAIAdapter

# Process variant through OpenSpliceAI
adapter = OpenSpliceAIAdapter()
result = adapter.analyze_variant(
    chrom="17",
    pos=43094077,
    ref="G", 
    alt="A",
    gene_id="BRCA1"
)

# Delta scores around variant position
delta_scores = {
    "43094070_acceptor": -0.85,  # Suppressed exon 11 start
    "43094180_donor": -0.78,     # Suppressed exon 11 end  
    "43093950_donor": +0.45,     # Enhanced upstream donor
    "43094250_acceptor": +0.52   # Enhanced downstream acceptor
}
```

#### Step 2: Convert to SpliceSite Objects
```python
from meta_spliceai.splice_engine.case_studies.analysis import SpliceSite

splice_sites = [
    # Suppressed exon boundaries (skipped exon)
    SpliceSite(43094070, 'acceptor', -0.85, True, False, '+', 'BRCA1'),
    SpliceSite(43094180, 'donor', -0.78, True, False, '+', 'BRCA1'),
    
    # Enhanced flanking sites (alternative splicing)
    SpliceSite(43093950, 'donor', +0.45, True, False, '+', 'BRCA1'),
    SpliceSite(43094250, 'acceptor', +0.52, True, False, '+', 'BRCA1')
]
```

#### Step 3: Detect Splicing Patterns
```python
from meta_spliceai.splice_engine.case_studies.analysis import SplicingPatternAnalyzer

analyzer = SplicingPatternAnalyzer()
patterns = analyzer.analyze_variant_impact(splice_sites, 43094077)

# Expected result: Exon skipping pattern
for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type.value}")           # exon_skipping
    print(f"Exon size: {pattern.coordinates['exon_size']}")   # 110bp
    print(f"Confidence: {pattern.confidence_score:.2f}")      # 0.87
    print(f"Severity: {pattern.severity_score:.2f}")          # 0.82
```

### Example 2: Intron Retention in DMD

#### Input VCF
```vcf
#CHROM  POS       ID  REF  ALT  QUAL  FILTER  INFO
X       32386854  .   C    T    99    PASS    GENE=DMD;INTRON=44
```

#### Delta Scores
```python
delta_scores = {
    "32386800_donor": -0.72,     # Suppressed intron 44 start
    "32389950_acceptor": -0.68,  # Suppressed intron 44 end
    # No alternative sites activated
}
```

#### SpliceSite Objects
```python
splice_sites = [
    SpliceSite(32386800, 'donor', -0.72, True, False, '+', 'DMD'),
    SpliceSite(32389950, 'acceptor', -0.68, True, False, '+', 'DMD')
]
```

#### Detection Result
```python
patterns = analyzer.analyze_variant_impact(splice_sites, 32386854)

# Expected: Intron retention
pattern = patterns[0]
print(f"Pattern: {pattern.pattern_type.value}")              # intron_retention
print(f"Intron size: {pattern.coordinates['intron_size']}")  # 3150bp
print(f"Consequence: {pattern.predicted_consequence}")       # "Retention of 3150bp intron"
```

### Example 3: Alternative 5' Splice Site in CFTR

#### Input VCF
```vcf
#CHROM  POS      ID  REF  ALT  QUAL  FILTER  INFO
7       117199644 .   G    A    99    PASS    GENE=CFTR;EXON=10
```

#### Delta Scores
```python
delta_scores = {
    "117199650_donor": -0.65,    # Suppressed canonical donor
    "117199635_donor": +0.78,    # Activated alternative donor (-15bp)
    "117199670_donor": +0.45     # Activated alternative donor (+20bp)
}
```

#### Detection Result
```python
splice_sites = [
    SpliceSite(117199650, 'donor', -0.65, True, False, '+', 'CFTR'),   # Canonical
    SpliceSite(117199635, 'donor', +0.78, False, False, '+', 'CFTR'),  # Alt -15bp
    SpliceSite(117199670, 'donor', +0.45, False, False, '+', 'CFTR')   # Alt +20bp
]

patterns = analyzer.analyze_variant_impact(splice_sites, 117199644)

# Expected: Two alternative 5'SS patterns
for pattern in patterns:
    if pattern.pattern_type.value == "alternative_5ss":
        shift = pattern.coordinates['shift_distance']
        print(f"Alternative 5'SS: {shift}bp shift")
        print(f"Confidence: {pattern.confidence_score:.2f}")
```

## Complete Integration Example

### Full Mutation Analysis Workflow
```python
class ClinicalSpliceAnalysis:
    def __init__(self):
        self.openspliceai = OpenSpliceAIAdapter()
        self.pattern_analyzer = SplicingPatternAnalyzer()
        
    def analyze_vcf_variant(self, vcf_record):
        """Complete analysis from VCF to clinical interpretation."""
        
        # Step 1: Extract variant information
        chrom = vcf_record.CHROM
        pos = vcf_record.POS
        ref = vcf_record.REF
        alt = str(vcf_record.ALT[0])
        gene_id = vcf_record.INFO.get('GENE', 'UNKNOWN')
        
        # Step 2: Generate delta scores
        openspliceai_result = self.openspliceai.analyze_variant(
            chrom=chrom, pos=pos, ref=ref, alt=alt, gene_id=gene_id
        )
        
        # Step 3: Convert to splice sites
        splice_sites = self._delta_scores_to_splice_sites(
            openspliceai_result.delta_scores, gene_id
        )
        
        # Step 4: Detect patterns
        patterns = self.pattern_analyzer.analyze_variant_impact(splice_sites, pos)
        
        # Step 5: Clinical interpretation
        clinical_impact = self._interpret_clinical_significance(patterns)
        
        return {
            'variant': f"{chrom}:{pos}{ref}>{alt}",
            'gene': gene_id,
            'delta_scores': openspliceai_result.delta_scores,
            'splice_patterns': patterns,
            'clinical_impact': clinical_impact
        }
    
    def _delta_scores_to_splice_sites(self, delta_scores, gene_id):
        """Convert delta score dictionary to SpliceSite objects."""
        splice_sites = []
        
        for position_type, delta in delta_scores.items():
            # Parse position and site type
            parts = position_type.split('_')
            position = int(parts[0])
            site_type = parts[1]  # 'donor' or 'acceptor'
            
            # Determine if canonical (from gene annotation) or cryptic
            is_canonical = self._is_canonical_site(position, site_type, gene_id)
            is_cryptic = not is_canonical and abs(delta) > 0.1
            
            splice_site = SpliceSite(
                position=position,
                site_type=site_type,
                delta_score=delta,
                is_canonical=is_canonical,
                is_cryptic=is_cryptic,
                strand='+',  # From gene annotation
                gene_id=gene_id
            )
            splice_sites.append(splice_site)
        
        return splice_sites
    
    def _interpret_clinical_significance(self, patterns):
        """Interpret clinical significance of detected patterns."""
        high_impact_patterns = [
            p for p in patterns 
            if p.severity_score > 0.7 and p.confidence_score > 0.8
        ]
        
        if not high_impact_patterns:
            return "Likely benign - no significant splicing impact"
        
        # Prioritize by clinical relevance
        clinical_priority = {
            "exon_skipping": "Pathogenic - likely loss of function",
            "intron_retention": "Pathogenic - frameshift or NMD",
            "alternative_5ss": "VUS - altered protein sequence",
            "alternative_3ss": "VUS - altered protein sequence",
            "cryptic_activation": "Pathogenic - aberrant splicing"
        }
        
        top_pattern = max(high_impact_patterns, key=lambda x: x.severity_score)
        return clinical_priority.get(top_pattern.pattern_type.value, "VUS")

# Usage example
analyzer = ClinicalSpliceAnalysis()

# Process VCF file
import pysam
vcf = pysam.VariantFile("variants.vcf")
for record in vcf:
    if record.INFO.get('GENE') in ['BRCA1', 'BRCA2', 'DMD', 'CFTR']:
        result = analyzer.analyze_vcf_variant(record)
        print(f"Variant: {result['variant']}")
        print(f"Gene: {result['gene']}")
        print(f"Clinical Impact: {result['clinical_impact']}")
        print("---")
```

## Database Comparison for Splice Analysis

### SpliceVarDB vs ClinVar

#### **SpliceVarDB** (Recommended for Research)
```python
# Advantages for splice analysis:
advantages = {
    "experimental_validation": "RT-PCR and RNA-seq validated splice effects",
    "detailed_annotations": "Specific splice pattern types documented",
    "quantitative_data": "PSI (Percent Spliced In) measurements",
    "tissue_specificity": "Tissue-specific splicing effects",
    "comprehensive_coverage": "~2,000 splice-affecting variants"
}

# Example SpliceVarDB entry
splicevardb_entry = {
    "variant": "BRCA1:c.4986+1G>A",
    "splice_effect": "Exon 16 skipping",
    "psi_change": -0.85,  # 85% reduction in inclusion
    "validation": "RT-PCR confirmed",
    "tissues": ["breast", "ovary", "lymphocytes"]
}
```

#### **ClinVar** (Better for Clinical Interpretation)
```python
# Advantages for clinical use:
advantages = {
    "clinical_significance": "Pathogenic/Benign classifications",
    "large_scale": "~2M variants with clinical annotations", 
    "expert_curation": "Professional society guidelines",
    "population_data": "Allele frequencies and penetrance",
    "regulatory_approval": "FDA-recognized for clinical use"
}

# Example ClinVar entry
clinvar_entry = {
    "variant": "NM_007294.3:c.4986+1G>A",
    "clinical_significance": "Pathogenic",
    "review_status": "criteria provided, multiple submitters",
    "condition": "Hereditary breast and ovarian cancer syndrome",
    "molecular_consequence": "splice donor variant"
}
```

### **Recommendation**: Hybrid Approach

```python
class HybridSpliceDatabase:
    def __init__(self):
        self.splicevardb = SpliceVarDBClient()
        self.clinvar = ClinVarClient()
    
    def validate_splice_prediction(self, variant, predicted_patterns):
        """Validate predictions against known data."""
        
        # Check SpliceVarDB for experimental validation
        experimental_data = self.splicevardb.lookup(variant)
        if experimental_data:
            return {
                "validation_source": "SpliceVarDB",
                "experimental_effect": experimental_data.splice_effect,
                "psi_change": experimental_data.psi_change,
                "prediction_accuracy": self._compare_prediction(
                    predicted_patterns, experimental_data
                )
            }
        
        # Fall back to ClinVar for clinical significance
        clinical_data = self.clinvar.lookup(variant)
        if clinical_data:
            return {
                "validation_source": "ClinVar", 
                "clinical_significance": clinical_data.significance,
                "review_status": clinical_data.review_status,
                "predicted_pathogenicity": self._predict_pathogenicity(
                    predicted_patterns
                )
            }
        
        return {"validation_source": "None", "status": "Novel variant"}
```

## Summary

**For your splice analysis workflow:**

1. **Primary Database**: **SpliceVarDB** for training/validation of splice pattern detection
2. **Secondary Database**: **ClinVar** for clinical interpretation and population context  
3. **Workflow**: VCF → OpenSpliceAI → Delta Scores → SplicingPatternAnalyzer → Clinical Interpretation
4. **Validation**: Cross-reference predictions with experimental data from SpliceVarDB

The SplicingPatternAnalyzer provides the computational framework, while SpliceVarDB offers the experimental ground truth for validating and improving your detection algorithms.
