# Splicing Pattern Analyzer - Usage Guide

## Overview

The Splicing Pattern Analyzer is a comprehensive module for detecting and characterizing alternative splicing patterns caused by genetic variants. It works in conjunction with the Cryptic Site Detector to provide complete splicing impact assessment.

## Core Modules

### 1. SplicingPatternAnalyzer
**Location**: `meta_spliceai/splice_engine/case_studies/analysis/splicing_pattern_analyzer.py`

**Purpose**: Detects and classifies alternative splicing patterns from splice site delta scores.

**Key Features**:
- Detects multiple pattern types: exon skipping, intron retention, alternative splice sites, cryptic activation
- Calculates pattern confidence scores based on delta score magnitudes
- Computes severity scores for clinical prioritization
- Groups patterns by gene for coordinated analysis

### 2. CrypticSiteDetector  
**Location**: `meta_spliceai/splice_engine/case_studies/analysis/cryptic_site_detector.py`

**Purpose**: Specialized detection and scoring of cryptic splice site activation patterns.

**Key Features**:
- Consensus sequence matching for donor/acceptor sites
- Polypyrimidine tract analysis for acceptor sites
- Position-weighted scoring matrices
- Confidence-based pattern ranking

## Integration into Mutation Workflows

### Basic Usage

```python
from meta_spliceai.splice_engine.case_studies.analysis import (
    SplicingPatternAnalyzer,
    SpliceSite,
    CrypticSiteDetector
)

# Initialize analyzer
analyzer = SplicingPatternAnalyzer()

# Create splice site objects from delta scores
splice_sites = [
    SpliceSite(
        position=1000,
        site_type='donor',
        delta_score=-0.8,
        is_canonical=True,
        is_cryptic=False,
        strand='+',
        gene_id='GENE1'
    ),
    SpliceSite(
        position=1100, 
        site_type='acceptor',
        delta_score=-0.7,
        is_canonical=True,
        is_cryptic=False,
        strand='+',
        gene_id='GENE1'
    )
]

# Analyze variant impact
variant_position = 1050
patterns = analyzer.analyze_variant_impact(splice_sites, variant_position)

# Process detected patterns
for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type.value}")
    print(f"Confidence: {pattern.confidence_score:.2f}")
    print(f"Severity: {pattern.severity_score:.2f}")
    print(f"Consequence: {pattern.predicted_consequence}")
```

### Integration with Mutation Analysis Workflow

```python
from meta_spliceai.splice_engine.case_studies.workflows import MutationAnalysisWorkflow

class EnhancedMutationWorkflow(MutationAnalysisWorkflow):
    def analyze_mutation(self, mutation_data):
        # Get delta scores from base analysis
        delta_scores = self.calculate_delta_scores(mutation_data)
        
        # Convert to SpliceSite objects
        splice_sites = []
        for key, delta in delta_scores.items():
            # Parse position and type from key
            position, site_type = self.parse_delta_key(key)
            
            # Determine if canonical or cryptic
            is_canonical = self.is_canonical_site(position, site_type)
            is_cryptic = not is_canonical
            
            splice_site = SpliceSite(
                position=position,
                site_type=site_type,
                delta_score=delta,
                is_canonical=is_canonical,
                is_cryptic=is_cryptic,
                strand=mutation_data.get('strand', '+'),
                gene_id=mutation_data.get('gene_id', 'UNKNOWN')
            )
            splice_sites.append(splice_site)
        
        # Detect patterns
        analyzer = SplicingPatternAnalyzer()
        patterns = analyzer.analyze_variant_impact(
            splice_sites, 
            mutation_data['position']
        )
        
        # Integrate cryptic site detection
        cryptic_detector = CrypticSiteDetector()
        cryptic_patterns = cryptic_detector.analyze_cryptic_activation_pattern(
            mutation_data['sequence'],
            mutation_data['mutated_sequence'],
            mutation_data['position']
        )
        
        return {
            'delta_scores': delta_scores,
            'splicing_patterns': patterns,
            'cryptic_patterns': cryptic_patterns
        }
```

## Pattern Types and Detection Logic

### Exon Skipping
- **Detection**: Suppressed acceptor-donor pairs (exon boundaries)
- **Requirements**: Both sites must be canonical and suppressed (delta < -0.1)
- **Confidence**: Based on magnitude of suppression and presence of enhanced flanking sites

### Intron Retention  
- **Detection**: Suppressed donor-acceptor pairs without alternatives
- **Requirements**: Consecutive sites forming intron boundaries, both suppressed
- **Confidence**: Higher for longer introns with strong suppression

### Alternative 5' Splice Sites
- **Detection**: Suppressed canonical donor with activated alternative nearby
- **Requirements**: Alternative within proximity window (default 100bp)
- **Confidence**: Based on relative strengths and distance

### Alternative 3' Splice Sites
- **Detection**: Suppressed canonical acceptor with activated alternative nearby
- **Requirements**: Alternative within proximity window
- **Confidence**: Based on relative strengths and distance

### Cryptic Activation
- **Detection**: New non-canonical sites with positive delta scores
- **Requirements**: Delta score > 0.1, marked as cryptic
- **Confidence**: Based on consensus sequence match and delta magnitude

## Configuration Parameters

### SplicingPatternAnalyzer Constants
```python
DELTA_SCORE_THRESHOLD = 0.1  # Minimum delta for significance
PROXIMITY_WINDOW = 100        # bp window for alternative sites
MIN_EXON_SIZE = 50           # Minimum exon size for skipping
MAX_EXON_SIZE = 500          # Maximum exon size for skipping  
MIN_INTRON_SIZE = 100        # Minimum intron size
MAX_INTRON_SIZE = 10000      # Maximum intron size
```

### CrypticSiteDetector Parameters
```python
min_distance = 20       # Minimum distance from known sites
max_distance = 300      # Maximum search distance
consensus_threshold = 0.7  # Minimum consensus match score
```

## Output Format

### SplicingPattern Object
```python
class SplicingPattern:
    pattern_type: SplicingPatternType  # Enum of pattern types
    affected_sites: List[SpliceSite]   # Sites involved in pattern
    confidence_score: float             # 0-1 confidence score
    coordinates: Dict[str, Any]         # Pattern-specific coordinates
    predicted_consequence: str          # Human-readable description
    
    @property
    def severity_score(self) -> float:
        """Calculate severity from affected sites."""
        # Computed from delta score magnitudes
```

### Pattern Summary Generation
```python
# Generate summary statistics
summary = analyzer.summarize_patterns(patterns)
print(f"Total patterns: {summary['total_patterns']}")
print(f"High confidence: {summary['high_confidence_count']}")
print(f"Pattern distribution: {summary['pattern_type_distribution']}")
```

## Testing and Validation

### Unit Tests
- **Location**: `tests/unit/test_splicing_pattern_analyzer.py`
- **Coverage**: 18 test cases covering all pattern types
- **Status**: All tests passing ✅

### Test Categories
1. Pattern detection (exon skipping, intron retention, alternatives)
2. Confidence calculation
3. Severity scoring  
4. Multi-gene handling
5. Edge cases (empty sites, single sites)
6. Delta threshold validation

### Running Tests
```bash
# Run splicing pattern analyzer tests
python -m pytest tests/unit/test_splicing_pattern_analyzer.py -v

# Run cryptic site detector tests  
python -m pytest tests/unit/test_cryptic_site_detector.py -v

# Run all analysis module tests
python -m pytest tests/unit/ -k "pattern or cryptic" -v
```

## Clinical Applications

### Variant Prioritization
```python
def prioritize_variants(patterns):
    """Prioritize variants by splicing impact."""
    # Sort by severity and confidence
    high_impact = [
        p for p in patterns 
        if p.severity_score > 0.7 and p.confidence_score > 0.8
    ]
    
    # Filter for clinically relevant patterns
    clinical_patterns = [
        p for p in high_impact
        if p.pattern_type in [
            SplicingPatternType.EXON_SKIPPING,
            SplicingPatternType.CRYPTIC_ACTIVATION
        ]
    ]
    
    return sorted(clinical_patterns, 
                  key=lambda x: x.severity_score, 
                  reverse=True)
```

### Meta-Model Training Data
```python
def prepare_training_features(patterns):
    """Extract features for meta-model training."""
    features = []
    for pattern in patterns:
        feature_dict = {
            'pattern_type': pattern.pattern_type.value,
            'confidence': pattern.confidence_score,
            'severity': pattern.severity_score,
            'num_affected_sites': len(pattern.affected_sites),
            'max_delta': max(abs(s.delta_score) for s in pattern.affected_sites),
            'pattern_span': pattern.coordinates.get('shift_distance', 0)
        }
        features.append(feature_dict)
    return pd.DataFrame(features)
```

## Best Practices

1. **Always validate splice site annotations** before pattern detection
2. **Use appropriate delta score thresholds** based on model confidence
3. **Consider gene context** when interpreting patterns
4. **Combine multiple detection methods** for comprehensive analysis
5. **Validate high-impact patterns** with experimental data when available

## Future Enhancements

- [ ] Integration with gene expression data
- [ ] Machine learning-based pattern ranking
- [ ] Tissue-specific splicing patterns
- [ ] Population frequency annotations
- [ ] Automated report generation

## Support

For questions or issues, please refer to:
- Technical documentation: `/docs/development/`
- API reference: `/meta_spliceai/splice_engine/case_studies/analysis/`
- Test examples: `/tests/unit/test_splicing_pattern_analyzer.py`
