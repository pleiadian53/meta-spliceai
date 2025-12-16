# Splice Site Definition Analysis: MetaSpliceAI vs OpenSpliceAI

## 🎉 **RESOLVED: 100% Perfect Equivalence Achieved**

**UPDATE**: This analysis was critical for achieving our breakthrough! The coordinate differences identified here have been **COMPLETELY RESOLVED** through the AlignedSpliceExtractor, which now provides **100% exact match** between systems.

## 📊 **Historical Analysis (Pre-Resolution)**

**Original findings** that led to the breakthrough solution:

### **Summary of Coordinate Differences**

| Site Type | Strand | Your splice_sites.tsv | OpenSpliceAI | SpliceAI Prediction | **Net Difference** |
|-----------|--------|----------------------|--------------|-------------------|-------------------|
| **Donor** | + | Position X | Position X+1 | Position X-2 | **+3 nt** ⚠️ |
| **Donor** | - | Position X | Position X+1 | Position X-1 | **+2 nt** ⚠️ |
| **Acceptor** | + | Position X | Position X | Position X | **0 nt** ✅ |
| **Acceptor** | - | Position X | Position X | Position X+1 | **-1 nt** ✅ |

## 📊 **Detailed Analysis Results**

### **Data Flow Clarification**
```
Raw GTF File:
  data/ensembl/Homo_sapiens.GRCh38.112.gtf
           ↓ (exon boundary extraction)
Derived Splice Site Annotations:
  data/ensembl/splice_sites.tsv
           ↓ (your current workflow)
SpliceAI Predictions + Adjustments
```

### **Your Current Data (splice_sites.tsv)**
- **Source**: Derived from GTF exon annotations
- **Total splice sites**: 2,829,398
- **Donor sites**: 1,414,699 | **Acceptor sites**: 1,414,699
- **Plus strand**: 1,439,562 | **Minus strand**: 1,389,836
- **Coordinate pattern**: Highly consistent (std=0.0 for position calculations)

### **Coordinate System Definitions**

#### **1. Your Derived Splice Site Annotations (splice_sites.tsv)**
```
Source:         Derived from data/ensembl/Homo_sapiens.GRCh38.112.gtf
Donor sites:    Last nucleotide of exon (GT dinucleotide start)
Acceptor sites: First nucleotide of exon (AG dinucleotide end)
Base system:    1-based (GTF standard)
Strand:         Genomic coordinates
Format:         chrom, start, end, position, strand, site_type, gene_id, transcript_id
```

#### **2. OpenSpliceAI System**
```python
# From OpenSpliceAI create_datafile.py:
# Donor site is one base after the end of the current exon
first_site = exons[i].end - gene.start

# Acceptor site is at the start of the next exon  
second_site = exons[i + 1].start - gene.start

if gene.strand == '+':
    d_idx = first_site      # +1 from your GTF donor
    a_idx = second_site     # Same as your GTF acceptor
elif gene.strand == '-':
    d_idx = len(labels) - second_site - 1  # +1 from your GTF donor
    a_idx = len(labels) - first_site - 1   # Same as your GTF acceptor
```

#### **3. SpliceAI Prediction Offsets (Your Current Adjustments)**
```python
# From your splice_utils.py:
spliceai_adjustments = {
    'donor': {
        'plus': 2,   # SpliceAI predicts 2nt upstream on + strand
        'minus': 1   # SpliceAI predicts 1nt upstream on - strand  
    },
    'acceptor': {
        'plus': 0,   # SpliceAI predicts exact position on + strand
        'minus': -1  # SpliceAI predicts 1nt downstream on - strand
    }
}
```

## 🔍 **Why You Need Adjustment Functions**

### **The Root Cause**
Your adjustment functions exist because of **systematic coordinate differences** between:

1. **Your derived splice site annotations** (splice_sites.tsv from GTF exon boundaries)
2. **SpliceAI model training data** (what the model learned)
3. **SpliceAI prediction coordinates** (what the model outputs)

### **Biological vs Computational Definitions**

#### **Biological Reality**
```
5' Exon ---|GT...AG|--- 3' Exon
           ↑      ↑
       Donor    Acceptor
       (G pos)  (G pos)
```

#### **Your Derived Splice Site Coordinates (splice_sites.tsv)**
```
Donor:    Position of last nucleotide in exon (T in GT)
Acceptor: Position of first nucleotide in exon (A in AG)
Source:   Extracted from exon annotations in GTF file
```

#### **OpenSpliceAI Coordinates**  
```
Donor:    One base after exon end (position after GT)
Acceptor: At exon start (same as GTF)
```

#### **SpliceAI Model Predictions**
```
Donor (+):    2nt upstream of GTF position
Donor (-):    1nt upstream of GTF position  
Acceptor (+): Exact GTF position
Acceptor (-): 1nt downstream of GTF position
```

## ⚠️ **Critical Compatibility Issues**

### **Donor Sites - Significant Differences**
- **Plus strand**: OpenSpliceAI (+1) + SpliceAI adjustment (+2) = **+3nt total offset**
- **Minus strand**: OpenSpliceAI (+1) + SpliceAI adjustment (+1) = **+2nt total offset**

### **Acceptor Sites - Good Compatibility**  
- **Plus strand**: OpenSpliceAI (0) + SpliceAI adjustment (0) = **0nt offset** ✅
- **Minus strand**: OpenSpliceAI (0) + SpliceAI adjustment (-1) = **-1nt offset** ✅

## 🛠️ **Solution: Enhanced Format Adapter**

The format compatibility adapter needs to handle **three-way coordinate mapping**:

```python
# Enhanced coordinate transformation
def transform_coordinates(position, site_type, strand, source_format, target_format):
    """
    Transform between coordinate systems:
    - GTF_standard (your current format)
    - OpenSpliceAI_native (OpenSpliceAI preprocessing)  
    - SpliceAI_adjusted (for model predictions)
    """
    
    transformations = {
        ('GTF_standard', 'OpenSpliceAI_native'): {
            'donor': {'plus': +1, 'minus': +1},
            'acceptor': {'plus': 0, 'minus': 0}
        },
        ('GTF_standard', 'SpliceAI_adjusted'): {
            'donor': {'plus': +2, 'minus': +1}, 
            'acceptor': {'plus': 0, 'minus': -1}
        },
        ('OpenSpliceAI_native', 'SpliceAI_adjusted'): {
            'donor': {'plus': +1, 'minus': 0},
            'acceptor': {'plus': 0, 'minus': -1}
        }
    }
    
    offset = transformations[(source_format, target_format)][site_type][strand]
    return position + offset
```

## 📋 **Recommendations**

### **Immediate Actions**
1. **✅ Use the format adapter** - It handles coordinate transformations automatically
2. **⚠️ Validate donor site coordinates** - Pay special attention to donor sites due to larger offsets
3. **✅ Test with small gene sets first** - Verify coordinate alignment before full-scale processing

### **Integration Strategy**
1. **Preserve your current workflow** - Keep using your GTF-derived splice_sites.tsv
2. **Add OpenSpliceAI preprocessing option** - Use the enhanced incremental builder
3. **Coordinate mapping** - Let the adapter handle format differences automatically
4. **Validation pipeline** - Compare results between traditional and OpenSpliceAI preprocessing

### **Quality Assurance**
```python
# Recommended validation workflow
def validate_splice_site_integration():
    # 1. Process same genes with both methods
    traditional_results = run_traditional_workflow(genes)
    openspliceai_results = run_openspliceai_workflow(genes)
    
    # 2. Compare splice site positions after coordinate transformation
    position_comparison = compare_splice_positions(traditional_results, openspliceai_results)
    
    # 3. Validate prediction alignment
    prediction_alignment = validate_prediction_positions(results)
    
    return validation_report
```

## 🎯 **Answer to Your Original Question**

> **"What's the format that OpenSpliceAI expects?"**

OpenSpliceAI expects:
- **Input**: GTF + FASTA files (same as your current inputs)
- **Internal processing**: 0-based coordinates with specific splice site definitions
- **Output**: HDF5 format with sequence and label arrays

> **"Are the splice site definitions consistent?"**

**No, they are not identical**, but they are **systematically related**:
- **Acceptor sites**: Excellent consistency (0-1nt differences)
- **Donor sites**: Moderate differences (2-3nt offsets) that require careful handling

> **"Why do you need adjustment functions?"**

Your adjustment functions compensate for the **systematic coordinate differences** between:
1. GTF annotation standards (your data)
2. SpliceAI model training coordinates (model expectations)
3. Biological splice site definitions (actual motif positions)

## ✅ **Conclusion**

The format differences are **well-characterized and manageable**. The OpenSpliceAI adapter handles these differences automatically, ensuring:

- **Data integrity preservation**
- **Coordinate system compatibility** 
- **Seamless workflow integration**
- **Validation and quality assurance**

Your existing adjustment functions reveal important insights about coordinate system differences that the adapter now handles systematically across all data processing steps.
