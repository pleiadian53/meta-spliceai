# Variant Splicing Analysis Documentation

## Overview

This directory contains comprehensive documentation for variant splicing analysis using OpenSpliceAI within the MetaSpliceAI framework. The documentation addresses key technical questions about delta score implementation and provides practical guidance for variant analysis workflows.

---

## 📚 Documentation Index

### Core Technical Documentation
- **[OpenSpliceAI Variant Analysis Guide](OPENSPLICEAI_VARIANT_ANALYSIS_GUIDE.md)** - **NEW** Complete technical guide with corrected understanding of delta scores, coverage windows, and masking logic
- **[Alternative Splicing Pattern Construction](ALTERNATIVE_SPLICING_PATTERN_CONSTRUCTION.md)** - **NEW** Guide for constructing splicing patterns from delta scores and validation datasets
- **[OpenSpliceAI Technical FAQ](OPENSPLICEAI_TECHNICAL_FAQ.md)** - Comprehensive Q&A covering delta score vectors, sequence handling, and implementation details
- **[Delta Score Implementation Guide](DELTA_SCORE_IMPLEMENTATION_GUIDE.md)** - Practical examples and code snippets for delta score interpretation

### Integration and Usage
- **[Integration Overview](../OPENSPLICEAI_INTEGRATION_GUIDE.md)** - Complete integration story and architecture
- **[Developer Onboarding](../DEVELOPER_ONBOARDING_CHECKLIST.md)** - 5-day structured learning plan
- **[AI Agent Prompts](../AI_AGENT_PROMPTS.md)** - Optimized prompts for AI agent adaptation

---

## 📚 **Documentation Structure**

### **Core Technical Documentation**

#### [OPENSPLICEAI_TECHNICAL_FAQ.md](./OPENSPLICEAI_TECHNICAL_FAQ.md)
**Comprehensive technical answers to 5 key questions about OpenSpliceAI delta scores:**

1. ❓ **Is Delta Score a vector of the same length/dimension as the queried sequence?**
   - ✅ YES - Delta scores calculated for every position, maxima reported
   - Technical details on sequence windows and vector processing

2. ❓ **Does the formula assume Alternative and Reference sequences are the same length?**
   - ✅ NO - Sophisticated indel handling with specialized algorithms
   - Code examples for deletion/insertion processing

3. ❓ **Does every position/nucleotide have a delta score?**
   - ✅ YES - Full vectors calculated, only maxima reported in output
   - Position reporting and search window details

4. ❓ **Do paired donor/acceptor sites represent predicted alternative splicing patterns?**
   - ⚠️ PARTIALLY - Individual sites reported, pattern inference needs additional logic
   - Limitations and recommended approaches

5. ❓ **What happens with different length sequences (like intron retention)?**
   - ✅ HANDLED - Sequence length differences supported, complex events need interpretation
   - Intron retention detection strategies

#### [DELTA_SCORE_IMPLEMENTATION_GUIDE.md](./DELTA_SCORE_IMPLEMENTATION_GUIDE.md)
**Practical implementation examples and code patterns:**

- 🔧 **Core Implementation Patterns**
  - Basic delta score calculation
  - Sequence length handling for indels
  - Vector-based analysis

- 🧬 **Alternative Splicing Pattern Detection**
  - Intron retention detection
  - Cryptic splice site identification
  - Exon skipping analysis

- 📊 **Data Analysis and Visualization**
  - Distribution analysis
  - Position pattern analysis
  - Statistical summaries

- 🔬 **MetaSpliceAI Integration**
  - Schema adapter usage
  - Format standardization
  - Workflow integration

---

## 🎯 **Quick Reference**

### **Key Technical Insights**

| Aspect | Answer | Implementation |
|--------|--------|----------------|
| **Vector Nature** | ✅ YES | Delta scores calculated for every position in sequence window |
| **Length Handling** | ✅ SMART | Specialized algorithms for insertions/deletions |
| **Position Coverage** | ✅ COMPLETE | All positions analyzed, maxima reported |
| **Splicing Patterns** | ⚠️ PARTIAL | Individual sites detected, pairing logic needed |
| **Complex Variants** | ⚠️ LIMITED | Simple indels supported, complex events need interpretation |

### **Critical Code Locations**

```python
# Core delta score calculation
from meta_spliceai.openspliceai.variant.utils import get_delta_scores

# Key function: get_delta_scores() in openspliceai/variant/utils.py
# Lines 352-540: Complete implementation
# Lines 494-506: Indel handling logic
# Lines 512-515: Maximum delta identification
```

### **Output Format**
```
ALT|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
```
Where:
- `DS_*`: Delta scores (probability changes)
- `DP_*`: Delta positions (distances from variant in bp)
- `AG/AL`: Acceptor Gain/Loss
- `DG/DL`: Donor Gain/Loss

---

## 🧬 **Biological Context**

### **Delta Score Interpretation**
- **Range**: -1.0 to +1.0 (probability differences)
- **Positive**: Splice site strengthening/creation
- **Negative**: Splice site weakening/loss
- **Threshold**: |0.2| typically considered significant

### **Alternative Splicing Modes**

| Mode | Detection Strategy | Required Analysis |
|------|-------------------|-------------------|
| **Intron Retention** | DS_DL + DS_AL at same intron | Transcript boundary checking |
| **Exon Skipping** | DS_DL upstream + DS_AL downstream | Exon flanking analysis |
| **Cryptic Sites** | DS_AG/DS_DG away from canonical | Distance thresholding |
| **Alternative 5'/3'** | DS_AG/DS_DG near canonical | Position clustering |

---

## 🔬 **Integration with MetaSpliceAI**

### **Schema Adapter Framework**
```python
from meta_spliceai.splice_engine.meta_models.core.schema_adapters import create_schema_adapter

# Convert OpenSpliceAI results to standard format
adapter = create_schema_adapter('openspliceai')
standardized_results = adapter.adapt_splice_annotations(delta_results)
```

### **Workflow Integration**
- **Base Models**: SpliceAI pre-trained models in `splice_prediction_workflow.py`
- **Variant Analysis**: OpenSpliceAI delta scores in `openspliceai/variant/`
- **Case Studies**: Integration in `case_studies/workflows/`

---

## 📊 **Performance Considerations**

### **Computational Requirements**
- **Memory**: ~10GB for full genome analysis
- **Processing**: GPU acceleration recommended
- **Batch Size**: 1000 variants per batch optimal

### **Optimization Strategies**
- Batch processing for large VCF files
- GPU utilization for model inference
- Chromosome-wise processing for memory efficiency
- Caching of reference sequences

---

## 🚀 **Usage Examples**

### **Basic Analysis**
```python
# Analyze variant splicing impact
delta_results = analyze_variant_splicing_impact(
    vcf_file="variants.vcf",
    ref_fasta="reference.fa", 
    annotations="grch38"
)

# Detect alternative splicing patterns
intron_retention = detect_intron_retention(delta_results)
cryptic_sites = detect_cryptic_splice_sites(delta_results)
```

### **High-Impact Variant Filtering**
```python
# Filter for high-impact variants
high_impact = [r for r in delta_results 
               if any(abs(r.get(f'DS_{t}', 0)) > 0.5 
                     for t in ['AG', 'AL', 'DG', 'DL'])]
```

### **Integration with Case Studies**
```python
# Use in case study workflows
from meta_spliceai.splice_engine.case_studies.workflows import MutationAnalysisWorkflow

workflow = MutationAnalysisWorkflow()
results = workflow.analyze_variants_with_openspliceai(vcf_file)
```

---

## 🔗 **Related Documentation**

### **Core MetaSpliceAI Documentation**
- [Case Studies Overview](../README.md)
- [Biological Context](../VARIANT_SPLICING_BIOLOGY_Q10_Q12.md)
- [OpenSpliceAI Integration](../DEV_OPENSPLICEAI_INTEGRATION.md)

### **Technical Implementation**
- [Schema Adapter Framework](../../meta_models/openspliceai_adapter/docs/SCHEMA_ADAPTER_FRAMEWORK.md)
- [Meta Models Overview](../../meta_models/README.md)
- [Workflow Integration](../../meta_models/workflows/README.md)

### **Development Resources**
- [Internal Dev Guide](../../../../tests/dev/OPENSPLICEAI_INTEGRATION_DEV_GUIDE.md)
- [AI Agent Prompts](../../../../tests/dev/AI_AGENT_PROMPTS.md)
- [Test Suite](../../../../tests/integration/openspliceai_adapter/)

---

## 📝 **Contributing**

When adding new variant splicing analysis features:

1. **Update Technical FAQ** for new implementation details
2. **Add Implementation Examples** to the guide
3. **Test Integration** with existing workflows
4. **Document Performance** characteristics
5. **Validate Biological** interpretation

---

## 📞 **Support**

For questions about variant splicing analysis:
- Review the Technical FAQ for implementation details
- Check the Implementation Guide for code examples
- Consult the biological context documentation
- Test with the provided example workflows

**Last Updated**: 2025-07-29  
**Version**: 1.0.0  
**Maintainer**: MetaSpliceAI Development Team
