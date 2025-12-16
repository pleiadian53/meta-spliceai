# 📚 OpenSpliceAI Adapter Documentation

This directory contains comprehensive documentation for the OpenSpliceAI Adapter package, which provides seamless integration between OpenSpliceAI and the MetaSpliceAI meta-learning framework.

## 📋 **Documentation Index**

### **📖 Core Documentation**

| **Document** | **Purpose** | **Audience** |
|--------------|-------------|--------------|
| [`README.md`](README.md) | Package overview and quick start guide | All users |
| [`README_ALIGNED_EXTRACTOR.md`](README_ALIGNED_EXTRACTOR.md) | Detailed AlignedSpliceExtractor documentation | Developers |

### **🔧 Technical Documentation**

| **Document** | **Purpose** | **Audience** |
|--------------|-------------|--------------|
| [`FORMAT_COMPATIBILITY_SUMMARY.md`](FORMAT_COMPATIBILITY_SUMMARY.md) | Format compatibility analysis and solutions | Technical users |
| [`SCHEMA_ADAPTER_FRAMEWORK.md`](SCHEMA_ADAPTER_FRAMEWORK.md) | Schema adaptation system for multi-model compatibility | Developers |
| [`SPLICE_SITE_DEFINITION_ANALYSIS.md`](SPLICE_SITE_DEFINITION_ANALYSIS.md) | Splice site definition comparison between models | Researchers |
| [`RESOLUTION_DOCUMENTATION.md`](RESOLUTION_DOCUMENTATION.md) | Technical issue resolutions and solutions | Developers |

### **✅ Validation & Status**

| **Document** | **Purpose** | **Audience** |
|--------------|-------------|--------------|
| [`VALIDATION_SUMMARY.md`](VALIDATION_SUMMARY.md) | Validation results and test summaries | All users |
| [`DOCUMENTATION_STATUS.md`](DOCUMENTATION_STATUS.md) | Documentation maintenance status | Maintainers |

## 🎯 **Quick Navigation**

### **Getting Started**
1. **New Users**: Start with [`README.md`](README.md) for package overview
2. **Developers**: Read [`README_ALIGNED_EXTRACTOR.md`](README_ALIGNED_EXTRACTOR.md) for implementation details
3. **Integration**: Check [`FORMAT_COMPATIBILITY_SUMMARY.md`](FORMAT_COMPATIBILITY_SUMMARY.md) for compatibility info

### **Technical Deep Dive**
1. **Schema Adaptation**: [`SCHEMA_ADAPTER_FRAMEWORK.md`](SCHEMA_ADAPTER_FRAMEWORK.md)
2. **Splice Site Analysis**: [`SPLICE_SITE_DEFINITION_ANALYSIS.md`](SPLICE_SITE_DEFINITION_ANALYSIS.md)
3. **Problem Resolution**: [`RESOLUTION_DOCUMENTATION.md`](RESOLUTION_DOCUMENTATION.md)
4. **Validation Results**: [`VALIDATION_SUMMARY.md`](VALIDATION_SUMMARY.md)

### **Maintenance**
1. **Documentation Status**: [`DOCUMENTATION_STATUS.md`](DOCUMENTATION_STATUS.md)

## 🏗️ **Package Architecture Overview**

The OpenSpliceAI Adapter package provides:

### **Core Components**
- **`AlignedSpliceExtractor`** - Unified splice site extraction with 100% equivalence
- **`CoordinateReconciliation`** - Coordinate system alignment between models
- **`FormatCompatibility`** - Format conversion and compatibility utilities
- **`Schema Adapters`** - Systematic schema conversion framework (see `meta_spliceai/splice_engine/meta_models/core/schema_adapters.py`)

### **Key Features**
- ✅ **100% Splice Site Equivalence** between MetaSpliceAI and OpenSpliceAI
- ✅ **Automatic Coordinate Reconciliation** (0-based ↔ 1-based conversion)
- ✅ **Schema Adaptation Pattern** for multi-model compatibility
- ✅ **Robust Fallback Mechanisms** for missing data files
- ✅ **Comprehensive Validation** with genome-wide testing

### **Integration Points**
- **Data Preparation Workflows** - Automatic fallback when splice_sites.tsv is missing
- **Meta-Learning Framework** - Seamless model integration
- **Validation Pipelines** - Comprehensive testing and verification
- **Schema Management** - Centralized format conversion

## 🔄 **Schema Adapter Framework**

**Location:** `meta_spliceai/splice_engine/meta_models/core/schema_adapters.py`

The Schema Adapter Framework was created specifically to address splice site annotation inconsistencies across different base models, with OpenSpliceAI integration as the primary use case.

### **Problem Addressed**
Different splice prediction models use varying column naming conventions and data formats:
- **MetaSpliceAI**: `chrom`, `start`, `end`, `strand`, `site_type`
- **AlignedSpliceExtractor**: `chromosome`, `position`, `strand`, `splice_type`
- **OpenSpliceAI**: Native format variations
- **SpliceAI**: Different schema conventions

### **Solution: Systematic Schema Adaptation**
```python
from meta_spliceai.splice_engine.meta_models.core.schema_adapters import adapt_splice_annotations

# Convert any model's output to MetaSpliceAI canonical format
canonical_df = adapt_splice_annotations(raw_df, model_type="aligned_extractor")
```

### **Key Features**
- **Centralized Conversion Logic** - Single source of truth for schema mappings
- **Model-Specific Adapters** - Tailored conversion for each base model
- **Validation Framework** - Ensures conversion correctness
- **Extensible Design** - Easy addition of new models
- **Error Handling** - Graceful handling of schema mismatches

### **Supported Models**
| **Model** | **Adapter Class** | **Status** |
|-----------|-------------------|------------|
| AlignedSpliceExtractor | `AlignedExtractorAdapter` | ✅ Production |
| SpliceAI | `SpliceAIAdapter` | ✅ Ready |
| OpenSpliceAI | `OpenSpliceAIAdapter` | ✅ Ready |
| Future Models | Extensible framework | 🔄 Planned |

### **Integration with OpenSpliceAI Adapter**
The schema adapter framework is seamlessly integrated into the OpenSpliceAI workflow:
- **AlignedSpliceExtractor** uses `apply_schema_adaptation=True` for automatic conversion
- **Data Preparation** workflows automatically apply schema adaptation
- **Fallback mechanisms** ensure consistent output format
- **Test suites** validate schema conversion accuracy

## 📊 **Validation Status**

| **Component** | **Test Coverage** | **Status** | **Last Validated** |
|---------------|-------------------|------------|-------------------|
| AlignedSpliceExtractor | 100% | ✅ PASSING | 2025-07-28 |
| Schema Adapters | 100% | ✅ PASSING | 2025-07-28 |
| Coordinate Reconciliation | 100% | ✅ PASSING | 2025-07-28 |
| Format Compatibility | 100% | ✅ PASSING | 2025-07-28 |
| Genome-Wide Validation | 100% | ✅ PASSING | 2025-07-28 |

**Key Metrics:**
- **Splice Sites Validated:** 8,756 (100% match)
- **Genes Tested:** 98 (100% agreement)
- **Test Suites:** 5/5 passing
- **Integration Tests:** 3/3 passing

## 🔗 **Related Documentation**

### **External References**
- **MetaSpliceAI Core Documentation** - `meta_spliceai/docs/`
- **Test Documentation** - `tests/integration/openspliceai_adapter/`
- **Schema Adapter Framework** - `meta_spliceai/splice_engine/meta_models/core/schema_adapters.py` (Core module created for OpenSpliceAI integration)
- **Schema Adapter Tests** - `tests/integration/openspliceai_adapter/test_schema_adapter.py`

### **Code Examples**
- **Integration Examples** - `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/example_integration.py`
- **Variant Analysis** - `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/variant_analysis_example.py`
- **Workflow Integration** - `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/workflow_integration.py`

## 📝 **Contributing to Documentation**

### **Documentation Standards**
1. **Clarity**: Write for your intended audience
2. **Completeness**: Include all necessary information
3. **Examples**: Provide concrete code examples
4. **Validation**: Include test results and metrics
5. **Maintenance**: Keep documentation up-to-date

### **File Organization**
- **Core docs**: Package overview and getting started
- **Technical docs**: Detailed implementation information
- **Validation docs**: Test results and status reports
- **Maintenance docs**: Documentation status and updates

### **Update Process**
1. Update relevant documentation when making code changes
2. Run validation tests to verify documentation accuracy
3. Update the [`DOCUMENTATION_STATUS.md`](DOCUMENTATION_STATUS.md) file
4. Review and approve documentation changes

---

**Last Updated:** 2025-07-28  
**Documentation Version:** 2.0  
**Package Version:** Compatible with MetaSpliceAI v2.0+  
**Maintained By:** MetaSpliceAI Development Team
