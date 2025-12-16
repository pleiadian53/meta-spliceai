# 🔄 Schema Adapter Framework

**Location:** `meta_spliceai/splice_engine/meta_models/core/schema_adapters.py`

The Schema Adapter Framework is a systematic solution for handling splice site annotation format inconsistencies across different base models, created specifically to address OpenSpliceAI integration challenges.

## 🎯 **Problem Statement**

### **The Challenge**
Different splice prediction models use varying column naming conventions and data formats, making integration complex and error-prone:

| **Model** | **Chromosome** | **Position** | **Strand** | **Site Type** |
|-----------|----------------|--------------|------------|---------------|
| **MetaSpliceAI** | `chrom` | `start`, `end` | `strand` | `site_type` |
| **AlignedSpliceExtractor** | `chromosome` | `position` | `strand` | `splice_type` |
| **OpenSpliceAI** | Various formats | Various formats | `strand` | Various formats |
| **SpliceAI** | `#CHROM` | `POS` | `strand` | `TYPE` |

### **Previous Approach: Ad-hoc Mapping**
Before the Schema Adapter Framework, format conversion was handled with scattered, manual column mapping:

```python
# Old approach - scattered throughout codebase
df_converted = df.rename(columns={
    'chromosome': 'chrom',
    'position': 'start',
    'splice_type': 'site_type'
})
df_converted['end'] = df_converted['start']  # Manual logic
```

**Problems with this approach:**
- ❌ Scattered conversion logic across multiple files
- ❌ Inconsistent handling of edge cases
- ❌ Difficult to maintain and extend
- ❌ No validation or error handling
- ❌ Hard to add new models

## ✅ **Solution: Schema Adapter Framework**

### **Design Principles**
1. **Centralized Logic** - Single source of truth for all schema conversions
2. **Model-Specific Adapters** - Tailored conversion for each base model
3. **Canonical Schema** - Standard MetaSpliceAI format as target
4. **Validation Framework** - Ensures conversion correctness
5. **Extensible Design** - Easy addition of new models

### **Architecture**

```python
# Base adapter class
class SpliceSchemaAdapter:
    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert input DataFrame to canonical MetaSpliceAI format"""
        pass
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame has required columns"""
        pass

# Model-specific adapters
class AlignedExtractorAdapter(SpliceSchemaAdapter):
    """Adapter for AlignedSpliceExtractor output format"""
    
class SpliceAIAdapter(SpliceSchemaAdapter):
    """Adapter for SpliceAI output format"""
    
class OpenSpliceAIAdapter(SpliceSchemaAdapter):
    """Adapter for OpenSpliceAI output format"""
```

## 🚀 **Usage Examples**

### **Basic Usage**
```python
from meta_spliceai.splice_engine.meta_models.core.schema_adapters import adapt_splice_annotations

# Convert AlignedSpliceExtractor output to canonical format
raw_df = extractor.extract_splice_sites(...)  # Returns original format
canonical_df = adapt_splice_annotations(raw_df, model_type="aligned_extractor")

# Result: DataFrame with canonical columns [chrom, start, end, strand, site_type, gene_id, transcript_id]
```

### **Integrated with AlignedSpliceExtractor**
```python
from meta_spliceai.splice_engine.meta_models.openspliceai_adapter import AlignedSpliceExtractor

# Automatic schema adaptation
extractor = AlignedSpliceExtractor(coordinate_system="splicesurveyor")
canonical_df = extractor.extract_splice_sites(
    gtf_file="annotations.gtf",
    fasta_file="genome.fa",
    apply_schema_adaptation=True  # Automatic conversion to canonical format
)
```

### **Manual Adapter Usage**
```python
from meta_spliceai.splice_engine.meta_models.core.schema_adapters import create_adapter

# Create specific adapter
adapter = create_adapter("aligned_extractor")
canonical_df = adapter.adapt(raw_df)

# Validate result
if adapter.validate(canonical_df):
    print("✅ Schema conversion successful!")
```

## 📊 **Canonical Schema Definition**

The framework defines a standard MetaSpliceAI schema that all models are converted to:

```python
METASPLICEAI_SCHEMA = SchemaDefinition(
    required_columns=['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id'],
    optional_columns=['score', 'sequence'],
    column_types={
        'chrom': 'string',
        'start': 'int64',
        'end': 'int64',
        'strand': 'string',
        'site_type': 'string',
        'gene_id': 'string',
        'transcript_id': 'string'
    }
)
```

## 🔧 **Supported Model Adapters**

### **AlignedExtractorAdapter**
**Status:** ✅ Production  
**Input Format:** `chromosome`, `position`, `strand`, `splice_type`, `gene_id`, `transcript_id`  
**Conversion Logic:**
- `chromosome` → `chrom`
- `position` → `start` and `end`
- `splice_type` → `site_type`

### **SpliceAIAdapter**
**Status:** ✅ Ready  
**Input Format:** `#CHROM`, `POS`, `strand`, `TYPE`, `GENE`, `TRANSCRIPT`  
**Conversion Logic:**
- `#CHROM` → `chrom`
- `POS` → `start` and `end`
- `TYPE` → `site_type`

### **OpenSpliceAIAdapter**
**Status:** ✅ Ready  
**Input Format:** Native OpenSpliceAI format  
**Conversion Logic:** Handles OpenSpliceAI-specific format variations

## 🧪 **Validation & Testing**

### **Test Coverage**
The Schema Adapter Framework has comprehensive test coverage:

```bash
# Run schema adapter tests
python tests/integration/openspliceai_adapter/test_schema_adapter.py
python tests/integration/openspliceai_adapter/test_schema_adapter_integration.py
```

### **Validation Results**
- ✅ **AlignedExtractor Adapter**: 100% conversion accuracy (28 splice sites tested)
- ✅ **Integration Testing**: 3/3 tests passing
- ✅ **Backward Compatibility**: Maintained (existing code unchanged)
- ✅ **Error Handling**: Graceful handling of invalid inputs

### **Test Scenarios**
1. **Basic Conversion** - Standard format conversion
2. **Edge Cases** - Missing columns, invalid data types
3. **Integration** - End-to-end workflow testing
4. **Performance** - Large dataset handling
5. **Backward Compatibility** - Existing code continues to work

## 🔄 **Integration with OpenSpliceAI Adapter**

The Schema Adapter Framework is deeply integrated into the OpenSpliceAI adapter workflow:

### **AlignedSpliceExtractor Integration**
```python
# In aligned_splice_extractor.py
def extract_splice_sites(self, ..., apply_schema_adaptation=False):
    # Extract splice sites in native format
    splice_sites_df = self._extract_raw_splice_sites(...)
    
    # Apply schema adaptation if requested
    if apply_schema_adaptation:
        splice_sites_df = adapt_splice_annotations(splice_sites_df, "aligned_extractor")
    
    return splice_sites_df
```

### **Data Preparation Integration**
```python
# In data_preparation.py
def prepare_splice_site_annotations(...):
    # OpenSpliceAI fallback with automatic schema adaptation
    splice_sites_df = extractor.extract_splice_sites(
        ...,
        apply_schema_adaptation=True  # Ensures canonical format
    )
```

### **Workflow Benefits**
- **Consistent Output** - All workflows produce canonical format
- **Transparent Integration** - Existing code works unchanged
- **Automatic Conversion** - No manual format handling required
- **Validation** - Built-in correctness checking

## 🚀 **Future Extensions**

### **Adding New Models**
The framework is designed for easy extension:

```python
class NewModelAdapter(SpliceSchemaAdapter):
    """Adapter for a new splice prediction model"""
    
    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implement model-specific conversion logic
        return converted_df
    
    def validate(self, df: pd.DataFrame) -> bool:
        # Implement model-specific validation
        return is_valid

# Register the new adapter
ADAPTER_REGISTRY["new_model"] = NewModelAdapter
```

### **Planned Enhancements**
- **Performance Optimization** - Vectorized operations for large datasets
- **Advanced Validation** - Statistical validation of conversion accuracy
- **Configuration Support** - User-customizable schema mappings
- **Logging Integration** - Detailed conversion logging and debugging

## 📝 **Development History**

### **Creation Context**
The Schema Adapter Framework was created during the OpenSpliceAI integration project to address:

1. **Immediate Need** - AlignedSpliceExtractor format compatibility with MetaSpliceAI
2. **Future Scalability** - Support for multiple base models (SpliceAI, OpenSpliceAI, etc.)
3. **Code Quality** - Replace scattered ad-hoc mapping with systematic approach
4. **Maintainability** - Centralized schema management

### **Key Milestones**
- **2025-07-28** - Initial framework design and implementation
- **2025-07-28** - AlignedExtractorAdapter implementation and testing
- **2025-07-28** - Integration with OpenSpliceAI adapter workflows
- **2025-07-28** - Comprehensive test suite and validation
- **2025-07-28** - Documentation and user guides

## 🔗 **Related Documentation**

- **[Package Overview](README.md)** - OpenSpliceAI adapter package documentation
- **[AlignedSpliceExtractor Guide](README_ALIGNED_EXTRACTOR.md)** - Core component documentation
- **[Format Compatibility](FORMAT_COMPATIBILITY_SUMMARY.md)** - Format analysis and solutions
- **[Validation Summary](VALIDATION_SUMMARY.md)** - Test results and metrics

## 📚 **References**

- **Source Code** - `meta_spliceai/splice_engine/meta_models/core/schema_adapters.py`
- **Test Suite** - `tests/integration/openspliceai_adapter/test_schema_adapter.py`
- **Integration Tests** - `tests/integration/openspliceai_adapter/test_schema_adapter_integration.py`
- **Usage Examples** - `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/example_integration.py`

---

**Created:** 2025-07-28 (OpenSpliceAI Integration Project)  
**Purpose:** Systematic schema adaptation for multi-model splice site annotation compatibility  
**Status:** ✅ Production Ready  
**Maintained By:** MetaSpliceAI Development Team
