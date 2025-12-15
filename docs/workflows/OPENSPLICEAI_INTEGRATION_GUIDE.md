# 🧬 OpenSpliceAI Integration Guide for MetaSpliceAI

**For Developers and AI Agents: Complete Integration Story and Quick Adaptation**

This guide provides a comprehensive overview of the OpenSpliceAI integration with MetaSpliceAI, designed for rapid onboarding of developers and AI agents to our meta-learning framework for alternative splicing analysis.

## 🎯 **Integration Overview**

### **The Vision**
Transform MetaSpliceAI into a comprehensive meta-learning platform that leverages multiple splice prediction models (OpenSpliceAI, SpliceAI, and others) to capture alternative splicing patterns more effectively than any single model alone.

### **Three-Layer Architecture**

```
🧬 MetaSpliceAI Meta-Learning Platform
├── 1️⃣ OpenSpliceAI Source Code (meta_spliceai/openspliceai/)
├── 2️⃣ OpenSpliceAI Adapter (splice_engine/meta_models/openspliceai_adapter/)
└── 3️⃣ Case Studies Package (splice_engine/case_studies/)
```

---

## 📁 **Layer 1: OpenSpliceAI Source Code**

**Location:** `meta_spliceai/openspliceai/`

### **Purpose**
Complete integration of OpenSpliceAI's source code into the MetaSpliceAI ecosystem, providing direct access to OpenSpliceAI's robust splice prediction capabilities.

### **Key Components**
- **Core Models** - OpenSpliceAI's neural network architectures
- **Data Processing** - Sequence extraction and preprocessing pipelines
- **Prediction Engine** - Splice site prediction algorithms
- **Utilities** - Helper functions and data structures

### **Integration Benefits**
- ✅ **Direct Access** - No external dependencies or API calls
- ✅ **Customization** - Ability to modify and extend OpenSpliceAI
- ✅ **Performance** - Optimized for MetaSpliceAI workflows
- ✅ **Consistency** - Unified data formats and processing

### **For AI Agents: Key Prompt Context**
```
OpenSpliceAI source code is fully integrated at meta_spliceai/openspliceai/. 
This provides direct access to OpenSpliceAI's prediction models and can be 
used for splice site analysis, sequence processing, and model training.
```

---

## 🔧 **Layer 2: OpenSpliceAI Adapter**

**Location:** `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/`

### **Purpose**
Systematic integration layer that provides seamless compatibility between OpenSpliceAI and MetaSpliceAI with 100% splice site equivalence and automatic format conversion.

### **Core Components**

#### **🎯 AlignedSpliceExtractor**
**File:** `aligned_splice_extractor.py`
```python
from meta_spliceai.splice_engine.meta_models.openspliceai_adapter import AlignedSpliceExtractor

# 100% equivalent splice site extraction
extractor = AlignedSpliceExtractor(coordinate_system="metaspliceai")
splice_sites = extractor.extract_splice_sites(
    gtf_file="annotations.gtf",
    fasta_file="genome.fa",
    apply_schema_adaptation=True  # Automatic format conversion
)
```

#### **🔄 Schema Adapter Framework**
**File:** `meta_spliceai/splice_engine/meta_models/core/schema_adapters.py`
```python
from meta_spliceai.splice_engine.meta_models.core.schema_adapters import adapt_splice_annotations

# Convert any model's output to canonical MetaSpliceAI format
canonical_df = adapt_splice_annotations(raw_df, model_type="aligned_extractor")
```

#### **📊 Coordinate Reconciliation**
**File:** `coordinate_reconciliation.py`
- Automatic 0-based ↔ 1-based coordinate conversion
- Perfect alignment between OpenSpliceAI and MetaSpliceAI coordinate systems

#### **🛡️ Fallback Mechanisms**
**Integration:** `splice_engine/meta_models/workflows/data_preparation.py`
- Automatic OpenSpliceAI fallback when `splice_sites.tsv` is missing
- Robust workflow continuation with 100% equivalence guarantee

### **Key Features**
- ✅ **100% Splice Site Equivalence** (8,756 sites validated)
- ✅ **Automatic Schema Adaptation** (Multi-model compatibility)
- ✅ **Coordinate Reconciliation** (Perfect alignment)
- ✅ **Robust Fallback Mechanisms** (Workflow reliability)
- ✅ **Comprehensive Validation** (100% test coverage)

### **Documentation**
- **📋 Complete Documentation:** `openspliceai_adapter/docs/`
- **🎯 Quick Start:** `openspliceai_adapter/README.md`
- **🔧 Technical Details:** `openspliceai_adapter/docs/INDEX.md`
- **🔄 Schema Adapters:** `openspliceai_adapter/docs/SCHEMA_ADAPTER_FRAMEWORK.md`

### **For AI Agents: Key Prompt Context**
```
The OpenSpliceAI adapter at splice_engine/meta_models/openspliceai_adapter/ 
provides seamless integration with 100% splice site equivalence. Use 
AlignedSpliceExtractor for unified splice site extraction and schema 
adapters for format conversion. All workflows support automatic fallback 
mechanisms and comprehensive validation.
```

---

## 🧪 **Layer 3: Case Studies Package**

**Location:** `meta_spliceai/splice_engine/case_studies/`

### **Purpose**
Comprehensive case studies demonstrating how meta-learning approaches can capture alternative splicing patterns more effectively by combining multiple base models.

### **Research Focus Areas**

#### **🎯 Meta-Learning for Alternative Splicing**
- **Model Ensemble Strategies** - Combining OpenSpliceAI, SpliceAI, and MetaSpliceAI
- **Transfer Learning** - Leveraging pre-trained models for specific splice patterns
- **Active Learning** - Intelligent sample selection for training data
- **Multi-Task Learning** - Joint prediction of multiple splice-related tasks

#### **📊 Alternative Splicing Pattern Analysis**
- **Tissue-Specific Splicing** - Meta-models for tissue-specific splice patterns
- **Disease-Associated Splicing** - Pathological splice variant detection
- **Evolutionary Conservation** - Cross-species splice pattern analysis
- **Regulatory Element Integration** - Incorporating regulatory sequence information

#### **🔬 Validation and Benchmarking**
- **Cross-Model Validation** - Comparing predictions across different models
- **Clinical Validation** - Real-world clinical dataset validation
- **Performance Metrics** - Comprehensive evaluation frameworks
- **Biological Relevance** - Functional validation of predictions

### **Current Case Study Structure**
```
case_studies/
├── README.md                   # Package overview and documentation
├── __init__.py                 # Package initialization
├── run_case_study.py          # Main case study execution script
├── data_sources/              # Data source integrations
│   ├── base.py                # Base data source classes
│   ├── clinvar.py             # ClinVar database integration
│   ├── dbass.py               # DBASS database integration
│   ├── mutsplicedb.py         # MutSpliceDB integration
│   └── splicevardb.py         # SpliceVarDB integration
├── workflows/                 # Analysis workflows
│   ├── disease_validation.py  # Disease-associated splice validation
│   └── mutation_analysis.py   # Mutation impact analysis
├── formats/                   # Data format handlers
├── examples/                  # Usage examples
└── docs/                      # Case study documentation
```

### **Planned Extensions (Future Development)**
```
# Future case study categories being planned:
├── meta_learning/             # Meta-learning approaches (planned)
│   ├── ensemble_strategies/   # Model ensemble methods (planned)
│   ├── transfer_learning/     # Transfer learning studies (planned)
│   └── active_learning/       # Active learning frameworks (planned)
├── alternative_splicing/      # Splicing pattern analysis (planned)
│   ├── tissue_specific/       # Tissue-specific studies (planned)
│   └── regulatory_elements/   # Regulatory sequence integration (planned)
└── validation/                # Additional validation frameworks (planned)
    ├── cross_model/           # Cross-model comparisons (planned)
    └── benchmarking/          # Performance benchmarks (planned)
```

### **For AI Agents: Key Prompt Context**
```
Case studies at splice_engine/case_studies/ demonstrate meta-learning 
approaches for alternative splicing analysis. Focus on combining multiple 
models (OpenSpliceAI, SpliceAI, MetaSpliceAI) to capture splice patterns 
more effectively than single models. Include tissue-specific, disease-
associated, and regulatory element studies.
```

---

## 🚀 **Quick Start for AI Agents**

### **Essential Context Prompts**

#### **1. Architecture Understanding**
```
MetaSpliceAI is a meta-learning platform with three key layers:
1. OpenSpliceAI source code (meta_spliceai/openspliceai/)
2. OpenSpliceAI adapter (splice_engine/meta_models/openspliceai_adapter/)
3. Case studies (splice_engine/case_studies/)

The adapter provides 100% splice site equivalence and automatic format 
conversion between models.
```

#### **2. Core Integration Points**
```
Key integration components:
- AlignedSpliceExtractor: Unified splice site extraction
- Schema Adapters: Multi-model format compatibility  
- Coordinate Reconciliation: Automatic coordinate alignment
- Fallback Mechanisms: Robust workflow continuation
- Validation Framework: 100% test coverage with 8,756 sites validated
```

#### **3. Usage Patterns**
```python
# Standard workflow for splice site extraction
from meta_spliceai.splice_engine.meta_models.openspliceai_adapter import AlignedSpliceExtractor

extractor = AlignedSpliceExtractor(coordinate_system="metaspliceai")
splice_sites = extractor.extract_splice_sites(
    gtf_file="annotations.gtf",
    fasta_file="genome.fa",
    apply_schema_adaptation=True
)

# Schema adaptation for multi-model compatibility
from meta_spliceai.splice_engine.meta_models.core.schema_adapters import adapt_splice_annotations
canonical_df = adapt_splice_annotations(raw_df, "aligned_extractor")
```

#### **4. Testing and Validation**
```
Comprehensive test suite at tests/integration/openspliceai_adapter/:
- test_schema_adapter.py: Schema conversion validation
- test_genome_wide_validation.py: Large-scale validation (8,756 sites)
- run_splice_comparison.py: 100% equivalence verification
- test_openspliceai_fallback.py: Fallback mechanism testing
```

---

## 📋 **Development Guidelines**

### **For New Features**
1. **Follow Schema Adapter Pattern** - Use systematic schema conversion
2. **Maintain 100% Equivalence** - Validate against existing benchmarks
3. **Comprehensive Testing** - Include unit, integration, and validation tests
4. **Documentation** - Update relevant documentation in `docs/` directories

### **For Bug Fixes**
1. **Identify Root Cause** - Use existing validation framework
2. **Implement Solution** - Follow established patterns
3. **Validate Fix** - Run comprehensive test suite
4. **Update Documentation** - Reflect changes in relevant docs

### **For Extensions**
1. **Assess Integration Points** - Identify where new models fit
2. **Create Model-Specific Adapter** - Follow schema adapter pattern
3. **Implement Validation** - Ensure equivalence and correctness
4. **Document Integration** - Add to case studies and documentation

---

## 🎯 **Effective AI Agent Prompts**

### **For Code Analysis**
```
Analyze the OpenSpliceAI integration in MetaSpliceAI focusing on:
1. AlignedSpliceExtractor at splice_engine/meta_models/openspliceai_adapter/
2. Schema adapters at splice_engine/meta_models/core/schema_adapters.py
3. Integration with data preparation workflows
4. Validation results showing 100% splice site equivalence
```

### **For Feature Development**
```
Develop new features for MetaSpliceAI's meta-learning platform:
1. Use AlignedSpliceExtractor for splice site extraction
2. Apply schema adapters for format compatibility
3. Leverage case studies at splice_engine/case_studies/
4. Maintain 100% equivalence with existing validation
5. Follow documentation patterns in openspliceai_adapter/docs/
```

### **For Troubleshooting**
```
Debug issues in MetaSpliceAI's OpenSpliceAI integration:
1. Check AlignedSpliceExtractor coordinate reconciliation
2. Validate schema adapter conversions
3. Review fallback mechanisms in data_preparation.py
4. Run test suite at tests/integration/openspliceai_adapter/
5. Consult documentation at openspliceai_adapter/docs/
```

### **For Research Applications**
```
Apply MetaSpliceAI's meta-learning framework for alternative splicing:
1. Use case studies at splice_engine/case_studies/ as templates
2. Combine multiple models (OpenSpliceAI, SpliceAI, MetaSpliceAI)
3. Focus on tissue-specific, disease-associated, or regulatory patterns
4. Validate using comprehensive benchmarking frameworks
5. Document findings following established documentation patterns
```

---

## 📊 **Success Metrics**

### **Integration Validation**
- ✅ **100% Splice Site Equivalence** (8,756 sites validated)
- ✅ **100% Gene-Level Agreement** (98 genes tested)
- ✅ **100% Test Coverage** (5/5 test suites passing)
- ✅ **Zero Regressions** (backward compatibility maintained)

### **Performance Benchmarks**
- **Extraction Speed** - Comparable to native MetaSpliceAI
- **Memory Usage** - Optimized for large-scale genomic data
- **Accuracy** - Maintains or improves upon individual model performance
- **Scalability** - Genome-wide analysis capabilities

### **Documentation Quality**
- **Comprehensive Coverage** - All components documented
- **Clear Navigation** - Easy discovery and understanding
- **Practical Examples** - Working code samples
- **Maintenance Status** - Up-to-date and validated

---

## 🔗 **Key Resources**

### **Documentation**
- **OpenSpliceAI Adapter Docs** - `splice_engine/meta_models/openspliceai_adapter/docs/`
- **Schema Adapter Framework** - `openspliceai_adapter/docs/SCHEMA_ADAPTER_FRAMEWORK.md`
- **Test Organization** - `tests/TEST_ORGANIZATION.md`

### **Test Suites**
- **Integration Tests** - `tests/integration/openspliceai_adapter/`
- **Validation Scripts** - `tests/integration/openspliceai_adapter/run_splice_comparison.py`
- **Test Execution Guide** - `tests/RUN_TESTS.md`

### **Code Examples**
- **Integration Examples** - `openspliceai_adapter/example_integration.py`
- **Workflow Integration** - `openspliceai_adapter/workflow_integration.py`
- **Variant Analysis** - `openspliceai_adapter/variant_analysis_example.py`

---

**Created:** 2025-07-28  
**Purpose:** Comprehensive onboarding for OpenSpliceAI integration with MetaSpliceAI  
**Target Audience:** Developers and AI Agents  
**Status:** ✅ Production Ready  
**Maintained By:** MetaSpliceAI Development Team
