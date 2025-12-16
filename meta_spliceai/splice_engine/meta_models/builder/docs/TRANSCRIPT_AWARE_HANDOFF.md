# Transcript-Aware Position Identification: Session Handoff Documentation

## 🎯 **CURRENT STATUS: READY FOR 5000-GENE MODEL INTEGRATION**

**Date**: 2025-08-20  
**Phase**: Implementation complete, ready for dataset builder integration  
**Next Step**: Integrate transcript-aware position identification into `incremental_builder.py` for 5000-gene meta model training

## 📊 **What Was Accomplished**

### **Core Problem Identified and Solved**
- **Issue**: Current 1000-gene meta model shows poor generalization (`meta_only` mode performs worse than `base_only` on unseen genes)
- **Root Cause**: Genomic-only position identification `['gene_id', 'position', 'strand']` oversimplifies biological reality
- **Solution**: Transcript-aware position identification `['gene_id', 'position', 'strand', 'transcript_id']` with `splice_type` as prediction target

### **Key Biological Insight**
Same genomic position can have **different splice site roles** across transcripts:
```
Position chr1:12345 in gene BRCA1:
├── Transcript BRCA1-001: DONOR site (TP)
├── Transcript BRCA1-002: NEITHER (TN) 
└── Transcript BRCA1-003: ACCEPTOR site (FP)

Current System: Forces single label → TP (loses biological context)
Enhanced System: Preserves all roles → Better training data for meta-learning
```

### **Critical Meta-Learning Logic**
- **Position ID**: `['gene_id', 'position', 'strand', 'transcript_id']` (features)
- **Prediction Target**: `splice_type` (what we want to predict)
- **Key Insight**: NOT grouping by `splice_type` (that would be circular!)

## 🏗️ **Modules Created and Located**

### **Primary Module**: `transcript_aware_positions.py`
**Location**: `meta_spliceai/splice_engine/meta_models/builder/transcript_aware_positions.py`

**Key Functions**:
```python
def get_position_grouping_columns(mode='genomic', include_splice_type=False, custom_columns=None)
    # Returns grouping columns based on mode
    
def resolve_transcript_specific_conflicts(df, mode='genomic', preserve_transcript_info=True)
    # Handles conflicts with transcript awareness
    
class TranscriptAwareConfig:
    # Configuration class for easy integration
```

**Key Constants**:
```python
POSITION_IDENTIFIER_COLUMNS = {
    'core': ['gene_id', 'position', 'strand'],
    'transcript_specific': ['gene_id', 'position', 'strand', 'transcript_id'],
    'splice_aware': ['gene_id', 'position', 'strand', 'transcript_id'],  # Identical to transcript_specific
    # ... additional specialized modes
}
```

### **Demo Script**: `transcript_aware_demo.py`
**Location**: `meta_spliceai/splice_engine/meta_models/builder/transcript_aware_demo.py`

**Usage**:
```bash
cd /home/bchiu/work/meta-spliceai
mamba activate surveyor
python meta_spliceai/splice_engine/meta_models/builder/transcript_aware_demo.py
```

**Features**:
- Demonstrates biological reality with synthetic data
- Compares all position identification modes
- Shows impact on meta-learning
- Provides integration strategy

### **Documentation**: `transcript_aware_strategy.md`
**Location**: `meta_spliceai/splice_engine/meta_models/builder/docs/transcript_aware_strategy.md`

## 🔧 **Position Identification Modes Available**

### **Mode 1: `genomic` (Current/Default)**
```python
group_cols = ['gene_id', 'position', 'strand']
```
- **Status**: Current system, backward compatible
- **Limitation**: Forces single `splice_type` per position across all transcripts
- **Use Case**: Existing workflows, legacy support

### **Mode 2: `transcript` (Recommended for 5000-gene model)**
```python
group_cols = ['gene_id', 'position', 'strand', 'transcript_id']
```
- **Status**: ✅ Ready for implementation
- **Advantage**: Preserves transcript-specific splice site roles
- **Use Case**: New meta model training, precision medicine

### **Mode 3: `splice_aware` (Identical to transcript)**
```python
group_cols = ['gene_id', 'position', 'strand', 'transcript_id']  # Same as transcript
```
- **Status**: ✅ Functionally identical to `transcript` mode
- **Semantic Difference**: Emphasizes meta-learning capability for variant effect prediction
- **Use Case**: When emphasizing variant effect learning perspective

### **Mode 4: `hybrid` (Transition Strategy)**
```python
group_cols = ['gene_id', 'position', 'strand']  # + transcript metadata
```
- **Status**: ✅ Available but requires additional implementation
- **Advantage**: Maintains current efficiency while preserving transcript information
- **Use Case**: Gradual migration, debugging, analysis

### **Mode 5: `splice_only` (Analysis Only)**
```python
group_cols = ['gene_id', 'position', 'strand', 'splice_type']
```
- **Status**: ✅ Available but NOT recommended for training
- **Use Case**: Analysis of splice site role diversity (not meta-learning)

## 🧪 **Validation Results**

### **Import Testing**
```bash
# Verified working:
from meta_spliceai.splice_engine.meta_models.builder.transcript_aware_positions import (
    TranscriptAwareConfig,
    get_position_grouping_columns,
    POSITION_IDENTIFIER_COLUMNS
)

config = TranscriptAwareConfig(mode='transcript')
grouping_cols = config.get_grouping_columns()
# Result: ['gene_id', 'position', 'strand', 'transcript_id']
```

### **Mode Consistency Verified**
- ✅ `transcript` and `splice_aware` modes produce identical results
- ✅ `genomic` mode unchanged (backward compatibility)
- ✅ All modes handle `splice_type` as prediction target (not grouping variable)

## 🚀 **NEXT STEPS: Integration Roadmap**

### **Phase 1: Dataset Builder Integration (IMMEDIATE)**

#### **Target File**: `incremental_builder.py`
**Location**: `meta_spliceai/splice_engine/meta_models/builder/incremental_builder.py`

**Required Changes**:
```python
# Add import
from .transcript_aware_positions import TranscriptAwareConfig

# Add parameter to build_base_dataset()
def build_base_dataset(
    gene_list: List[str],
    analysis_tsv_dir: Path,
    output_dir: Path,
    position_id_mode: str = 'genomic',  # NEW PARAMETER
    **kwargs
):
    # Add configuration logic
    if position_id_mode != 'genomic':
        config = TranscriptAwareConfig(mode=position_id_mode)
        group_cols = config.get_grouping_columns()
    else:
        group_cols = ['gene_id', 'position', 'strand']  # Current behavior
        
    # Use group_cols in dataset construction logic
```

#### **Target File**: `sequence_data_utils.py` 
**Location**: `meta_spliceai/splice_engine/meta_models/workflows/sequence_data_utils.py`

**Required Changes**:
```python
# Replace hardcoded group_cols
def load_and_process_sequences(
    file_path,
    position_id_mode='genomic',  # NEW PARAMETER
    **kwargs
):
    if position_id_mode in ['transcript', 'splice_aware']:
        group_cols = ['gene_id', 'position', 'strand', 'transcript_id']
    else:
        group_cols = ['gene_id', 'position', 'strand']  # Current behavior
```

### **Phase 2: Command-Line Interface**

#### **Add CLI Parameter**
```python
# In incremental_builder.py main() function
parser.add_argument(
    '--position-id-mode', 
    choices=['genomic', 'transcript', 'splice_aware', 'hybrid'],
    default='genomic',
    help='Position identification strategy for meta-learning'
)
```

#### **Training Command Example**
```bash
# Enhanced 5000-gene model training
python incremental_builder.py \
    --gene-list 5000_error_selected_genes.txt \
    --position-id-mode transcript \
    --output-dir results/train_5000_genes_transcript_aware \
    --verbose
```

### **Phase 3: Validation and Testing**

#### **Validation Strategy**
1. **Small-scale test**: Run both `genomic` and `transcript` modes on 10-20 genes
2. **Compare datasets**: Verify position expansion and data quality
3. **Performance test**: Train small models and compare generalization
4. **Full-scale deployment**: 5000-gene model with transcript-aware mode

#### **Success Metrics**
- **Technical**: Position expansion factor (expect ~1.2-2x increase)
- **Biological**: Positions with multiple splice types captured
- **Meta-learning**: `meta_only` mode outperforms `base_only` on unseen genes

## 📋 **Integration Checklist**

### **Required Code Changes**
- [ ] Add transcript-aware import to `incremental_builder.py`
- [ ] Add `position_id_mode` parameter to `build_base_dataset()`
- [ ] Update `sequence_data_utils.py` with configurable grouping
- [ ] Add CLI parameter for position identification mode
- [ ] Update any hardcoded `group_cols` references

### **Testing Requirements**
- [ ] Test imports in surveyor environment
- [ ] Validate small-scale dataset construction
- [ ] Compare genomic vs transcript mode outputs
- [ ] Verify backward compatibility (genomic mode unchanged)
- [ ] Test CLI parameter functionality

### **Documentation Updates**
- [ ] Update `incremental_builder.py` docstrings
- [ ] Add usage examples to builder documentation
- [ ] Update training workflow documentation

## 🔍 **Key Files to Modify**

### **Primary Integration Points**
1. **`incremental_builder.py`** (lines ~300-400): Add transcript-aware configuration
2. **`sequence_data_utils.py`** (lines ~570-580): Replace hardcoded group_cols
3. **`dataset_builder.py`** (optional): Add transcript-aware support

### **Files Already Ready**
- ✅ `transcript_aware_positions.py`: Core functionality complete
- ✅ `transcript_aware_demo.py`: Working demonstration
- ✅ `docs/transcript_aware_strategy.md`: Complete documentation

## 🎯 **Expected Impact on 5000-Gene Meta Model**

### **Training Data Improvements**
- **Biological Accuracy**: Captures transcript-specific splice site roles
- **Meta-Learning Capability**: Enables variant effect prediction
- **Data Quality**: Better representation of alternative splicing complexity

### **Model Performance Expectations**
- **Generalization**: Improved performance on unseen genes
- **Meta-Only Mode**: Should finally outperform base-only mode
- **Variant Effects**: Foundation for disease-specific adaptation

### **Computational Considerations**
- **Dataset Size**: Expect 1.2-2x increase in training examples
- **Memory**: Proportional increase in memory requirements
- **Training Time**: Slightly longer due to larger dataset

## 🚨 **Critical Success Factors**

### **Must Verify**
1. **Coordinate Consistency**: Ensure transcript_id mapping is correct
2. **Data Quality**: Validate no duplicate or missing positions
3. **Backward Compatibility**: Genomic mode produces identical results
4. **Performance**: Meta model actually improves with transcript-aware data

### **Risk Mitigation**
- **Start Small**: Test with 10-20 genes before full 5000-gene training
- **Parallel Validation**: Run both modes and compare results
- **Rollback Plan**: Keep genomic mode as fallback option

## 📞 **Handoff Context**

### **Environment Setup**
```bash
cd /home/bchiu/work/meta-spliceai
mamba activate surveyor
```

### **Quick Validation**
```python
from meta_spliceai.splice_engine.meta_models.builder.transcript_aware_positions import TranscriptAwareConfig
config = TranscriptAwareConfig(mode='transcript')
print(config.get_grouping_columns())  # Should print: ['gene_id', 'position', 'strand', 'transcript_id']
```

### **Current Working Directory Structure**
```
builder/
├── transcript_aware_positions.py      # ✅ Core module (465 lines)
├── transcript_aware_demo.py           # ✅ Demo script (323 lines)  
├── docs/
│   ├── transcript_aware_strategy.md   # ✅ Strategy documentation
│   └── TRANSCRIPT_AWARE_HANDOFF.md    # ✅ This handoff doc
├── incremental_builder.py             # 🎯 TARGET for integration (1225 lines)
└── ... (other builder files)
```

## 🎉 **Ready State Confirmation**

- ✅ **Modules**: All transcript-aware modules created and tested
- ✅ **Logic**: Meta-learning logic validated (splice_type as target, not grouping)
- ✅ **Documentation**: Complete strategy and implementation docs
- ✅ **Validation**: Import testing and functionality verification complete
- ✅ **Location**: Properly organized in builder/ package
- 🎯 **Next**: Ready for integration into `incremental_builder.py`

**The transcript-aware position identification system is READY for integration into the 5000-gene meta model training pipeline!** 🚀
