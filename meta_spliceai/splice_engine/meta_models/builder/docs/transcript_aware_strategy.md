# Enhanced Transcript-Aware Position Identification: Implementation Strategy

## 🎯 Executive Summary

**Problem**: Current workflows use genomic-only position identification (`['gene_id', 'position', 'strand']`), forcing a single splice site label per position across all transcripts. This oversimplifies biological reality and may contribute to the **meta model generalization failure** we observed.

**Solution**: Implement configurable position identification modes that respect the biological reality of alternative splicing while maintaining backward compatibility.

## 🧬 Biological Motivation

The same genomic position can have **different splice site roles** across transcripts:

```
Position chr1:12345 in gene BRCA1:
├── Transcript BRCA1-001: DONOR site (TP)
├── Transcript BRCA1-002: NEITHER (TN) 
└── Transcript BRCA1-003: ACCEPTOR site (FP)

Current System: Forces single label → TP (loses biological context)
Proposed System: Preserves all roles → Better training data quality
```

## 🚀 Implementation Phases

### Phase 1: Transcript-Aware Mode (Immediate Impact)
**Goal**: Enable meta-learning to capture variant effects on splicing patterns

**Implementation**:
- Add `--position-id-mode transcript` or `--position-id-mode splice_aware` flag
- Group by `['gene_id', 'position', 'strand', 'transcript_id']`
- **Key Insight**: `splice_type` is the prediction TARGET, not part of position identification
- **Advantage**: Enables meta-learning while preserving full biological context
- **Impact**: Better training data for 5000-gene meta model

**Code Changes**:
```python
# Current (genomic-only)
group_cols = ['gene_id', 'position', 'strand']

# Enhanced (transcript-aware for meta-learning)  
if position_id_mode in ['transcript', 'splice_aware']:
    group_cols = ['gene_id', 'position', 'strand', 'transcript_id']
    # splice_type remains as prediction target, not grouping variable
```

### Phase 2: Hybrid Mode (Preserve Information)
**Goal**: Maintain current ML efficiency while preserving transcript context as metadata

**Implementation**:
- Keep genomic-only grouping for deduplication
- Add transcript metadata columns: `transcript_ids`, `transcript_count`, `observed_splice_types`
- **Advantage**: Backward compatible, enriches data without changing structure
- **Impact**: Better debugging and biological interpretation

### Phase 3: Advanced Modes (Specialized Use Cases)
**Goal**: Support specialized analysis and transition strategies

**Modes Available**:
- **`transcript`/`splice_aware`**: Full transcript-aware analysis (identical implementation)
- **`hybrid`**: Genomic grouping + transcript metadata preservation  
- **`splice_only`**: Group by splice_type for analysis (not recommended for training)

**Implementation**: All modes now correctly treat `splice_type` as prediction target
**Impact**: Enables isoform-specific predictions, clinical applications, variant effect modeling

## 📋 Configuration Framework

```python
from meta_spliceai.splice_engine.meta_models.builder.transcript_aware_positions import TranscriptAwareConfig

# Current behavior (backward compatible)
config_genomic = TranscriptAwareConfig(mode='genomic')

# Transcript-aware mode (recommended for 5000-gene model)
config_transcript = TranscriptAwareConfig(
    mode='transcript',  # or 'splice_aware' - identical implementation
    preserve_transcript_info=True,
    enable_complexity_analysis=True
)

# Hybrid mode (transition strategy)
config_hybrid = TranscriptAwareConfig(
    mode='hybrid',
    preserve_transcript_info=True,
    backward_compatible=True
)
```

## 🔧 Integration Points

### 1. Dataset Builder (`builder/`)
```python
# In dataset_builder.py and incremental_builder.py
from .transcript_aware_positions import TranscriptAwareConfig

def build_training_dataset(
    analysis_tsv_dir,
    output_path,
    position_id_mode='genomic',  # New parameter
    transcript_aware_config=None,
    **kwargs
):
    if position_id_mode != 'genomic':
        # Use enhanced position identification
        config = TranscriptAwareConfig(mode=position_id_mode)
        group_cols = config.get_grouping_columns()
    else:
        # Current behavior
        group_cols = ['gene_id', 'position', 'strand']
```

### 2. Inference Workflow
```python
# In main_inference_workflow.py
parser.add_argument('--position-id-mode', 
                   choices=['genomic', 'transcript', 'hybrid', 'splice_aware'],
                   default='genomic',
                   help='Position identification strategy')
```

### 3. Sequence Data Utils
```python
# In sequence_data_utils.py - replace current hardcoded group_cols
def load_and_process_sequences(
    file_path,
    position_id_mode='genomic',
    **kwargs
):
    if position_id_mode in ['transcript', 'splice_aware']:
        # Both modes use identical grouping for meta-learning
        group_cols = ['gene_id', 'position', 'strand', 'transcript_id']
        # splice_type remains as prediction target
    elif position_id_mode == 'hybrid':
        group_cols = ['gene_id', 'position', 'strand']
        # + transcript metadata preservation logic
    else:  # genomic (current)
        group_cols = ['gene_id', 'position', 'strand']
```

## 🎯 Connection to 5000-Gene Meta Model

### Current Meta Model Issues
1. **Training Data Oversimplification**: Genomic-only grouping loses transcript-specific patterns
2. **Generalization Failure**: `meta_only` mode performs worse than `base_only` on unseen genes
3. **Biological Unrealism**: Same position forced to single label across all transcripts

### Enhanced 5000-Gene Model Strategy
1. **Phase 1 Implementation**: Use `transcript` or `splice_aware` mode for training dataset assembly
2. **Key Insight**: Position ID `['gene_id', 'position', 'strand', 'transcript_id']` + target `splice_type`
3. **Expected Improvements**:
   - Enables meta-learning to capture variant effects on splicing patterns
   - Better capture of alternative splicing complexity
   - Improved meta model generalization to unseen genes
   - `meta_only` mode finally outperforming `base_only` mode
4. **Validation**: Use existing diagnostic tools to compare performance

## 🔄 Backward Compatibility Strategy

### Immediate (No Breaking Changes)
1. **Default Behavior**: All modes default to `'genomic'` (current system)
2. **Optional Enhancement**: Add transcript metadata without changing core structure
3. **Gradual Migration**: Existing models continue to work unchanged

### Transition Period
1. **Dual Validation**: Run both genomic and splice-aware modes in parallel
2. **Performance Comparison**: Use diagnostic tools to measure improvement
3. **Documentation**: Clear migration guides and biological justification

### Long-term
1. **New Default**: Recommend `transcript`/`splice_aware` mode for new projects
2. **Clinical Mode**: Same implementation as research mode (both enable variant effect prediction)
3. **Legacy Support**: Maintain `genomic` mode for backward compatibility

## 🚀 Expected Benefits

### Immediate (Phase 1: Transcript-Aware)
- ✅ **Better Training Data**: Enables meta-learning for variant effect prediction
- ✅ **Logical Consistency**: `splice_type` as prediction target, not position identifier
- ✅ **Backward Compatible**: Existing workflows continue unchanged
- ✅ **Full Biological Context**: Captures complete transcript-specific patterns

### Short-term (Phase 2: Hybrid)
- ✅ **Enhanced Debugging**: Transcript metadata for better analysis
- ✅ **Biological Insight**: Understanding of position complexity
- ✅ **Improved Validation**: Better diagnostic capabilities

### Long-term (Phase 3: Full Transcript)
- ✅ **Precision Medicine**: Isoform-specific predictions
- ✅ **Clinical Applications**: Disease-specific splicing analysis
- ✅ **Novel Discovery**: Identification of rare splicing patterns
- ✅ **Meta Learning Success**: Proper generalization to unseen genes

## 📊 Success Metrics

### Technical Metrics
- **Position Expansion Factor**: How much biological complexity is captured
- **Meta Model Performance**: `meta_only` vs `base_only` on unseen genes
- **Generalization Quality**: Error rates across diverse gene sets

### Biological Metrics
- **Splice Site Role Preservation**: Positions with multiple splice types
- **Isoform Coverage**: Transcript-specific pattern capture
- **Clinical Relevance**: Disease-associated variant interpretation

The enhanced transcript-aware position identification directly addresses the **core limitation** that may be causing meta model generalization failure, positioning the 5000-gene model for success.
