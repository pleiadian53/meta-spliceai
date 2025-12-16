# Chromosome-Aware CV Completion Roadmap

**Comprehensive guide for completing the chromosome-aware cross-validation workflow to match SpliceAI's evaluation methodology**

---

## Executive Summary

The `run_loco_cv_multiclass_scalable.py` script provides a foundation for chromosome-aware cross-validation but lacks several critical components compared to the mature `run_gene_cv_sigmoid.py` workflow. This document outlines the missing functionality and provides a roadmap for implementing SpliceAI-compatible evaluation.

## Current Status Analysis

### ✅ **Implemented Features**
- Basic Leave-One-Chromosome-Out (LOCO) CV framework
- Memory-efficient chunked data loading infrastructure
- Sparse k-mer feature representation support
- XGBoost multiclass training with overfitting monitoring
- Basic ROC/PR curve generation
- Feature selection and leakage detection
- Calibration support (binary and per-class)

### ❌ **Missing Critical Components**

#### 1. **Inference Workflow Integration**
- **Issue**: No integration with the inference workflow for real-world evaluation
- **Impact**: Cannot evaluate trained models on new genomic regions
- **Gene CV Equivalent**: Lines 1637-1665 in `run_gene_cv_sigmoid.py`

#### 2. **Post-Training Evaluation Pipeline**
- **Issue**: Missing comprehensive meta vs base model comparison
- **Impact**: Cannot assess meta-model improvements over base predictions
- **Gene CV Equivalent**: Lines 1584-1733 in `run_gene_cv_sigmoid.py`

#### 3. **Scalability Features Not Fully Implemented**
- **Issue**: Chunked loading and sparse k-mers are scaffolded but not complete
- **Impact**: Cannot handle large datasets (20K+ genes) efficiently
- **Status**: `SCALABILITY_UTILS_AVAILABLE = False` (lines 83-88)

#### 4. **SpliceAI-Compatible Evaluation**
- **Issue**: No support for SpliceAI's specific chromosome split and paralog filtering
- **Impact**: Cannot directly compare results with SpliceAI benchmarks
- **Requirement**: New feature needed

---

## Phase 1: Core Infrastructure Completion

### 1.1 Implement Missing Scalability Modules

**Priority**: High  
**Estimated Effort**: 2-3 weeks

**Required Files to Create:**
```
meta_spliceai/splice_engine/meta_models/training/
├── scalability_utils.py
├── chunked_datasets.py
└── sparse_feature_utils.py
```

**Key Functions Needed:**
```python
# scalability_utils.py
def select_features(X, y, max_features=1000, method='model', **kwargs):
    """Feature selection for large datasets"""
    
def optimize_memory_usage(df, categorical_cols=None):
    """Optimize DataFrame memory usage"""
    
def create_sparse_kmer_matrix(df, kmer_cols):
    """Convert k-mer features to sparse representation"""

# chunked_datasets.py  
def load_dataset_chunked(dataset_path, chunksize=10000, **kwargs):
    """Memory-efficient chunked dataset loading"""
    
def process_chunk_for_cv(chunk, feature_cols, label_col):
    """Process individual data chunks for CV"""

# sparse_feature_utils.py
def combine_sparse_dense_features(X_sparse, X_dense):
    """Efficiently combine sparse and dense feature matrices"""
```

### 1.2 Integrate Inference Workflow

**Priority**: High  
**Estimated Effort**: 1-2 weeks

**Implementation Steps:**
1. Add inference workflow imports to `run_loco_cv_multiclass_scalable.py`
2. Implement `generate_per_nucleotide_meta_scores()` function
3. Add inference-compatible model saving format
4. Create chromosome-specific inference evaluation

**Code Template:**
```python
# Add to run_loco_cv_multiclass_scalable.py after line 1670
try:
    from meta_spliceai.splice_engine.meta_models.training.meta_evaluation_utils import generate_per_nucleotide_meta_scores
    
    score_tensor_path = generate_per_nucleotide_meta_scores(
        dataset_path=args.dataset,
        run_dir=out_dir,
        sample=diag_sample,
        output_format="parquet",
        verbose=args.verbose,
    )
    
    logger.info(f"✅ Per-nucleotide meta-scores generated: {score_tensor_path}")
    
except Exception as e:
    logger.warning(f"⚠️ Per-nucleotide score generation failed: {e}")
```

### 1.3 Implement Post-Training Evaluation

**Priority**: High  
**Estimated Effort**: 1 week

**Required Functions:**
```python
def run_chromosome_aware_evaluation(
    dataset_path: str,
    model_dir: Path,
    test_chromosomes: List[str],
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """Run comprehensive evaluation on held-out chromosomes"""
    
def compare_meta_vs_base_chromosome_level(
    dataset_path: str,
    model_dir: Path,
    output_path: Path
) -> pd.DataFrame:
    """Generate chromosome-level meta vs base comparison"""
```

---

## Phase 2: SpliceAI-Compatible Evaluation

### 2.1 Implement SpliceAI Chromosome Split

**Priority**: Medium  
**Estimated Effort**: 1 week

**SpliceAI Configuration:**
- **Training Chromosomes**: 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, X, Y
- **Test Chromosomes**: 1, 3, 5, 7, 9

**Implementation:**
```python
def create_spliceai_splits(
    df: pd.DataFrame,
    chrom_col: str = "chrom",
    gene_col: str = "gene_id"
) -> Tuple[np.ndarray, np.ndarray]:
    """Create SpliceAI-compatible train/test splits"""
    
    # SpliceAI test chromosomes
    test_chroms = ['1', '3', '5', '7', '9']
    train_chroms = ['2', '4', '6', '8', '10', '11', '12', '13', '14', 
                   '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
    
    test_mask = df[chrom_col].isin(test_chroms)
    train_mask = df[chrom_col].isin(train_chroms)
    
    return train_mask, test_mask

# Add CLI argument
parser.add_argument(
    "--spliceai-split",
    action="store_true", 
    help="Use SpliceAI's chromosome split (train: 2,4,6,8,10-22,X,Y; test: 1,3,5,7,9)"
)
```

### 2.2 Implement Paralog Filtering

**Priority**: Medium  
**Estimated Effort**: 2 weeks

**Requirements:**
1. Integration with Ensembl BioMart API
2. Paralog detection and filtering
3. Gene similarity analysis

**Implementation Plan:**
```python
# Create new module: paralog_filter.py
class ParalogFilter:
    def __init__(self, ensembl_release: str = "109"):
        self.ensembl_release = ensembl_release
        self.biomart_server = f"http://ensembl.org/biomart"
    
    def get_gene_paralogs(self, gene_ids: List[str]) -> Dict[str, List[str]]:
        """Query Ensembl BioMart for gene paralogs"""
        
    def filter_paralogous_genes(
        self, 
        test_genes: List[str], 
        train_genes: List[str]
    ) -> List[str]:
        """Remove test genes that have paralogs in training set"""
        
    def calculate_sequence_similarity(
        self, 
        gene1: str, 
        gene2: str
    ) -> float:
        """Calculate sequence similarity between genes"""

# Usage in main script
if args.filter_paralogs:
    paralog_filter = ParalogFilter()
    filtered_test_genes = paralog_filter.filter_paralogous_genes(
        test_genes, train_genes
    )
    logger.info(f"Filtered {len(test_genes) - len(filtered_test_genes)} paralogous genes")
```

### 2.3 Implement SpliceAI-Style Metrics

**Priority**: Medium  
**Estimated Effort**: 1-2 weeks

**Required Metrics:**
1. **Site-level metrics**: Top-k accuracy, AUPRC for donor/acceptor detection
2. **Variant-level metrics**: PR-AUC for Δ-score classification
3. **Chromosome-specific performance**: Per-chromosome breakdown

**Implementation:**
```python
def calculate_spliceai_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray, 
    y_proba: np.ndarray,
    chromosomes: np.ndarray,
    genes: np.ndarray
) -> Dict[str, Any]:
    """Calculate SpliceAI-compatible evaluation metrics"""
    
    metrics = {
        'site_level': {
            'donor_auprc': calculate_donor_auprc(y_true, y_proba),
            'acceptor_auprc': calculate_acceptor_auprc(y_true, y_proba),
            'top_k_accuracy': calculate_top_k_accuracy(y_true, y_proba, k=5),
        },
        'chromosome_level': {},
        'gene_level': {}
    }
    
    # Per-chromosome metrics
    for chrom in np.unique(chromosomes):
        chrom_mask = chromosomes == chrom
        metrics['chromosome_level'][chrom] = {
            'accuracy': accuracy_score(y_true[chrom_mask], y_pred[chrom_mask]),
            'f1_macro': f1_score(y_true[chrom_mask], y_pred[chrom_mask], average='macro')
        }
    
    return metrics
```

---

## Phase 3: Advanced Features

### 3.1 Variant-Level Evaluation

**Priority**: Low  
**Estimated Effort**: 2-3 weeks

**Requirements:**
1. Integration with variant databases (ClinVar, gnomAD)
2. Δ-score calculation for variants
3. Pathogenicity prediction evaluation

### 3.2 Enhanced Visualization

**Priority**: Low  
**Estimated Effort**: 1 week

**Features:**
- Chromosome-specific performance plots
- Paralog filtering impact analysis
- SpliceAI comparison dashboards

---

## Implementation Priority Matrix

| Component | Priority | Effort | Dependencies | Impact |
|-----------|----------|--------|--------------|--------|
| Scalability Modules | High | 2-3 weeks | None | High |
| Inference Integration | High | 1-2 weeks | Scalability | High |
| Post-Training Evaluation | High | 1 week | Inference | High |
| SpliceAI Chromosome Split | Medium | 1 week | None | Medium |
| Paralog Filtering | Medium | 2 weeks | BioMart API | Medium |
| SpliceAI Metrics | Medium | 1-2 weeks | Chromosome Split | Medium |
| Variant Evaluation | Low | 2-3 weeks | External DBs | Low |
| Enhanced Visualization | Low | 1 week | All above | Low |

---

## Recommended Implementation Sequence

### **Sprint 1 (3-4 weeks): Core Infrastructure**
1. Implement scalability modules (`scalability_utils.py`, `chunked_datasets.py`)
2. Add inference workflow integration
3. Implement post-training evaluation pipeline
4. **Deliverable**: Fully functional chromosome-aware CV with inference support

### **Sprint 2 (2-3 weeks): SpliceAI Compatibility**
1. Implement SpliceAI chromosome split
2. Add paralog filtering capability
3. Implement SpliceAI-style metrics
4. **Deliverable**: SpliceAI-compatible evaluation workflow

### **Sprint 3 (2-3 weeks): Advanced Features** (Optional)
1. Variant-level evaluation
2. Enhanced visualization
3. Performance optimization
4. **Deliverable**: Production-ready chromosome-aware CV system

---

## Testing Strategy

### Unit Tests Required
```python
# test_chromosome_cv.py
def test_spliceai_chromosome_split():
    """Test SpliceAI chromosome splitting logic"""
    
def test_paralog_filtering():
    """Test paralog detection and filtering"""
    
def test_scalability_modules():
    """Test chunked loading and sparse features"""

# Integration tests
def test_full_chromosome_cv_pipeline():
    """Test complete chromosome CV workflow"""
    
def test_spliceai_compatibility():
    """Test SpliceAI-compatible evaluation"""
```

### Validation Datasets
1. **Small test dataset**: 1000 genes across all chromosomes
2. **Medium validation dataset**: 5000 genes with known paralogs
3. **Large production dataset**: 20K+ genes for scalability testing

---

## Success Criteria

### **Phase 1 Success Metrics:**
- [ ] Chromosome CV completes on 20K+ gene datasets
- [ ] Memory usage < 32GB for large datasets
- [ ] Inference workflow integration functional
- [ ] Performance matches gene-aware CV quality

### **Phase 2 Success Metrics:**
- [ ] SpliceAI chromosome split implemented and tested
- [ ] Paralog filtering reduces test set appropriately
- [ ] Metrics match SpliceAI paper methodology
- [ ] Results comparable to published SpliceAI benchmarks

### **Phase 3 Success Metrics:**
- [ ] Variant-level evaluation functional
- [ ] Comprehensive visualization suite
- [ ] Production deployment ready
- [ ] Documentation complete

---

## Resource Requirements

### **Development Resources:**
- **Primary Developer**: 1 FTE for 8-12 weeks
- **Code Review**: Senior developer, 2-4 hours/week
- **Testing**: QA engineer, 1-2 days/week during sprints 2-3

### **Infrastructure Requirements:**
- **Compute**: 64GB RAM, 16+ cores for large dataset testing
- **Storage**: 500GB for intermediate datasets and results
- **External APIs**: Ensembl BioMart access for paralog queries

### **External Dependencies:**
- Ensembl BioMart API (paralog filtering)
- ClinVar/gnomAD databases (variant evaluation)
- Updated genomic reference files

---

## Risk Assessment

### **High Risk Items:**
1. **Scalability module complexity**: Chunked loading may require significant optimization
2. **BioMart API reliability**: External dependency for paralog filtering
3. **Memory constraints**: Large sparse matrices may exceed system limits

### **Mitigation Strategies:**
1. **Incremental implementation**: Build and test scalability modules iteratively
2. **API fallbacks**: Cache paralog data locally, provide manual override options
3. **Memory monitoring**: Implement memory usage tracking and automatic chunking

---

## Conclusion

Completing the chromosome-aware CV workflow will provide a robust, SpliceAI-compatible evaluation system that enables direct comparison with published benchmarks. The phased approach ensures incremental progress while maintaining system stability.

**Next Immediate Actions:**
1. Create `scalability_utils.py` module with feature selection functions
2. Implement basic chunked dataset loading
3. Add inference workflow integration hooks
4. Begin unit test development

This roadmap provides a clear path to a production-ready chromosome-aware cross-validation system that meets both internal needs and external benchmark compatibility requirements.
