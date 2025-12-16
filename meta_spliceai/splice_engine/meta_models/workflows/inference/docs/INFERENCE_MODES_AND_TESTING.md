# 🎯 **Inference Modes & Comprehensive Testing Guide**

A complete guide to the three inference modes and comprehensive test cases covering all scenarios.

---

## 🎯 **Inference Modes**

The workflow supports three distinct inference modes to balance recalibration accuracy and computational efficiency:

### **1. `hybrid` (Default - Recommended)**
- **Strategy**: Combines base model predictions with meta-model recalibration for uncertain positions
- **Threshold**: Applies meta-model to positions with base model confidence between 0.02-0.80
- **Use Case**: Production inference balancing accuracy and efficiency
- **Output**: Complete predictions with selective meta-model improvements
- **Performance**: Optimal balance of accuracy improvement and computational cost

### **2. `base_only`**
- **Strategy**: Uses only base SpliceAI model predictions without meta-model recalibration
- **Threshold**: No uncertainty-based filtering
- **Use Case**: Baseline comparison, high-throughput scenarios, resource-constrained environments
- **Output**: Pure base model predictions
- **Performance**: Fastest execution, baseline accuracy

### **3. `meta_only`**
- **Strategy**: Forces meta-model recalibration on all positions (experimental)
- **Threshold**: Applies meta-model to all positions regardless of base model confidence
- **Use Case**: Research validation, maximum recalibration testing
- **Output**: All positions processed through meta-model
- **Performance**: Highest computational cost, maximum recalibration coverage

---

## 🧬 **Gene Processing Scenarios**

The workflow intelligently handles different gene scenarios:

### **Scenario 1: Genes with Unseen Positions**
- **Description**: Genes present in training data but with positions missing due to TN downsampling
- **Detection**: Genes found in `gene_features.tsv` with some existing artifacts
- **Processing**: Loads existing artifacts, generates features for unseen positions only
- **Artifacts**: Uses pre-computed `analysis_sequences_*.tsv` files
- **Performance**: Fast (reuses existing base model results)

### **Scenario 2A: Unseen Genes with Existing Artifacts**
- **Description**: Genes not in training data but with pre-computed base model artifacts
- **Detection**: Genes found in artifact directories but not in training coverage
- **Processing**: Loads existing artifacts, applies full meta-model inference
- **Artifacts**: Uses pre-computed `analysis_sequences_*.tsv` files
- **Performance**: Medium (feature enrichment required)

### **Scenario 2B: Completely Unprocessed Genes**
- **Description**: Genes not in training data and no existing artifacts
- **Detection**: Genes not found in any artifact directories
- **Processing**: Triggers full base model pipeline + feature enrichment
- **Artifacts**: Generates new `analysis_sequences_*.tsv` files with 57-column structure
- **Performance**: Slow (full pipeline execution) but optimized with caching
- **Optimization**: Uses "reuse-first" strategy for genomic resources

---

## 🧪 **Comprehensive Test Cases**

The following test cases validate all three inference modes across different gene scenarios:

### **Test Case 1: Scenario 1 Genes (Unseen Positions)**

These genes are present in training data but have positions missing due to TN downsampling:

```bash
# Find genes with unseen positions
python -c "
import pandas as pd
import polars as pl

# Load training gene manifest
train_manifest = pd.read_csv('train_pc_1000_3mers/master/gene_manifest.csv')
print('Training genes (first 10):')
print(train_manifest['gene_id'].head(10).tolist())

# Select test genes (confirmed in training)
test_genes_scenario1 = ['ENSG00000157764', 'ENSG00000139618', 'ENSG00000156006']
print(f'Scenario 1 test genes: {test_genes_scenario1}')
"
```

#### **Test 1A: Hybrid Mode (Default)**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000157764,ENSG00000139618,ENSG00000156006 \
    --output-dir results/test_scenario1_hybrid \
    --inference-mode hybrid \
    --uncertainty-low 0.02 \
    --uncertainty-high 0.80 \
    --verbose
```

#### **Test 1B: Base Only Mode**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000157764,ENSG00000139618,ENSG00000156006 \
    --output-dir results/test_scenario1_base_only \
    --inference-mode base_only \
    --verbose
```

#### **Test 1C: Meta Only Mode**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000157764,ENSG00000139618,ENSG00000156006 \
    --output-dir results/test_scenario1_meta_only \
    --inference-mode meta_only \
    --verbose
```

### **Test Case 2: Scenario 2B Genes (Completely Unprocessed)**

These genes are not in training data and have no existing artifacts:

```bash
# Find genes not in training but present in gene_features.tsv
python -c "
import pandas as pd
import polars as pl

# Load training gene manifest
train_manifest = pd.read_csv('train_pc_1000_3mers/master/gene_manifest.csv')
training_genes = set(train_manifest['gene_id'].tolist())

# Load gene features
gene_features = pd.read_csv('data/ensembl/spliceai_analysis/gene_features.tsv', sep='\t')
all_genes = set(gene_features['gene_id'].tolist())

# Find genes NOT in training (unseen test genes)
unseen_genes = list(all_genes - training_genes)
print(f'Total unseen genes: {len(unseen_genes)}')

# Select test genes (confirmed NOT in training)
test_genes_scenario2b = ['ENSG00000142611', 'ENSG00000261456', 'ENSG00000268895']
print(f'Scenario 2B test genes: {test_genes_scenario2b}')

# Verify they're not in training
for gene in test_genes_scenario2b:
    in_training = gene in training_genes
    print(f'{gene}: in_training={in_training}')
"
```

#### **Test 2A: Hybrid Mode (Default)**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000142611,ENSG00000261456,ENSG00000268895 \
    --output-dir results/test_scenario2b_hybrid \
    --inference-mode hybrid \
    --uncertainty-low 0.02 \
    --uncertainty-high 0.80 \
    --verbose
```

#### **Test 2B: Base Only Mode**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000142611,ENSG00000261456,ENSG00000268895 \
    --output-dir results/test_scenario2b_base_only \
    --inference-mode base_only \
    --verbose
```

#### **Test 2C: Meta Only Mode**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000142611,ENSG00000261456,ENSG00000268895 \
    --output-dir results/test_scenario2b_meta_only \
    --inference-mode meta_only \
    --verbose
```

### **Test Case 3: Mixed Scenario Testing**

Test both scenarios together to validate scenario detection:

#### **Test 3A: Mixed Genes - Hybrid Mode**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000157764,ENSG00000142611,ENSG00000139618,ENSG00000261456 \
    --output-dir results/test_mixed_hybrid \
    --inference-mode hybrid \
    --verbose
```

---

## 📊 **Expected Validation Results**

### **Performance Metrics to Validate**

1. **Inference Mode Behavior**:
   - `base_only`: Meta-model usage = 0%, fastest execution
   - `hybrid`: Meta-model usage = 2-5%, balanced performance
   - `meta_only`: Meta-model usage = 100%, slowest execution

2. **Scenario Detection**:
   - Scenario 1 genes: Fast processing, reuse existing artifacts
   - Scenario 2B genes: Slower processing, generate new artifacts

3. **Output Consistency**:
   - All modes produce valid predictions
   - `hybrid` combines base + meta predictions appropriately
   - Column structure consistent across all modes

### **Success Criteria**

✅ **Mode Validation**:
- [ ] `base_only` completes without meta-model calls
- [ ] `hybrid` applies meta-model only to uncertain positions
- [ ] `meta_only` applies meta-model to all positions
- [ ] Default mode is correctly set to `hybrid`

✅ **Scenario Validation**:
- [ ] Scenario 1 genes process quickly with existing artifacts
- [ ] Scenario 2B genes trigger full artifact generation
- [ ] Generated artifacts have 57-column structure matching training data
- [ ] "Reuse-first" optimizations work for global genomic files

✅ **Integration Validation**:
- [ ] Recalibrated predictions properly integrated with base predictions
- [ ] Uncertainty thresholds correctly applied (0.02-0.80 default)
- [ ] Results saved to organized directory structure
- [ ] Performance reports generated with accurate statistics

---

## 🔧 **Debugging & Troubleshooting**

### **Common Issues**

1. **Column Mismatch Errors**: Ensure inference artifacts have same 57-column structure as training
2. **Missing Artifacts**: Verify Scenario 2B artifact generation is working
3. **Performance Issues**: Check that caching optimizations are active
4. **Mode Confusion**: Validate meta-model usage percentages match expected mode behavior

### **Validation Commands**

```bash
# Check results structure
ls -la results/test_*/

# Validate performance reports
cat results/test_*/performance_report.txt

# Check meta-model usage statistics
grep "Meta-model usage" results/test_*/performance_report.txt

# Validate column consistency
python -c "
import pandas as pd
base_df = pd.read_parquet('results/test_scenario1_hybrid/selective_inference/predictions/*/base_model_predictions.parquet')
meta_df = pd.read_parquet('results/test_scenario1_hybrid/selective_inference/predictions/*/meta_model_predictions.parquet')
print(f'Base model columns: {base_df.shape[1]}')
print(f'Meta model columns: {meta_df.shape[1]}')
print('Column consistency:', base_df.columns.equals(meta_df.columns))
"
```

---

## 🎯 **Production Deployment Checklist**

Before deploying to production, validate:

- [ ] All three inference modes work correctly
- [ ] Both gene scenarios (1 and 2B) process successfully
- [ ] Performance optimizations are active
- [ ] Output structure is organized and consistent
- [ ] Meta-model integration works as expected
- [ ] Default `hybrid` mode balances accuracy and efficiency
- [ ] Error handling covers edge cases
- [ ] Documentation is complete and accurate

The comprehensive test suite above validates all critical functionality needed for production deployment.