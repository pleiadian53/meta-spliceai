# 🧬 **Main-Entry Meta-Model Inference Workflow**

A comprehensive, production-ready command-line tool for performing meta-model inference on arbitrary genes using pre-trained meta-models.

## 🎯 **Overview**

The **Main-Entry Inference Workflow** (`main_inference_workflow.py`) is a unified, practical script that simplifies meta-model inference for both research and production use cases. It provides:

- **🔧 Flexible Parameterization**: Arbitrary models, datasets, and target genes
- **⚡ Efficient Selective Processing**: Reuses confident base model predictions
- **📊 Complete Coverage Capability**: Optional full genomic coverage
- **📁 Structured Data Management**: Organized outputs with gene manifests
- **🚀 Performance Optimization**: Parallel processing and memory efficiency
- **🔍 Comprehensive Reporting**: Detailed statistics and performance metrics

---

## 🏗️ **Architecture Integration**

```
┌─────────────────────────────────────────────────────────────┐
│               Main-Entry Workflow                           │
│           (main_inference_workflow.py)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Workflow Orchestration                    │    │
│  │    • Gene manifest management                       │    │
│  │    • Parallel processing coordination               │    │
│  │    • Performance reporting                          │    │
│  │    • Error handling & logging                       │    │
│  │  ┌───────────────────────────────────────────────┐   │    │
│  │  │         Enhanced Meta-Score Generation        │   │    │
│  │  │      (meta_evaluation_utils.py)               │   │    │
│  │  │  ┌─────────────────────────────────────────┐  │   │    │
│  │  │  │      Selective Meta Inference           │  │   │    │
│  │  │  │   (selective_meta_inference.py)         │  │   │    │
│  │  │  └─────────────────────────────────────────┘  │   │    │
│  │  └───────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

The main-entry workflow acts as a **high-level orchestrator** that:
- Manages the complete lifecycle of inference tasks
- Coordinates with existing selective inference components
- Provides user-friendly command-line interface
- Handles data management and result aggregation

---

## 🚀 **Quick Start**

### **Basic Usage**

```bash
# Run inference on specific genes
python main_inference_workflow.py \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000104435,ENSG00000006420 \
    --output-dir ./inference_results
```

### **Run as Module (Recommended)**

```bash
# From project root
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000104435 \
    --output-dir inference_results
```

---

## 📋 **Command-Line Interface**

### **Required Arguments**

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Path to pre-trained meta-model (.pkl file) | `results/model_multiclass.pkl` |
| `--genes` OR `--genes-file` | Target genes (comma-separated OR file path) | `ENSG001,ENSG002` OR `genes.txt` |

### **Key Optional Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--training-dataset` | `None` | Path to original training dataset directory |
| `--output-dir` | `./inference_results` | Output directory for all results |
| `--uncertainty-low` | `0.02` | Lower uncertainty threshold for selective processing |
| `--uncertainty-high` | `0.80` | Upper uncertainty threshold for selective processing |
| `--complete-coverage` | `False` | Generate predictions for ALL positions |
| `--strategy` | `SELECTIVE` | Position selection strategy |
| `--parallel-workers` | `1` | Number of parallel workers |
| `--verbose / -v` | `1` | Verbosity level (use `-v`, `-vv`, or `-vvv`) |

### **Full Argument Reference**

```bash
python main_inference_workflow.py --help
```

---

## 📊 **Processing Strategies**

### **1. SELECTIVE (Default)**
- **Best for**: Most use cases requiring efficiency
- **Behavior**: Meta-model recalibration only on uncertain base model predictions
- **Memory**: ~80-95% reduction compared to complete coverage
- **Speed**: ~88% of positions processed selectively

### **2. COMPLETE**
- **Best for**: Comprehensive analysis requiring every position
- **Behavior**: Generate meta-model predictions for all positions
- **Memory**: Full feature matrix generation
- **Speed**: Slowest but most comprehensive

### **3. TRAINING_GAPS**
- **Best for**: Genes from the training dataset with known gaps
- **Behavior**: Focus on positions not included in original training
- **Memory**: Moderate (depends on training coverage)
- **Speed**: Faster than COMPLETE, more targeted than SELECTIVE

### **4. UNCERTAINTY_FOCUSED**
- **Best for**: High-precision analysis of ambiguous regions
- **Behavior**: Prioritize highest-uncertainty positions
- **Memory**: Minimal (only most uncertain positions)
- **Speed**: Fastest, but limited coverage

---

## 🎯 **Practical Use Cases**

### **1. Research Analysis on Known Error-Prone Genes**

```bash
# Use genes identified from training analysis
python main_inference_workflow.py \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes-file top_error_prone_genes.txt \
    --strategy SELECTIVE \
    --output-dir error_prone_analysis \
    --verbose
```

**Expected outputs:**
- Individual gene prediction files
- Comparative statistics (meta vs base model)
- Performance efficiency metrics

### **2. High-Throughput Production Pipeline**

```bash
# Process large gene lists with parallel workers
python main_inference_workflow.py \
    --model production_model.pkl \
    --training-dataset production_dataset \
    --genes-file large_gene_list.txt \
    --parallel-workers 8 \
    --strategy UNCERTAINTY_FOCUSED \
    --output-dir production_results \
    --no-cleanup
```

**Expected outputs:**
- Scalable processing across multiple genes
- Detailed performance reports
- Organized result hierarchy for downstream analysis

### **3. Complete Genomic Coverage Analysis**

```bash
# Comprehensive analysis with full position coverage
python main_inference_workflow.py \
    --model comprehensive_model.pkl \
    --training-dataset comprehensive_dataset \
    --genes ENSG00000104435 \
    --complete-coverage \
    --strategy COMPLETE \
    --uncertainty-low 0.01 \
    --uncertainty-high 0.90 \
    --output-dir comprehensive_analysis \
    -vv
```

**Expected outputs:**
- Complete probability tensors for every nucleotide
- Detailed confidence statistics
- Full genomic span coverage verification

### **4. New Gene Discovery Pipeline**

```bash
# Analyze genes not in training data
python main_inference_workflow.py \
    --model discovery_model.pkl \
    --genes-file novel_genes.txt \
    --strategy SELECTIVE \
    --max-positions 50000 \
    --output-dir discovery_results \
    --force-recompute
```

**Expected outputs:**
- Predictions for completely novel genes
- Base model vs meta-model performance comparison
- Uncertainty distribution analysis

---

## 📁 **Output Structure**

The workflow creates a comprehensive, organized output structure:

```
inference_results/
├── inference_workflow.log              # Detailed execution log
├── gene_manifest.json                  # Gene processing tracking
├── inference_summary.json              # Overall results summary
├── performance_report.txt              # Performance analysis
├── genes/                              # Individual gene results
│   ├── ENSG00000104435/
│   │   ├── ENSG00000104435_predictions.parquet
│   │   └── ENSG00000104435_statistics.json
│   └── ENSG00000006420/
│       ├── ENSG00000006420_predictions.parquet
│       └── ENSG00000006420_statistics.json
└── selective_inference/                # Intermediate artifacts
    ├── analysis_sequences_*.parquet
    ├── splice_positions_enhanced_*.parquet
    └── gene_manifest.csv
```

### **Key Output Files**

| File | Description | Contents |
|------|-------------|----------|
| `gene_manifest.json` | Processing tracker | Gene status, configurations, timestamps |
| `inference_summary.json` | Results overview | Success rates, statistics, performance |
| `*_predictions.parquet` | Per-gene predictions | Complete probability tensors |
| `*_statistics.json` | Per-gene statistics | Class distributions, confidence metrics |
| `performance_report.txt` | Performance analysis | Runtime, efficiency, resource usage |

---

## 🔍 **Gene Manifest System**

### **Purpose**
The gene manifest tracks processing status to:
- **Avoid redundant computation** for already-processed genes
- **Enable incremental processing** of large gene lists
- **Maintain configuration consistency** across runs
- **Provide audit trail** for research reproducibility

### **Manifest Structure**

```json
{
  "created": "2024-01-15 10:30:00",
  "model_path": "results/model_multiclass.pkl",
  "training_dataset_path": "train_pc_1000_3mers",
  "config": {
    "uncertainty_threshold_low": 0.02,
    "uncertainty_threshold_high": 0.80,
    "selective_strategy": "SELECTIVE"
  },
  "processed_genes": {
    "ENSG00000104435": {
      "success": true,
      "output_file": "genes/ENSG00000104435/ENSG00000104435_predictions.parquet",
      "processing_time": 45.2,
      "statistics": {...},
      "processed_timestamp": "2024-01-15 10:35:00"
    }
  },
  "performance_summary": {...}
}
```

### **Incremental Processing**

```bash
# First run: Process initial genes
python main_inference_workflow.py --genes GENE1,GENE2 --output-dir results

# Second run: Add more genes (reuses existing results)
python main_inference_workflow.py --genes GENE1,GENE2,GENE3,GENE4 --output-dir results

# Force recomputation if needed
python main_inference_workflow.py --genes GENE1 --output-dir results --force-recompute
```

---

## ⚡ **Performance Optimization**

### **Memory Efficiency**

The workflow implements several memory optimization strategies:

1. **Selective Featurization**: Only generates features for uncertain positions
2. **Streaming Processing**: Processes genes individually to avoid memory accumulation
3. **Intermediate Cleanup**: Removes temporary files unless `--no-cleanup` specified
4. **Chunked Operations**: Breaks large operations into manageable chunks

### **Computational Efficiency**

```bash
# Example performance characteristics (typical gene ~50,000 bp)
Strategy          Memory Usage    Processing Time    Coverage
SELECTIVE         ~5-20% of full  Fast (30-60s)     ~12% positions
UNCERTAINTY_FOC   ~2-10% of full  Very fast (15-30s) ~5% positions  
TRAINING_GAPS     ~10-40% of full Medium (60-120s)  ~20% positions
COMPLETE          100% of full    Slow (300-600s)   100% positions
```

### **Parallel Processing**

```bash
# Utilize multiple CPU cores
python main_inference_workflow.py \
    --genes-file large_list.txt \
    --parallel-workers 4 \
    --output-dir parallel_results
```

**Guidelines:**
- **Workers = CPU cores**: Good starting point
- **Workers < genes**: Each worker processes multiple genes sequentially
- **Workers > genes**: Some workers will be idle
- **Memory consideration**: More workers = higher peak memory usage

---

## 🔬 **Advanced Configuration**

### **Dynamic Uncertainty Thresholds**

Fine-tune selective processing behavior:

```bash
# Conservative (more meta-model usage)
--uncertainty-low 0.05 --uncertainty-high 0.75

# Aggressive (minimal meta-model usage)  
--uncertainty-low 0.01 --uncertainty-high 0.90

# Balanced (default)
--uncertainty-low 0.02 --uncertainty-high 0.80
```

### **Position Limits**

Control computational scope:

```bash
# Standard genes
--max-positions 10000

# Large genes (e.g., dystrophin)
--max-positions 100000

# Quick analysis
--max-positions 1000
```

### **Verbosity Levels**

```bash
-q           # Quiet (errors only)
(default)    # Normal (progress + summary)
-v           # Verbose (detailed progress)
-vv          # Very verbose (debug info)
-vvv         # Maximum verbosity (all details)
```

---

## 🧪 **Example Workflows**

### **Workflow 1: Reproduce Demo Results**

```bash
# Use error-prone genes identified in training analysis
python main_inference_workflow.py \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000104435,ENSG00000006420,ENSG00000166986 \
    --output-dir demo_reproduction \
    --verbose
```

**Expected outcomes:**
- Clear meta-model advantage over base model
- F1-score improvements for donor/acceptor classes  
- Efficiency gains from selective processing

### **Workflow 2: Novel Gene Analysis**

```bash
# Analyze genes not in training data
python main_inference_workflow.py \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --genes ENSG00000198888,ENSG00000198899 \
    --strategy SELECTIVE \
    --output-dir novel_gene_analysis \
    --complete-coverage \
    -v
```

**Expected outcomes:**
- Complete probability tensors for novel genes
- Base model uncertainty identification
- Meta-model recalibration on uncertain positions

### **Workflow 3: High-Throughput Research**

```bash
# Process many genes efficiently
echo "ENSG00000104435
ENSG00000006420  
ENSG00000166986
ENSG00000198888
ENSG00000198899" > research_genes.txt

python main_inference_workflow.py \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes-file research_genes.txt \
    --parallel-workers 2 \
    --strategy UNCERTAINTY_FOCUSED \
    --output-dir research_pipeline \
    --performance-reporting
```

**Expected outcomes:**
- Efficient processing of multiple genes
- Comprehensive performance statistics
- Research-ready organized outputs

---

## 🔧 **Troubleshooting**

### **Common Issues**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Memory Error** | OOM during processing | Use `--strategy UNCERTAINTY_FOCUSED` or reduce `--max-positions` |
| **Model Not Found** | FileNotFoundError | Check model path relative to current directory |
| **No Genes Processed** | Empty results | Verify gene IDs match training data format |
| **Slow Performance** | Long processing times | Increase `--parallel-workers` or use selective strategy |

### **Debugging Steps**

1. **Check Prerequisites**
   ```bash
   # Verify model exists
   ls -la results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl
   
   # Verify training dataset
   ls -la train_pc_1000_3mers/
   ```

2. **Run with Increased Verbosity**
   ```bash
   python main_inference_workflow.py \
       --model YOUR_MODEL \
       --genes YOUR_GENES \
       --output-dir debug_run \
       -vvv
   ```

3. **Check Output Logs**
   ```bash
   # Review detailed execution log
   cat debug_run/inference_workflow.log
   
   # Check error patterns
   grep -i error debug_run/inference_workflow.log
   ```

4. **Test with Single Gene**
   ```bash
   # Isolate issues with minimal test
   python main_inference_workflow.py \
       --model YOUR_MODEL \
       --genes ENSG00000104435 \
       --output-dir single_gene_test \
       --verbose
   ```

### **Performance Optimization**

| Scenario | Recommendation |
|----------|----------------|
| **Large genes (>100kb)** | Use `--strategy UNCERTAINTY_FOCUSED` + `--max-positions 50000` |
| **Many small genes** | Use `--parallel-workers 4-8` |
| **Memory-constrained** | Use `--strategy SELECTIVE` + `--cleanup-intermediates` |
| **Time-constrained** | Use `--strategy UNCERTAINTY_FOCUSED` + parallel processing |

---

## 📖 **Integration with Existing Tools**

### **With Demo Scripts**

The main workflow complements existing demo scripts:

```bash
# Use main workflow for batch processing
python main_inference_workflow.py --genes-file genes.txt --output-dir batch_results

# Use demo script for detailed analysis  
python demo_accuracy_evaluation.py --genes ENSG00000104435 --output-dir detailed_analysis
```

### **With Test Suite**

```bash
# Verify system health before main workflow
python run_inference_tests.py --quick

# Run main workflow
python main_inference_workflow.py --model MODEL --genes GENES --output-dir results

# Validate results with specific tests
python test_inference_accuracy.py results/
```

### **With Selective Inference Components**

The main workflow automatically leverages:
- `selective_meta_inference.py` for core processing logic
- `meta_evaluation_utils.py` for enhanced score generation  
- `splice_inference_workflow.py` for feature preparation
- `splice_prediction_workflow.py` for base model execution

---

## 🎯 **Best Practices**

### **For Research Use**

1. **Start with known genes** from training analysis
2. **Use selective strategy** for initial exploration  
3. **Enable complete coverage** for final publication results
4. **Document configuration** in research notes
5. **Keep intermediate files** (`--no-cleanup`) for reproducibility

### **For Production Use**

1. **Use parallel processing** for throughput
2. **Implement error handling** for robust pipelines
3. **Monitor resource usage** with performance reporting
4. **Use incremental processing** for large datasets
5. **Clean up intermediates** for storage efficiency

### **For Comparative Analysis**

1. **Consistent uncertainty thresholds** across experiments
2. **Document model and dataset versions**
3. **Use identical gene lists** for fair comparison
4. **Include base model baseline** results
5. **Report efficiency metrics** alongside accuracy

---

## 📚 **Additional Resources**

- **Demo Scripts**: `meta_spliceai/splice_engine/meta_models/workflows/inference/`
- **Test Suite**: `meta_spliceai/splice_engine/meta_models/tests/inference_workflow/`
- **Architecture Documentation**: `SELECTIVE_INFERENCE_IMPLEMENTATION.md`
- **Training Workflows**: `meta_spliceai/splice_engine/meta_models/training/`

---

## 🆘 **Support**

For issues, feature requests, or questions:

1. **Check the troubleshooting section** above
2. **Review test suite results** to verify system health
3. **Run with maximum verbosity** (`-vvv`) for detailed diagnostics
4. **Check existing demo scripts** for working examples
5. **Consult the architecture documentation** for system design details

---

**The Main-Entry Inference Workflow provides a production-ready, comprehensive solution for meta-model inference that bridges the gap between research prototypes and practical applications.**