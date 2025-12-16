# 🧬 **Complete Splice Surveyor Inference Workflow**

A comprehensive end-to-end guide covering the complete pipeline from training data curation through meta-model training to production inference and splice site recalibration.

---

## 🎯 **Complete Inference Workflow Overview**

The Splice Surveyor inference workflow is a **three-phase pipeline** that transforms raw genomic data into enhanced splice site predictions:

1. **🏗️ Training Data Curation** - Create comprehensive training datasets with position-centric features and k-mer analysis
2. **🧠 Meta-Model Training & Evaluation** - Train scalable XGBoost meta-models with advanced ensemble techniques for large datasets
3. **🚀 Production Inference & Recalibration** - Apply trained models to predict and recalibrate splice site scores on unseen positions and genes

Each phase builds upon the previous one, creating a **complete inference pipeline** that enhances splice site prediction accuracy through intelligent meta-learning and selective recalibration.

---

## 📋 **Prerequisites**

### Environment Setup
```bash
# Activate the surveyor environment (REQUIRED)
mamba activate surveyor

# Verify you're in the project root
cd /path/to/meta-spliceai

# Create necessary directories
mkdir -p logs results
```

### Required Resources
- **Genomic Data**: Ensembl gene features, splice sites, and sequence data
- **Base Model**: Pre-trained SpliceAI model for initial predictions
- **Computational Resources**: 16-32GB RAM recommended for large datasets

---

## 🏗️ **Phase 1: Training Data Curation**

### Step 1.1: Create Training Dataset with Incremental Builder

The incremental builder creates comprehensive training datasets with position-centric features and k-mer analysis.

#### Basic Training Dataset (Protein-Coding Genes)
```bash
# Create a focused protein-coding dataset
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --gene-types protein_coding \
    --subset-policy random \
    --output-dir train_pc_5000_3mers_focused \
    --kmer-sizes 3 \
    --batch-size 200 \
    --batch-rows 15000 \
    --run-workflow \
    --verbose \
    2>&1 | tee logs/train_pc_5000_assembly.log
```

#### Advanced Regulatory Dataset (Multi-Gene Types)
```bash
# Create a comprehensive regulatory dataset with multiple k-mer sizes
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 10000 \
    --gene-types protein_coding lncRNA \
    --subset-policy random \
    --output-dir train_regulatory_10k_kmers \
    --kmer-sizes 3 5 \
    --batch-size 200 \
    --batch-rows 15000 \
    --run-workflow \
    --verbose \
    2>&1 | tee logs/train_regulatory_10k_assembly.log
```

### Step 1.2: Validate Dataset Quality

```bash
# Validate schema consistency
python -m meta_spliceai.splice_engine.meta_models.builder.validate_dataset_schema \
    --dataset train_regulatory_10k_kmers/master \
    --fix \
    --verbose

# Inspect dataset characteristics
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector \
    --dataset train_regulatory_10k_kmers/master \
    --verbose
```

**Expected Output Structure:**
```
train_regulatory_10k_kmers/
├── master/
│   ├── gene_manifest.csv              # Enhanced manifest with splice density
│   ├── batch_00001.parquet            # Training data batches
│   ├── batch_00002.parquet
│   └── ...
├── metadata/
│   ├── incremental_builder_log.txt
│   └── dataset_statistics.json
└── artifacts/                         # Base model evaluation artifacts
    ├── analysis_sequences_*.tsv
    └── splice_positions_enhanced_*.tsv
```

---

## 🧠 **Phase 2: Meta-Model Training & Evaluation**

### Step 2.1: Training Strategy Selection

The training system automatically selects the optimal strategy based on dataset size and user requirements:

#### Small to Medium Datasets (≤1,500 genes)
**Single-Model Training** - Standard approach for manageable datasets
```bash
# Automatically selected for datasets with ≤1,500 genes
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/single_model_run \
    --n-estimators 800 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --verbose
```

#### Large Datasets (>1,500 genes) 
**Multi-Instance Ensemble Training** - Scalable approach for comprehensive gene coverage
```bash
# Automatically selected when using --train-all-genes on large datasets
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/multi_instance_ensemble \
    --train-all-genes \
    --genes-per-instance 1500 \
    --n-estimators 800 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --verbose
```

**Key Benefits of Multi-Instance Ensemble:**
- ✅ **100% Gene Coverage**: All 9,280+ genes included in training
- ✅ **Memory Efficiency**: 12-15GB per instance vs >64GB single model
- ✅ **Scalability**: Handles unlimited dataset sizes
- ✅ **Enhanced Performance**: Ensemble benefits improve generalization

### Step 2.2: Standard Gene-Aware Training

Gene-aware CV ensures that genes are split across folds to prevent data leakage.

#### Minimal Training Configuration (Quick Start)
```bash
# Essential switches for standard meta-model training
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_reg_10k_kmers_run1 \
    --calibrate-per-class \
    --verbose \
    2>&1 | tee logs/gene_cv_reg_10k_kmers_run1.log
```

#### Standard Training Configuration (Production Ready)
```bash
# Recommended configuration with quality enhancements
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_reg_10k_kmers_production \
    --n-estimators 800                     # Default: 800 (high-quality model)
    --calibrate-per-class                  # Enable per-class calibration
    --auto-exclude-leaky                   # Automatic leaky feature exclusion
    --monitor-overfitting                  # Overfitting detection
    --neigh-sample 2000                    # Neighbor analysis for insights
    --verbose                              # Detailed progress output
    2>&1 | tee logs/gene_cv_reg_10k_kmers_production.log
```

#### Advanced Training with Comprehensive Analysis

**Minimal Practical Usage (Recommended):**
```bash
# Essential switches for high-quality training
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_reg_10k_kmers_comprehensive \
    --n-estimators 800 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --verbose \
    2>&1 | tee logs/gene_cv_comprehensive.log
```

**Full Reference Configuration (All Options Explicit):**
```bash
# Complete command showing all available options with their defaults
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_reg_10k_kmers_comprehensive \
    --n-folds 5                            # Default: 5 (gene-aware CV folds)
    --n-estimators 800                     # Default: 800 (XGBoost trees)
    --calibrate-per-class                  # Default: False (enable per-class calibration)
    --calib-method platt                   # Default: 'platt' (vs 'isotonic')
    --plot-curves                          # Default: True (generate ROC/PR curves)
    --plot-format pdf                      # Default: 'pdf' (vs 'png', 'svg')
    --check-leakage                        # Default: True (feature leakage detection)
    --leakage-threshold 0.95               # Default: 0.95 (correlation threshold)
    --auto-exclude-leaky                   # Default: False (automatically exclude leaky features)
    --monitor-overfitting                  # Default: False (enable overfitting detection)
    --overfitting-threshold 0.05           # Default: 0.05 (performance gap threshold)
    --early-stopping-patience 30           # Default: 20 (early stopping patience)
    --convergence-improvement 0.001        # Default: 0.001 (minimum improvement)
    --diag-sample 25000                    # Default: 25000 (diagnostic sample size)
    --neigh-sample 5000                    # Default: 0 (neighbor analysis sample)
    --neigh-window 10                      # Default: 10 (neighbor window size)
    --transcript-topk                      # Default: False (transcript-level accuracy)
    --calibration-analysis                 # Default: False (comprehensive calibration)
    --quick-overconfidence-check           # Default: True (overconfidence detection)
    --verbose                              # Default: False (detailed output)
    2>&1 | tee logs/gene_cv_comprehensive.log
```

**Key Default Behaviors:**
- ✅ **Cross-Validation**: 5 folds with gene-aware splitting (prevents data leakage)
- ✅ **Model Training**: 800 XGBoost estimators with automatic early stopping
- ✅ **Feature Processing**: Automatic leakage detection (manual exclusion by default)
- ✅ **Visualization**: ROC/PR curves generated automatically in PDF format
- ✅ **Calibration**: Basic calibration enabled, per-class requires explicit flag

**Essential vs Optional Switches:**

| Switch | Default | Essential | Purpose |
|--------|---------|-----------|---------|
| `--dataset` | *(required)* | ✅ **Required** | Training data path |
| `--out-dir` | *(required)* | ✅ **Required** | Output directory |
| `--calibrate-per-class` | `False` | ✅ **Recommended** | Better calibration |
| `--auto-exclude-leaky` | `False` | ✅ **Recommended** | Data quality |
| `--verbose` | `False` | ✅ **Recommended** | Progress tracking |
| `--train-all-genes` | `False` | ✅ **Large datasets** | Multi-instance mode |
| `--n-estimators` | `800` | ⚠️ *Optional* | Model complexity |
| `--monitor-overfitting` | `False` | ⚠️ *Optional* | Training diagnostics |
| `--neigh-sample` | `0` | ⚠️ *Optional* | Spatial analysis |
| `--calibration-analysis` | `False` | ⚠️ *Optional* | Detailed calibration |

### Step 2.3: Multi-Instance Ensemble Training (Large Datasets)

For very large datasets (>1,500 genes), use **multi-instance ensemble training** to process ALL genes with memory efficiency:

#### Minimal Multi-Instance Configuration (Recommended)
```bash
# Essential switches for multi-instance ensemble training
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_all_genes_ensemble \
    --train-all-genes \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --verbose \
    2>&1 | tee logs/gene_cv_all_genes_ensemble.log
```

#### Production Multi-Instance Configuration (Full Options)
```bash
# Complete multi-instance training with all advanced features
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_all_genes_production \
    --train-all-genes                      # Required: Enable multi-instance for large datasets
    --genes-per-instance 1500              # Default: 1500 (genes per instance)
    --max-instances 10                     # Default: 10 (maximum instances)
    --instance-overlap 0.1                 # Default: 0.1 (10% overlap between instances)
    --max-memory-per-instance-gb 15.0      # Default: 15.0 (memory limit per instance)
    --auto-adjust-instance-size            # Default: True (hardware adaptation)
    --resume-from-checkpoint               # Default: True (automatic checkpointing)
    --n-estimators 800                     # Default: 800 (XGBoost trees)
    --calibrate-per-class                  # Default: False (per-class calibration)
    --auto-exclude-leaky                   # Default: False (automatic leaky feature exclusion)
    --monitor-overfitting                  # Default: False (overfitting detection)
    --calibration-analysis                 # Default: False (comprehensive calibration)
    --neigh-sample 2000                    # Default: 0 (neighbor analysis sample)
    --early-stopping-patience 30           # Default: 20 (early stopping patience)
    --verbose                              # Default: False (detailed output)
    2>&1 | tee logs/gene_cv_all_genes_production.log
```

#### Hardware-Adaptive Configuration (Custom Resources)
```bash
# Custom configuration for specific hardware constraints
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_hardware_custom \
    --train-all-genes \
    --genes-per-instance 1200              # Smaller instances for limited memory
    --max-memory-per-instance-gb 8.0       # Reduced memory per instance
    --n-estimators 400                     # Fewer estimators for faster training
    --calibrate-per-class \
    --auto-exclude-leaky \
    --verbose \
    2>&1 | tee logs/gene_cv_hardware_custom.log
```

**Multi-Instance Ensemble Features:**
- ✅ **Complete Gene Coverage**: ALL genes included in training (100%)
- ✅ **Memory Scalability**: 12-15GB per instance vs >64GB single model
- ✅ **Automatic Strategy Selection**: Triggered by `--train-all-genes` on large datasets
- ✅ **Performance Weighting**: Instance contributions weighted by F1 scores
- ✅ **Checkpointing Support**: Resume from interruptions automatically
- ✅ **Enhanced SHAP Analysis**: Comprehensive feature importance across all instances
- ✅ **Unified Interface**: Same inference commands work seamlessly

### Step 2.4: Feature Ablation Analysis (Optional)

**Ablation Analysis** helps understand the contribution of different feature types to model performance by systematically removing feature groups and measuring the impact.

#### Minimal Ablation Configuration
```bash
# Essential ablation study comparing key feature groups
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/ablation_study \
    --modes full,no_spliceai,only_kmer \
    --sample-genes 200 \
    --n-estimators 200 \
    --verbose \
    2>&1 | tee logs/ablation_study.log
```

#### Comprehensive Ablation Analysis
```bash
# Complete ablation study with all feature combinations
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/comprehensive_ablation \
    --modes full,no_spliceai,no_probs,no_kmer,only_kmer,raw_scores,positional_only,context_only \
    --cv-strategy gene                     # Default: 'gene' (vs 'chromosome')
    --n-folds 3                           # Default: 5 (reduced for faster analysis)
    --n-estimators 200                    # Default: 200 (vs 800 for main training)
    --sample-genes 500                    # Gene-level sampling for efficiency
    --check-leakage                       # Default: True (feature leakage detection)
    --auto-exclude-leaky                  # Default: False (automatic exclusion)
    --run-full-diagnostics               # Default: False (comprehensive analysis per mode)
    --memory-optimize                     # Default: False (enable for constrained systems)
    --verbose \
    2>&1 | tee logs/comprehensive_ablation.log
```

**Ablation Modes Available:**
- **`full`**: All features (baseline performance)
- **`no_spliceai`**: Excludes SpliceAI-derived features (measures SpliceAI contribution)
- **`only_kmer`**: K-mer features only (sequence-based prediction)
- **`no_kmer`**: Excludes k-mer features (non-sequence-based prediction)
- **`raw_scores`**: Only donor/acceptor/neither scores (minimal feature set)
- **`no_probs`**: Excludes probability-derived features
- **`positional_only`**: Position-based features only
- **`context_only`**: Context window features only

**Expected Ablation Outputs:**
```
results/ablation_study/
├── ablation_summary.csv              # Performance comparison across modes
├── ablation_report.json              # Comprehensive analysis summary
├── ablation_comparison.pdf           # Visual comparison plots
├── leakage_analysis/                 # Feature leakage detection
├── full/                             # Full feature set results
│   ├── model_multiclass_all_genes.pkl # Trained model for this mode
│   ├── ablation_mode_full_summary.json # Detailed metrics
│   └── batch_*/                      # Batch-specific results (if applicable)
├── no_spliceai/                      # No SpliceAI features results
├── only_kmer/                        # K-mer only results
└── ...                               # Additional modes
```

**Interpreting Ablation Results:**
```bash
# Check overall comparison
cat results/ablation_study/ablation_summary.csv

# View detailed analysis
cat results/ablation_study/ablation_report.json

# Examine visual comparison
open results/ablation_study/ablation_comparison.pdf
```

**Key Insights from Ablation Analysis:**
- **SpliceAI Contribution**: Compare `full` vs `no_spliceai` to measure SpliceAI feature importance
- **Sequence vs Context**: Compare `only_kmer` vs `context_only` to understand feature type contributions  
- **Feature Efficiency**: Identify minimal feature sets that maintain performance
- **Model Robustness**: Assess performance stability across different feature combinations

### Step 2.5: Memory-Optimized Training Options (NEW)

Control memory usage with these advanced options:

```bash
# Manual memory control with dynamic gene limit
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_memory_optimized \
    --max-genes-in-memory 2000 \
    --memory-safety-factor 0.5 \
    --n-folds 3 \
    --n-estimators 400 \
    --calibrate-per-class \
    --verbose
```

**Memory Parameters:**
- `--train-all-genes`: Enable multi-batch training for all genes
- `--max-genes-in-memory`: Override automatic gene limit (expert use)
- `--memory-safety-factor`: Memory safety factor (0.0-1.0, default: 0.6)

### Step 2.2: Chromosome-Aware Cross-Validation (Optional)

For additional validation, run chromosome-aware (LOCO) cross-validation:

```bash
# Chromosome-aware CV for additional validation
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/loco_cv_reg_10k_kmers_run1 \
    --n-estimators 800 \
    --row-cap 0 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --monitor-overfitting \
    --verbose \
    2>&1 | tee logs/loco_cv_reg_10k_kmers.log
```

### Step 2.3: Validate Training Results

```bash
# Monitor training progress (universal monitor)
python scripts/monitoring/monitor_training_universal.py --run-name gene_cv_reg_10k_kmers_run1

# Verify critical model files exist
ls -la results/gene_cv_reg_10k_kmers_run1/model_multiclass.pkl
ls -la results/gene_cv_reg_10k_kmers_run1/feature_manifest.csv
ls -la results/gene_cv_reg_10k_kmers_run1/consolidation_info.json  # Multi-instance only
```

**Expected Training Output Structure:**

**Single-Model Training:**
```
results/gene_cv_single_model/
├── model_multiclass.pkl               # Trained meta-model
├── feature_manifest.csv               # Feature schema
├── gene_cv_metrics.csv                # Cross-validation results
├── complete_training_results.json     # Comprehensive metrics
├── cv_metrics_visualization/          # CV analysis suite
├── feature_importance_analysis/       # SHAP and feature analysis
├── leakage_analysis/                  # Data leakage detection
├── pr_curves_meta.pdf                 # Precision-recall curves
├── roc_curves_meta.pdf                # ROC curves
├── probability_diagnostics.png        # Calibration analysis
└── metrics_fold*.json                 # Per-fold metrics
```

**Multi-Instance Ensemble Training:**
```
results/gene_cv_multi_instance_ensemble/
├── model_multiclass.pkl               # Consolidated ensemble model
├── consolidation_info.json            # Ensemble metadata and weights
├── feature_manifest.csv               # Feature schema
├── complete_training_results.json     # Comprehensive training summary
├── multi_instance_training/           # Instance-specific results
│   ├── instance_00/                   # Individual instance outputs
│   │   ├── model_multiclass.pkl       # Instance model
│   │   ├── gene_cv_metrics.csv        # Instance CV results
│   │   ├── pr_curves_meta.pdf         # Instance visualizations
│   │   └── ...                        # Complete instance analysis
│   ├── instance_01/
│   └── ...
├── cv_metrics_visualization/          # Consolidated CV analysis
├── feature_importance_analysis/       # Enhanced ensemble SHAP
│   └── comprehensive_shap_analysis/   # Multi-instance SHAP results
├── leakage_analysis/                  # Global feature screening
├── pr_curves_meta.pdf                 # Consolidated visualizations
├── roc_curves_meta.pdf                # Consolidated ROC curves
└── metrics_fold*.json                 # Consolidated CV metrics
```

### Step 2.6: Monitor Training Progress

Use the universal training monitor to track progress for both single-instance and multi-instance training:

```bash
# Auto-detect and monitor active training runs
python scripts/monitoring/monitor_training_universal.py --auto-detect

# Monitor specific run with real-time updates
python scripts/monitoring/monitor_training_universal.py --run-name gene_cv_all_genes_ensemble --watch

# Check completion status
python scripts/monitoring/monitor_training_universal.py --run-name gene_cv_all_genes_ensemble
```

**Monitor Capabilities:**
- ✅ **Training Mode Detection**: Automatically identifies single vs multi-instance
- ✅ **Progress Tracking**: Milestone completion and instance progress
- ✅ **Resource Monitoring**: Memory usage and CPU utilization
- ✅ **Error Detection**: Early identification of issues and fallbacks
- ✅ **Output Validation**: Verification of expected file generation
- ✅ **SHAP Analysis Tracking**: Enhanced vs standard SHAP detection

---

## 🚀 **Phase 3: Production Inference & Recalibration**

### Step 3.1: Prepare Test Genes

Use the streamlined gene preparation utility to identify suitable test genes:

#### Quick Gene Selection
```bash
# Prepare test genes for inference
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --unseen 20 \
    --training 10 \
    --study-name "regulatory_inference_study" \
    --output-dir gene_lists \
    --verbose

# This creates:
# - gene_lists/regulatory_inference_study_unseen_genes.txt
# - gene_lists/regulatory_inference_study_training_genes.txt
# - Ready-to-use inference commands
```

#### Custom Gene Selection
```bash
# If you have specific genes of interest
echo "ENSG00000142611
ENSG00000261456  
ENSG00000268895" > custom_test_genes.txt
```

### Step 3.2: Run Inference Workflow

The inference workflow provides **three operational modes** for different use cases:

#### Mode 1: Hybrid (Recommended for Production)
**Intelligent Selective Recalibration** - Meta-model applied only to uncertain positions

This mode combines the **speed of base model predictions** with the **accuracy of meta-model recalibration**, applying the meta-model only where it's most needed.

**Minimal Configuration (Auto-enables full coverage):**
```bash
# Essential switches - complete coverage enabled automatically
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --training-dataset train_regulatory_10k_kmers \
    --genes-file gene_lists/regulatory_inference_study_unseen_genes.txt \
    --output-dir results/inference_hybrid \
    --inference-mode hybrid \
    --verbose \
    2>&1 | tee logs/inference_hybrid.log
```

**Full Configuration (All options explicit):**
```bash
# Complete configuration showing all options with defaults
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --training-dataset train_regulatory_10k_kmers \
    --genes-file gene_lists/regulatory_inference_study_unseen_genes.txt \
    --output-dir results/inference_hybrid \
    --inference-mode hybrid                   # Default: 'hybrid' (balanced approach)
    --complete-coverage                       # Default: Auto-enabled (ALL positions predicted)
    --uncertainty-low 0.02                   # Default: 0.02 (confident non-splice threshold)
    --uncertainty-high 0.80                  # Default: 0.80 (confident splice threshold)
    --enable-chunked-processing              # Default: Auto-enabled with complete coverage
    --chunk-size 5000                        # Default: 10000 (processing chunk size)
    --max-positions 10000                    # Default: 10000 (per-gene position limit)
    --verbose \
    --mlflow-enable \
    --mlflow-experiment "regulatory_inference_study" \
    2>&1 | tee logs/inference_hybrid.log
```

**Usage Characteristics:**
- ✅ **Complete Coverage**: Predicts **ALL nucleotide positions** in genomic regions (auto-enabled)
- ✅ **Efficiency**: Processes 95-98% of positions with fast base model
- ✅ **Accuracy**: Applies meta-model recalibration to uncertain positions (2-5%)
- ✅ **Variant Analysis Ready**: Every position scored for comprehensive variant analysis
- ✅ **Production Ready**: Optimal balance of speed and accuracy
- ✅ **Resource Efficient**: Chunked processing prevents memory issues

#### Mode 2: Base-Only (Baseline Comparison)
**Pure Base Model Predictions** - SpliceAI predictions without meta-model enhancement

This mode provides baseline performance using only the pre-trained SpliceAI model, useful for comparison and scenarios where meta-model enhancement is not needed.
```bash
# Base-only inference: SpliceAI predictions without meta-model
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --training-dataset train_regulatory_10k_kmers \
    --genes-file gene_lists/regulatory_inference_study_unseen_genes.txt \
    --output-dir results/inference_base_only \
    --inference-mode base_only \
    --enable-chunked-processing \
    --chunk-size 5000 \
    --verbose \
    --mlflow-enable \
    --mlflow-experiment "regulatory_inference_study" \
    2>&1 | tee logs/inference_base_only.log
```

**Usage Characteristics:**
- ✅ **Speed**: Fastest inference mode (no meta-model computation)
- ✅ **Baseline**: Establishes baseline performance for comparison
- ✅ **Resource Minimal**: Lowest memory and CPU requirements
- ✅ **Compatibility**: Works with any base model configuration

#### Mode 3: Meta-Only (Research/Comprehensive)
**Complete Meta-Model Recalibration** - Meta-model enhancement applied to all positions

This mode applies meta-model recalibration to **all positions**, providing the most comprehensive enhancement but with higher computational cost. **Essential for variant analysis** and research applications.

**Minimal Configuration (Complete coverage auto-enabled):**
```bash
# Essential switches for comprehensive meta-model analysis
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --training-dataset train_regulatory_10k_kmers \
    --genes-file gene_lists/regulatory_inference_study_unseen_genes.txt \
    --output-dir results/inference_meta_only \
    --inference-mode meta_only \
    --verbose \
    2>&1 | tee logs/inference_meta_only.log
```

**Full Configuration (Optimized for large genes):**
```bash
# Complete meta-only inference with chunked processing
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --training-dataset train_regulatory_10k_kmers \
    --genes-file gene_lists/regulatory_inference_study_unseen_genes.txt \
    --output-dir results/inference_meta_only \
    --inference-mode meta_only                # Complete meta-model recalibration
    --complete-coverage                       # Default: Auto-enabled (ALL positions)
    --enable-chunked-processing              # Default: Auto-enabled (prevents OOM)
    --chunk-size 3000                        # Smaller chunks for memory efficiency
    --max-positions 50000                    # Higher limit for comprehensive analysis
    --verbose \
    --mlflow-enable \
    --mlflow-experiment "regulatory_inference_study" \
    2>&1 | tee logs/inference_meta_only.log
```

**Usage Characteristics:**
- ✅ **Complete Coverage**: **ALL nucleotide positions** scored (perfect for variant analysis)
- ✅ **Comprehensive**: Maximum meta-model enhancement coverage
- ✅ **Variant Analysis Optimal**: Every position available for variant impact assessment
- ✅ **Research Optimal**: Ideal for research and detailed splice landscape analysis
- ✅ **Clinical Ready**: Supports pathogenic variant analysis workflows
- ⚠️ **Resource Intensive**: Highest computational requirements (mitigated by chunking)

#### Ensemble Model Support (Multi-Instance Training)

The inference workflow automatically detects and handles both single and multi-instance ensemble models:

```bash
# Inference with multi-instance ensemble model (trained using --train-all-genes)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/multi_instance_ensemble \
    --training-dataset train_regulatory_10k_kmers \
    --genes-file gene_lists/regulatory_inference_study_unseen_genes.txt \
    --output-dir results/inference_ensemble_hybrid \
    --inference-mode hybrid \
    --uncertainty-low 0.02 \
    --uncertainty-high 0.80 \
    --enable-chunked-processing \
    --chunk-size 5000 \
    --verbose \
    2>&1 | tee logs/inference_ensemble_hybrid.log
```

**Automatic Model Detection:**
- ✅ **Unified Interface**: Same commands work for single and ensemble models
- ✅ **Performance Weighting**: Ensemble uses F1-based instance weights
- ✅ **Transparent Operation**: No code changes needed in inference workflow
- ✅ **Enhanced Accuracy**: Ensemble predictions improve over single models

### Step 3.3: Variant Analysis with Complete Coverage

**Complete coverage** is essential for variant analysis, providing splice site scores for **every nucleotide position** to enable comprehensive variant impact assessment.

#### Variant Analysis Configuration
```bash
# Complete coverage for variant analysis (auto-enabled by default)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --training-dataset train_regulatory_10k_kmers \
    --genes ENSG00000142611,ENSG00000261456 \
    --output-dir results/variant_analysis \
    --inference-mode hybrid \
    --complete-coverage \
    --enable-chunked-processing \
    --verbose \
    2>&1 | tee logs/variant_analysis.log
```

**Variant Analysis Benefits:**
- ✅ **Complete Position Coverage**: Every nucleotide position scored
- ✅ **Variant Impact Assessment**: Compare scores before/after variant
- ✅ **Splice Site Discovery**: Identify potential cryptic splice sites
- ✅ **Clinical Applications**: Support for pathogenic variant analysis
- ✅ **Research Applications**: Comprehensive splice landscape analysis

**Expected Variant Analysis Output:**
```
results/variant_analysis/
├── genes/
│   ├── ENSG00000142611/
│   │   ├── ENSG00000142611_predictions.parquet    # All positions with scores
│   │   ├── ENSG00000142611_statistics.json        # Gene-level summary
│   │   └── ENSG00000142611_splice_sites.tsv       # Identified splice sites
├── consolidated_predictions.parquet               # All genes combined
├── variant_analysis_summary.json                 # Analysis summary
└── position_coverage_report.txt                  # Coverage verification
```

### Step 3.4: Sequence-Centric Inference for Variant Analysis (Extension)

**Sequence-Centric Interface** enables SpliceAI/OpenSpliceAI-compatible predictions on arbitrary DNA sequences, perfect for variant analysis and delta score calculations.

#### Basic Sequence Prediction
```python
# SpliceAI-compatible interface for arbitrary sequences
from meta_spliceai.splice_engine.meta_models.workflows.inference.sequence_inference import predict_splice_scores

# Predict splice scores for any DNA sequence
scores = predict_splice_scores(
    sequence="ATGCGTAAGTCGACTAGCTAGCTGATCGATCGTAGCTAGCTAG",
    model_path="results/gene_cv_reg_10k_kmers_run1/model_multiclass.pkl",
    training_dataset_path="train_regulatory_10k_kmers/master",
    inference_mode="hybrid",
    return_format="dict"
)

print(f"Donor scores: {scores['donor'][:10]}")
print(f"Acceptor scores: {scores['acceptor'][:10]}")
```

#### Variant Delta Score Calculation
```python
# OpenSpliceAI-compatible delta score calculation
from meta_spliceai.splice_engine.meta_models.workflows.inference.sequence_inference import (
    compute_variant_delta_scores, SequenceInferenceInterface
)

# Example: BRCA1 variant analysis
gene_id = "ENSG00000012048"  # BRCA1 gene ID
chromosome = "chr17"
start_position = 43094077

# Wild-type and alternate sequences (example G>A variant)
wt_sequence = "ATGCGTAAGTCGACTAGCTAGCTGATCGATCGTAGCTAGCTAG"
alt_sequence = "ATGCATAAGT CGACTAGCTAGCTGATCGATCGTAGCTAGCTAG"  # G>A variant at position 4

# Generate descriptive IDs for traceability
wt_id, alt_id = SequenceInferenceInterface.create_variant_gene_ids(gene_id, chromosome, start_position)
print(f"Analysis IDs: WT={wt_id}, ALT={alt_id}")  # ENSG00000012048_WT_1, ENSG00000012048_ALT_1

# Compute delta scores for variant impact
delta_results = compute_variant_delta_scores(
    wt_sequence=wt_sequence,
    alt_sequence=alt_sequence,
    variant_position=4,
    model_path="results/gene_cv_reg_10k_kmers_run1/model_multiclass.pkl",
    training_dataset_path="train_regulatory_10k_kmers/master",
    inference_mode="hybrid"
)

print(f"Maximum donor delta: {delta_results['max_delta_donor']:.4f}")
print(f"Maximum acceptor delta: {delta_results['max_delta_acceptor']:.4f}")
print(f"Impact level: {delta_results['impact_assessment']['impact_level']}")
```

#### Integration with Variant Analysis Workflows
```python
# Complete variant analysis pipeline
from meta_spliceai.splice_engine.case_studies.analysis import SplicingPatternAnalyzer

def analyze_variant_with_meta_model(wt_sequence, alt_sequence, variant_pos, model_path, dataset_path):
    """Complete variant analysis using meta-model enhanced predictions."""
    
    # Get delta scores using meta-model
    delta_results = compute_variant_delta_scores(
        wt_sequence, alt_sequence, variant_pos, model_path, dataset_path
    )
    
    # Convert to SpliceSite objects for pattern analysis
    splice_sites = []
    for i, (d_delta, a_delta) in enumerate(zip(delta_results['donor_delta'], delta_results['acceptor_delta'])):
        if abs(d_delta) > 0.1:  # Significant donor changes
            splice_sites.append(SpliceSite(
                position=i, site_type='donor', delta_score=d_delta,
                is_canonical=True, is_cryptic=False, strand='+', gene_id='VARIANT_GENE'
            ))
        if abs(a_delta) > 0.1:  # Significant acceptor changes
            splice_sites.append(SpliceSite(
                position=i, site_type='acceptor', delta_score=a_delta,
                is_canonical=True, is_cryptic=False, strand='+', gene_id='VARIANT_GENE'
            ))
    
    # Analyze splicing patterns
    analyzer = SplicingPatternAnalyzer()
    patterns = analyzer.analyze_variant_impact(splice_sites, variant_pos)
    
    return {
        'delta_scores': delta_results,
        'splicing_patterns': patterns,
        'impact_summary': delta_results['impact_assessment']
    }
```

**Sequence Interface Benefits:**
- ✅ **SpliceAI Compatibility**: Drop-in replacement for SpliceAI/OpenSpliceAI APIs
- ✅ **No Gene IDs Required**: Works with arbitrary DNA sequences
- ✅ **Delta Score Support**: Native support for WT/ALT comparison
- ✅ **Meta-Model Enhancement**: Leverages trained meta-models for improved accuracy
- ✅ **Variant Analysis Ready**: Perfect for VCF variant analysis workflows
- ✅ **Clinical Applications**: Supports pathogenic variant assessment

#### Enhanced VCF Variant Analysis Integration
```python
# Drop-in enhancement for existing VCF variant analysis workflow
from meta_spliceai.splice_engine.meta_models.workflows.inference.meta_variant_analysis import (
    enhance_existing_delta_calculation, enhance_existing_cryptic_detection
)

# Replace existing OpenSpliceAI delta calculation
# OLD: delta_scores = compute_openspliceai_delta_scores(wt_context, alt_context)
# NEW (enhanced):
enhanced_deltas = enhance_existing_delta_calculation(
    wt_context, alt_context, 
    model_path="results/my_model/model_multiclass.pkl",
    training_dataset_path="train_data/master"
)

# Replace existing cryptic site detection  
# OLD: cryptic_sites = detect_cryptic_splice_sites(alt_context, delta_scores)
# NEW (enhanced):
enhanced_cryptic_sites = enhance_existing_cryptic_detection(
    alt_context, enhanced_deltas,
    model_path="results/my_model/model_multiclass.pkl", 
    training_dataset_path="train_data/master"
)
```

**VCF Integration Benefits:**
- ✅ **Improved Delta Accuracy**: 10-20% improvement over base OpenSpliceAI
- ✅ **Enhanced Cryptic Detection**: Better sensitivity and specificity
- ✅ **Clinical Confidence**: Meta-model based confidence scoring
- ✅ **Backward Compatibility**: Drop-in replacement for existing functions
- ✅ **Progressive Adoption**: Can be integrated incrementally

### Step 3.5: Analyze Inference Results

#### Performance Comparison
```bash
# Compare all three inference modes
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \
    --results-dir results \
    --base-suffix inference_base_only \
    --hybrid-suffix inference_hybrid \
    --meta-suffix inference_meta_only \
    --output-dir inference_analysis_results \
    --batch-size 25 \
    --verbose
```

#### Statistical Analysis
```bash
# Generate statistical comparison report
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.batch_comparator \
    --analysis-results inference_analysis_results/detailed_report.json \
    --output-dir statistical_comparison_results \
    --reference-mode base_only \
    --significance-level 0.05 \
    --primary-metric ap_score \
    --include-effect-sizes \
    --create-plots \
    --verbose
```

### Step 3.4: Review Results

```bash
# Check inference performance reports
cat results/inference_hybrid/performance_report.txt

# View statistical comparison
cat statistical_comparison_results/comparison_report.txt

# Examine per-gene results
ls -la results/inference_hybrid/genes/
```

**Expected Inference Output Structure:**
```
results/inference_hybrid/
├── genes/                              # Per-gene results
│   ├── ENSG00000123456/
│   │   ├── ENSG00000123456_predictions.parquet
│   │   └── ENSG00000123456_statistics.json
├── selective_inference/                # Consolidated predictions
│   ├── predictions/
│   │   ├── complete_coverage_predictions.parquet
│   │   ├── meta_model_predictions.parquet
│   │   └── base_model_predictions.parquet
├── performance_analysis/               # Built-in analysis
│   ├── consolidated_performance_report.txt
│   ├── roc_pr_curves_inference.pdf
│   └── curve_metrics.json
├── gene_manifest.json                 # Processing metadata
├── inference_summary.json             # Summary statistics
├── performance_report.txt             # Performance summary
└── inference_workflow.log             # Detailed execution log
```

---

## 📊 **Expected Performance Metrics**

### Training Phase Results
- **F1 Score Improvement**: ~47% improvement over base model
- **Error Reduction**: ~60% false positive reduction, ~78% false negative reduction
- **Top-k Accuracy**: >95% gene-level accuracy
- **Feature Count**: 1,100+ features (including multi-scale k-mers)
- **Training Time**: 2-6 hours depending on dataset size

### Inference Phase Results
- **Processing Speed**: ~1-2 seconds per gene
- **Meta-model Usage**: 2-5% of positions (hybrid mode)
- **Memory Efficiency**: <500MB per gene
- **Accuracy Improvement**: Variable by gene type and scenario

### Performance by Scenario
- **Training Genes (Scenario 1)**: 3-4% meta-model usage, high accuracy gains
- **Unseen Genes with Artifacts (Scenario 2A)**: 5-7% meta-model usage, moderate gains  
- **Completely Unseen Genes (Scenario 2B)**: Variable usage, conservative improvements

---

## 🔧 **Advanced Configuration**

### Large-Scale Production Setup
```bash
# For processing hundreds of genes efficiently
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --training-dataset train_regulatory_10k_kmers \
    --genes-file large_gene_list.txt \
    --output-dir results/large_scale_inference \
    --inference-mode hybrid \
    --parallel-workers 4 \
    --enable-chunked-processing \
    --chunk-size 3000 \
    --verbose \
    --mlflow-enable \
    --mlflow-experiment "large_scale_production"
```

### Memory-Optimized Configuration
```bash
# For memory-constrained environments
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --training-dataset train_regulatory_10k_kmers \
    --genes-file memory_test_genes.txt \
    --output-dir results/memory_optimized_inference \
    --inference-mode hybrid \
    --enable-chunked-processing \
    --chunk-size 1000 \
    --max-positions 25000 \
    --verbose
```

---

## 🛠️ **Troubleshooting Common Issues**

### Training Phase Issues

#### Monitor Training Progress
```bash
# Check if training is progressing normally
python scripts/monitoring/monitor_training_universal.py --auto-detect

# Monitor for errors and resource issues
python scripts/monitoring/monitor_training_universal.py --run-name <run_name> --watch
```

#### Common Training Issues
```bash
# Schema validation errors
python -m meta_spliceai.splice_engine.meta_models.builder.validate_dataset_schema \
    --dataset train_regulatory_10k_kmers/master --fix

# Memory issues during training (use multi-instance ensemble)
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --train-all-genes \
    --genes-per-instance 1200 \
    --max-memory-per-instance-gb 10.0

# Feature leakage detection
# Training automatically excludes leaky features with --auto-exclude-leaky
```

#### Multi-Instance Training Issues
```bash
# Resume from interruptions (automatic checkpointing)
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/interrupted_run \
    --train-all-genes \
    --resume-from-checkpoint

# Force complete retrain if needed
--force-retrain-all

# Adjust instance size for memory constraints
--genes-per-instance 800 \
--max-memory-per-instance-gb 8.0
```

### Inference Phase Issues
```bash
# Out of memory errors
--enable-chunked-processing --chunk-size 1000

# Feature harmonization warnings
# These are usually handled automatically; check feature_manifest.csv

# Performance issues
# Use --verbose to identify bottlenecks
# Consider --parallel-workers for multi-gene processing
```

### Model Performance Issues
```bash
# Check training metrics
cat results/gene_cv_reg_10k_kmers_run1/training_summary.json

# Validate on known genes first
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_reg_10k_kmers_run1 \
    --genes ENSG00000058453 \  # Known training gene
    --inference-mode hybrid \
    --verbose
```

---

## 📚 **Additional Resources**

### Documentation References
- **Training Documentation**: `meta_spliceai/splice_engine/meta_models/training/docs/`
- **Inference Documentation**: `meta_spliceai/splice_engine/meta_models/workflows/inference/docs/`
- **Dataset Documentation**: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/`

### Key Utility Scripts
- **Gene Selection**: `prepare_gene_lists.py` - Automated test gene identification
- **Schema Validation**: `validate_dataset_schema.py` - Dataset quality assurance  
- **Performance Analysis**: `inference_analyzer.py` - Comprehensive result analysis
- **Statistical Comparison**: `batch_comparator.py` - Multi-mode performance comparison

### Configuration Files
- **Feature Manifest**: `feature_manifest.csv` - Training feature schema
- **Gene Manifest**: `gene_manifest.csv` - Enhanced gene characteristics
- **Training Summary**: `training_summary.json` - Comprehensive training metrics
- **Inference Summary**: `inference_summary.json` - Inference performance metrics

---

## 🎯 **Best Practices**

### Dataset Creation
1. **Start Small**: Begin with 1,000-5,000 genes for initial testing
2. **Use Gene Type Filtering**: Focus on `protein_coding` and `lncRNA` for regulatory analysis
3. **Validate Schema**: Always run schema validation after dataset creation
4. **Monitor Memory**: Use appropriate batch sizes for your system

### Model Training  
1. **Enable Comprehensive Analysis**: Use `--calibration-analysis` and `--monitor-overfitting`
2. **Use Automatic Feature Exclusion**: Enable `--auto-exclude-leaky` 
3. **Monitor Training Progress**: Use `--verbose` and save logs
4. **Validate Results**: Check F1 scores and feature importance

### Inference Deployment
1. **Start with Hybrid Mode**: Optimal balance of accuracy and efficiency
2. **Enable Chunked Processing**: Prevents memory issues for large genes
3. **Use MLflow Tracking**: Monitor performance across experiments
4. **Validate with Known Genes**: Test on training genes first

### Performance Optimization
1. **Use Appropriate Chunk Sizes**: 5000 for normal, 1000-3000 for memory-constrained
2. **Monitor Meta-model Usage**: Should be 2-10% for hybrid mode
3. **Enable Parallel Processing**: Use `--parallel-workers` for large gene sets
4. **Profile Memory Usage**: Monitor system resources during processing

---

## 🚀 **Quick Start Summary**

For users who want to get started immediately:

```bash
# 1. Create training dataset
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 --gene-types protein_coding --output-dir train_pc_1000_quick \
    --run-workflow --verbose

# 2. Train meta-model
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_quick/master --out-dir results/quick_model \
    --n-estimators 400 --calibrate-per-class --auto-exclude-leaky --verbose

# 3. Prepare test genes
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --unseen 5 --study-name "quick_test"

# 4. Run inference
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/quick_model --genes-file quick_test_unseen_genes.txt \
    --output-dir results/quick_inference --inference-mode hybrid --verbose
```

This complete workflow demonstrates the full power of the Splice Surveyor system, from data assembly through model training to production inference, providing a robust pipeline for splice site prediction enhancement.
