# 🧬 Error Model Installation Guide

**Complete setup guide for the MetaSpliceAI Error Model package**

The error model (`meta_spliceai.splice_engine.meta_models.error_model`) provides multi-modal transformer-based error analysis for splice site predictions, featuring DNABERT integration and comprehensive interpretability analysis.

## 📋 Prerequisites

- **Python**: 3.10+ 
- **System**: Linux/macOS (Windows via WSL)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **GPU**: Optional but recommended (NVIDIA with CUDA 12.2+)

## 🚀 Quick Start

### **1. Environment Setup**

```bash
# Install Mamba (fast conda alternative)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3
source $HOME/miniforge3/bin/activate

# Clone repository
git clone <repository-url>
cd meta-spliceai

# Create environment with error model dependencies
mamba env create -f environment.yml
mamba activate surveyor

# Install package in editable mode
pip install -e . --no-deps
```

### **2. Verify Installation**

```bash
# Test core error model imports
python -c "
from meta_spliceai.splice_engine.meta_models.error_model import ErrorModelConfig, TransformerTrainer
from transformers import AutoTokenizer
print('✅ Error model imports successful!')

# Test DNABERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M')
print('✅ DNABERT tokenizer loaded!')
"
```

### **3. Quick Test Run**

```bash
# Quick CPU test (requires meta-model data)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/error_model_test \
    --device cpu \
    --batch-size 4 \
    --num-epochs 1 \
    --max-ig-samples 20 \
    --error-type fp \
    --skip-ig
```

## 🔧 Key Dependencies

The error model requires these HuggingFace ecosystem packages (automatically included in `environment.yml`):

```yaml
# HuggingFace ecosystem for error model (DNABERT, transformers)
- transformers>=4.45.0,<4.56.0    # Latest stable, compatible with PyTorch 2.7.1
- tokenizers>=0.20.0,<0.21.0      # Fast tokenizers, compatible with transformers 4.45+
- datasets>=3.0.0,<4.0.0          # HuggingFace datasets (optional but useful)
- accelerate>=1.0.0,<2.0.0        # Hardware acceleration for transformers
```

## 🖥️ Hardware Configurations

### **CPU-Only Setup**
```bash
# Default installation works on CPU
mamba env create -f environment.yml
# Uses torch==2.7.1 (CPU version)
```

### **Single GPU Setup**
```bash
# Install GPU-enabled PyTorch
mamba activate surveyor
mamba install -c pytorch -c nvidia -c conda-forge pytorch=2.7 pytorch-cuda=12.6

# Verify GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### **Multi-GPU Setup**
```bash
# Same as single GPU - error model automatically detects multiple GPUs
# Uses DataParallel for multi-GPU training
```

## 📊 Error Model Architecture

The error model implements a **multi-modal transformer architecture**:

### **Components:**
- **Primary Input**: DNA sequences (contextual sequences around splice sites)
- **Secondary Input**: Numerical features (SpliceAI scores, context features, etc.)
- **Foundation Model**: DNABERT-2-117M (default) or other transformer models
- **Multi-Modal Fusion**: Concatenates sequence embeddings with feature embeddings
- **Output**: Binary classification (True Positive vs Error)

### **Supported Foundation Models:**
- ✅ **DNABERT/DNABERT-2** (default): `zhihan1996/DNABERT-2-117M`
- ✅ **HyenaDNA**: `LongSafari/hyenadna-medium-450k-seqlen`
- ✅ **Nucleotide Transformer**: `InstaDeepAI/nucleotide-transformer-500m-human-ref`
- ✅ **ESM models**: For protein sequences (if applicable)

## 🎯 Usage Examples

### **Basic Error Analysis**
```bash
# Analyze False Positives vs True Positives
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/fp_analysis \
    --error-type fp \
    --splice-type donor \
    --device auto \
    --batch-size 16 \
    --num-epochs 5
```

### **Multi-GPU Training**
```bash
# Automatically uses all available GPUs
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/multigpu_training \
    --error-type fn \
    --device auto \
    --batch-size 32 \
    --num-epochs 10 \
    --use-mixed-precision
```

### **Custom Foundation Model**
```bash
# Use HyenaDNA instead of DNABERT
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/hyenadna_analysis \
    --model-name "LongSafari/hyenadna-medium-450k-seqlen" \
    --error-type fp \
    --context-length 512 \
    --device auto
```

## 🔍 Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | Required | Meta-model artifacts directory |
| `--output-dir` | Required | Output directory for results |
| `--error-type` | `fp` | Error type: `fp`, `fn`, `fp_vs_tp`, `fn_vs_tp` |
| `--splice-type` | `any` | Splice type: `donor`, `acceptor`, `any` |
| `--model-name` | `zhihan1996/DNABERT-2-117M` | Foundation model |
| `--context-length` | `200` | Total sequence context (±100 nt) |
| `--batch-size` | `16` | Per-device batch size |
| `--num-epochs` | `10` | Training epochs |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, `cuda:0` |
| `--use-mixed-precision` | `True` | Enable FP16 for GPU acceleration |
| `--max-ig-samples` | `500` | Samples for Integrated Gradients analysis |
| `--skip-training` | `False` | Skip training, use existing model |
| `--skip-ig` | `False` | Skip interpretability analysis |

## 🧪 Data Requirements

The error model expects meta-model artifacts in this structure:

```
data/ensembl/spliceai_eval/meta_models/
├── analysis_sequences_simple.parquet      # Primary sequences and features
├── splice_positions_enhanced_simple.parquet  # Position metadata
└── [other facets...]
```

### **Required Columns:**
- `context_sequence`: DNA sequences around splice sites
- `pred_type`: Prediction labels (FP, FN, TP, TN)
- Additional numerical features (scores, context features, etc.)

## 🎯 Output Structure

```
results/error_model_analysis/
├── models/
│   ├── final_model/           # Trained transformer model
│   └── checkpoints/           # Training checkpoints
├── ig_analysis/
│   ├── attributions.parquet   # Token-level attributions
│   └── analysis_results.json  # Pattern analysis
├── visualizations/
│   ├── token_frequency_comparison.png
│   ├── attribution_distribution.png
│   └── positional_analysis.png
├── logs/
│   └── workflow.log          # Complete execution log
└── report.md                 # Summary report
```

## 🚨 Troubleshooting

### **Common Issues:**

#### **1. Keras 3.0 Compatibility Error**
```bash
# Error: "Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers"
# Solution: Install tf-keras compatibility package
mamba activate surveyor
pip install tf-keras
```

#### **2. Import Errors**
```bash
# If you see "No module named 'transformers'"
mamba activate surveyor
mamba install transformers tokenizers datasets accelerate -c conda-forge

# Alternative with pip
pip install transformers tokenizers datasets accelerate
```

#### **3. Missing Captum for Integrated Gradients**
```bash
# If you see "No module named 'captum'"
mamba activate surveyor
pip install captum
# Note: May downgrade numpy to 1.26.4 for compatibility
```

#### **4. CUDA Out of Memory**
```bash
# Reduce batch size and context length
--batch-size 8 --context-length 150
```

#### **5. Model Download Issues**
```bash
# Pre-download DNABERT model
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M')"
```

#### **6. Data Path Issues**
```bash
# Verify data structure
ls -la data/ensembl/spliceai_eval/meta_models/
```

## 🔗 Related Documentation

- **Main Installation**: `docs/installation/INSTALLATION.md`
- **Error Model README**: `meta_spliceai/splice_engine/meta_models/error_model/README.md`
- **Workflow Guide**: `meta_spliceai/splice_engine/meta_models/error_model/WORKFLOW_GUIDE.md`
- **Usage Examples**: `meta_spliceai/splice_engine/meta_models/error_model/USAGE_EXAMPLES.md`

## 🆘 Support

For error model specific issues:
1. Check the workflow logs in `output-dir/logs/workflow.log`
2. Verify hardware detection in the diagnostic output
3. Test with minimal parameters first (`--skip-ig --num-epochs 1`)
4. Ensure data artifacts are properly formatted

---

**🎯 Quick Reference**: The error model provides state-of-the-art interpretable analysis of splice site prediction errors using multi-modal transformer architectures with full CPU/GPU support.
