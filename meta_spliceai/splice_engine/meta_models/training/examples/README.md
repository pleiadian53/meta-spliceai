# Meta-Model Training Examples

This directory contains example scripts demonstrating how to use MetaSpliceAI's meta-model training system with different approaches and algorithms.

## 📋 Available Examples

### **Deep Learning CV Examples**
- **[deep_learning_cv_examples.py](deep_learning_cv_examples.py)** - Complete examples for deep learning-based gene-aware CV
  - TabNet multi-class classification
  - TensorFlow MLP multi-class networks
  - Multi-modal transformer models
  - Custom parameter configurations

### **CV Evaluation Integration**
- **[cv_evaluation_integration.py](cv_evaluation_integration.py)** - Examples for integrating CV evaluation workflows

## 🚀 Quick Start

### **Deep Learning CV Examples**

```bash
# Run all deep learning examples
python meta_spliceai/splice_engine/meta_models/training/examples/deep_learning_cv_examples.py

# Or run individual examples by modifying the script
```

### **Prerequisites**

```bash
# Install deep learning dependencies (optional)
pip install pytorch-tabnet tensorflow

# Activate environment
mamba activate surveyor
```

## 📚 **Example Categories**

### **1. Deep Learning Models**
- **TabNet**: Attention-based feature selection for k-mer features
- **TensorFlow MLP**: Multi-class neural networks
- **Multi-Modal Transformers**: Sequence + tabular feature integration

### **2. Traditional ML Models**
- **XGBoost**: Gradient boosting (see main training guide)
- **CatBoost**: Categorical feature optimization
- **LightGBM**: Memory-efficient gradient boosting
- **Random Forest**: Interpretable baseline models

### **3. Evaluation Approaches**
- **Gene-Aware CV**: Gene boundary preservation
- **Chromosome-Aware CV**: Chromosome-level evaluation (deferred)
- **LOCO-CV**: Leave-One-Chromosome-Out evaluation

## 🔧 **Customization**

### **Modifying Examples**

Each example script can be customized by:

1. **Dataset Path**: Update the `dataset` variable
2. **Output Directory**: Modify the `base_output` path
3. **Model Parameters**: Adjust `algorithm-params` JSON strings
4. **Sample Size**: Change `max-variants` for different dataset sizes

### **Adding New Examples**

To add new examples:

1. Create a new Python script in this directory
2. Follow the existing pattern with clear documentation
3. Include error handling and progress reporting
4. Update this README with the new example

## 📊 **Expected Outputs**

### **Deep Learning CV Examples**
Each example generates:
- **Model files**: `trained_model.pkl`
- **Metrics**: `fold_metrics.csv`, `aggregate_metrics.json`
- **Visualizations**: ROC/PR curves, confusion matrices
- **Reports**: `cv_report.txt` with comprehensive analysis

### **Directory Structure**
```
results/deep_learning_cv/
├── tabnet/
│   ├── trained_model.pkl
│   ├── fold_metrics.csv
│   ├── aggregate_metrics.json
│   ├── cv_report.txt
│   └── *.pdf (visualizations)
├── tf_mlp_multiclass/
│   └── ...
└── multimodal_transformer/
    └── ...
```

## 🎯 **Use Cases**

### **Research & Development**
- **Model Comparison**: Compare different deep learning architectures
- **Parameter Tuning**: Test various hyperparameter configurations
- **Feature Analysis**: Understand which features are most important
- **Performance Evaluation**: Assess model generalization across genes

### **Production Training**
- **Custom Models**: Train models with specific requirements
- **Large-Scale Training**: Handle datasets with thousands of genes
- **Multi-Modal Data**: Integrate sequence and tabular features
- **Interpretable Models**: Use attention-based models for feature insights

## 🔗 **Related Documentation**

### **Main Training Guides**
- **[COMPREHENSIVE_TRAINING_GUIDE.md](../docs/COMPREHENSIVE_TRAINING_GUIDE.md)** - Complete training reference
- **[gene_aware_cv/](../docs/gene_aware_cv/)** - Gene-aware CV documentation
- **[chrom_aware_cv/](../docs/chrom_aware_cv/)** - Chromosome-aware CV (deferred)

### **Deep Learning Specific**
- **[DEEP_LEARNING_CV_README.md](../docs/gene_aware_cv/DEEP_LEARNING_CV_README.md)** - Deep learning CV guide
- **[DEEP_LEARNING_CV_SOLUTION_SUMMARY.md](../docs/gene_aware_cv/DEEP_LEARNING_CV_SOLUTION_SUMMARY.md)** - Technical details

### **Algorithm Support**
- **[models.py](../models.py)** - Available model implementations
- **[run_gene_cv_deep_learning.py](../run_gene_cv_deep_learning.py)** - Deep learning CV module
- **[run_gene_cv_sigmoid.py](../run_gene_cv_sigmoid.py)** - Traditional ML CV module

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install pytorch-tabnet tensorflow
   ```

2. **Memory Issues**
   ```bash
   # Reduce sample size in examples
   --max-variants 5000  # Instead of 10000
   ```

3. **Dataset Path Issues**
   ```bash
   # Update dataset path in example scripts
   dataset = "path/to/your/dataset"
   ```

4. **GPU Issues**
   ```bash
   # Use CPU-only mode
   --algorithm-params '{"device": "cpu"}'
   ```

### **Getting Help**

- Check the main training documentation
- Review error messages in the example output
- Verify dataset format and availability
- Ensure all dependencies are installed

---

*Last updated: January 2025*
