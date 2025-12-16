# Post-Training Analysis Tools

**📁 Location**: `meta_spliceai/splice_engine/meta_models/analysis/post_training/`

This directory contains **follow-up analysis scripts** that can be run after completing CV training runs.

## 🎯 **Quick Access**

### **CV Run Comparison**
```bash
# Direct usage
python -m meta_spliceai.splice_engine.meta_models.analysis.post_training.compare_cv_runs \
    --run1 results/gene_cv_run_1 \
    --run2 results/gene_cv_run_2 \
    --output comparison_results

# Convenience script
./scripts/post_training_analysis.sh compare-cv results/gene_cv_run_1 results/gene_cv_run_2
```

## 📋 **Available Tools**

- **`compare_cv_runs.py`** - Compare CV runs for reproducibility assessment
- **`README.md`** - Comprehensive documentation and usage examples

## 🔗 **Documentation**

For detailed documentation, see: [`post_training/README.md`](post_training/README.md)

## 🚀 **Why This Organization?**

The analysis subpackage was getting large, so we organized post-training tools into a dedicated subdirectory for:

- ✅ **Better discoverability** - Easy to find post-training tools
- ✅ **Cleaner structure** - Separates post-training from other analysis
- ✅ **Future scalability** - Room to add more post-training tools
- ✅ **Clear purpose** - Obvious what these tools are for

---

**💡 Tip**: Use the convenience script `./scripts/post_training_analysis.sh` for easy access to these tools! 