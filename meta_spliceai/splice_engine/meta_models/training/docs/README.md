# Meta-Model Training Documentation Index

**Last Updated:** September 2025  
**Status:** ✅ **CONSOLIDATED & CURRENT**

---

## 📚 Current Documentation (Use These)

### 🎯 Primary Guides

| Document | Purpose | Status | Use For |
|----------|---------|---------|---------|
| **[Comprehensive Training Guide](COMPREHENSIVE_TRAINING_GUIDE.md)** | Complete training workflow | ✅ Current | All training scenarios |
| **[Multi-Instance Ensemble Training](MULTI_INSTANCE_ENSEMBLE_TRAINING.md)** | Large-scale training architecture | ✅ Current | >2K gene datasets |
| **[Memory Scalability Lessons](MEMORY_SCALABILITY_LESSONS.md)** | Memory optimization complete guide | ✅ Current | Memory troubleshooting |

### 🔧 Supporting Documentation

| Document | Purpose | Status | Use For |
|----------|---------|---------|---------|
| **[Utility Scripts Reference](UTILITY_SCRIPTS_REFERENCE.md)** | Supporting tools documentation | ✅ Current | Analysis utilities |
| **[Utility Scripts Quick Reference](UTILITY_SCRIPTS_QUICK_REFERENCE.md)** | Essential commands | ✅ Current | Quick lookup |
| **[Unified Training System](UNIFIED_TRAINING_SYSTEM.md)** | Architecture overview | ✅ Current | System understanding |

### 🧠 Base Model Integration

| Document | Purpose | Status | Use For |
|----------|---------|---------|---------|
| **[Base Models Overview](base_models/README.md)** | Base model system overview | ✅ Current | Base model understanding |
| **[SpliceAI Integration](base_models/SPLICEAI_INTEGRATION.md)** | SpliceAI model loading and usage | ✅ Current | SpliceAI integration |
| **[Model Loading Architecture](base_models/MODEL_LOADING_ARCHITECTURE.md)** | Technical architecture details | ✅ Current | System architecture |
| **[Base Model Resource Management](base_models/BASE_MODEL_RESOURCE_MANAGEMENT.md)** | Resource management analysis | ✅ Current | Resource management |

---

## 🗂️ Legacy Documentation (Superseded)

### ❌ Outdated/Fragmented (Do Not Use)

| Document | Status | Replaced By | Reason |
|----------|---------|-------------|--------|
| **gene_aware_cv/gene_cv_sigmoid.md** | ❌ Outdated | Comprehensive Training Guide | Missing recent fixes |
| **gene_aware_cv/gene_aware_evaluation.md** | ❌ Outdated | Comprehensive Training Guide | Incomplete coverage |
| **large_scale_meta_model_training.md** | ❌ Outdated | Multi-Instance Ensemble Training | Pre-breakthrough |
| **BATCH_ENSEMBLE_TRAINING.md** | ❌ Superseded | Multi-Instance Ensemble Training | Replaced architecture |
| **COMPLETE_META_MODEL_WORKFLOW.md** | ❌ Fragmented | Comprehensive Training Guide | Outdated workflow |
| **Meta_Model_Architecture_Overview.md** | ❌ Outdated | Comprehensive Training Guide | Missing recent changes |
| ~~**oom_issues.md**~~ | ✅ **MERGED** | Memory Scalability Lessons | Content consolidated |

### 🔄 Specialized Documentation (Still Valid)

| Document | Purpose | Status | Use For |
|----------|---------|---------|---------|
| **chrom_aware_cv/chromosome_aware_evaluation.md** | LOCO-CV methodology | ✅ Valid | Chromosome-aware CV |
| **loco_cv_multiclass_scalable.md** | LOCO-CV implementation | ✅ Valid | Out-of-domain evaluation |
| **calibrated_sigmoid_workflow.md** | Calibration details | ✅ Valid | Calibration understanding |
| **shap_analysis_troubleshooting.md** | SHAP debugging | ✅ Valid | SHAP issues |

---

## 🚀 Quick Navigation

### For New Users
1. **Start Here:** [Comprehensive Training Guide](COMPREHENSIVE_TRAINING_GUIDE.md)
2. **Large Datasets:** [Multi-Instance Ensemble Training](MULTI_INSTANCE_ENSEMBLE_TRAINING.md)
3. **Memory Issues:** [Memory Scalability Lessons](MEMORY_SCALABILITY_LESSONS.md)

### For Experienced Users
- **Quick Commands:** [Utility Scripts Quick Reference](UTILITY_SCRIPTS_QUICK_REFERENCE.md)
- **Architecture Details:** [Unified Training System](UNIFIED_TRAINING_SYSTEM.md)
- **Advanced Analysis:** [Utility Scripts Reference](UTILITY_SCRIPTS_REFERENCE.md)

### For Troubleshooting
- **Memory Problems:** [Memory Scalability Lessons](MEMORY_SCALABILITY_LESSONS.md)
- **SHAP Issues:** [shap_analysis_troubleshooting.md](shap_analysis_troubleshooting.md)
- **Schema Errors:** Use `validate_dataset_schema.py --fix`

---

## 🎯 Major Breakthroughs Documented

### ✅ Gene-Aware Sampling Memory Fix (January 2025)
**Problem:** Row-cap conflicts caused 21× memory amplification  
**Solution:** Systematic memory optimization across all pipeline phases  
**Impact:** True memory efficiency achieved, gene structure preserved  
**Documentation:** [Memory Scalability Lessons](MEMORY_SCALABILITY_LESSONS.md)

### ✅ Multi-Instance Ensemble Training (January 2025)
**Problem:** Cannot train on >2K genes due to memory constraints  
**Solution:** Multiple instances with intelligent gene distribution  
**Impact:** 100% gene coverage with memory efficiency  
**Documentation:** [Multi-Instance Ensemble Training](MULTI_INSTANCE_ENSEMBLE_TRAINING.md)

### ✅ Unified Training Architecture (January 2025)
**Problem:** Fragmented training approaches with inconsistent outputs  
**Solution:** Strategy pattern with orchestrated execution  
**Impact:** Clean, modular, extensible training system  
**Documentation:** [Unified Training System](UNIFIED_TRAINING_SYSTEM.md)

---

## 📋 Documentation Maintenance

### Current Status
- ✅ **3 Primary Guides:** Up-to-date, comprehensive, tested
- ✅ **3 Supporting Docs:** Current and maintained  
- ❌ **7 Legacy Docs:** Outdated, should not be used
- ✅ **4 Specialized Docs:** Still valid for specific use cases

### Maintenance Policy
- **Primary Guides:** Updated with each major breakthrough
- **Supporting Docs:** Updated as features evolve
- **Legacy Docs:** Marked for eventual removal
- **Specialized Docs:** Maintained for specific use cases

---

**For all meta-model training needs, start with the [Comprehensive Training Guide](COMPREHENSIVE_TRAINING_GUIDE.md) - it contains everything you need to know!** 🚀