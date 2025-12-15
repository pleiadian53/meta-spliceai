# GPU Environment Setup Scripts

This directory contains comprehensive scripts and guides for setting up and testing GPU support for MetaSpliceAI.

## 📁 **Directory Structure**

```
scripts/gpu_env_setup/
├── README.md                    # This file
├── test_gpu_installation.sh     # Basic GPU installation verification
├── test_gpu_performance.py      # Comprehensive GPU performance testing
├── GPU_TESTING_GUIDE.md         # Complete testing guide and troubleshooting
├── install_gpu_environment.sh   # Automated GPU environment setup
└── verify_gpu_setup.py          # Detailed GPU setup verification
```

## 🚀 **Quick Start**

### **1. Basic GPU Installation Test**
```bash
# Test if GPU components are properly installed
./scripts/gpu_env_setup/test_gpu_installation.sh
```

### **2. Performance Benchmark**
```bash
# Run comprehensive GPU performance tests
python scripts/gpu_env_setup/test_gpu_performance.py
```

### **3. Automated Setup (Optional)**
```bash
# Automated GPU environment setup
./scripts/gpu_env_setup/install_gpu_environment.sh
```

## 📋 **Script Descriptions**

### **`test_gpu_installation.sh`**
**Purpose:** Quick verification of GPU installation
- ✅ NVIDIA driver check
- ✅ PyTorch CUDA runtime
- ✅ TensorFlow GPU support
- ✅ XGBoost GPU support
- ✅ CUDA toolkit verification

### **`test_gpu_performance.py`**
**Purpose:** Comprehensive performance testing
- 🚀 TensorFlow matrix operations
- 🚀 PyTorch matrix operations
- 🚀 XGBoost training benchmarks
- 📊 Performance timing and reporting

### **`GPU_TESTING_GUIDE.md`**
**Purpose:** Complete testing documentation
- 📖 Step-by-step testing procedures
- 🔧 Troubleshooting common issues
- 📊 Performance benchmarks
- 🎯 Integration with MetaSpliceAI

### **`install_gpu_environment.sh`**
**Purpose:** Automated environment setup
- 🏗️ Environment creation
- 📦 Package installation
- ✅ Verification testing
- 🔧 Error handling

### **`verify_gpu_setup.py`**
**Purpose:** Detailed setup verification
- 🔍 Deep system analysis
- 📊 Hardware compatibility
- ⚙️ Configuration validation
- 📝 Detailed reporting

## 🎯 **Use Cases**

### **For New GPU Setup:**
1. Run `install_gpu_environment.sh` for automated setup
2. Run `test_gpu_installation.sh` for basic verification
3. Run `test_gpu_performance.py` for performance testing

### **For Existing Setup Verification:**
1. Run `test_gpu_installation.sh` for quick check
2. Run `verify_gpu_setup.py` for detailed analysis
3. Run `test_gpu_performance.py` for performance validation

### **For Troubleshooting:**
1. Check `GPU_TESTING_GUIDE.md` for common issues
2. Run `verify_gpu_setup.py` for detailed diagnostics
3. Follow troubleshooting steps in the guide

## 🔧 **Integration with MetaSpliceAI**

### **Pre-Training Verification:**
```bash
# Before running MetaSpliceAI training
./scripts/gpu_env_setup/test_gpu_installation.sh
python scripts/gpu_env_setup/test_gpu_performance.py

# If all tests pass, proceed with training
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset your_dataset \
    --tree-method gpu_hist \
    --verbose
```

### **Performance Monitoring:**
```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Run performance tests after training
python scripts/gpu_env_setup/test_gpu_performance.py
```

## 📊 **Expected Results**

### **On GPU Machine:**
- ✅ All installation tests pass
- ✅ Performance tests complete successfully
- 🚀 Significant speedup compared to CPU
- 📊 GPU utilization > 50% during training

### **On Non-GPU Machine:**
- ℹ️ GPU tests show "not available"
- ✅ CPU fallback works correctly
- ⚠️ Performance tests may be slower
- 📝 Clear indication of GPU requirements

## 🎉 **Success Criteria**

Your GPU setup is ready when:
- ✅ All basic tests pass (`test_gpu_installation.sh`)
- ✅ Performance tests pass (`test_gpu_performance.py`)
- ✅ GPU utilization > 50% during training
- ✅ No memory errors during stress tests

## 📚 **Documentation**

- **Main GPU Setup Guide:** `docs/gpu_environment_setup.md`
- **Testing Guide:** `scripts/gpu_env_setup/GPU_TESTING_GUIDE.md`
- **Troubleshooting:** See troubleshooting section in testing guide

## 🤝 **Contributing**

To add new GPU testing scripts:
1. Place them in this directory
2. Update this README.md
3. Add appropriate documentation
4. Test on both GPU and non-GPU environments

---

**Note:** These scripts are designed to work on both GPU and non-GPU machines, providing appropriate feedback for each environment. 