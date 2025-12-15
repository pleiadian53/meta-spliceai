# GPU Testing Guide for MetaSpliceAI

This guide provides comprehensive testing procedures to verify your GPU installation for MetaSpliceAI.

## 🚀 **Quick Start Testing**

### **1. Basic Installation Test**
```bash
# Activate your environment
mamba activate surveyor

# Run basic GPU test
./test_gpu_installation.sh
```

### **2. Performance Test**
```bash
# Run comprehensive performance test
python test_gpu_performance.py
```

## 📋 **Test Scripts Overview**

### **`test_gpu_installation.sh` - Basic Installation Check**
**Purpose:** Quick verification that GPU components are properly installed

**Tests:**
- ✅ NVIDIA driver (`nvidia-smi`)
- ✅ PyTorch CUDA runtime
- ✅ TensorFlow GPU support
- ✅ XGBoost GPU support
- ✅ Full CUDA toolkit (optional)

**Expected Output:**
```bash
🔍 GPU Installation Test
========================
1. NVIDIA Driver Test:
   ✅ nvidia-smi found
   Tesla T4, 535.247.01, 15109
2. CUDA Runtime Test:
   ✅ PyTorch CUDA runtime available
3. TensorFlow GPU Test:
   ✅ TensorFlow GPU support available
4. XGBoost GPU Test:
   ✅ XGBoost GPU support available
5. Full CUDA Toolkit Test:
   ℹ️  Full CUDA toolkit not installed (runtime libraries only)
================================
✅ GPU installation test completed!
```

### **`test_gpu_performance.py` - Performance Benchmark**
**Purpose:** Comprehensive performance testing with actual computations

**Tests:**
- 🚀 TensorFlow matrix multiplication (10K x 10K)
- 🚀 PyTorch matrix multiplication (10K x 10K)
- 🚀 XGBoost training (10K samples, 100 features)
- 🔧 CUDA toolkit verification (if installed)

**Expected Output:**
```bash
🚀 MetaSpliceAI GPU Performance Test
==================================================
1. TensorFlow GPU Test:
   Available GPUs: 4
   Computation time: 0.15s
   Result: 100000000.00
2. PyTorch GPU Test:
   CUDA available: True
   GPU count: 4
   Computation time: 0.12s
   Result: 100000000.00
3. XGBoost GPU Test:
   Training time: 2.34s
   Model trained successfully
4. CUDA Toolkit Test:
   ℹ️  nvcc not found (runtime libraries only)

==================================================
📊 Test Results Summary:
==================================================
tensorflow      ✅ PASS
pytorch         ✅ PASS
xgboost         ✅ PASS
cuda_toolkit    ✅ PASS

🎉 All GPU tests passed! Your setup is ready for MetaSpliceAI.
```

## 🎯 **Testing Scenarios**

### **Scenario 1: Basic GPU Setup (Recommended)**
```bash
# Install with runtime libraries only
mamba env create -f environment.yml
mamba activate surveyor

# Test installation
./test_gpu_installation.sh
python test_gpu_performance.py
```

**Expected Results:**
- ✅ All basic tests pass
- ✅ Performance tests pass
- ℹ️ CUDA toolkit test shows "runtime libraries only"

### **Scenario 2: Full CUDA Toolkit Setup**
```bash
# Install with full CUDA toolkit
mamba create -n surveyor-gpu-complete python=3.10
mamba activate surveyor-gpu-complete
mamba install -c conda-forge cudatoolkit=12.1 cudnn=8.9
mamba install -c conda-forge -c pytorch -c nvidia \
    "pytorch=2.7.1" "pytorch-cuda=12.1" "tensorflow=2.19.0"

# Test installation
./test_gpu_installation.sh
python test_gpu_performance.py
```

**Expected Results:**
- ✅ All basic tests pass
- ✅ Performance tests pass
- ✅ CUDA toolkit test shows nvcc version

### **Scenario 3: Multi-GPU Setup**
```bash
# For systems with multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Test installation
./test_gpu_installation.sh
python test_gpu_performance.py
```

**Expected Results:**
- ✅ Multiple GPUs detected
- ✅ Performance tests use specified GPUs

## 🔧 **Troubleshooting Common Issues**

### **Issue: "nvidia-smi not found"**
```bash
# Check if NVIDIA driver is installed
lspci | grep -i nvidia

# Install driver if needed
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

### **Issue: "CUDA runtime not available"**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.version.cuda)"

# Reinstall with CUDA support
mamba install -c pytorch pytorch=2.7.1 pytorch-cuda=12.1
```

### **Issue: "TensorFlow GPU not detected"**
```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reinstall TensorFlow
mamba install tensorflow=2.19.0
```

### **Issue: "XGBoost GPU test failed"**
```bash
# Check XGBoost installation
python -c "import xgboost as xgb; print(xgb.__version__)"

# Reinstall XGBoost
mamba install xgboost=3.0.1
```

### **Issue: "Out of memory errors"**
```bash
# Reduce test data size in test_gpu_performance.py
# Change matrix size from 10000 to 5000
a = tf.random.normal([5000, 5000])  # Smaller test
```

## 📊 **Performance Benchmarks**

### **Expected Performance on Tesla T4**

| Test | Expected Time | Notes |
|------|---------------|-------|
| TensorFlow (10K x 10K) | 0.1-0.3s | Matrix multiplication |
| PyTorch (10K x 10K) | 0.1-0.3s | Matrix multiplication |
| XGBoost (10K samples) | 1-5s | 100 trees, 100 features |

### **Performance Comparison: CPU vs GPU**

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| TensorFlow MM | 15s | 0.2s | 75x |
| PyTorch MM | 12s | 0.15s | 80x |
| XGBoost Training | 45s | 3s | 15x |

## 🎯 **Integration with MetaSpliceAI**

### **Verify GPU Usage in Training**
```bash
# Run gene-aware CV with GPU acceleration
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gpu_test \
    --tree-method gpu_hist \
    --n-estimators 500 \
    --verbose

# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

### **Expected GPU Utilization**
- **XGBoost training**: 80-95% GPU utilization
- **Data preprocessing**: 20-40% GPU utilization
- **Model evaluation**: 60-80% GPU utilization

## 🔍 **Advanced Testing**

### **Memory Stress Test**
```python
# Create larger test in test_gpu_performance.py
def test_memory_stress():
    """Test GPU memory under stress."""
    import torch
    
    # Try to allocate large tensors
    device = torch.device('cuda:0')
    tensors = []
    
    for i in range(10):
        try:
            tensor = torch.randn(5000, 5000, device=device)
            tensors.append(tensor)
            print(f"Allocated tensor {i+1}")
        except RuntimeError as e:
            print(f"Memory exhausted at tensor {i+1}: {e}")
            break
    
    # Clean up
    del tensors
    torch.cuda.empty_cache()
```

### **Multi-GPU Load Balancing**
```python
# Test multiple GPUs
def test_multi_gpu():
    """Test multiple GPU utilization."""
    import torch
    
    gpu_count = torch.cuda.device_count()
    print(f"Testing {gpu_count} GPUs")
    
    for i in range(gpu_count):
        device = torch.device(f'cuda:{i}')
        start = time.time()
        a = torch.randn(5000, 5000, device=device)
        b = torch.mm(a, a)
        torch.cuda.synchronize(device)
        print(f"GPU {i}: {time.time() - start:.2f}s")
```

## 📝 **Test Results Documentation**

### **Template for Test Report**
```markdown
# GPU Test Report

**Date:** [Date]
**System:** [System specifications]
**Environment:** [Conda environment name]

## Installation Test Results
- [ ] NVIDIA Driver: [Version]
- [ ] PyTorch CUDA: [Version]
- [ ] TensorFlow GPU: [Available/Not Available]
- [ ] XGBoost GPU: [Available/Not Available]
- [ ] CUDA Toolkit: [Full/Runtime only]

## Performance Test Results
- [ ] TensorFlow: [Time]s
- [ ] PyTorch: [Time]s
- [ ] XGBoost: [Time]s

## Issues Found
[List any issues and resolutions]

## Recommendations
[Any recommendations for optimization]
```

## 🎉 **Success Criteria**

Your GPU setup is ready for MetaSpliceAI when:

✅ **All basic tests pass** (`test_gpu_installation.sh`)
✅ **All performance tests pass** (`test_gpu_performance.py`)
✅ **GPU utilization > 50%** during training
✅ **No memory errors** during stress tests
✅ **Multi-GPU support** (if applicable)

## 🚀 **Next Steps**

After successful GPU testing:

1. **Run MetaSpliceAI training** with GPU acceleration
2. **Monitor performance** during training
3. **Optimize batch sizes** for your GPU memory
4. **Consider multi-GPU** training for large datasets

Your GPU setup is now ready to accelerate MetaSpliceAI training and analysis! 🎯 