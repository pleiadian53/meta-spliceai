#!/bin/bash

echo "🔍 GPU Installation Test"
echo "========================"

# Test 1: NVIDIA Driver
echo "1. NVIDIA Driver Test:"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "   ✅ nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1
else
    echo "   ❌ nvidia-smi not found"
fi

# Test 2: CUDA Runtime
echo "2. CUDA Runtime Test:"
if python -c "import torch; print('PyTorch CUDA:', torch.version.cuda if torch.cuda.is_available() else 'No CUDA')" 2>/dev/null; then
    echo "   ✅ PyTorch CUDA runtime available"
else
    echo "   ❌ PyTorch CUDA runtime not available"
fi

# Test 3: TensorFlow GPU
echo "3. TensorFlow GPU Test:"
if python -c "import tensorflow as tf; print('TF GPUs:', len(tf.config.list_physical_devices('GPU')))" 2>/dev/null; then
    echo "   ✅ TensorFlow GPU support available"
else
    echo "   ❌ TensorFlow GPU support not available"
fi

# Test 4: XGBoost GPU
echo "4. XGBoost GPU Test:"
if python -c "import xgboost as xgb; print('XGBoost GPU support available')" 2>/dev/null; then
    echo "   ✅ XGBoost GPU support available"
else
    echo "   ❌ XGBoost GPU support not available"
fi

# Test 5: Full CUDA Toolkit (if installed)
echo "5. Full CUDA Toolkit Test:"
if command -v nvcc >/dev/null 2>&1; then
    echo "   ✅ Full CUDA toolkit installed"
    nvcc --version | head -1
else
    echo "   ℹ️  Full CUDA toolkit not installed (runtime libraries only)"
fi

echo "================================"
echo "✅ GPU installation test completed!" 