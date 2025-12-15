#!/usr/bin/env python3
"""
Comprehensive GPU Performance Test for MetaSpliceAI
"""

import time
import numpy as np
import sys

def test_tensorflow_gpu():
    """Test TensorFlow GPU performance."""
    print("1. TensorFlow GPU Test:")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   Available GPUs: {len(gpus)}")
        
        if gpus:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Performance test
            with tf.device('/GPU:0'):
                start = time.time()
                a = tf.random.normal([10000, 10000])
                b = tf.matmul(a, a)
                result = tf.reduce_sum(b)
                print(f"   Computation time: {time.time() - start:.2f}s")
                print(f"   Result: {result.numpy():.2f}")
            return True
        else:
            print("   ❌ No GPUs detected")
            return False
    except Exception as e:
        print(f"   ❌ TensorFlow GPU test failed: {e}")
        return False

def test_pytorch_gpu():
    """Test PyTorch GPU performance."""
    print("2. PyTorch GPU Test:")
    try:
        import torch
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            start = time.time()
            a = torch.randn(10000, 10000, device=device)
            b = torch.mm(a, a)
            result = torch.sum(b)
            torch.cuda.synchronize()
            print(f"   Computation time: {time.time() - start:.2f}s")
            print(f"   Result: {result.item():.2f}")
            return True
        else:
            print("   ❌ CUDA not available")
            return False
    except Exception as e:
        print(f"   ❌ PyTorch GPU test failed: {e}")
        return False

def test_xgboost_gpu():
    """Test XGBoost GPU performance."""
    print("3. XGBoost GPU Test:")
    try:
        import xgboost as xgb
        
        # Generate test data
        X = np.random.rand(10000, 100)
        y = np.random.randint(3, size=10000)
        
        start = time.time()
        model = xgb.XGBClassifier(
            tree_method='gpu_hist',
            n_estimators=100,
            objective='multi:softprob',
            device='cuda:0'
        )
        model.fit(X, y)
        training_time = time.time() - start
        
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Model trained successfully")
        return True
    except Exception as e:
        print(f"   ❌ XGBoost GPU test failed: {e}")
        return False

def test_cuda_toolkit():
    """Test full CUDA toolkit installation."""
    print("4. CUDA Toolkit Test:")
    try:
        import subprocess
        
        # Test nvcc
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"   ✅ {version_line}")
            return True
        else:
            print("   ℹ️  nvcc not found (runtime libraries only)")
            return False
    except Exception as e:
        print(f"   ℹ️  CUDA toolkit test skipped: {e}")
        return False

def main():
    """Run all GPU tests."""
    print("🚀 MetaSpliceAI GPU Performance Test")
    print("=" * 50)
    
    results = {
        'tensorflow': test_tensorflow_gpu(),
        'pytorch': test_pytorch_gpu(),
        'xgboost': test_xgboost_gpu(),
        'cuda_toolkit': test_cuda_toolkit()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:15} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All GPU tests passed! Your setup is ready for MetaSpliceAI.")
    else:
        print("\n⚠️  Some tests failed. Check the installation instructions.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 