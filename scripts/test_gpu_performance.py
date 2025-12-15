#!/usr/bin/env python3
"""
GPU Performance Test for MetaSpliceAI on Tesla T4 setup.

This script tests GPU acceleration performance for key components:
- TensorFlow operations
- PyTorch operations  
- XGBoost GPU training
- Memory management

Optimized for Tesla T4 with CUDA 12.2.
"""

import time
import os
import sys
from typing import Dict, List, Tuple


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print("="*60)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"🧪 {title}")
    print("-"*40)


def check_gpu_setup() -> Dict[str, any]:
    """Check basic GPU setup and return information."""
    print_header("GPU Hardware Detection")
    
    gpu_info = {}
    
    # Check nvidia-smi
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        gpu_data = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu_data.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_mb': int(parts[2]),
                        'driver': parts[3]
                    })
        
        gpu_info['gpu_count'] = len(gpu_data)
        gpu_info['gpus'] = gpu_data
        gpu_info['nvidia_smi_available'] = True
        
        print(f"✅ Found {len(gpu_data)} NVIDIA GPU(s):")
        for gpu in gpu_data:
            print(f"   GPU {gpu['index']}: {gpu['name']} ({gpu['memory_mb']} MB)")
        
        if gpu_data:
            print(f"   Driver Version: {gpu_data[0]['driver']}")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ nvidia-smi not available: {e}")
        gpu_info['nvidia_smi_available'] = False
        gpu_info['gpu_count'] = 0
    
    return gpu_info


def test_tensorflow_gpu() -> Dict[str, any]:
    """Test TensorFlow GPU performance."""
    print_section("TensorFlow GPU Performance")
    
    results = {}
    
    try:
        import tensorflow as tf
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        results['gpus_detected'] = len(gpus)
        results['tensorflow_version'] = tf.__version__
        
        print(f"📦 TensorFlow Version: {tf.__version__}")
        print(f"🎮 GPUs Detected: {len(gpus)}")
        
        if len(gpus) > 0:
            # Enable memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print("✅ Enabled GPU memory growth")
            
            # Test computation performance
            print("\n🏃 Running performance test...")
            
            # CPU test
            with tf.device('/CPU:0'):
                start_time = time.time()
                a = tf.random.normal([5000, 5000])
                b = tf.matmul(a, a)
                result_cpu = tf.reduce_sum(b)
                cpu_time = time.time() - start_time
            
            # GPU test  
            with tf.device('/GPU:0'):
                start_time = time.time()
                a = tf.random.normal([5000, 5000])
                b = tf.matmul(a, a)
                result_gpu = tf.reduce_sum(b)
                gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            results['cpu_time'] = cpu_time
            results['gpu_time'] = gpu_time
            results['speedup'] = speedup
            results['success'] = True
            
            print(f"   CPU Time: {cpu_time:.3f}s")
            print(f"   GPU Time: {gpu_time:.3f}s")
            print(f"   Speedup: {speedup:.1f}x")
            
            if speedup > 2:
                print("   ✅ GPU acceleration working well!")
            elif speedup > 1:
                print("   ⚠️  GPU acceleration modest")
            else:
                print("   ❌ GPU slower than CPU (check setup)")
                
        else:
            print("❌ No GPUs detected by TensorFlow")
            results['success'] = False
            
    except ImportError:
        print("❌ TensorFlow not available")
        results['success'] = False
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        results['success'] = False
    
    return results


def test_pytorch_gpu() -> Dict[str, any]:
    """Test PyTorch GPU performance."""
    print_section("PyTorch GPU Performance")
    
    results = {}
    
    try:
        import torch
        
        results['pytorch_version'] = torch.__version__
        results['cuda_available'] = torch.cuda.is_available()
        results['gpu_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        print(f"📦 PyTorch Version: {torch.__version__}")
        print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🎮 GPU Count: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
            
            print("\n🏃 Running performance test...")
            
            # CPU test
            start_time = time.time()
            a = torch.randn(5000, 5000)
            b = torch.mm(a, a)
            result_cpu = torch.sum(b)
            cpu_time = time.time() - start_time
            
            # GPU test
            device = torch.device('cuda:0')
            start_time = time.time()
            a = torch.randn(5000, 5000, device=device)
            b = torch.mm(a, a)
            result_gpu = torch.sum(b)
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            results['cpu_time'] = cpu_time
            results['gpu_time'] = gpu_time  
            results['speedup'] = speedup
            results['success'] = True
            
            print(f"   CPU Time: {cpu_time:.3f}s")
            print(f"   GPU Time: {gpu_time:.3f}s")
            print(f"   Speedup: {speedup:.1f}x")
            
            if speedup > 5:
                print("   ✅ Excellent GPU acceleration!")
            elif speedup > 2:
                print("   ✅ Good GPU acceleration")
            else:
                print("   ⚠️  Limited GPU acceleration")
                
        else:
            print("❌ CUDA not available")
            results['success'] = False
            
    except ImportError:
        print("❌ PyTorch not available")
        results['success'] = False
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        results['success'] = False
    
    return results


def test_xgboost_gpu() -> Dict[str, any]:
    """Test XGBoost GPU training performance."""
    print_section("XGBoost GPU Performance")
    
    results = {}
    
    try:
        import xgboost as xgb
        import numpy as np
        
        print(f"📦 XGBoost Version: {xgb.__version__}")
        
        # Generate test data
        n_samples = 10000
        n_features = 100
        X = np.random.rand(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 3, size=n_samples)
        
        print(f"🔢 Test data: {n_samples} samples, {n_features} features")
        
        # CPU training
        print("\n🏃 CPU Training...")
        start_time = time.time()
        model_cpu = xgb.XGBClassifier(
            tree_method='hist',
            n_estimators=100,
            objective='multi:softprob',
            random_state=42
        )
        model_cpu.fit(X, y)
        cpu_time = time.time() - start_time
        
        # GPU training
        print("🏃 GPU Training...")
        start_time = time.time()
        model_gpu = xgb.XGBClassifier(
            tree_method='gpu_hist',
            n_estimators=100,
            objective='multi:softprob',
            device='cuda:0',
            random_state=42
        )
        model_gpu.fit(X, y)
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        results['cpu_time'] = cpu_time
        results['gpu_time'] = gpu_time
        results['speedup'] = speedup
        results['success'] = True
        results['xgboost_version'] = xgb.__version__
        
        print(f"   CPU Time: {cpu_time:.3f}s")
        print(f"   GPU Time: {gpu_time:.3f}s") 
        print(f"   Speedup: {speedup:.1f}x")
        
        if speedup > 3:
            print("   ✅ Excellent XGBoost GPU acceleration!")
        elif speedup > 1.5:
            print("   ✅ Good XGBoost GPU acceleration")
        else:
            print("   ⚠️  Limited XGBoost GPU acceleration")
            
    except ImportError:
        print("❌ XGBoost not available")
        results['success'] = False
    except Exception as e:
        print(f"❌ XGBoost GPU test failed: {e}")
        results['success'] = False
        
    return results


def test_memory_management() -> Dict[str, any]:
    """Test GPU memory management."""
    print_section("GPU Memory Management")
    
    results = {}
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Check initial memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
            max_memory = torch.cuda.get_device_properties(device).total_memory
            
            print(f"🔋 GPU Memory Info:")
            print(f"   Total Memory: {max_memory / 1e9:.1f} GB")
            print(f"   Initial Used: {initial_memory / 1e6:.1f} MB")
            
            # Allocate and deallocate memory
            print("\n🧪 Memory allocation test...")
            
            # Allocate large tensor
            large_tensor = torch.randn(2000, 2000, device=device)
            after_alloc = torch.cuda.memory_allocated(device)
            
            print(f"   After allocation: {after_alloc / 1e6:.1f} MB")
            
            # Free memory
            del large_tensor
            torch.cuda.empty_cache()
            after_free = torch.cuda.memory_allocated(device)
            
            print(f"   After freeing: {after_free / 1e6:.1f} MB")
            
            results['total_memory_gb'] = max_memory / 1e9
            results['memory_test_success'] = True
            results['memory_management_working'] = after_free <= initial_memory + 1e6  # Allow 1MB tolerance
            
            if results['memory_management_working']:
                print("   ✅ Memory management working correctly")
            else:
                print("   ⚠️  Possible memory leak detected")
                
        else:
            print("❌ No CUDA available for memory test")
            results['memory_test_success'] = False
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        results['memory_test_success'] = False
        
    return results


def generate_summary(results: Dict[str, Dict]) -> None:
    """Generate a summary of all test results."""
    print_header("Performance Summary")
    
    # Check if any tests succeeded
    has_gpu = any(result.get('success', False) for result in results.values())
    
    if not has_gpu:
        print("❌ No GPU acceleration detected!")
        print("   • Check NVIDIA drivers are installed")
        print("   • Verify CUDA compatibility")
        print("   • Ensure GPU-enabled packages are installed")
        return
    
    print("✅ GPU Acceleration Summary:")
    
    # TensorFlow summary
    if 'tensorflow' in results and results['tensorflow'].get('success'):
        tf_speedup = results['tensorflow'].get('speedup', 0)
        print(f"   🧠 TensorFlow: {tf_speedup:.1f}x speedup")
    
    # PyTorch summary
    if 'pytorch' in results and results['pytorch'].get('success'):
        torch_speedup = results['pytorch'].get('speedup', 0)
        print(f"   🔥 PyTorch: {torch_speedup:.1f}x speedup")
    
    # XGBoost summary
    if 'xgboost' in results and results['xgboost'].get('success'):
        xgb_speedup = results['xgboost'].get('speedup', 0)
        print(f"   🌲 XGBoost: {xgb_speedup:.1f}x speedup")
    
    # Memory summary
    if 'memory' in results and results['memory'].get('memory_test_success'):
        total_mem = results['memory'].get('total_memory_gb', 0)
        print(f"   💾 GPU Memory: {total_mem:.1f} GB available")
    
    # Multi-GPU info
    gpu_count = results.get('gpu_info', {}).get('gpu_count', 0)
    if gpu_count > 1:
        print(f"   🎮 Multi-GPU: {gpu_count} GPUs available for parallel processing")
    
    print("\n🚀 Recommendations for MetaSpliceAI:")
    print("   • Use --tree-method gpu_hist for XGBoost training")
    print("   • Enable GPU acceleration in TensorFlow models")
    print("   • Consider multi-GPU parallel CV folds")
    
    if gpu_count >= 4:
        print("   • Your 4-GPU setup is excellent for parallel training!")


def main():
    """Run comprehensive GPU performance tests."""
    print("🔍 MetaSpliceAI GPU Performance Test")
    print("Optimized for Tesla T4 with CUDA 12.2")
    print("="*60)
    
    # Store all results
    results = {}
    
    # Run tests
    results['gpu_info'] = check_gpu_setup()
    results['tensorflow'] = test_tensorflow_gpu()
    results['pytorch'] = test_pytorch_gpu()
    results['xgboost'] = test_xgboost_gpu()
    results['memory'] = test_memory_management()
    
    # Generate summary
    generate_summary(results)
    
    print(f"\n{'='*60}")
    print("🎯 GPU Performance Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main() 