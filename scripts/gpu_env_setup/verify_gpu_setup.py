#!/usr/bin/env python3
"""
Detailed GPU Setup Verification for MetaSpliceAI
This script provides comprehensive analysis of GPU setup and compatibility.
"""

import sys
import subprocess
import platform
import os
from pathlib import Path
import json
from typing import Dict, List, Any, Optional

def run_command(cmd: List[str], capture_output: bool = True) -> Dict[str, Any]:
    """Run a command and return results."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=30)
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Command timed out'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def check_system_info() -> Dict[str, Any]:
    """Check basic system information."""
    print("🔍 System Information")
    print("=" * 50)
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'machine': platform.machine()
    }
    
    for key, value in info.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return info

def check_nvidia_driver() -> Dict[str, Any]:
    """Check NVIDIA driver installation."""
    print("\n🔍 NVIDIA Driver Check")
    print("=" * 50)
    
    result = run_command(['nvidia-smi'])
    
    if result['success']:
        print("  ✅ NVIDIA driver found")
        
        # Parse driver information
        lines = result['stdout'].split('\n')
        driver_info = {}
        
        for line in lines:
            if 'Driver Version' in line:
                driver_info['version'] = line.split(':')[1].strip()
            elif 'CUDA Version' in line:
                driver_info['cuda_version'] = line.split(':')[1].strip()
            elif 'Tesla T4' in line or 'RTX' in line or 'GTX' in line:
                driver_info['gpu_model'] = line.split('|')[1].strip()
        
        for key, value in driver_info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        return {'status': 'available', 'info': driver_info}
    else:
        print("  ❌ NVIDIA driver not found")
        print(f"  Error: {result.get('error', result.get('stderr', 'Unknown error'))}")
        return {'status': 'not_available', 'error': result.get('error', result.get('stderr', 'Unknown error'))}

def check_cuda_toolkit() -> Dict[str, Any]:
    """Check CUDA toolkit installation."""
    print("\n🔍 CUDA Toolkit Check")
    print("=" * 50)
    
    # Check nvcc
    nvcc_result = run_command(['nvcc', '--version'])
    
    if nvcc_result['success']:
        print("  ✅ CUDA compiler (nvcc) found")
        version_line = nvcc_result['stdout'].split('\n')[0]
        print(f"  Version: {version_line}")
        
        # Check other CUDA tools
        tools = ['cuda-gdb', 'cuda-memcheck', 'cuda-profiler']
        available_tools = []
        
        for tool in tools:
            result = run_command([tool, '--version'])
            if result['success']:
                available_tools.append(tool)
        
        print(f"  Available tools: {', '.join(available_tools) if available_tools else 'None'}")
        
        return {
            'status': 'full_toolkit',
            'nvcc_version': version_line,
            'available_tools': available_tools
        }
    else:
        print("  ℹ️  CUDA compiler (nvcc) not found")
        print("  This indicates runtime libraries only (sufficient for MetaSpliceAI)")
        return {'status': 'runtime_only'}

def check_python_packages() -> Dict[str, Any]:
    """Check Python package installations."""
    print("\n🔍 Python Package Check")
    print("=" * 50)
    
    packages = {
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'xgboost': 'XGBoost',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'polars': 'Polars'
    }
    
    results = {}
    
    for package, display_name in packages.items():
        try:
            module = __import__(package)
            
            # Get version
            if hasattr(module, '__version__'):
                version = module.__version__
            else:
                version = 'Unknown'
            
            # Check GPU support
            gpu_support = False
            gpu_info = {}
            
            if package == 'torch':
                gpu_support = hasattr(module, 'cuda') and module.cuda.is_available()
                if gpu_support:
                    gpu_info = {
                        'cuda_version': module.version.cuda,
                        'device_count': module.cuda.device_count(),
                        'devices': [module.cuda.get_device_name(i) for i in range(module.cuda.device_count())]
                    }
            
            elif package == 'tensorflow':
                gpu_devices = module.config.list_physical_devices('GPU')
                gpu_support = len(gpu_devices) > 0
                if gpu_support:
                    gpu_info = {
                        'device_count': len(gpu_devices),
                        'devices': [str(device) for device in gpu_devices]
                    }
            
            elif package == 'xgboost':
                # XGBoost GPU support is built-in
                gpu_support = True
                gpu_info = {'built_in_support': True}
            
            status = "✅" if gpu_support else "⚠️"
            print(f"  {status} {display_name}: {version}")
            
            if gpu_support and gpu_info:
                for key, value in gpu_info.items():
                    print(f"    {key.replace('_', ' ').title()}: {value}")
            
            results[package] = {
                'version': version,
                'gpu_support': gpu_support,
                'gpu_info': gpu_info
            }
            
        except ImportError:
            print(f"  ❌ {display_name}: Not installed")
            results[package] = {'status': 'not_installed'}
        except Exception as e:
            print(f"  ⚠️  {display_name}: Error checking - {e}")
            results[package] = {'status': 'error', 'error': str(e)}
    
    return results

def check_environment_variables() -> Dict[str, Any]:
    """Check relevant environment variables."""
    print("\n🔍 Environment Variables")
    print("=" * 50)
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_HOME',
        'LD_LIBRARY_PATH',
        'PATH'
    ]
    
    results = {}
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}: {value}")
            results[var] = value
        else:
            print(f"  {var}: Not set")
            results[var] = None
    
    return results

def check_gpu_memory() -> Dict[str, Any]:
    """Check GPU memory availability."""
    print("\n🔍 GPU Memory Check")
    print("=" * 50)
    
    result = run_command(['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'])
    
    if result['success']:
        lines = result['stdout'].strip().split('\n')
        for i, line in enumerate(lines):
            total, free, used = map(int, line.split(', '))
            print(f"  GPU {i}:")
            print(f"    Total: {total} MB")
            print(f"    Used: {used} MB")
            print(f"    Free: {free} MB")
            print(f"    Utilization: {(used/total)*100:.1f}%")
        
        return {'status': 'available', 'gpus': len(lines)}
    else:
        print("  ❌ Cannot check GPU memory (no NVIDIA driver)")
        return {'status': 'not_available'}

def run_performance_tests() -> Dict[str, Any]:
    """Run basic performance tests."""
    print("\n🚀 Performance Tests")
    print("=" * 50)
    
    results = {}
    
    # Test 1: TensorFlow
    try:
        import tensorflow as tf
        import time
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("  Testing TensorFlow GPU performance...")
            start = time.time()
            
            with tf.device('/GPU:0'):
                a = tf.random.normal([5000, 5000])
                b = tf.matmul(a, a)
                result = tf.reduce_sum(b)
            
            tf_time = time.time() - start
            print(f"  TensorFlow (5K x 5K): {tf_time:.3f}s")
            results['tensorflow'] = {'time': tf_time, 'success': True}
        else:
            print("  ⚠️  No TensorFlow GPUs available")
            results['tensorflow'] = {'success': False, 'reason': 'no_gpus'}
    except Exception as e:
        print(f"  ❌ TensorFlow test failed: {e}")
        results['tensorflow'] = {'success': False, 'error': str(e)}
    
    # Test 2: PyTorch
    try:
        import torch
        import time
        
        if torch.cuda.is_available():
            print("  Testing PyTorch GPU performance...")
            start = time.time()
            
            device = torch.device('cuda:0')
            a = torch.randn(5000, 5000, device=device)
            b = torch.mm(a, a)
            result = torch.sum(b)
            torch.cuda.synchronize()
            
            torch_time = time.time() - start
            print(f"  PyTorch (5K x 5K): {torch_time:.3f}s")
            results['pytorch'] = {'time': torch_time, 'success': True}
        else:
            print("  ⚠️  PyTorch CUDA not available")
            results['pytorch'] = {'success': False, 'reason': 'no_cuda'}
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")
        results['pytorch'] = {'success': False, 'error': str(e)}
    
    return results

def generate_report(all_results: Dict[str, Any]) -> None:
    """Generate a comprehensive report."""
    print("\n📊 Comprehensive Report")
    print("=" * 50)
    
    # Overall status
    gpu_available = all_results.get('nvidia_driver', {}).get('status') == 'available'
    cuda_toolkit = all_results.get('cuda_toolkit', {}).get('status') == 'full_toolkit'
    
    print(f"GPU Available: {'✅ Yes' if gpu_available else '❌ No'}")
    print(f"CUDA Toolkit: {'✅ Full' if cuda_toolkit else 'ℹ️ Runtime Only'}")
    
    # Package status
    packages = all_results.get('python_packages', {})
    gpu_packages = ['torch', 'tensorflow', 'xgboost']
    
    print("\nGPU Package Status:")
    for pkg in gpu_packages:
        if pkg in packages:
            pkg_info = packages[pkg]
            if isinstance(pkg_info, dict) and pkg_info.get('gpu_support'):
                print(f"  ✅ {pkg.upper()}: GPU support available")
            else:
                print(f"  ⚠️  {pkg.upper()}: No GPU support")
        else:
            print(f"  ❌ {pkg.upper()}: Not installed")
    
    # Performance results
    perf_results = all_results.get('performance_tests', {})
    if perf_results:
        print("\nPerformance Test Results:")
        for test, result in perf_results.items():
            if result.get('success'):
                print(f"  ✅ {test.title()}: {result.get('time', 'N/A'):.3f}s")
            else:
                print(f"  ❌ {test.title()}: Failed")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if not gpu_available:
        print("  • Install NVIDIA driver for GPU acceleration")
        print("  • Consider using CPU-only setup for development")
    elif not cuda_toolkit:
        print("  • Current setup is sufficient for MetaSpliceAI")
        print("  • Install full CUDA toolkit only if doing CUDA development")
    else:
        print("  • Full GPU setup detected - ready for all workloads")
    
    # Check for missing packages
    missing_packages = []
    for pkg in gpu_packages:
        if pkg not in packages or packages[pkg].get('status') == 'not_installed':
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"  • Install missing packages: {', '.join(missing_packages)}")

def main():
    """Main function to run all checks."""
    print("🚀 MetaSpliceAI GPU Setup Verification")
    print("=" * 60)
    
    all_results = {}
    
    # Run all checks
    all_results['system_info'] = check_system_info()
    all_results['nvidia_driver'] = check_nvidia_driver()
    all_results['cuda_toolkit'] = check_cuda_toolkit()
    all_results['python_packages'] = check_python_packages()
    all_results['environment_variables'] = check_environment_variables()
    all_results['gpu_memory'] = check_gpu_memory()
    all_results['performance_tests'] = run_performance_tests()
    
    # Generate comprehensive report
    generate_report(all_results)
    
    print("\n" + "=" * 60)
    print("✅ GPU setup verification completed!")
    
    # Save results to file
    output_file = "gpu_setup_verification.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

if __name__ == "__main__":
    main() 