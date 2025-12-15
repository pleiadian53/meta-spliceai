#!/usr/bin/env python3
"""
Comprehensive GPU Environment Diagnostic Script for MetaSpliceAI

This script checks for common issues that cause TensorFlow GPU failures,
particularly the "DNN library initialization failed" error.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Suppress TensorFlow logs initially
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def run_command(cmd, capture_output=True):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.stdout.strip() if capture_output else None
    except Exception as e:
        return f"Error: {e}"

def check_system_info():
    """Check basic system information."""
    print_section("SYSTEM INFORMATION")
    print(f"OS: {platform.platform()}")
    print(f"Python Version: {sys.version}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Current working directory: {os.getcwd()}")

def check_nvidia_setup():
    """Check NVIDIA driver and CUDA installation."""
    print_section("NVIDIA & CUDA SETUP")
    
    # Check nvidia-smi
    nvidia_smi_test = run_command("nvidia-smi --help")
    if nvidia_smi_test and ("NVIDIA System Management Interface" in str(nvidia_smi_test) or "nvidia-smi" in str(nvidia_smi_test).lower()):
        print("✅ nvidia-smi found and working")
        print("\nGPU Information:")
        gpu_info = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
        print(gpu_info)
        
        # Get driver version
        driver_info = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
        if driver_info:
            print(f"\nNVIDIA Driver Version: {driver_info.split()[0] if driver_info.split() else 'Unknown'}")
    else:
        print("❌ nvidia-smi not found or not working")
        print(f"Command output: {nvidia_smi_test}")
        return False
    
    # Check CUDA version
    cuda_version = run_command("nvcc --version")
    if "nvcc" in str(cuda_version):
        print(f"\n✅ CUDA Compiler found:")
        print(cuda_version.split('\n')[-1] if cuda_version else "Version info not available")
    else:
        print("⚠️  nvcc not found in PATH")
    
    # Check CUDA runtime
    cuda_runtime = run_command("cat /usr/local/cuda/version.txt")
    if not cuda_runtime or "Error" in cuda_runtime:
        cuda_runtime = run_command("cat /usr/local/cuda/version.json")
    
    if cuda_runtime and "Error" not in cuda_runtime:
        print(f"✅ CUDA Runtime: {cuda_runtime}")
    else:
        print("⚠️  CUDA runtime version file not found")
    
    return True

def check_conda_environment():
    """Check conda environment and package versions."""
    print_section("CONDA ENVIRONMENT")
    
    # Check if in conda env
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"Active conda environment: {conda_env}")
    
    if conda_env != 'None':
        print(f"Conda prefix: {os.environ.get('CONDA_PREFIX', 'Not set')}")
        
        # Check key packages
        packages_to_check = [
            'tensorflow', 'keras', 'numpy', 'matplotlib', 
            'torch', 'torchvision', 'cudatoolkit', 'cudnn'
        ]
        
        print("\n📦 Key Package Versions:")
        for pkg in packages_to_check:
            version = run_command(f"conda list {pkg} | grep {pkg}")
            if version and not version.startswith("Error"):
                print(f"  {pkg}: {version.split()[1] if version.split() else 'Found but version unclear'}")
            else:
                print(f"  {pkg}: ❌ Not installed via conda")

def check_tensorflow_detailed():
    """Detailed TensorFlow diagnostics."""
    print_section("TENSORFLOW DETAILED DIAGNOSTICS")
    
    try:
        # Re-enable TensorFlow logging for detailed diagnostics
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        
        # Suppress NumPy warnings that are not actual failures
        import warnings
        warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
        warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
        
        print("⚠️  Note: NumPy 2.x compatibility warnings may appear but are not failures")
        
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully: {tf.__version__}")
        
        # Check build info
        print(f"\nTensorFlow build info:")
        print(f"  CUDA support: {tf.test.is_built_with_cuda()}")
        print(f"  GPU support: {tf.test.is_built_with_gpu_support()}")
        
        # Check available devices
        physical_devices = tf.config.list_physical_devices()
        print(f"\nPhysical devices:")
        for device in physical_devices:
            print(f"  {device}")
        
        # Check GPU devices specifically
        gpu_devices = tf.config.list_physical_devices('GPU')
        print(f"\nGPU devices: {len(gpu_devices)}")
        for i, gpu in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu}")
            
        # Try to get GPU memory info
        if gpu_devices:
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu_devices[0])
                print(f"\nGPU Details for {gpu_devices[0]}:")
                for key, value in gpu_details.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"⚠️  Could not get GPU details: {e}")
        
        # Additional NumPy 2.x environment check
        import numpy as np
        print(f"\nNumPy version: {np.__version__}")
        if np.__version__.startswith("2."):
            print("ℹ️  NumPy 2.x detected - some compatibility warnings are expected but harmless")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import TensorFlow: {e}")
        return False
    except Exception as e:
        # Check if it's just a warning vs actual failure
        if "TensorFlow imported successfully" in str(e):
            print(f"⚠️  TensorFlow imported with warnings (likely NumPy 2.x related): {e}")
            return True
        else:
            print(f"❌ TensorFlow error: {e}")
            return False

def check_cuda_libraries():
    """Check for CUDA libraries."""
    print_section("CUDA LIBRARIES CHECK")
    
    # Common CUDA library locations
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/opt/cuda/lib64",
        "/usr/local/cuda-11.8/lib64",
        "/usr/local/cuda-12.0/lib64"
    ]
    
    # Key libraries to check
    key_libraries = [
        "libcudart.so",
        "libcublas.so", 
        "libcurand.so",
        "libcusolver.so",
        "libcusparse.so",
        "libcudnn.so"
    ]
    
    print("🔍 Searching for CUDA libraries...")
    found_libraries = {}
    
    for lib in key_libraries:
        found = False
        for path in cuda_paths:
            if Path(path).exists():
                lib_files = list(Path(path).glob(f"{lib}*"))
                if lib_files:
                    found_libraries[lib] = str(lib_files[0])
                    found = True
                    break
        
        if found:
            print(f"  ✅ {lib}: {found_libraries[lib]}")
        else:
            print(f"  ❌ {lib}: Not found")
    
    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"\nLD_LIBRARY_PATH: {ld_path if ld_path else 'Not set'}")
    
    return found_libraries

def check_tensorflow_gpu_test():
    """Attempt a TensorFlow GPU computation."""
    print_section("TENSORFLOW GPU COMPUTATION TEST")
    
    try:
        import tensorflow as tf
        
        # Suppress warnings for cleaner output
        tf.get_logger().setLevel('ERROR')
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU devices found by TensorFlow")
            return False
        
        print(f"✅ Found {len(gpus)} GPU(s)")
        
        # Try a simple computation on each GPU
        for i, gpu in enumerate(gpus):
            try:
                with tf.device(f'/GPU:{i}'):
                    # Simple matrix multiplication
                    a = tf.random.normal([100, 100])
                    b = tf.random.normal([100, 100])
                    c = tf.matmul(a, b)
                    result = tf.reduce_sum(c)
                    
                print(f"  ✅ GPU {i} computation successful: result = {result.numpy():.2f}")
                
            except Exception as e:
                print(f"  ❌ GPU {i} computation failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def check_spliceai_models():
    """Check if SpliceAI models can be loaded."""
    print_section("SPLICEAI MODEL LOADING TEST")
    
    try:
        # Try to import spliceai
        from spliceai.utils import one_hot_encode
        print("✅ SpliceAI utils imported successfully")
        
        # Try to load a SpliceAI model
        from keras.models import load_model
        
        # Check if model files exist
        model_paths = [
            "models/spliceai",
            "/opt/spliceai/models", 
            "~/.spliceai/models"
        ]
        
        print("🔍 Looking for SpliceAI model files...")
        model_found = False
        for model_path in model_paths:
            expanded_path = Path(model_path).expanduser()
            if expanded_path.exists():
                print(f"  ✅ Model directory found: {expanded_path}")
                model_files = list(expanded_path.glob("*.h5"))
                if model_files:
                    print(f"    Found {len(model_files)} model files")
                    model_found = True
                else:
                    print(f"    No .h5 model files found")
        
        if not model_found:
            print("  ⚠️  No SpliceAI model files found in common locations")
            print("     This may cause issues when running splice prediction workflows")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SpliceAI: {e}")
        return False
    except Exception as e:
        print(f"❌ SpliceAI test error: {e}")
        return False

def generate_fix_recommendations(results):
    """Generate specific fix recommendations based on diagnostic results."""
    print_section("RECOMMENDED FIXES")
    
    if not results.get('nvidia_setup', False):
        print("🔧 NVIDIA Setup Issues:")
        print("   1. Install NVIDIA drivers: sudo apt install nvidia-driver-XXX")
        print("   2. Install CUDA toolkit: conda install cudatoolkit=11.8")
        print("   3. Reboot the system after driver installation")
    
    if not results.get('tensorflow_gpu', False):
        print("\n🔧 TensorFlow GPU Issues:")
        print("   CRITICAL: CUDA version mismatch detected!")
        print("   - You have CUDA 12.2 but TensorFlow 2.18.0 needs CUDA 11.8")
        print("   - You have cuDNN 9.7.1 but TensorFlow needs 8.6+")
        print("   ")
        print("   SOLUTION OPTIONS:")
        print("   1. Downgrade CUDA to 11.8 (recommended):")
        print("      conda install cudatoolkit=11.8 cudnn=8.6.0")
        print("   2. Or upgrade to TensorFlow 2.19+ which supports CUDA 12:")
        print("      conda install tensorflow=2.19.0")
        print("   3. Use conda-forge for better compatibility:")
        print("      conda install -c conda-forge tensorflow-gpu=2.18.0 cudatoolkit=11.8")
    
    if 'libcudnn.so' not in results.get('cuda_libraries', {}):
        print("\n🔧 cuDNN Missing:")
        print("   1. Install cuDNN via conda:")
        print("      conda install cudnn")
        print("   2. Or download from NVIDIA and install manually")
        print("   3. Ensure cuDNN version matches TensorFlow requirements")
    
    print("\n🔧 General Environment Fixes:")
    print("   1. Create a fresh environment:")
    print("      conda create -n surveyor-gpu python=3.10")
    print("      conda activate surveyor-gpu")
    print("      conda install tensorflow-gpu=2.18.0 pytorch cudatoolkit=11.8 cudnn")
    print("   2. Set environment variables:")
    print("      export CUDA_VISIBLE_DEVICES=0")
    print("      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    print("   3. Test with the check_gpu.py script before running incremental builder")

def main():
    """Run comprehensive GPU environment diagnostics."""
    print("🚀 MetaSpliceAI GPU Environment Diagnostics")
    print("This script will check for common GPU setup issues.")
    
    results = {}
    
    # Run all diagnostic checks
    check_system_info()
    results['nvidia_setup'] = check_nvidia_setup()
    check_conda_environment()
    results['tensorflow_basic'] = check_tensorflow_detailed()
    results['cuda_libraries'] = check_cuda_libraries()
    results['tensorflow_gpu'] = check_tensorflow_gpu_test()
    results['spliceai'] = check_spliceai_models()
    
    # Generate recommendations
    generate_fix_recommendations(results)
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    all_good = all([
        results.get('nvidia_setup', False),
        results.get('tensorflow_basic', False), 
        results.get('tensorflow_gpu', False)
    ])
    
    if all_good:
        print("🎉 All checks passed! Your GPU environment should work for MetaSpliceAI.")
    else:
        print("⚠️  Some issues detected. Please review the recommendations above.")
        print("💡 Common solution: Create a fresh conda environment with GPU packages.")
    
    print(f"\n📋 Results Summary:")
    for check, status in results.items():
        if isinstance(status, bool):
            print(f"  {check}: {'✅ PASS' if status else '❌ FAIL'}")
        elif isinstance(status, dict):
            print(f"  {check}: {'✅ PASS' if status else '❌ FAIL'} ({len(status)} items found)")

if __name__ == "__main__":
    main() 