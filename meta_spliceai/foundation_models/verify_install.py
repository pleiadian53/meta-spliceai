"""
Verify installation of required dependencies for meta-spliceai foundation models.

This module provides utilities to check if all required dependencies are correctly installed
and configured, along with providing system information for debugging.
"""

import sys
import platform
import logging
from importlib import import_module
from typing import Dict, List, Tuple, Optional, Any
from packaging.version import parse as parse_version

logger = logging.getLogger(__name__)

# Required packages with minimum versions
REQUIREMENTS = {
    'torch': '1.7.0',
    'tensorflow': '2.0.0',
    'numpy': '1.19.0',
    'pandas': '1.0.0',
    'sklearn': '0.23.0',
    'matplotlib': '3.3.0',
}

# Optional packages that enhance functionality but aren't strictly required
OPTIONAL_PACKAGES = {
    'einops': '0.3.0',         # For HyenaDNA model
    'tqdm': '4.45.0',          # For progress bars
    'tensorboard': '2.2.0',    # For training visualization
    'biopython': '1.78',       # For sequence handling
    'pybedtools': '0.8.0',     # For genomic data processing
}


def check_package(package_name: str, min_version: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if a package is installed and meets the minimum version requirement.
    
    Args:
        package_name: Name of the package to check
        min_version: Minimum version required
        
    Returns:
        Tuple of (is_available, installed_version, error_message)
    """
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        # Special case for TensorFlow which has a different version attribute
        if package_name == 'tensorflow' and version == 'unknown':
            version = getattr(module, 'VERSION', 'unknown')
        
        # If we can import but version is unknown, consider it a success
        if version == 'unknown':
            return True, version, None
            
        # Simple version check using semantic versioning
        if parse_version(version) < parse_version(min_version):
            return False, version, f"Installed version {version} is older than required {min_version}"
            
        return True, version, None
    
    except ImportError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Error checking {package_name}: {str(e)}"


def check_pytorch_cuda() -> Tuple[bool, str]:
    """
    Check if PyTorch has CUDA support.
    
    Returns:
        Tuple of (is_available, message)
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            message = f"CUDA available: {device_count} devices. First device: {device_name}"
        else:
            message = "CUDA not available. Using CPU only."
        return cuda_available, message
    except Exception as e:
        return False, f"Error checking CUDA: {str(e)}"


def check_tensorflow_gpu() -> Tuple[bool, str]:
    """
    Check if TensorFlow has GPU support.
    
    Returns:
        Tuple of (is_available, message)
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_count = len(gpus)
            # Create a simple string representation of GPU info
            gpu_info = [f"GPU:{i}" for i in range(gpu_count)]
            message = f"TensorFlow GPU available: {gpu_count} devices. {', '.join(gpu_info)}"
            return True, message
        else:
            message = "TensorFlow GPU not available. Using CPU only."
            return False, message
    except Exception as e:
        return False, f"Error checking TensorFlow GPU: {str(e)}"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_implementation': platform.python_implementation(),
    }
    
    # Add memory info if psutil is available
    try:
        import psutil
        vm = psutil.virtual_memory()
        info['total_memory_gb'] = round(vm.total / (1024 ** 3), 2)
        info['available_memory_gb'] = round(vm.available / (1024 ** 3), 2)
    except ImportError:
        pass
    
    return info


def verify_installation(include_optional: bool = False, 
                        verbose: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify all required (and optionally optional) dependencies.
    
    Args:
        include_optional: Whether to check optional packages
        verbose: Whether to print results to stdout
        
    Returns:
        Tuple of (all_required_available, detailed_results)
    """
    if verbose:
        print("\n=== Verifying MetaSpliceAI Foundation Model Dependencies ===\n")
    
    # Check required packages
    results = {'required': {}, 'optional': {}, 'system': get_system_info()}
    all_required_available = True
    
    for package, min_version in REQUIREMENTS.items():
        available, version, error = check_package(package, min_version)
        results['required'][package] = {
            'available': available,
            'version': version,
            'error': error,
            'required_version': min_version
        }
        
        if not available:
            all_required_available = False
            
        if verbose:
            status = "✅" if available else "❌"
            version_str = f"v{version}" if version else "Not Found"
            print(f"{status} {package:<12} {version_str:<10} (Required: {min_version})")
            if error and not available:
                print(f"   └── Error: {error}")
    
    # Check optional packages if requested
    if include_optional:
        if verbose:
            print("\nOptional Packages:")
            
        for package, min_version in OPTIONAL_PACKAGES.items():
            available, version, error = check_package(package, min_version)
            results['optional'][package] = {
                'available': available,
                'version': version,
                'error': error,
                'required_version': min_version
            }
            
            if verbose:
                status = "✅" if available else "⚠️"  # Warning for optional
                version_str = f"v{version}" if version else "Not Found"
                print(f"{status} {package:<12} {version_str:<10} (Recommended: {min_version})")
    
    # Check hardware acceleration
    cuda_available, cuda_message = check_pytorch_cuda()
    tf_gpu_available, tf_message = check_tensorflow_gpu()
    
    results['gpu'] = {
        'pytorch_cuda': {'available': cuda_available, 'message': cuda_message},
        'tensorflow_gpu': {'available': tf_gpu_available, 'message': tf_message}
    }
    
    if verbose:
        print("\nHardware Acceleration:")
        print(f"{'✅' if cuda_available else '⚠️'} PyTorch: {cuda_message}")
        print(f"{'✅' if tf_gpu_available else '⚠️'} TensorFlow: {tf_message}")
        
        print("\nSystem Information:")
        for key, value in results['system'].items():
            print(f"- {key}: {value}")
            
        print("\nVerification Summary:")
        if all_required_available:
            print("✅ All required dependencies are installed correctly.")
        else:
            print("❌ Some required dependencies are missing or have incorrect versions.")
            print("   Please install the missing dependencies before using this package.")
    
    return all_required_available, results


def run_quick_test() -> bool:
    """
    Run a quick test of PyTorch and TensorFlow functionality.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("\n=== Running Quick Functionality Test ===\n")
    all_tests_passed = True
    
    # Test PyTorch
    try:
        print("Testing PyTorch...")
        import torch
        
        # Create and manipulate tensors
        x = torch.rand(5, 3)
        y = torch.rand(5, 3)
        z = x + y
        
        # Test a simple neural network
        linear = torch.nn.Linear(3, 1)
        output = linear(x)
        
        print("✅ PyTorch is working correctly\n")
    except Exception as e:
        print(f"❌ PyTorch test failed: {str(e)}\n")
        all_tests_passed = False
    
    # Test TensorFlow
    try:
        print("Testing TensorFlow...")
        import tensorflow as tf
        
        # Create and manipulate tensors
        x = tf.random.normal((5, 3))
        y = tf.random.normal((5, 3))
        z = x + y
        
        # Test a simple neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(3,))
        ])
        output = model(x)
        
        print("✅ TensorFlow is working correctly\n")
    except Exception as e:
        print(f"❌ TensorFlow test failed: {str(e)}\n")
        all_tests_passed = False
    
    # Overall result
    if all_tests_passed:
        print("✅ All functionality tests passed!")
    else:
        print("❌ Some functionality tests failed!")
        
    return all_tests_passed


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Verify meta-spliceai foundation model dependencies")
    parser.add_argument("--check-optional", action="store_true", help="Also check optional packages")
    parser.add_argument("--test", action="store_true", help="Run functionality tests")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()
    
    # Verify installation
    all_required, results = verify_installation(
        include_optional=args.check_optional,
        verbose=not args.quiet
    )
    
    # Run tests if requested
    if args.test and all_required:
        all_tests_passed = run_quick_test()
        
    # Exit with appropriate status code
    sys.exit(0 if all_required else 1)
