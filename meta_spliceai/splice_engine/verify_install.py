"""
Verify installation of required dependencies for meta-spliceai splice_engine.

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

# Required packages with minimum versions based on surveyor environment
REQUIREMENTS = {
    'polars': '1.16.0',
    'pandas': '2.2.2',
    'pybedtools': '0.9.1',
    'gffutils': '0.12',
    'biopython': '1.83',  # Updated to match actual version
    'tqdm': '4.67.1',
    'tensorflow': '2.18.0',
    'keras': '3.5.0',
    'spliceai': '1.3.1',
    'pyfaidx': '0.8.1.3',
    'xgboost': '2.1.1',
    'shap': '0.46.0'
}

# Optional packages that enhance functionality but aren't strictly required
OPTIONAL_PACKAGES = {
    'h5py': '3.12.1',
    'tensorboard': '2.18.0',
    'rich': '13.9.4',
    'matplotlib': '3.9.2',
    'seaborn': '0.13.2'
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
        # Special case for biopython which has a different import name
        if package_name == 'biopython':
            module = import_module('Bio')
        else:
            module = import_module(package_name)
            
        version = getattr(module, '__version__', 'unknown')
        
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

def check_bedtools() -> Tuple[bool, str]:
    """
    Check if bedtools is installed and accessible.
    
    Returns:
        Tuple of (is_available, message)
    """
    try:
        import subprocess
        result = subprocess.run(['bedtools', '--version'], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            return True, f"bedtools found: {result.stdout.strip()}"
        else:
            return False, ("bedtools command failed. Please ensure bedtools is installed.\n"
                         "   Install via conda: conda install -c bioconda bedtools\n"
                         "   Or via apt: sudo apt-get install bedtools")
    except FileNotFoundError:
        return False, ("bedtools not found in PATH. Please install bedtools:\n"
                      "   Install via conda: conda install -c bioconda bedtools\n"
                      "   Or via apt: sudo apt-get install bedtools")
    except Exception as e:
        return False, f"Error checking bedtools: {str(e)}"

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
        print("\n=== Verifying MetaSpliceAI Splice Engine Dependencies ===\n")
    
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
    
    # Check bedtools
    bedtools_available, bedtools_message = check_bedtools()
    results['required']['bedtools'] = {
        'available': bedtools_available,
        'version': 'system',
        'error': None if bedtools_available else bedtools_message
    }
    
    if verbose:
        status = "✅" if bedtools_available else "❌"
        print(f"{status} {'bedtools':<12} {'system':<10} (Required: system)")
        if not bedtools_available:
            print(f"   └── Error: {bedtools_message}")
    
    if not bedtools_available:
        all_required_available = False
    
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
    
    if verbose:
        print("\nSystem Information:")
        for key, value in results['system'].items():
            print(f"  {key}: {value}")
    
    return all_required_available, results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run verification
    success, results = verify_installation(include_optional=True, verbose=True)
    
    if not success:
        print("\n❌ Some required dependencies are missing or outdated.")
        sys.exit(1)
    else:
        print("\n✅ All required dependencies are properly installed.")
