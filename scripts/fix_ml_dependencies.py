#!/usr/bin/env python3
"""
ML Dependencies Fixer for MetaSpliceAI

This script helps resolve complex version conflicts between TensorFlow, PyTorch,
NumPy, and other ML libraries in conda environments.
"""

import subprocess
import sys
import os
from typing import Dict, List, Tuple, Optional

# Known working combinations for different TensorFlow versions
TENSORFLOW_COMPATIBILITY = {
    "2.18.0": {
        "cuda": "11.8",
        "cudnn": "8.6.0",
        "numpy": ">=1.23,<2.0",
        "keras": "3.5.0",
        "python": ">=3.9,<=3.11"
    },
    "2.17.0": {
        "cuda": "11.8",
        "cudnn": "8.6.0", 
        "numpy": ">=1.23,<2.0",
        "keras": "3.4.1",
        "python": ">=3.9,<=3.11"
    },
    "2.16.1": {
        "cuda": "11.8",
        "cudnn": "8.6.0",
        "numpy": ">=1.23,<1.27",
        "keras": "3.3.3",
        "python": ">=3.9,<=3.11"
    }
}

# PyTorch compatibility with CUDA versions
PYTORCH_COMPATIBILITY = {
    "2.3.0": {
        "cuda": ["11.8", "12.1"],
        "numpy": ">=1.21,<2.0",
        "python": ">=3.8,<=3.11"
    },
    "2.2.0": {
        "cuda": ["11.8", "12.1"],
        "numpy": ">=1.21,<2.0", 
        "python": ">=3.8,<=3.11"
    }
}

def run_command(cmd: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def get_conda_package_info(package: str) -> Optional[str]:
    """Get installed version of a conda package."""
    success, output = run_command(f"conda list {package}")
    if success:
        for line in output.split('\n'):
            if line.strip().startswith(package) and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    return None

def check_current_environment():
    """Check current environment packages and versions."""
    print("🔍 CHECKING CURRENT ENVIRONMENT")
    print("=" * 50)
    
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Conda Environment: {env_name}")
    
    # Key packages to check
    packages = [
        'python', 'tensorflow', 'tensorflow-gpu', 'keras', 
        'pytorch', 'numpy', 'cudatoolkit', 'cudnn', 'matplotlib'
    ]
    
    current_versions = {}
    print("\n📦 Current Package Versions:")
    
    for pkg in packages:
        version = get_conda_package_info(pkg)
        current_versions[pkg] = version
        status = "✅" if version else "❌"
        print(f"  {status} {pkg}: {version or 'Not installed'}")
    
    return current_versions

def detect_conflicts(versions: Dict[str, Optional[str]]) -> List[str]:
    """Detect potential version conflicts."""
    conflicts = []
    
    tf_version = versions.get('tensorflow') or versions.get('tensorflow-gpu')
    numpy_version = versions.get('numpy')
    cuda_version = versions.get('cudatoolkit')
    
    # Check TensorFlow-NumPy compatibility
    if tf_version and numpy_version:
        if tf_version.startswith('2.18') and numpy_version.startswith('1.26'):
            pass  # This is good
        elif tf_version.startswith('2.18') and numpy_version.startswith('2.'):
            conflicts.append("TensorFlow 2.18.0 may have issues with NumPy 2.x - use NumPy 1.26.x")
        elif tf_version.startswith('2.17') and numpy_version.startswith('2.'):
            conflicts.append("TensorFlow 2.17.x not compatible with NumPy 2.x - use NumPy 1.26.x")
    
    # Check CUDA compatibility
    if tf_version and cuda_version:
        if tf_version.startswith('2.18') and not cuda_version.startswith('11.8'):
            conflicts.append(f"TensorFlow 2.18.0 requires CUDA 11.8, found {cuda_version}")
    
    # Check missing cuDNN
    if versions.get('cudatoolkit') and not versions.get('cudnn'):
        conflicts.append("CUDA toolkit found but cuDNN missing - required for GPU operations")
    
    # Check missing TensorFlow GPU
    if versions.get('tensorflow') and not versions.get('tensorflow-gpu') and versions.get('cudatoolkit'):
        conflicts.append("CPU TensorFlow with CUDA toolkit - may want tensorflow-gpu instead")
    
    return conflicts

def generate_fix_commands(versions: Dict[str, Optional[str]], target_tf_version: str = "2.18.0") -> List[str]:
    """Generate conda commands to fix the environment."""
    commands = []
    
    # Get target compatibility info
    tf_compat = TENSORFLOW_COMPATIBILITY.get(target_tf_version, TENSORFLOW_COMPATIBILITY["2.18.0"])
    
    # Start with base packages
    base_packages = [
        f"python=3.10",
        f"tensorflow-gpu={target_tf_version}",
        f"keras={tf_compat['keras']}",
        f"cudatoolkit={tf_compat['cuda']}",
        f"cudnn={tf_compat['cudnn']}",
        "numpy>=1.26,<2.0",  # Safe range
    ]
    
    # Add PyTorch if needed
    pytorch_version = versions.get('pytorch')
    if pytorch_version or input("\nInstall PyTorch? (y/n): ").lower().startswith('y'):
        base_packages.extend([
            "pytorch=2.3.0",
            "torchvision",
            "pytorch-cuda=11.8"
        ])
    
    # Generate installation commands
    commands.append("# Remove conflicting packages")
    commands.append("conda remove tensorflow tensorflow-gpu keras numpy pytorch --yes")
    
    commands.append("\n# Install compatible versions")
    conda_cmd = "conda install -c conda-forge -c pytorch " + " ".join(base_packages)
    commands.append(conda_cmd)
    
    commands.append("\n# Install additional ML packages")
    commands.append("conda install matplotlib seaborn scikit-learn pandas polars")
    
    commands.append("\n# Install MetaSpliceAI specific packages")
    pip_packages = [
        "spliceai==1.3.1",
        "shap>=0.46.0", 
        "xgboost>=3.0.1",
        "biopython>=1.83",
        "pybedtools>=0.9.1"
    ]
    commands.append(f"pip install {' '.join(pip_packages)}")
    
    return commands

def create_clean_environment_script():
    """Create a script to set up a clean GPU environment."""
    script_content = '''#!/bin/bash
# Clean GPU Environment Setup for MetaSpliceAI
set -e

echo "🚀 Creating clean GPU environment for MetaSpliceAI"

# Remove existing environment if it exists
conda env remove -n surveyor-gpu --yes 2>/dev/null || true

# Create new environment with specific versions
echo "📦 Creating conda environment..."
conda create -n surveyor-gpu python=3.10 --yes

# Activate environment
echo "🔧 Activating environment..."
conda activate surveyor-gpu

# Install GPU packages from conda-forge (more reliable)
echo "🎯 Installing TensorFlow GPU stack..."
conda install -c conda-forge -c pytorch \\
    tensorflow-gpu=2.18.0 \\
    keras=3.5.0 \\
    pytorch=2.3.0 \\
    torchvision \\
    pytorch-cuda=11.8 \\
    cudatoolkit=11.8 \\
    cudnn=8.6.0 \\
    numpy=1.26.4 \\
    --yes

# Install data science packages
echo "📊 Installing data science packages..."
conda install -c conda-forge \\
    matplotlib=3.9.2 \\
    seaborn=0.13.2 \\
    scikit-learn \\
    pandas=2.2.2 \\
    polars=1.31.0 \\
    pyarrow=14.0.2 \\
    --yes

# Install bioinformatics packages
echo "🧬 Installing bioinformatics packages..."
conda install -c bioconda bedtools=2.30.0 --yes

# Install remaining packages via pip
echo "🐍 Installing additional Python packages..."
pip install \\
    spliceai==1.3.1 \\
    shap==0.46.0 \\
    xgboost==3.0.1 \\
    biopython==1.83 \\
    pybedtools==0.9.1 \\
    gffutils==0.12 \\
    pyfaidx==0.8.1.3 \\
    h5py==3.12.1 \\
    rich==13.9.4 \\
    tqdm==4.67.1 \\
    openpyxl

# Test installation
echo "🧪 Testing installation..."
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} GPU available: {len(tf.config.list_physical_devices(\"GPU\"))}')"
python -c "import torch; print(f'PyTorch {torch.__version__} GPU available: {torch.cuda.is_available()}')"

echo "✅ Clean GPU environment setup complete!"
echo "To use: conda activate surveyor-gpu"
'''
    
    with open('setup_clean_gpu_env.sh', 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod('setup_clean_gpu_env.sh', 0o755)
    print("📄 Created setup_clean_gpu_env.sh script")

def main():
    """Main function to diagnose and fix ML dependencies."""
    print("🔧 MetaSpliceAI ML Dependencies Fixer")
    print("=" * 50)
    
    # Check current environment
    current_versions = check_current_environment()
    
    # Detect conflicts
    print("\n🔍 DETECTING CONFLICTS")
    print("=" * 50)
    conflicts = detect_conflicts(current_versions)
    
    if conflicts:
        print("❌ Found potential issues:")
        for i, conflict in enumerate(conflicts, 1):
            print(f"  {i}. {conflict}")
    else:
        print("✅ No obvious conflicts detected")
    
    # Check if TensorFlow can import and use GPU
    print("\n🧪 TESTING TENSORFLOW GPU")
    print("=" * 50)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU devices found: {len(gpus)}")
        
        if gpus:
            print("✅ TensorFlow can see GPU(s)")
            # Try a simple computation
            try:
                with tf.device(gpus[0].name):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    c = tf.matmul(a, b)
                print("✅ GPU computation test passed")
            except Exception as e:
                print(f"❌ GPU computation failed: {e}")
                conflicts.append("TensorFlow imports but cannot perform GPU computations")
        else:
            print("❌ No GPU devices found by TensorFlow")
            conflicts.append("TensorFlow cannot find GPU devices")
            
    except Exception as e:
        print(f"❌ TensorFlow import failed: {e}")
        conflicts.append("TensorFlow cannot be imported")
    
    # Show recommendations
    if conflicts:
        print(f"\n🔧 RECOMMENDED FIXES")
        print("=" * 50)
        
        print("Option 1: Fix current environment")
        fix_commands = generate_fix_commands(current_versions)
        for cmd in fix_commands:
            print(f"  {cmd}")
        
        print(f"\nOption 2: Create clean environment")
        create_clean_environment_script()
        print(f"  Run: bash setup_clean_gpu_env.sh")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"  For GPU issues, Option 2 (clean environment) is usually more reliable")
        print(f"  The script will create 'surveyor-gpu' environment with tested versions")
        
    else:
        print(f"\n🎉 ENVIRONMENT LOOKS GOOD")
        print("=" * 50)
        print("Your environment appears to be properly configured!")
        print("If you're still having issues, try running:")
        print("  python scripts/diagnose_gpu_environment.py")

if __name__ == "__main__":
    main() 