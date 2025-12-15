#!/bin/bash

# MetaSpliceAI Installation Test Script
# This script verifies that all components are properly installed and working

set -e

echo "🧪 MetaSpliceAI Installation Test"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}❌ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    elif [ "$status" = "INFO" ]; then
        echo -e "${BLUE}ℹ️  $message${NC}"
    fi
}

# Function to test command existence
test_command() {
    local cmd=$1
    local name=$2
    if command -v "$cmd" >/dev/null 2>&1; then
        print_status "PASS" "$name is installed"
        return 0
    else
        print_status "FAIL" "$name is not installed"
        return 1
    fi
}

# Function to test Python import
test_python_import() {
    local module=$1
    local name=$2
    if python -c "import $module" 2>/dev/null; then
        print_status "PASS" "$name can be imported"
        return 0
    else
        print_status "FAIL" "$name cannot be imported"
        return 1
    fi
}

# Function to get Python package version
get_package_version() {
    local package=$1
    python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown"
}

echo "📋 System Information"
echo "===================="
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"
echo "Python: $(python --version 2>&1 || echo 'not found')"
echo ""

# Test 1: Environment Management Tools
echo "🔧 Environment Management Tools"
echo "=============================="

test_command "conda" "Conda"
test_command "mamba" "Mamba"
test_command "poetry" "Poetry"

# Test 2: System Dependencies
echo ""
echo "🧬 System Dependencies"
echo "====================="

test_command "bedtools" "Bedtools"
if command -v bedtools >/dev/null 2>&1; then
    echo "  Version: $(bedtools --version 2>&1 | head -1)"
fi

# Test 3: Python Environment
echo ""
echo "🐍 Python Environment"
echo "==================="

if [ -n "$CONDA_DEFAULT_ENV" ]; then
    print_status "PASS" "Conda environment active: $CONDA_DEFAULT_ENV"
else
    print_status "WARN" "No conda environment detected"
fi

# Test 4: Core Python Packages
echo ""
echo "📦 Core Python Packages"
echo "======================"

# Test essential packages
packages=(
    "numpy:NumPy"
    "pandas:Pandas"
    "tensorflow:TensorFlow"
    "torch:PyTorch"
    "keras:Keras"
    "sklearn:scikit-learn"
    "scipy:SciPy"
    "matplotlib:Matplotlib"
    "seaborn:Seaborn"
    "polars:Polars"
    "pyarrow:PyArrow"
    "numba:Numba"
    "shap:SHAP"
    "xgboost:XGBoost"
    "Bio:Biopython"
    "gffutils:gffutils"
    "pybedtools:pybedtools"
    "transformers:HuggingFace Transformers"
    "tokenizers:HuggingFace Tokenizers"
    "accelerate:HuggingFace Accelerate"
    "captum:Captum (Interpretability)"
    "mlflow:MLflow"
    "spliceai:SpliceAI"
    "pyfaidx:PyFaidx"
    "rich:Rich"
    "tqdm:TQDM"
    "h5py:HDF5 Python"
)

all_packages_ok=true
for package_info in "${packages[@]}"; do
    IFS=':' read -r module name <<< "$package_info"
    if test_python_import "$module" "$name"; then
        version=$(get_package_version "$module")
        echo "  Version: $version"
    else
        all_packages_ok=false
    fi
done

# Test 5: MetaSpliceAI Package
echo ""
echo "🎯 MetaSpliceAI Package"
echo "========================="

if test_python_import "meta_spliceai" "MetaSpliceAI"; then
    version=$(python -c "import meta_spliceai; print(getattr(meta_spliceai, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "  Version: $version"
else
    all_packages_ok=false
fi

# Test 6: GPU Support (if available)
echo ""
echo "🎮 GPU Support"
echo "============="

if command -v nvidia-smi >/dev/null 2>&1; then
    # Get GPU information
    gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$gpu_info" ]; then
        gpu_count=$(echo "$gpu_info" | wc -l)
        print_status "PASS" "$gpu_count NVIDIA GPU(s) detected"
        
        # Show detailed GPU information
        echo "  GPU Details:"
        while IFS=',' read -r index name memory; do
            index=$(echo "$index" | xargs)
            name=$(echo "$name" | xargs)  
            memory=$(echo "$memory" | xargs)
            echo "    GPU $index: $name ($memory MB)"
        done <<< "$gpu_info"
        
        # Show CUDA and driver version
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null)
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1 2>/dev/null)
        if [ -n "$cuda_version" ]; then
            echo "  CUDA Version: $cuda_version"
        fi
        if [ -n "$driver_version" ]; then
            echo "  Driver Version: $driver_version"
        fi
        
        # Test TensorFlow GPU
        if python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null; then
            tf_gpu_count=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null)
            if [ "$tf_gpu_count" -gt 0 ]; then
                print_status "PASS" "TensorFlow GPU support: $tf_gpu_count GPU(s) detected"
            else
                print_status "WARN" "TensorFlow GPU support: No GPUs detected"
            fi
        else
            print_status "FAIL" "TensorFlow GPU support: Cannot import TensorFlow"
        fi
        
        # Test PyTorch GPU
        if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null; then
            if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                torch_gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
                print_status "PASS" "PyTorch GPU support: $torch_gpu_count GPU(s) detected"
                
                # Test XGBoost GPU support
                if python -c "import xgboost as xgb; import numpy as np; X=np.random.rand(100,10); y=np.random.randint(2,size=100); xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1).fit(X,y)" 2>/dev/null; then
                    print_status "PASS" "XGBoost GPU support: GPU training successful"
                else
                    print_status "WARN" "XGBoost GPU support: GPU training failed"
                fi
            else
                print_status "WARN" "PyTorch GPU support: No GPUs detected"
            fi
        else
            print_status "FAIL" "PyTorch GPU support: Cannot import PyTorch"
        fi
    else
        print_status "INFO" "nvidia-smi command failed or no GPUs found"
    fi
else
    print_status "INFO" "No NVIDIA GPU detected (CPU-only system)"
fi

# Test 7: Memory and Disk Space
echo ""
echo "💾 System Resources"
echo "=================="

# Check available memory
if command -v free >/dev/null 2>&1; then
    total_mem=$(free -g | grep '^Mem:' | awk '{print $2}')
    available_mem=$(free -g | grep '^Mem:' | awk '{print $7}')
    echo "  Total Memory: ${total_mem}GB"
    echo "  Available Memory: ${available_mem}GB"
    
    if [ "$total_mem" -ge 8 ]; then
        print_status "PASS" "Sufficient memory for MetaSpliceAI"
    else
        print_status "WARN" "Low memory detected (8GB+ recommended)"
    fi
fi

# Check disk space
if command -v df >/dev/null 2>&1; then
    available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    echo "  Available Disk Space: ${available_space}GB"
    
    if [ "$available_space" -ge 10 ]; then
        print_status "PASS" "Sufficient disk space"
    else
        print_status "WARN" "Low disk space detected (10GB+ recommended)"
    fi
fi

# Test 8: Optional Dependencies
echo ""
echo "🔧 Optional Dependencies"
echo "======================="

# Test optional packages (don't fail if missing)
optional_packages=(
    "pyspark:PySpark (distributed processing)"
    "pyBigWig:pyBigWig (BigWig file handling)"
    "wandb:Weights & Biases (experiment tracking)"
)

for package_info in "${optional_packages[@]}"; do
    IFS=':' read -r module name <<< "$package_info"
    if test_python_import "$module" "$name"; then
        version=$(get_package_version "$module")
        echo "  Version: $version"
    else
        print_status "INFO" "$name not installed (optional)"
    fi
done

# Test 9: Command Line Tools
echo ""
echo "🛠️  Command Line Tools"
echo "====================="

# Test if splice surveyor commands work
if python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --help >/dev/null 2>&1; then
    print_status "PASS" "Incremental builder command available"
else
    print_status "FAIL" "Incremental builder command not available"
    all_packages_ok=false
fi

if python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid --help >/dev/null 2>&1; then
    print_status "PASS" "Training command available"
else
    print_status "FAIL" "Training command not available"
    all_packages_ok=false
fi

# Test error model workflow
if python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow --help >/dev/null 2>&1; then
    print_status "PASS" "Error model workflow command available"
else
    print_status "FAIL" "Error model workflow command not available"
    all_packages_ok=false
fi

# Summary
echo ""
echo "📊 Test Summary"
echo "=============="

if [ "$all_packages_ok" = true ]; then
    print_status "PASS" "All core components are working!"
    echo ""
    echo "🎉 Installation appears to be successful!"
    echo "You can now run MetaSpliceAI analyses."
    echo ""
    echo "Next steps:"
    echo "1. Read the documentation in docs/installation/"
    echo "2. Try running a sample analysis"
    echo "3. Check the GPU environment setup if using GPU"
    exit 0
else
    print_status "FAIL" "Some components are missing or not working"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "1. Check the installation guide in docs/installation/"
    echo "2. Verify your conda/mamba environment is activated"
    echo "3. Try reinstalling problematic packages"
    echo "4. Check system requirements"
    exit 1
fi 