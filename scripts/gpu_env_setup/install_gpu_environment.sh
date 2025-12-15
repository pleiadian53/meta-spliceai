#!/bin/bash

# GPU Environment Setup Script for MetaSpliceAI
# This script automates the installation of GPU-enabled environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect GPU setup
detect_gpu_setup() {
    print_status "Detecting GPU setup..."
    
    if command_exists nvidia-smi; then
        print_success "NVIDIA driver found"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1
        return 0
    else
        print_warning "NVIDIA driver not found - this is a CPU-only machine"
        return 1
    fi
}

# Function to check CUDA compatibility
check_cuda_compatibility() {
    print_status "Checking CUDA compatibility..."
    
    if command_exists nvidia-smi; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_success "CUDA Version: $cuda_version"
        
        # Check if CUDA version is supported
        if [[ "$cuda_version" == "12.2" || "$cuda_version" == "12.1" || "$cuda_version" == "11.8" ]]; then
            print_success "CUDA version $cuda_version is supported"
            return 0
        else
            print_warning "CUDA version $cuda_version may have compatibility issues"
            return 1
        fi
    else
        print_warning "Cannot check CUDA version - no NVIDIA driver"
        return 1
    fi
}

# Function to create GPU environment
create_gpu_environment() {
    local env_name=${1:-"surveyor-gpu"}
    local install_method=${2:-"basic"}
    
    print_status "Creating GPU environment: $env_name"
    
    if [[ "$install_method" == "complete" ]]; then
        print_status "Installing complete CUDA toolkit..."
        
        # Create environment with full CUDA toolkit
        mamba create -n "$env_name" python=3.10 bedtools poetry -c conda-forge -c bioconda -y
        
        # Activate environment
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate "$env_name"
        
        # Install full CUDA toolkit
        mamba install -c conda-forge cudatoolkit=12.1 cudnn=8.9 -y
        
        # Install GPU-enabled packages
        mamba install -c conda-forge -c pytorch -c nvidia \
            "pytorch=2.7.1" \
            "pytorch-cuda=12.1" \
            "tensorflow=2.19.0" \
            "xgboost=3.0.1" \
            "scikit-learn=1.7.0" \
            "pandas=2.3.1" \
            "polars=1.31.0" \
            "numpy=2.1.3" -y
        
    else
        print_status "Installing basic GPU support (runtime libraries only)..."
        
        # Create environment with basic GPU support
        mamba create -n "$env_name" python=3.10 bedtools poetry -c conda-forge -c bioconda -y
        
        # Activate environment
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate "$env_name"
        
        # Install GPU-enabled packages (runtime libraries only)
        mamba install -c conda-forge -c pytorch -c nvidia \
            "pytorch=2.7.1" \
            "pytorch-cuda=12.1" \
            "tensorflow=2.19.0" \
            "xgboost=3.0.1" \
            "scikit-learn=1.7.0" \
            "pandas=2.3.1" \
            "polars=1.31.0" \
            "numpy=2.1.3" -y
    fi
    
    print_success "GPU environment created successfully"
}

# Function to install MetaSpliceAI dependencies
install_meta_spliceai_deps() {
    local env_name=${1:-"surveyor-gpu"}
    
    print_status "Installing MetaSpliceAI dependencies..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$env_name"
    
    # Configure poetry to not create virtual environments
    poetry config virtualenvs.create false
    
    # Install MetaSpliceAI dependencies
    poetry install --no-deps
    
    print_success "MetaSpliceAI dependencies installed"
}

# Function to verify installation
verify_installation() {
    local env_name=${1:-"surveyor-gpu"}
    
    print_status "Verifying GPU installation..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$env_name"
    
    # Run basic GPU test
    if [[ -f "scripts/gpu_env_setup/test_gpu_installation.sh" ]]; then
        ./scripts/gpu_env_setup/test_gpu_installation.sh
    else
        print_warning "GPU test script not found, running basic verification..."
        
        # Basic verification
        python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
        python -c "import tensorflow as tf; print('TF GPUs:', len(tf.config.list_physical_devices('GPU')))"
        python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env-name NAME     Environment name (default: surveyor-gpu)"
    echo "  -m, --method METHOD     Installation method: basic|complete (default: basic)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Installation Methods:"
    echo "  basic                   Install runtime libraries only (recommended for MetaSpliceAI)"
    echo "  complete                Install full CUDA toolkit (for development work)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Basic GPU setup"
    echo "  $0 -e my-gpu-env        # Custom environment name"
    echo "  $0 -m complete          # Full CUDA toolkit installation"
}

# Main script
main() {
    local env_name="surveyor-gpu"
    local install_method="basic"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env-name)
                env_name="$2"
                shift 2
                ;;
            -m|--method)
                install_method="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate installation method
    if [[ "$install_method" != "basic" && "$install_method" != "complete" ]]; then
        print_error "Invalid installation method: $install_method"
        print_error "Use 'basic' or 'complete'"
        exit 1
    fi
    
    print_status "Starting GPU environment setup..."
    print_status "Environment name: $env_name"
    print_status "Installation method: $install_method"
    
    # Check if mamba is available
    if ! command_exists mamba; then
        print_error "mamba is not installed. Please install mamba first."
        print_error "Visit: https://github.com/conda-forge/miniforge"
        exit 1
    fi
    
    # Detect GPU setup
    if detect_gpu_setup; then
        check_cuda_compatibility
    else
        print_warning "Proceeding with CPU-only setup (GPU packages will be installed but not functional)"
    fi
    
    # Create GPU environment
    create_gpu_environment "$env_name" "$install_method"
    
    # Install MetaSpliceAI dependencies
    install_meta_spliceai_deps "$env_name"
    
    # Verify installation
    verify_installation "$env_name"
    
    print_success "GPU environment setup completed!"
    print_status "To activate the environment:"
    echo "  mamba activate $env_name"
    print_status "To test GPU performance:"
    echo "  python scripts/gpu_env_setup/test_gpu_performance.py"
}

# Run main function with all arguments
main "$@" 