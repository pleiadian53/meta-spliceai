#!/bin/bash

# Migration Script: Conda to Mamba for MetaSpliceAI
# This script helps migrate from conda-managed to mamba-managed environment

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

# Function to backup current environment
backup_environment() {
    local env_name=${1:-"surveyor"}
    
    print_status "Backing up current conda environment..."
    
    if conda env list | grep -q "^$env_name "; then
        backup_file="environment-conda-backup-$(date +%Y%m%d_%H%M%S).yml"
        conda env export --no-builds -n "$env_name" > "$backup_file"
        print_success "Environment backed up to: $backup_file"
        return 0
    else
        print_warning "Environment '$env_name' not found in conda"
        return 1
    fi
}

# Function to install mamba
install_mamba() {
    print_status "Installing mamba..."
    
    if command_exists mamba; then
        print_success "mamba is already installed"
        return 0
    fi
    
    # Check if miniforge is already installed
    if [[ -d "$HOME/miniforge3" ]]; then
        print_status "Miniforge found at $HOME/miniforge3"
        source "$HOME/miniforge3/bin/activate"
        return 0
    fi
    
    # Install miniforge
    print_status "Installing Miniforge (includes mamba)..."
    
    # Download and install miniforge
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3
    
    # Source the installation
    source $HOME/miniforge3/bin/activate
    
    print_success "mamba installed successfully"
}

# Function to remove old environment
remove_old_environment() {
    local env_name=${1:-"surveyor"}
    
    print_status "Removing old conda environment..."
    
    if conda env list | grep -q "^$env_name "; then
        print_warning "This will remove the existing '$env_name' environment"
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n "$env_name"
            print_success "Old environment removed"
        else
            print_warning "Skipping environment removal"
            return 1
        fi
    else
        print_status "No existing '$env_name' environment found"
    fi
}

# Function to create new mamba environment
create_mamba_environment() {
    local env_name=${1:-"surveyor"}
    
    print_status "Creating new mamba environment..."
    
    # Check if environment.yml exists
    if [[ ! -f "environment.yml" ]]; then
        print_error "environment.yml not found in current directory"
        print_error "Please run this script from the meta-spliceai root directory"
        exit 1
    fi
    
    # Create environment
    mamba env create -f environment.yml
    
    print_success "New mamba environment created"
}

# Function to verify new environment
verify_environment() {
    local env_name=${1:-"surveyor"}
    
    print_status "Verifying new environment..."
    
    # Activate environment
    mamba activate "$env_name"
    
    # Run basic tests
    print_status "Running installation tests..."
    if [[ -f "docs/installation/test_installation.sh" ]]; then
        ./docs/installation/test_installation.sh
    else
        print_warning "Installation test script not found"
    fi
    
    # Test GPU if available
    if command_exists nvidia-smi; then
        print_status "Testing GPU setup..."
        if [[ -f "scripts/gpu_env_setup/test_gpu_installation.sh" ]]; then
            ./scripts/gpu_env_setup/test_gpu_installation.sh
        else
            print_warning "GPU test script not found"
        fi
    else
        print_status "No GPU detected, skipping GPU tests"
    fi
    
    print_success "Environment verification completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env-name NAME     Environment name (default: surveyor)"
    echo "  -k, --keep-old         Keep the old conda environment"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "This script will:"
    echo "  1. Backup your current conda environment"
    echo "  2. Install mamba (if not already installed)"
    echo "  3. Remove the old conda environment (unless --keep-old)"
    echo "  4. Create a new mamba environment from environment.yml"
    echo "  5. Verify the new environment"
    echo ""
    echo "Examples:"
    echo "  $0                      # Migrate default 'surveyor' environment"
    echo "  $0 -e my-env           # Migrate custom environment name"
    echo "  $0 -k                  # Keep old environment (backup only)"
}

# Main script
main() {
    local env_name="surveyor"
    local keep_old=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env-name)
                env_name="$2"
                shift 2
                ;;
            -k|--keep-old)
                keep_old=true
                shift
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
    
    print_status "Starting conda to mamba migration..."
    print_status "Environment name: $env_name"
    print_status "Keep old environment: $keep_old"
    
    # Check if we're in the right directory
    if [[ ! -f "environment.yml" ]]; then
        print_error "environment.yml not found"
        print_error "Please run this script from the meta-spliceai root directory"
        exit 1
    fi
    
    # Step 1: Backup current environment
    backup_environment "$env_name"
    
    # Step 2: Install mamba
    install_mamba
    
    # Step 3: Remove old environment (unless --keep-old)
    if [[ "$keep_old" == "false" ]]; then
        remove_old_environment "$env_name"
    else
        print_status "Skipping environment removal (--keep-old specified)"
    fi
    
    # Step 4: Create new mamba environment
    create_mamba_environment "$env_name"
    
    # Step 5: Verify new environment
    verify_environment "$env_name"
    
    print_success "Migration completed successfully!"
    print_status "To activate your new environment:"
    echo "  mamba activate $env_name"
    print_status "To test GPU performance:"
    echo "  python scripts/gpu_env_setup/test_gpu_performance.py"
}

# Run main function with all arguments
main "$@" 