# Installation Scripts

This directory contains scripts for installing and setting up MetaSpliceAI environments.

## 📁 **Directory Structure**

```
scripts/installation/
├── README.md                    # This file
├── migrate_conda_to_mamba.sh    # Migrate from conda to mamba environment
└── [future installation scripts]
```

## 🚀 **Available Scripts**

### **`migrate_conda_to_mamba.sh` - Conda to Mamba Migration**

**Purpose:** Automatically migrate from conda-managed to mamba-managed environment

**Features:**
- 🔄 **Automated migration** from conda to mamba
- 💾 **Environment backup** with timestamp
- 🛡️ **Safe environment removal** with confirmation
- ✅ **Automatic verification** after migration
- 🚀 **GPU testing integration** for GPU machines

**Usage:**
```bash
# Basic migration
./scripts/installation/migrate_conda_to_mamba.sh

# Custom environment name
./scripts/installation/migrate_conda_to_mamba.sh -e my-env-name

# Keep old environment (backup only)
./scripts/installation/migrate_conda_to_mamba.sh -k

# Show help
./scripts/installation/migrate_conda_to_mamba.sh -h
```

**What it does:**
1. 📋 Backs up current conda environment
2. ⚙️ Installs mamba (if not already installed)
3. 🗑️ Removes old conda environment (with confirmation)
4. 🏗️ Creates new mamba environment from `environment.yml`
5. ✅ Verifies new environment with comprehensive tests
6. 🚀 Tests GPU setup if GPU is available

## 🎯 **Use Cases**

### **For New Users:**
- Use `environment.yml` directly with mamba
- See main installation guide: `docs/installation/INSTALLATION.md`

### **For Existing Conda Users:**
- Use `migrate_conda_to_mamba.sh` for easy transition
- Benefits: faster package installation, better GPU support

### **For GPU Machines:**
- After migration, test GPU setup with `scripts/gpu_env_setup/`
- See GPU setup guide: `docs/gpu_environment_setup.md`

## 🔧 **Integration with Other Scripts**

### **Related Scripts:**
- **GPU Setup:** `scripts/gpu_env_setup/` - GPU environment setup and testing
- **Installation Testing:** `docs/installation/test_installation.sh` - Basic installation verification

### **Documentation:**
- **Main Installation Guide:** `docs/installation/INSTALLATION.md`
- **GPU Setup Guide:** `docs/gpu_environment_setup.md`

## 🤝 **Contributing**

To add new installation scripts:
1. Place them in this directory
2. Update this README.md
3. Add appropriate documentation
4. Test on both GPU and non-GPU environments
5. Update references in main installation guide

## 📊 **Migration Benefits**

| Aspect | Conda | Mamba | Improvement |
|--------|-------|-------|-------------|
| **Installation Speed** | 🐌 Slow | ⚡ Fast | 10x faster |
| **GPU Package Installation** | 🐌 Slow | 🚀 Fast | 5-10x faster |
| **Dependency Resolution** | ⚠️ Sometimes conflicts | 🛡️ Reliable | Fewer conflicts |
| **CUDA Compatibility** | ⚠️ Sometimes issues | ✅ Better | Improved compatibility |

---

**Note:** These scripts are designed to work on both GPU and non-GPU machines, providing appropriate feedback for each environment. 