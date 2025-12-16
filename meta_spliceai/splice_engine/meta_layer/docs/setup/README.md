# Setup Documentation

This directory contains guides for setting up different compute environments for meta-layer training.

## ⚠️ IMPORTANT: Storage Requirements

**RunPods instances need 50GB+ Volume Disk!** The default 20GB is insufficient.

See [RUNPODS_STORAGE_REQUIREMENTS.md](./RUNPODS_STORAGE_REQUIREMENTS.md) for details.

## Available Guides

| Guide | Purpose | Audience |
|-------|---------|----------|
| [RUNPODS_COMPLETE_SETUP.md](./RUNPODS_COMPLETE_SETUP.md) | Complete RunPods VM setup including SSH, environment, data, and GitHub access | AI agents & developers |
| [RUNPODS_DISK_CONFIGURATION.md](./RUNPODS_DISK_CONFIGURATION.md) | **⭐ How to configure Volume Disk size (step-by-step)** | All |
| [RUNPODS_STORAGE_REQUIREMENTS.md](./RUNPODS_STORAGE_REQUIREMENTS.md) | Storage requirements and quota troubleshooting | All |

## Quick Links

### For RunPods GPU Training

1. **Initial Setup**: [RUNPODS_COMPLETE_SETUP.md](./RUNPODS_COMPLETE_SETUP.md)
2. **Data Transfer**: [DATA_TRANSFER_GUIDE.md](../experiments/DATA_TRANSFER_GUIDE.md)
3. **Training Guide**: [GPU_TRAINING_GUIDE.md](../experiments/GPU_TRAINING_GUIDE.md)
4. **Experiment Queue**: [GPU_EXPERIMENTS.md](../wishlist/GPU_EXPERIMENTS.md)

### Setup Checklist

```
□ SSH access configured
□ Environment installed (Miniforge + metaspliceai)
□ Data transferred (SpliceVarDB + FASTA)
□ GitHub push access configured
□ Verification script passes
```

---

## Environment Files Reference

| File | Platform | Use Case |
|------|----------|----------|
| `environment.yml` | All | Base environment (may have platform-specific issues) |
| `environment-linux.yml` | Linux | Linux-specific (no macOS packages) |
| `environment-runpods-minimal.yml` | RunPods | Minimal deps for GPU training |

---

*For local M1 Mac development, see the main project README.*

