# Meta-SpliceAI RunPods Setup Scripts

This directory contains utilities for setting up and managing RunPods instances for GPU training.

---

## 🚀 Quick Start

### First Time Setup

```bash
# 1. Test the SSH manager
./test_runpod_manager.sh

# 2. Add a new pod configuration
./runpod_ssh_manager.sh add meta-spliceai

# 3. Connect to the pod
ssh runpod-meta-spliceai-a40-100gb
```

### For Other Projects

```bash
# Install across all projects
cd ~/work/scripts
./install_runpod_manager.sh

# Then use from any project
cd ~/work/genai-lab
./scripts/runpod_manager.sh add genai-lab
```

---

## 📁 Files in This Directory

### Main Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `runpod_ssh_manager.sh` | SSH config manager | `./runpod_ssh_manager.sh [command]` |
| `test_runpod_manager.sh` | Test suite | `./test_runpod_manager.sh` |

### Documentation

| File | Description |
|------|-------------|
| `RUNPOD_SSH_MANAGER_GUIDE.md` | Complete guide with examples |
| `RUNPOD_QUICK_REFERENCE.md` | Single-page cheat sheet |
| `RUNPODS_COMPLETE_SETUP.md` | Full pod setup workflow |
| `MODEL_TRANSFER_GUIDE.md` | Transferring trained models |
| `RUNPODS_DISK_CONFIGURATION.md` | Disk space management |
| `RUNPODS_STORAGE_REQUIREMENTS.md` | Storage requirements |
| `README.md` | This file |

---

## 🎯 Common Commands

### SSH Configuration

```bash
# Add new pod
./runpod_ssh_manager.sh add meta-spliceai

# List all pods
./runpod_ssh_manager.sh list

# Remove pod
./runpod_ssh_manager.sh remove

# Show history
./runpod_ssh_manager.sh history

# Interactive menu
./runpod_ssh_manager.sh
```

### Backup & Restore

```bash
# List backups
./runpod_ssh_manager.sh backups

# Restore from backup
./runpod_ssh_manager.sh restore
```

### Testing

```bash
# Run test suite
./test_runpod_manager.sh
```

---

## 📚 Documentation Quick Links

### For First-Time Users

1. **Start Here**: [RUNPODS_COMPLETE_SETUP.md](../../meta_spliceai/splice_engine/meta_layer/docs/setup/RUNPODS_COMPLETE_SETUP.md)
   - Complete walkthrough of pod setup
   - Environment installation
   - Data transfer

2. **SSH Manager**: [RUNPOD_SSH_MANAGER_GUIDE.md](./RUNPOD_SSH_MANAGER_GUIDE.md)
   - SSH config management
   - Multi-project setup
   - Advanced features

3. **Quick Reference**: [RUNPOD_QUICK_REFERENCE.md](./RUNPOD_QUICK_REFERENCE.md)
   - One-page cheat sheet
   - Common workflows
   - Troubleshooting

### For Experienced Users

- **Model Transfer**: [MODEL_TRANSFER_GUIDE.md](../../meta_spliceai/splice_engine/meta_layer/docs/setup/MODEL_TRANSFER_GUIDE.md)
- **Storage Issues**: [RUNPODS_STORAGE_REQUIREMENTS.md](../../meta_spliceai/splice_engine/meta_layer/docs/setup/RUNPODS_STORAGE_REQUIREMENTS.md)

---

## 🔧 Installation Across Projects

To use the SSH manager from any project in your workspace:

```bash
cd ~/work/scripts
./install_runpod_manager.sh
```

This creates symlinks in:
- `genai-lab/scripts/`
- `causal-bio-lab/scripts/`
- `ehr-sequencing/scripts/`
- `cf-ensemble/scripts/`
- `loinc-predictor/scripts/`
- `biographlab/scripts/`

---

## 🌐 Multi-Project Workflow

### Example: Using Multiple Pods

```bash
# Meta-SpliceAI pod
cd ~/work/meta-spliceai
./scripts/setup/runpod_ssh_manager.sh add meta-spliceai
ssh runpod-meta-spliceai-a40

# GenAI Lab pod
cd ~/work/genai-lab
./scripts/runpod_manager.sh add genai-lab
ssh runpod-genai-rtx4090

# All configs are in ~/.ssh/config
# List from anywhere
~/work/scripts/runpod_manager.sh list
```

---

## 🎓 Tutorial: Complete Pod Setup

### Step 1: Get a RunPods Instance

1. Go to [RunPods](https://www.runpod.io)
2. Select GPU (A40, RTX 4090, A100, etc.)
3. Choose disk size (minimum 50GB, recommended 100GB)
4. Deploy

### Step 2: Configure SSH

```bash
cd ~/work/meta-spliceai
./scripts/setup/runpod_ssh_manager.sh add meta-spliceai
```

Enter details from RunPods dashboard:
- **Hostname**: From "SSH over exposed TCP"
- **Port**: From "SSH over exposed TCP"
- **Nickname**: `a40-100gb` (or your GPU/disk)
- **SSH Key**: `~/.ssh/id_ed25519`

### Step 3: Connect and Setup

```bash
# Connect
ssh runpod-meta-spliceai-a40-100gb

# Install Miniforge
cd /workspace
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
/workspace/miniforge3/bin/conda init bash
source ~/.bashrc

# Clone and setup
git clone https://github.com/pleiadian53/meta-spliceai.git
cd meta-spliceai
mamba env create -f environment-runpods-minimal.yml
mamba activate metaspliceai
pip install -e .
```

### Step 4: Transfer Data

```bash
# From local machine
rsync -avzP ~/work/meta-spliceai/data/ runpod-meta-spliceai-a40-100gb:/workspace/meta-spliceai/data/
```

### Step 5: Start Training

```bash
# On pod
ssh runpod-meta-spliceai-a40-100gb
tmux new -s training
cd /workspace/meta-spliceai
mamba activate metaspliceai
python train.py

# Detach: Ctrl+B, D
```

### Step 6: Clean Up When Done

```bash
# From local machine
./scripts/setup/runpod_ssh_manager.sh remove
# Select pod to remove
```

---

## 🔍 Troubleshooting

### Script Not Executable

```bash
chmod +x ./runpod_ssh_manager.sh
chmod +x ./test_runpod_manager.sh
```

### Connection Issues

```bash
# Verify pod is running
# Check hostname/port from dashboard
# Test manually
ssh -v runpod-meta-spliceai-a40-100gb
```

### Update Pod Details

```bash
# Just run add again with same project name
./runpod_ssh_manager.sh add meta-spliceai
# Choose "y" to update
```

### Restore Broken Config

```bash
./runpod_ssh_manager.sh restore
```

---

## 🔐 Security Notes

The SSH manager uses:
- `StrictHostKeyChecking=no`: Safe for ephemeral RunPods instances
- `UserKnownHostsFile=/dev/null`: Pods change frequently
- Automatic backups before any changes
- No credentials stored (uses SSH keys)

---

## 📦 File Locations

| Item | Location |
|------|----------|
| SSH Config | `~/.ssh/config` |
| Backups | `~/.ssh/config_backups/` |
| History | `~/.ssh/runpod_history.json` |
| Manager | `~/work/meta-spliceai/scripts/setup/runpod_ssh_manager.sh` |
| Universal | `~/work/scripts/runpod_manager.sh` |

---

## 🚀 Next Steps

After setup:

1. **Start Training**: See [GPU_TRAINING_GUIDE.md](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/GPU_TRAINING_GUIDE.md)
2. **Transfer Models**: See [MODEL_TRANSFER_GUIDE.md](../../meta_spliceai/splice_engine/meta_layer/docs/setup/MODEL_TRANSFER_GUIDE.md)
3. **Monitor Progress**: Use tmux sessions
4. **Backup Results**: Regular rsync from pod

---

## 📞 Support

For issues:
1. Check [RUNPOD_QUICK_REFERENCE.md](./RUNPOD_QUICK_REFERENCE.md)
2. Run `./test_runpod_manager.sh`
3. Check `./runpod_ssh_manager.sh history`
4. Contact development team

---

**Last Updated**: January 27, 2026  
**Version**: 1.0.0
