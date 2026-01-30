# Quick Start Guide - Meta-SpliceAI Scripts

**Fast reference for common tasks** - Bookmark this page!

---

## 🎯 Most Common Tasks

### 1. Train a Model

#### Full Genome Training
```bash
cd ~/work/meta-spliceai/scripts/training
./start_full_genome_run.sh
```

#### Single Chromosome (for testing)
```bash
cd ~/work/meta-spliceai/scripts/training
./run_single_chromosome.sh 21  # Chromosome 21 is small, good for testing
```

#### Multi-GPU Training
```bash
cd ~/work/meta-spliceai/scripts
python run_multi_gpu_training.py --config configs/multi_gpu.yaml
```

#### Monitor Training
```bash
cd ~/work/meta-spliceai/scripts/monitoring
./monitor_meta_training.sh
```

---

### 2. Test/Validate

#### Complete Pipeline Test
```bash
cd ~/work/meta-spliceai/scripts/testing
./test_complete_pipeline.sh
```

#### Test Inference
```bash
cd ~/work/meta-spliceai/scripts/testing
python test_inference_modes.py
```

#### Validate Environment
```bash
cd ~/work/meta-spliceai
python verify_setup.py
```

---

### 3. Work with RunPods

**All RunPods utilities are now centralized!**

```bash
cd ~/work/meta-spliceai/runpods

# First time setup
cat START_HERE.md

# Quick pod setup
./scripts/quick_pod_setup.sh meta-spliceai

# SSH management
./scripts/runpod_ssh_manager.sh add meta-spliceai
./scripts/runpod_ssh_manager.sh list

# Connect
ssh runpod-meta-spliceai-a40-100gb
```

**See**: [../runpods/README.md](../runpods/README.md) for complete documentation

---

### 4. Analyze Data

#### Quick Splice Site Analysis
```bash
cd ~/work/meta-spliceai/scripts/data
./quick_splice_analysis.sh
```

#### Gene Patterns
```bash
cd ~/work/meta-spliceai/scripts/data
./analyze_gene_patterns.sh
```

#### Training Data Analysis
```bash
cd ~/work/meta-spliceai/scripts
python analyze_training_data.py --dataset path/to/dataset
```

---

### 5. Setup Environment

#### GPU Environment
```bash
cd ~/work/meta-spliceai/scripts/gpu_env_setup
./install_gpu_environment.sh
```

#### Verify GPU
```bash
cd ~/work/meta-spliceai/scripts/gpu_env_setup
python verify_gpu_setup.py
```

#### Jupyter Notebook (Remote)
```bash
cd ~/work/meta-spliceai/scripts/notebook
./setup_jupyter_remote.sh
```

---

## 🔥 Quick Commands

### Check Status

```bash
# GPU status
nvidia-smi

# Environment status
cd ~/work/meta-spliceai
python verify_setup.py

# Package versions
cd ~/work/meta-spliceai/scripts
python check_versions.py
```

### Data Transfer (RunPods)

```bash
# Upload to pod
rsync -avzP ~/work/meta-spliceai/data/ \
    runpod-meta-spliceai-a40:/workspace/data/

# Download from pod
rsync -avzP runpod-meta-spliceai-a40:/workspace/results/ \
    ~/work/meta-spliceai/results/
```

### Monitoring

```bash
# Watch training
watch -n 60 'tail -100 logs/training.log'

# Monitor GPU
watch -n 5 nvidia-smi

# Check disk space
df -h
```

---

## 📂 Where Things Are

### By Task

| I want to... | Go to... |
|--------------|----------|
| Train models | `scripts/training/` |
| Run tests | `scripts/testing/` |
| Analyze data | `scripts/data/` or `scripts/analysis/` |
| Setup GPU | `scripts/gpu_env_setup/` |
| Work with RunPods | `../runpods/` |
| Monitor training | `scripts/monitoring/` |
| Clean up | `scripts/maintenance/` |
| Evaluate models | `scripts/evaluation/` |

### Key Files

```bash
# Main README
scripts/README.md

# This guide
scripts/QUICK_START.md

# Reorganization plan
scripts/REORGANIZATION_PLAN.md

# RunPods
runpods/START_HERE.md
runpods/README.md

# Project
README.md
GETTING_STARTED.md
PROJECT_SUMMARY.md
```

---

## 💡 Workflows

### Workflow 1: Local Development & Testing

```bash
# 1. Setup environment
cd ~/work/meta-spliceai
python verify_setup.py

# 2. Run quick test
cd scripts/testing
python test_inference_modes.py

# 3. Analyze results
cd ../data
./quick_splice_analysis.sh
```

### Workflow 2: RunPods Training

```bash
# 1. Setup pod
cd ~/work/meta-spliceai/runpods
./scripts/quick_pod_setup.sh meta-spliceai

# 2. Transfer data
rsync -avzP ~/work/meta-spliceai/data/ \
    runpod-meta-spliceai-a40:/workspace/data/

# 3. Connect and train
ssh runpod-meta-spliceai-a40
cd /workspace/meta-spliceai
mamba activate metaspliceai
./scripts/training/start_full_genome_run.sh

# 4. Monitor (from local, in another terminal)
ssh runpod-meta-spliceai-a40 'tail -f logs/training.log'

# 5. Download results
rsync -avzP runpod-meta-spliceai-a40:/workspace/results/ ./results/
```

### Workflow 3: Full Pipeline

```bash
# 1. Verify setup
python verify_setup.py

# 2. Prepare data
cd scripts/data
./quick_splice_analysis.sh

# 3. Train model
cd ../training
./run_single_chromosome.sh 21

# 4. Evaluate
cd ../evaluation
python evaluate_model.py --checkpoint latest

# 5. Test inference
cd ../testing
python test_inference_modes.py
```

---

## 🆘 Troubleshooting

### Common Issues

#### GPU Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Diagnose
cd ~/work/meta-spliceai/scripts
python diagnose_gpu_environment.py
```

#### Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Use smaller batch size
# Edit training script or config file

# Or use memory optimization
cd scripts/scaling_solutions/memory_optimization
./flexible_build.sh
```

#### Can't Find Script
```bash
# Search by name
find ~/work/meta-spliceai/scripts -name "*pattern*"

# Search by content
grep -r "keyword" ~/work/meta-spliceai/scripts
```

#### RunPods Connection Issues
```bash
# List pods
cd ~/work/meta-spliceai/runpods
./scripts/runpod_ssh_manager.sh list

# Test connection
ssh -v runpod-meta-spliceai-a40

# Update config
./scripts/runpod_ssh_manager.sh add meta-spliceai
# Choose 'y' to update
```

---

## 🚀 Pro Tips

### Speed Up Workflows

```bash
# Alias for common directories
alias cdms='cd ~/work/meta-spliceai'
alias cdscripts='cd ~/work/meta-spliceai/scripts'
alias cdrunpods='cd ~/work/meta-spliceai/runpods'

# Alias for common commands
alias gpu='watch -n 5 nvidia-smi'
alias runpod-connect='ssh runpod-meta-spliceai-a40'
```

### Use tmux on RunPods

```bash
# Start tmux session
ssh runpod-meta-spliceai-a40 -t "tmux new -s train"

# Detach: Ctrl+B, D

# Reattach later
ssh runpod-meta-spliceai-a40 -t "tmux attach -t train"

# Kill session
ssh runpod-meta-spliceai-a40 -t "tmux kill-session -t train"
```

### Monitor Multiple Things

```bash
# Split terminal (use tmux or screen)
# Window 1: Training log
tail -f logs/training.log

# Window 2: GPU utilization
watch -n 5 nvidia-smi

# Window 3: Disk space
watch -n 60 df -h
```

---

## 📚 Learn More

### Documentation

- **scripts/README.md** - Complete directory guide
- **scripts/REORGANIZATION_PLAN.md** - Future improvements
- **runpods/START_HERE.md** - RunPods complete guide
- **GETTING_STARTED.md** - Project setup
- **PROJECT_SUMMARY.md** - Project overview

### Subdirectory READMEs

- `scripts/training/README.md` (if exists)
- `scripts/testing/README.md` (if exists)
- `scripts/evaluation/README.md` (if exists)
- `runpods/docs/` - Detailed RunPods documentation

---

## 🔖 Bookmark These Commands

```bash
# Navigate to project
cd ~/work/meta-spliceai

# Verify setup
python verify_setup.py

# Check GPU
nvidia-smi

# Scripts directory
cd scripts

# RunPods utilities
cd runpods

# Quick reference
cat scripts/QUICK_START.md

# Complete guide
cat scripts/README.md
```

---

## ✨ What's Next?

1. **Explore**: Browse `scripts/README.md` for complete inventory
2. **Learn**: Read subdirectory READMEs
3. **Practice**: Try the workflows above
4. **Customize**: Add your own aliases and shortcuts
5. **Contribute**: Improve documentation as you learn

---

**Last Updated**: January 27, 2026  
**For detailed documentation**: See [README.md](./README.md)

Happy coding! 🚀
