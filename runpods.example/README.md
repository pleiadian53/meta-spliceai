# RunPods Setup for meta-spliceai

**This is an EXAMPLE** - Copy and customize for your setup.

---

## ⚠️ Important: This Directory is NOT Tracked

The `runpods/` directory contains user-specific configuration and is excluded from git via `.gitignore`.

This `runpods.example/` directory provides templates you can copy.

---

## 🚀 Quick Setup

### Step 1: Copy Example Directory

```bash
cd ~/work/meta-spliceai
cp -r runpods.example runpods
```

### Step 2: Use Scripts

```bash
cd runpods/scripts
./runpod_ssh_manager.sh add meta-spliceai
```

**Enter when prompted**:
- Hostname: `ssh.runpods.io` (from RunPods dashboard)
- Port: `12345` (from RunPods dashboard)
- Nickname: `a40-48gb` (or your GPU type)
- SSH Key: Press Enter for default

**Result**: SSH alias `runpod-meta-spliceai-a40-48gb` created

### Step 3: Connect

```bash
ssh runpod-meta-spliceai-a40-48gb
```

---

## 📋 What's Included

### Scripts (Self-Contained)

```
runpods.example/scripts/
├── runpod_ssh_manager.sh      # SSH config manager
├── quick_pod_setup.sh          # Automated setup
└── test_runpod_manager.sh      # Test suite
```

### Environment Configuration

```
runpods.example/environment-runpods-minimal.yml  # Minimal conda env for pods
```

---

## 🔧 Execution Model

### LOCAL (Your Machine)

Scripts run on **your local machine**:
- `runpod_ssh_manager.sh` - Configures SSH
- `quick_pod_setup.sh` - Automated setup
- `test_runpod_manager.sh` - Tests

**Modifies**: `~/.ssh/config` on your machine

### POD (RunPods Instance)

After SSH'ing to pod:

```bash
# 1. Setup GitHub SSH (for cloning private repos and pushing)
# See: runpods.example/docs/GITHUB_SSH_SETUP.md for detailed guide
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github -N ""
cat ~/.ssh/id_ed25519_github.pub  # Add this to https://github.com/settings/ssh/new

# 2. Configure SSH and Git
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  StrictHostKeyChecking no
EOF

git config --global user.name "pleiadian53"
git config --global user.email "pleiadian53@users.noreply.github.com"

# 3. Clone and setup environment
cd /workspace
git clone git@github.com:pleiadian53/meta-spliceai.git
cd meta-spliceai
mamba env create -f environment-runpods-minimal.yml
mamba activate metaspliceai
```

---

## 🔒 Privacy

### Why runpods/ is NOT in Git

- ❌ User-specific paths
- ❌ SSH configuration history
- ❌ Personal workflow customizations

### What IS Shared

- ✅ `runpods.example/` - This template
- ✅ Public setup guides in `runpods/docs/`

---

## 📚 Usage Workflow

### 1. Initial Setup (Once)

```bash
cd ~/work/meta-spliceai
cp -r runpods.example runpods
cd runpods/scripts
```

### 2. Configure Pod Access (Per Pod)

```bash
./runpod_ssh_manager.sh add meta-spliceai
```

### 3. Connect

```bash
ssh runpod-meta-spliceai-a40-48gb
```

### 4. Setup Environment (On Pod)

```bash
cd /workspace
git clone git@github.com:pleiadian53/meta-spliceai.git
cd meta-spliceai
mamba env create -f environment-runpods-minimal.yml
mamba activate metaspliceai
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers einops accelerate
pip install -e .
```

### 5. Transfer Data (From Local)

```bash
rsync -avzP ~/work/meta-spliceai/data/ \
  runpod-meta-spliceai-a40-48gb:/workspace/data/
```

### 6. Start Training (On Pod)

```bash
tmux new -s training
cd /workspace/meta-spliceai
mamba activate metaspliceai
python train.py --config configs/meta_layer_training.yaml
```

---

## 💡 Tips

- **Use tmux**: Sessions survive disconnection
- **Monitor GPU**: `watch -n 1 nvidia-smi`
- **Check costs**: RunPods dashboard
- **Terminate when done**: Only pay for compute time

---

## 📖 Additional Documentation

For more details, see the comprehensive documentation in `runpods/docs/`:

- `START_HERE.md` - Complete overview
- `RUNPODS_COMPLETE_SETUP.md` - Detailed setup guide
- `RUNPOD_QUICK_REFERENCE.md` - Command cheat sheet
- `RUNPOD_SSH_MANAGER_GUIDE.md` - SSH manager details
- `MODEL_TRANSFER_GUIDE.md` - Transferring results back
- `RUNPODS_STORAGE_REQUIREMENTS.md` - Disk space planning

---

**Created**: January 28, 2026  
**Status**: Template - Copy to `runpods/` and customize
