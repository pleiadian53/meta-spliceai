# RunPods Disk Configuration Guide

**Created**: December 16, 2025  
**Verified**: Against RunPods official documentation  
**Purpose**: Step-by-step guide to configure sufficient storage on RunPods

---

## ✅ Verification Summary

The instructions provided are **ACCURATE**. Here's the verified information:

| Item | Status | Notes |
|------|--------|-------|
| Volume Disk location | ✅ Verified | Found in "Customize Deployment" section |
| Container vs Volume Disk | ✅ Verified | Volume = /workspace (persistent) |
| Pricing structure | ✅ Verified | See corrected rates below |

---

## 📊 Storage Pricing (Official - December 2025)

### Pod Storage (Volume Disk)

| State | Cost | Example (200GB) |
|-------|------|-----------------|
| **Running** Pod | $0.10/GB/month | $20/month |
| **Stopped** Pod | $0.20/GB/month | $40/month |

⚠️ **Important**: Stopped pods cost **2x** for storage! If you're not using the pod, consider using Network Volumes instead.

### Network Volumes (Alternative)

| Size | Cost | Example (200GB) |
|------|------|-----------------|
| Under 1TB | $0.07/GB/month | $14/month |
| Over 1TB | $0.05/GB/month | $50/month for 1TB |

**Network Volume Advantages**:
- Persists even if you delete the Pod
- Can be attached to different Pods
- Lower cost for stopped storage
- Shareable across multiple Pods

---

## 🖥️ Step-by-Step: Configure Volume Disk

### Step 1: Select Your GPU

From the RunPods dashboard (as shown in your screenshot):

1. Go to **Pods** → **Deploy a Pod**
2. Select your GPU (e.g., **A40** at $0.40/hr with 48GB VRAM)
3. Click **Deploy** on your chosen GPU card

### Step 2: Find the Storage Configuration

After clicking Deploy, you'll see the **Deployment Configuration** page:

1. Scroll down to **"Customize Deployment"** section
2. Look for **"Storage"** or **"Volume"** settings
3. You should see **two sliders**:
   - **Container Disk**: Default ~20-50GB (temporary, OS/libraries)
   - **Volume Disk**: This is `/workspace` (persistent) ← **INCREASE THIS ONE**

### Step 3: Set Volume Disk Size

**Recommended sizes for Meta-SpliceAI GPU training:**

| Use Case | Recommended Size | Monthly Cost (Running) |
|----------|-----------------|------------------------|
| Basic experiments | 100GB | $10/month |
| **Standard (Recommended)** | **200GB** | **$20/month** |
| Extensive training + checkpoints | 500GB | $50/month |
| Multiple large datasets | 1TB | $100/month |

**For Meta-SpliceAI, 200GB provides:**
- Miniforge + environment (~15GB)
- PyTorch CUDA (~3GB)
- Meta-SpliceAI repo (~3GB)
- FASTA reference (~3GB)
- SpliceVarDB (~7MB)
- Model checkpoints (~500MB each × many)
- Plenty of headroom for experiments

### Step 4: Review and Deploy

1. Check the updated cost estimate (should include storage)
2. Click **"Continue"** or **"Deploy"**
3. Wait for the Pod to initialize

---

## 🔄 Modifying Storage on Existing Pod

If you already have a Pod and need more space:

### Option A: Edit Pod (⚠️ Data Loss Risk)

1. Go to **Pods** in the RunPods console
2. Click the **three dots (⋮)** next to your Pod
3. Select **"Edit Pod"**
4. Increase the **Volume Disk** size
5. Click **Save**

⚠️ **WARNING**: Editing a running Pod **will reset it**, erasing all data NOT in `/workspace`. Since `/workspace` IS the volume disk, your data should be safe, but always back up important files first.

### Option B: Create Network Volume (Better for Persistence)

1. Go to **Storage** → **New Network Volume**
2. Select datacenter (same region as your Pod)
3. Set name: `metaspliceai-data`
4. Set size: 200GB or more
5. Click **Create**

Then attach to your Pod:
1. Edit Pod or create new Pod
2. Under storage, select your Network Volume
3. It will mount at `/workspace` or a custom path

---

## 💡 Best Practices

### For Short-Term Experiments

Use **Volume Disk** (simple, attached to Pod):
- 200GB Volume Disk
- Stop Pod when not using (but remember 2x storage cost when stopped)

### For Long-Term Projects

Use **Network Volume** (persistent, transferable):
- Create 500GB Network Volume
- Attach to Pods as needed
- Data persists even if you delete the Pod
- Lower cost when idle

### Data Backup Strategy

```bash
# Backup important files to your local machine periodically:
rsync -avz --progress \
    -e "ssh -p PORT -i ~/.ssh/id_ed25519" \
    root@HOST:/workspace/meta-spliceai/models/ \
    ~/backups/runpods/models/

# Or sync to cloud storage:
# (Install rclone on RunPods)
rclone sync /workspace/meta-spliceai/models/ remote:metaspliceai-backups/
```

---

## 📋 Quick Reference

### Verify Your Quota

```bash
# Check actual usage (NOT the misleading df -h output)
du -sh /workspace/*

# Test write capacity
dd if=/dev/zero of=/workspace/test_write bs=1M count=100 && rm /workspace/test_write
```

### Check Your Pod's Configuration

```bash
# See mounted volumes
df -h | grep workspace

# Check environment size
du -sh /workspace/miniforge3
```

---

## 🎯 Recommended Configuration for Meta-SpliceAI

Based on the screenshot showing GPU options:

### Budget Option
| Setting | Value | Cost |
|---------|-------|------|
| GPU | A40 (48GB VRAM) | $0.40/hr |
| Volume Disk | 200GB | +$20/month |
| **Total (10hr/day, 20 days)** | | **~$80 + $20 = $100/month** |

### Performance Option
| Setting | Value | Cost |
|---------|-------|------|
| GPU | RTX 5090 (32GB VRAM) | $0.89/hr |
| Volume Disk | 200GB | +$20/month |
| **Total (10hr/day, 20 days)** | | **~$178 + $20 = $198/month** |

---

## 📚 Related Documentation

- [RUNPODS_STORAGE_REQUIREMENTS.md](./RUNPODS_STORAGE_REQUIREMENTS.md) - Minimum storage requirements
- [RUNPODS_COMPLETE_SETUP.md](./RUNPODS_COMPLETE_SETUP.md) - Full setup guide
- [Official RunPods Docs: Storage](https://docs.runpod.io/pods/storage/sync-volumes)
- [Official RunPods Docs: Pricing](https://docs.runpod.io/pods/pricing)

---

*This guide was verified against RunPods official documentation as of December 2025.*

