# Model Transfer Guide: RunPods to Local Machine

This guide documents how to transfer trained models and their weights from RunPods back to your local machine.

## Overview

After training models on RunPods, you'll have checkpoint files (`.pt` files) that contain:
- Model architecture state dictionary
- Training configuration
- Evaluation metrics
- Optimizer state (optional)

## Trained Model Locations

All trained models are saved under the following directory structure:

```
/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/
└── YYYYMMDD_HHMMSS/                    # Timestamp of training run
    └── checkpoints/
        ├── binary_classifier_*.pt       # Binary classification models
        ├── delta_predictor_*.pt         # Delta prediction models
        └── hyenadna_*.pt                 # HyenaDNA-based models
```

## Method 1: SCP (Secure Copy)

### Prerequisites

1. Get your RunPods SSH connection details from the RunPods dashboard
2. Ensure your local SSH key is configured

### Transfer Commands

From your **local machine**, run:

```bash
# Define variables
RUNPOD_USER="root"
RUNPOD_HOST="your-pod-hostname"  # e.g., ssh.runpod.io or the IP address
RUNPOD_PORT="your-pod-port"      # e.g., 22 or the custom port shown in dashboard

# Single file transfer
scp -P $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/YYYYMMDD_HHMMSS/checkpoints/model_name.pt ./local_destination/

# Transfer entire checkpoints directory
scp -r -P $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/YYYYMMDD_HHMMSS/checkpoints/ ./local_checkpoints/

# Transfer all experiment checkpoints
scp -r -P $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/ ./all_experiments/
```

### Example with Actual Path

```bash
# Example: Transfer binary classifier model from Dec 17 experiment
scp -P 22 root@ssh.runpod.io:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/20251217_214354/checkpoints/binary_classifier_binary_classifier_\(multi-step_step_1\).pt ./models/

# Transfer entire December 17 experiment
scp -r -P 22 root@ssh.runpod.io:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/20251217_*/ ./local_experiments/
```

## Method 2: rsync (Recommended for Large Files)

`rsync` provides better handling for large files and can resume interrupted transfers:

```bash
# Sync checkpoints with progress
rsync -avzP -e "ssh -p $RUNPOD_PORT" \
    $RUNPOD_USER@$RUNPOD_HOST:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/ \
    ./local_experiments/

# Sync only new/changed files (useful for incremental backups)
rsync -avzP --update -e "ssh -p $RUNPOD_PORT" \
    $RUNPOD_USER@$RUNPOD_HOST:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/ \
    ./local_experiments/
```

### rsync Options Explained

- `-a`: Archive mode (preserves permissions, timestamps, etc.)
- `-v`: Verbose output
- `-z`: Compress during transfer
- `-P`: Show progress and allow resume
- `--update`: Skip files that are newer on receiver

## Method 3: Transfer via Git LFS (For Model Versioning)

For model versioning and team sharing, consider using Git LFS:

```bash
# On RunPods - install Git LFS and track .pt files
cd /workspace/meta-spliceai
git lfs install
git lfs track "*.pt"
git add .gitattributes

# Add and commit models
git add data/mane/GRCh38/openspliceai_eval/meta_layer_dev/*/checkpoints/*.pt
git commit -m "Add trained model checkpoints"
git push

# On local machine - pull with LFS
git lfs pull
```

## Listing Available Models on RunPods

Before transferring, list available models:

```bash
# SSH into RunPods first, then run:
find /workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/ \
    -name "*.pt" \
    -exec ls -lh {} \;

# Or get a summary
find /workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/ \
    -name "*.pt" | wc -l
```

## Model Checkpoint Contents

Each `.pt` checkpoint file contains:

```python
{
    'model_state_dict': model.state_dict(),  # Model weights
    'config': {
        'name': 'Experiment Name',
        'hidden_dim': 256,
        'n_layers': 8,
        'batch_size': 64,
        # ... other hyperparameters
    },
    'metrics': {
        'roc_auc': 0.718,     # For classifiers
        'pr_auc': 0.652,      # For classifiers
        'f1': 0.65,           # For classifiers
        'correlation': 0.42,   # For regression
        'best_threshold': 0.3  # For classifiers
    }
}
```

## Loading Models Locally

After transferring, load models with:

```python
import torch
from pathlib import Path

# Load checkpoint
checkpoint_path = Path("./local_experiments/20251217_214354/checkpoints/binary_classifier.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Inspect contents
print("Config:", checkpoint['config'])
print("Metrics:", checkpoint['metrics'])

# Reconstruct model
from meta_spliceai.splice_engine.meta_layer.models import BinaryClassifier

model = BinaryClassifier(
    hidden_dim=checkpoint['config']['hidden_dim'],
    n_layers=checkpoint['config'].get('n_layers', 8)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
# ...
```

## Automation Script

Create a helper script for easy transfers:

```bash
#!/bin/bash
# save_models.sh - Transfer models from RunPods

# Configuration (update these)
RUNPOD_USER="root"
RUNPOD_HOST="ssh.runpod.io"  # or your pod's hostname
RUNPOD_PORT="22"             # or your pod's SSH port
LOCAL_DIR="./trained_models"

# Source and destination
REMOTE_PATH="/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/"

# Create local directory
mkdir -p "$LOCAL_DIR"

# Transfer with rsync
echo "Transferring models from RunPods..."
rsync -avzP -e "ssh -p $RUNPOD_PORT" \
    "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_PATH}" \
    "$LOCAL_DIR/"

echo "Transfer complete! Models saved to: $LOCAL_DIR"

# List transferred files
echo ""
echo "Transferred checkpoints:"
find "$LOCAL_DIR" -name "*.pt" -exec ls -lh {} \;
```

## Transferring Logs

Don't forget to transfer experiment logs for documentation:

```bash
# Transfer logs
scp -P $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST:/workspace/meta-spliceai/logs/gpu_exp_*.log ./local_logs/
```

## Best Practices

1. **Always verify transfers**: Check file sizes match between source and destination
2. **Create backups**: Keep copies of important model checkpoints
3. **Document experiments**: Transfer along with experiment documentation
4. **Clean up RunPods**: After confirmed transfer, remove old checkpoints to save disk space
5. **Use compression**: For very large models, compress before transfer:
   ```bash
   # On RunPods
   tar -czvf models.tar.gz /workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_layer_dev/
   
   # Transfer
   scp -P $RUNPOD_PORT root@host:models.tar.gz ./
   
   # Extract locally
   tar -xzvf models.tar.gz
   ```

## Troubleshooting

### Connection Timeout
```bash
# Use longer timeout
scp -o ConnectTimeout=60 -P $RUNPOD_PORT ...
```

### Permission Denied
```bash
# Ensure SSH key is loaded
ssh-add ~/.ssh/id_ed25519

# Or specify key explicitly
scp -i ~/.ssh/id_ed25519 -P $RUNPOD_PORT ...
```

### Transfer Interrupted
```bash
# Use rsync which can resume
rsync -avzP --partial -e "ssh -p $RUNPOD_PORT" ...
```

## See Also

- [RUNPODS_COMPLETE_SETUP.md](./RUNPODS_COMPLETE_SETUP.md) - Initial RunPods setup
- [PATH_RESOLUTION_ISSUES.md](./PATH_RESOLUTION_ISSUES.md) - Path configuration
- [EXP_2025_12_17_RUNPODS_A40_50K.md](../experiments/EXP_2025_12_17_RUNPODS_A40_50K.md) - Experiment documentation

