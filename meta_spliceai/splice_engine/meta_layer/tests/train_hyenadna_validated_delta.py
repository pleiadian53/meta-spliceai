#!/usr/bin/env python3
"""
Training Script for HyenaDNA Validated Delta Predictor.

This script provides a complete training pipeline for fine-tuning HyenaDNA
on splice variant effect prediction using the validated delta target strategy.

Features:
- Configurable fine-tuning (freeze all, unfreeze last N layers)
- Discriminative learning rates
- Early stopping with best model restoration
- Mixed precision training
- Gradient accumulation for memory efficiency
- Comprehensive logging and checkpointing

Usage:
    # Transfer learning (frozen encoder)
    python train_hyenadna_validated_delta.py --mode frozen
    
    # Fine-tuning last 2 layers
    python train_hyenadna_validated_delta.py --mode finetune --unfreeze 2
    
    # Deep fine-tuning (last 4 layers)
    python train_hyenadna_validated_delta.py --mode finetune --unfreeze 4 --lr 3e-5
    
    # Custom configuration
    python train_hyenadna_validated_delta.py \
        --model hyenadna-medium-160k \
        --mode finetune \
        --unfreeze 2 \
        --lr 5e-5 \
        --encoder-lr-mult 0.1 \
        --epochs 50 \
        --batch-size 16

Example tmux session:
    tmux new-session -d -s hyenadna_train
    tmux send-keys -t hyenadna_train "cd /workspace/meta-spliceai && \\
        source /workspace/miniforge3/etc/profile.d/conda.sh && \\
        conda activate metaspliceai && \\
        python -m meta_spliceai.splice_engine.meta_layer.tests.train_hyenadna_validated_delta \\
            --mode finetune --unfreeze 2 2>&1 | tee logs/hyenadna_finetune.log" Enter
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model settings
    model_name: str = 'hyenadna-medium-160k'
    hidden_dim: int = 256
    freeze_encoder: bool = False
    unfreeze_last_n: int = 2
    dropout: float = 0.1
    
    # Data settings
    max_train: int = 25000
    max_test: int = 1000
    context_size: int = 501
    val_split: float = 0.15
    
    # Training settings
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 5e-5
    encoder_lr_mult: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation: int = 4
    use_amp: bool = True
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-5
    
    # Other
    device: str = 'cuda'
    seed: int = 42
    
    def __post_init__(self):
        if self.freeze_encoder:
            self.unfreeze_last_n = 0


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class Sample:
    """Training/test sample."""
    alt_seq: str
    ref_base: str
    alt_base: str
    target_delta: np.ndarray
    classification: str


class ValidatedDeltaDataset(Dataset):
    """Dataset for validated delta prediction."""
    
    def __init__(self, samples: List[Sample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            'alt_seq': torch.tensor(one_hot_seq(s.alt_seq), dtype=torch.float32),
            'ref_base': torch.tensor(one_hot_base(s.ref_base), dtype=torch.float32),
            'alt_base': torch.tensor(one_hot_base(s.alt_base), dtype=torch.float32),
            'target_delta': torch.tensor(s.target_delta, dtype=torch.float32),
            'classification': s.classification
        }


def one_hot_seq(seq: str) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding [4, L]."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    one_hot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        idx = mapping.get(base, 4)
        if idx < 4:
            one_hot[idx, i] = 1.0
        else:
            one_hot[:, i] = 0.25  # N = uniform
    return one_hot


def one_hot_base(base: str) -> np.ndarray:
    """Convert single base to one-hot [4]."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros(4, dtype=np.float32)
    idx = mapping.get(base.upper(), -1)
    if idx >= 0:
        one_hot[idx] = 1.0
    else:
        one_hot[:] = 0.25
    return one_hot


def load_splicevardb_data(
    config: TrainingConfig
) -> Tuple[List[Sample], List[Sample]]:
    """
    Load and prepare SpliceVarDB data with validated delta targets.
    
    Returns
    -------
    Tuple[List[Sample], List[Sample]]
        Training and test samples
    """
    # Import dependencies
    from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb
    from meta_spliceai.splice_engine.base_models.loaders import load_ensemble_models
    from meta_spliceai.system.genomic_resources import GenomicResources
    
    logger.info("Loading resources...")
    
    # Load variants
    loader = load_splicevardb(genome_build='GRCh38')
    variants = loader.load_all()
    logger.info(f"Loaded {len(variants):,} variants")
    
    # Load FASTA
    resources = GenomicResources.get_instance()
    fasta_path = os.environ.get('SS_FASTA_PATH', 
        resources.get_reference_genome_path('GRCh38', release=112))
    
    import pysam
    fasta = pysam.FastaFile(fasta_path)
    
    # Load base models for delta computation
    base_models = load_ensemble_models(tissue='all', device=config.device)
    
    # Filter and classify
    by_class = {}
    for v in variants:
        if v.classification not in ['Splice-altering', 'Normal']:
            continue
        by_class.setdefault(v.classification, []).append(v)
    
    logger.info(f"  Splice-altering: {len(by_class.get('Splice-altering', []))}")
    logger.info(f"  Normal: {len(by_class.get('Normal', []))}")
    
    # Train/test split by chromosome
    test_chroms = {'21', '22'}
    train_variants = []
    test_variants = []
    
    for cls, vs in by_class.items():
        for v in vs:
            chrom = v.chrom.replace('chr', '')
            if chrom in test_chroms:
                test_variants.append(v)
            else:
                train_variants.append(v)
    
    logger.info(f"Train/test split: {len(train_variants)} / {len(test_variants)}")
    
    # Balance training data
    train_sa = [v for v in train_variants if v.classification == 'Splice-altering']
    train_normal = [v for v in train_variants if v.classification == 'Normal']
    
    n_each = min(len(train_sa), len(train_normal), config.max_train // 2)
    train_sa = train_sa[:n_each]
    train_normal = train_normal[:n_each]
    train_variants = train_sa + train_normal
    np.random.shuffle(train_variants)
    
    logger.info(f"Balanced training: {n_each} each = {len(train_variants)} total")
    
    # Prepare samples
    train_samples = prepare_samples(
        train_variants, config.context_size, fasta, base_models, 
        config.device, "Train samples"
    )
    
    test_sa = [v for v in test_variants if v.classification == 'Splice-altering'][:config.max_test // 2]
    test_normal = [v for v in test_variants if v.classification == 'Normal'][:config.max_test // 2]
    test_variants = test_sa + test_normal
    
    test_samples = prepare_samples(
        test_variants, config.context_size, fasta, base_models,
        config.device, "Test samples"
    )
    
    logger.info(f"Prepared: {len(train_samples)} train, {len(test_samples)} test")
    
    return train_samples, test_samples


def prepare_samples(
    variants: List,
    context_size: int,
    fasta,
    base_models: List,
    device: str,
    desc: str
) -> List[Sample]:
    """Prepare samples with validated delta targets."""
    samples = []
    
    for v in tqdm(variants, desc=desc):
        try:
            # Get sequences
            chrom = v.chrom if v.chrom.startswith('chr') else f'chr{v.chrom}'
            pos = v.pos - 1  # 0-based
            half_ctx = context_size // 2
            
            # Reference sequence
            ref_seq = fasta.fetch(chrom, pos - half_ctx, pos + half_ctx + 1)
            if len(ref_seq) != context_size:
                continue
            
            # Create alternate sequence
            alt_seq = ref_seq[:half_ctx] + v.alt_allele[0] + ref_seq[half_ctx + 1:]
            
            # Compute target delta
            if v.classification == 'Splice-altering':
                target_delta = compute_base_delta(ref_seq, alt_seq, base_models, device)
            else:
                target_delta = np.zeros(3, dtype=np.float32)
            
            samples.append(Sample(
                alt_seq=alt_seq,
                ref_base=v.ref_allele[0] if v.ref_allele else 'N',
                alt_base=v.alt_allele[0] if v.alt_allele else 'N',
                target_delta=target_delta,
                classification=v.classification
            ))
            
        except Exception as e:
            continue
    
    return samples


def compute_base_delta(
    ref_seq: str, 
    alt_seq: str, 
    models: List, 
    device: str
) -> np.ndarray:
    """Compute delta from base model predictions."""
    # Extend sequences for base model (needs ~10K context)
    # For now, use the short sequence with padding
    pad_len = 5000 - len(ref_seq) // 2
    ref_padded = 'N' * pad_len + ref_seq + 'N' * pad_len
    alt_padded = 'N' * pad_len + alt_seq + 'N' * pad_len
    
    def predict(seq):
        x = torch.tensor(one_hot_seq(seq), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = [m(x).cpu() for m in models]
            avg = torch.mean(torch.stack(preds), dim=0)
            probs = F.softmax(avg.permute(0, 2, 1), dim=-1)
        return probs[0].numpy()
    
    ref_probs = predict(ref_padded)
    alt_probs = predict(alt_padded)
    delta = alt_probs - ref_probs
    
    # Get max delta near center
    center = len(delta) // 2
    window = 50
    center_delta = delta[max(0, center-window):min(len(delta), center+window+1)]
    max_idx = np.abs(center_delta).sum(axis=1).argmax()
    
    return center_delta[max_idx].astype(np.float32)


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Trainer for HyenaDNA validated delta model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_samples: List[Sample],
        test_samples: List[Sample]
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Split training data for validation
        n_val = int(len(train_samples) * config.val_split)
        self.val_samples = train_samples[:n_val]
        self.train_samples = train_samples[n_val:]
        self.test_samples = test_samples
        
        logger.info(f"Split: {len(self.train_samples)} train, {len(self.val_samples)} val, "
                   f"{len(self.test_samples)} test")
        
        # Create data loaders
        self.train_loader = DataLoader(
            ValidatedDeltaDataset(self.train_samples),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            ValidatedDeltaDataset(self.val_samples),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Setup optimizer
        if not config.freeze_encoder and config.unfreeze_last_n > 0:
            self.optimizer = model.get_optimizer(
                base_lr=config.learning_rate,
                encoder_lr_mult=config.encoder_lr_mult,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
        # Metrics history
        self.history = {'train_loss': [], 'val_loss': [], 'correlation': []}
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            alt_seq = batch['alt_seq'].to(self.device)
            ref_base = batch['ref_base'].to(self.device)
            alt_base = batch['alt_base'].to(self.device)
            target = batch['target_delta'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                pred = self.model(alt_seq, ref_base, alt_base)
                loss = F.mse_loss(pred, target)
                loss = loss / self.config.gradient_accumulation
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (i + 1) % self.config.gradient_accumulation == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                alt_seq = batch['alt_seq'].to(self.device)
                ref_base = batch['ref_base'].to(self.device)
                alt_base = batch['alt_base'].to(self.device)
                target = batch['target_delta'].to(self.device)
                
                with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                    pred = self.model(alt_seq, ref_base, alt_base)
                    loss = F.mse_loss(pred, target)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for s in tqdm(self.test_samples, desc="Evaluating", leave=False):
                alt_seq = torch.tensor(one_hot_seq(s.alt_seq), dtype=torch.float32).unsqueeze(0).to(self.device)
                ref_base = torch.tensor(one_hot_base(s.ref_base), dtype=torch.float32).unsqueeze(0).to(self.device)
                alt_base = torch.tensor(one_hot_base(s.alt_base), dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                    pred = self.model(alt_seq, ref_base, alt_base)
                
                pred = pred.float().cpu().numpy()[0]
                
                results.append({
                    'classification': s.classification,
                    'target_max': np.abs(s.target_delta).max(),
                    'pred_max': np.abs(pred).max()
                })
        
        # Compute metrics
        sa_results = [r for r in results if r['classification'] == 'Splice-altering']
        sa_target = [r['target_max'] for r in sa_results]
        sa_pred = [r['pred_max'] for r in sa_results]
        
        corr, pval = pearsonr(sa_target, sa_pred) if len(sa_target) > 1 else (0, 1)
        
        all_pred = [r['pred_max'] for r in results]
        all_is_sa = [1 if r['classification'] == 'Splice-altering' else 0 for r in results]
        
        auc = roc_auc_score(all_is_sa, all_pred) if len(set(all_is_sa)) > 1 else 0.5
        ap = average_precision_score(all_is_sa, all_pred) if len(set(all_is_sa)) > 1 else 0.5
        
        return {
            'correlation': corr,
            'p_value': pval,
            'roc_auc': auc,
            'pr_auc': ap,
            'n_test': len(results)
        }
    
    def train(self) -> Dict[str, float]:
        """Full training loop with early stopping."""
        logger.info(f"\nStarting training for {self.config.epochs} epochs...")
        logger.info(f"Early stopping: patience={self.config.patience}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                           f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                           f"patience={self.patience_counter}/{self.config.patience}")
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}! Best val_loss={self.best_val_loss:.6f}")
                break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model weights")
        
        elapsed = (time.time() - start_time) / 60
        logger.info(f"Training completed in {elapsed:.1f} minutes")
        
        # Final evaluation
        metrics = self.evaluate()
        metrics['elapsed_minutes'] = elapsed
        metrics['best_val_loss'] = self.best_val_loss
        metrics['stopped_epoch'] = len(self.history['train_loss'])
        
        return metrics
    
    def save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'metrics': metrics,
            'history': self.history
        }
        
        torch.save(save_dict, path)
        logger.info(f"Saved checkpoint: {path}")
        
        # Also save metrics as JSON
        json_path = Path(path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                           for k, v in metrics.items()}
            }, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train HyenaDNA Validated Delta Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transfer learning (frozen encoder)
  python train_hyenadna_validated_delta.py --mode frozen
  
  # Fine-tuning last 2 layers
  python train_hyenadna_validated_delta.py --mode finetune --unfreeze 2
  
  # Deep fine-tuning with custom settings
  python train_hyenadna_validated_delta.py \\
      --mode finetune \\
      --unfreeze 4 \\
      --model hyenadna-medium-160k \\
      --lr 3e-5 \\
      --epochs 50
        """
    )
    
    # Mode
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['frozen', 'finetune', 'full'],
                        help='Training mode: frozen (encoder frozen), finetune (last N layers), full (all trainable)')
    
    # Model
    parser.add_argument('--model', type=str, default='hyenadna-medium-160k',
                        help='HyenaDNA model variant')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--unfreeze', type=int, default=2,
                        help='Number of encoder layers to unfreeze (for finetune mode)')
    
    # Training
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Base learning rate')
    parser.add_argument('--encoder-lr-mult', type=float, default=0.1,
                        help='LR multiplier for encoder (e.g., 0.1 = 10x lower)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='Gradient accumulation steps')
    
    # Data
    parser.add_argument('--max-train', type=int, default=25000,
                        help='Maximum training samples')
    parser.add_argument('--max-test', type=int, default=1000,
                        help='Maximum test samples')
    parser.add_argument('--context-size', type=int, default=501,
                        help='Context window size')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configure based on mode
    freeze_encoder = args.mode == 'frozen'
    unfreeze_last_n = args.unfreeze if args.mode == 'finetune' else 0
    if args.mode == 'full':
        unfreeze_last_n = 100  # Unfreeze all
    
    config = TrainingConfig(
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        freeze_encoder=freeze_encoder,
        unfreeze_last_n=unfreeze_last_n,
        max_train=args.max_train,
        max_test=args.max_test,
        context_size=args.context_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        encoder_lr_mult=args.encoder_lr_mult,
        weight_decay=args.weight_decay,
        gradient_accumulation=args.grad_accum,
        patience=args.patience,
        device=args.device,
        seed=args.seed
    )
    
    # Log configuration
    logger.info("=" * 80)
    logger.info(f"HyenaDNA Validated Delta Training")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Freeze encoder: {config.freeze_encoder}")
    logger.info(f"Unfreeze last N: {config.unfreeze_last_n}")
    logger.info(f"Learning rate: {config.learning_rate} (encoder mult: {config.encoder_lr_mult})")
    logger.info(f"Batch size: {config.batch_size} (accum: {config.gradient_accumulation})")
    logger.info(f"Epochs: {config.epochs} (patience: {config.patience})")
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    logger.info("\nLoading data...")
    train_samples, test_samples = load_splicevardb_data(config)
    
    # Create model
    logger.info("\nCreating model...")
    from meta_spliceai.splice_engine.meta_layer.models.hyenadna_validated_delta import (
        create_hyenadna_model
    )
    
    model = create_hyenadna_model(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        freeze_encoder=config.freeze_encoder,
        unfreeze_last_n=config.unfreeze_last_n,
        dropout=config.dropout
    )
    
    model = model.to(config.device)
    
    # Train
    trainer = Trainer(model, config, train_samples, test_samples)
    metrics = trainer.train()
    
    # Log results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Correlation: r = {metrics['correlation']:.4f} (p = {metrics['p_value']:.2e})")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"Time: {metrics['elapsed_minutes']:.1f} minutes")
    
    # Save checkpoint
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(os.environ.get('META_SPLICEAI_ROOT', '.')) / 'data' / 'checkpoints'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = f"hyenadna_{args.mode}_{config.unfreeze_last_n}_{timestamp}.pt"
    checkpoint_path = output_dir / checkpoint_name
    
    trainer.save_checkpoint(str(checkpoint_path), metrics)
    
    return metrics


if __name__ == '__main__':
    main()

