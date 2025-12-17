"""
Training and evaluation script for Position Localization (Multi-Step Step 3).

This script trains a position localizer model to predict WHERE splice effects
occur for splice-altering variants.

Usage
-----
# Quick test (CPU, small dataset)
python -m meta_spliceai.splice_engine.meta_layer.tests.test_position_localization --quick

# Full training (GPU recommended)
python -m meta_spliceai.splice_engine.meta_layer.tests.test_position_localization \
    --device cuda --epochs 50 --batch-size 64

# With effect type conditioning (requires Step 2 predictions)
python -m meta_spliceai.splice_engine.meta_layer.tests.test_position_localization \
    --use-effect-type

See Also
--------
- docs/methods/MULTI_STEP_FRAMEWORK.md
- data/position_labels.py
- models/position_localizer.py
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for position localization."""
    
    # Data
    max_train: int = 10000
    max_test: int = 1000
    context_size: int = 501
    test_chromosomes: List[str] = None
    
    # Model
    hidden_dim: int = 128
    n_layers: int = 6
    dropout: float = 0.1
    mode: str = 'attention'  # 'attention' or 'segmentation'
    use_effect_type: bool = False
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    patience: int = 7
    
    # Target generation
    delta_threshold: float = 0.1
    attention_sigma: float = 3.0
    
    # Device
    device: str = 'auto'
    
    # Output
    output_dir: str = None
    save_checkpoint: bool = True
    
    def __post_init__(self):
        if self.test_chromosomes is None:
            self.test_chromosomes = ['21', '22']
        
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'


# =============================================================================
# DATASET
# =============================================================================

@dataclass
class PositionSample:
    """Single sample for position localization training."""
    alt_seq: str
    ref_base: str
    alt_base: str
    position_target: np.ndarray  # [L] attention or binary mask
    peak_position: int
    effect_type: str
    classification: str
    variant_id: str


class PositionLocalizationDataset(Dataset):
    """Dataset for position localization training."""
    
    def __init__(self, samples: List[PositionSample]):
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        
        return {
            'alt_seq': torch.tensor(one_hot_seq(s.alt_seq), dtype=torch.float32),
            'ref_base': torch.tensor(one_hot_base(s.ref_base), dtype=torch.float32),
            'alt_base': torch.tensor(one_hot_base(s.alt_base), dtype=torch.float32),
            'position_target': torch.tensor(s.position_target, dtype=torch.float32),
            'peak_position': torch.tensor(s.peak_position, dtype=torch.long),
            'effect_type_idx': torch.tensor(EFFECT_TYPE_TO_IDX.get(s.effect_type, 0), dtype=torch.long)
        }


# Effect type mapping for conditioning
EFFECT_TYPE_TO_IDX = {
    'donor_gain': 0,
    'donor_loss': 1,
    'acceptor_gain': 2,
    'acceptor_loss': 3,
    'unknown': 4
}


def one_hot_seq(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    seq = seq.upper()
    indices = [mapping.get(b, 0) for b in seq]
    oh = np.zeros((4, len(seq)), dtype=np.float32)
    oh[indices, np.arange(len(seq))] = 1
    return oh


def one_hot_base(base: str) -> np.ndarray:
    """One-hot encode a single base."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    oh = np.zeros(4, dtype=np.float32)
    oh[mapping.get(base.upper(), 0)] = 1
    return oh


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_samples(
    variants: List,
    max_n: int,
    context_size: int,
    fasta,
    base_models: List,
    base_device: str,
    delta_threshold: float = 0.1,
    attention_sigma: float = 3.0,
    desc: str = "Preparing"
) -> List[PositionSample]:
    """
    Prepare position localization samples.
    
    For each splice-altering variant:
    1. Extract sequence context
    2. Run base model to find delta peaks
    3. Create attention target distribution
    """
    from meta_spliceai.splice_engine.meta_layer.data.position_labels import (
        derive_position_labels_from_delta,
        create_position_attention_target
    )
    
    samples = []
    half = context_size // 2
    
    for v in tqdm(variants[:max_n * 2], desc=desc):  # Sample extra to account for failures
        if len(samples) >= max_n:
            break
        
        try:
            chrom = str(v.chrom)
            pos = v.position
            
            # Get extended sequence for base model (10K flanking)
            ext_start = max(0, pos - 5500 - 1)
            ext_end = pos + 5500
            ref_ext = str(fasta[chrom][ext_start:ext_end].seq).upper()
            
            if len(ref_ext) < 11000:
                continue
            
            # Create alt sequence
            var_pos_in_ext = pos - ext_start - 1
            alt_ext = (
                ref_ext[:var_pos_in_ext] + 
                v.alt_allele + 
                ref_ext[var_pos_in_ext + len(v.ref_allele):]
            )
            
            # Get position labels from base model delta
            affected_positions = derive_position_labels_from_delta(
                ref_ext, alt_ext, base_models, base_device,
                threshold=delta_threshold
            )
            
            if not affected_positions:
                # No significant delta found, skip
                continue
            
            # Create attention target
            # Note: The delta analysis is on the base model output space
            # We need to map back to our context window
            
            # For simplicity, compute delta on context window directly
            ctx_start = max(0, pos - half - 1)
            ctx_end = pos + half
            
            ref_ctx = str(fasta[chrom][ctx_start:ctx_end].seq).upper()
            var_pos_in_ctx = pos - ctx_start - 1
            alt_ctx = (
                ref_ctx[:var_pos_in_ctx] + 
                v.alt_allele + 
                ref_ctx[var_pos_in_ctx + len(v.ref_allele):]
            )
            
            # Adjust length if needed
            if len(alt_ctx) != context_size:
                if len(alt_ctx) > context_size:
                    alt_ctx = alt_ctx[:context_size]
                else:
                    alt_ctx = alt_ctx.ljust(context_size, 'N')
            
            # Map affected positions to context window
            # Using the peak from extended analysis
            peak_pos = affected_positions[0].position
            peak_effect_type = affected_positions[0].effect_type
            
            # Map peak to context window coordinates
            # (approximate - peak position in output is relative to center)
            center_in_ctx = context_size // 2
            
            # Create attention target centered on variant position
            # (since we don't have exact mapping, use heuristic)
            attention_target = np.zeros(context_size, dtype=np.float32)
            
            # Place Gaussian at positions where delta is significant
            x = np.arange(context_size)
            
            # Use variant position as anchor, apply offset from affected positions
            for ap in affected_positions[:3]:  # Top 3 affected positions
                # Estimate position in context window
                # (This is approximate - in production, need proper coordinate mapping)
                offset = ap.position - (len(ref_ext) // 2)  # Offset from center
                ctx_pos = center_in_ctx + offset
                
                if 0 <= ctx_pos < context_size:
                    weight = abs(ap.delta_value)
                    gaussian = np.exp(-0.5 * ((x - ctx_pos) / attention_sigma) ** 2)
                    attention_target += weight * gaussian
            
            # Normalize
            if attention_target.sum() > 0:
                attention_target = attention_target / attention_target.sum()
            else:
                # Fallback: uniform around center
                attention_target = np.exp(-0.5 * ((x - center_in_ctx) / attention_sigma) ** 2)
                attention_target = attention_target / attention_target.sum()
            
            # Find peak position
            peak_in_ctx = attention_target.argmax()
            
            samples.append(PositionSample(
                alt_seq=alt_ctx,
                ref_base=v.ref_allele[0] if v.ref_allele else 'N',
                alt_base=v.alt_allele[0] if v.alt_allele else 'N',
                position_target=attention_target,
                peak_position=peak_in_ctx,
                effect_type=peak_effect_type,
                classification=v.classification,
                variant_id=v.get_coordinate_key()
            ))
            
        except Exception as e:
            logger.debug(f"Failed to process variant: {e}")
            continue
    
    logger.info(f"Prepared {len(samples)} samples")
    return samples


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    mode: str = 'attention'
) -> float:
    """Train for one epoch."""
    from meta_spliceai.splice_engine.meta_layer.models.position_localizer import (
        attention_cross_entropy_loss,
        segmentation_loss
    )
    
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in train_loader:
        alt_seq = batch['alt_seq'].to(device)
        ref_base = batch['ref_base'].to(device)
        alt_base = batch['alt_base'].to(device)
        target = batch['position_target'].to(device)
        
        optimizer.zero_grad()
        
        if mode == 'attention':
            logits = model(alt_seq, ref_base, alt_base, return_logits=True)
            loss = attention_cross_entropy_loss(logits, target)
        else:  # segmentation
            logits = model(alt_seq, ref_base, alt_base, return_logits=True)
            loss = segmentation_loss(logits, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    mode: str = 'attention'
) -> Tuple[float, Dict[str, float]]:
    """Validate the model."""
    from meta_spliceai.splice_engine.meta_layer.models.position_localizer import (
        attention_cross_entropy_loss,
        segmentation_loss
    )
    
    model.eval()
    total_loss = 0
    n_batches = 0
    
    # Metrics
    hit_top1 = 0
    hit_top3 = 0
    total = 0
    total_distance = 0
    
    with torch.no_grad():
        for batch in val_loader:
            alt_seq = batch['alt_seq'].to(device)
            ref_base = batch['ref_base'].to(device)
            alt_base = batch['alt_base'].to(device)
            target = batch['position_target'].to(device)
            target_peak = batch['peak_position'].to(device)
            
            if mode == 'attention':
                logits = model(alt_seq, ref_base, alt_base, return_logits=True)
                loss = attention_cross_entropy_loss(logits, target)
                attention = F.softmax(logits, dim=-1)
            else:
                logits = model(alt_seq, ref_base, alt_base, return_logits=True)
                loss = segmentation_loss(logits, target)
                attention = torch.sigmoid(logits)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Get predicted peaks
            _, top3_positions = torch.topk(attention, k=3, dim=-1)
            pred_peak = top3_positions[:, 0]
            
            # Calculate metrics
            distance = (pred_peak - target_peak).abs()
            hit_top1 += (distance <= 5).sum().item()
            
            for i in range(alt_seq.shape[0]):
                tp = target_peak[i].item()
                any_hit = any(
                    abs(top3_positions[i, k].item() - tp) <= 5
                    for k in range(3)
                )
                if any_hit:
                    hit_top3 += 1
            
            total_distance += distance.float().sum().item()
            total += alt_seq.shape[0]
    
    metrics = {
        'hit_rate_top1': hit_top1 / total if total > 0 else 0,
        'hit_rate_top3': hit_top3 / total if total > 0 else 0,
        'mean_distance': total_distance / total if total > 0 else 0
    }
    
    return total_loss / n_batches, metrics


def run_experiment(config: TrainingConfig) -> Dict:
    """Run the full training experiment."""
    
    logger.info("=" * 60)
    logger.info("Position Localization Training (Multi-Step Step 3)")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Hidden dim: {config.hidden_dim}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    # Load dependencies
    logger.info("\nLoading dependencies...")
    
    try:
        import pyfaidx
        from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import (
            SpliceVarDBLoader
        )
        from meta_spliceai.splice_engine.meta_layer.models.position_localizer import (
            create_position_localizer
        )
    except ImportError as e:
        logger.error(f"Failed to import dependencies: {e}")
        return {'error': str(e)}
    
    # Load SpliceVarDB
    logger.info("\nLoading SpliceVarDB...")
    loader = SpliceVarDBLoader(genome_build='GRCh38')
    
    # Get train/test split
    train_variants, test_variants = loader.get_train_test_split(
        test_chromosomes=config.test_chromosomes
    )
    
    # Filter to splice-altering only
    train_variants = [v for v in train_variants if v.is_splice_altering]
    test_variants = [v for v in test_variants if v.is_splice_altering]
    
    logger.info(f"Splice-altering variants: {len(train_variants)} train, {len(test_variants)} test")
    
    # Load reference genome
    logger.info("\nLoading reference genome...")
    from meta_spliceai.system.config import find_project_root
    
    project_root = Path(find_project_root(__file__))
    fasta_paths = [
        project_root / "data" / "mane" / "GRCh38" / "hg38.fa",
        Path("/workspace/meta-spliceai/data/mane/GRCh38/hg38.fa"),
    ]
    
    fasta = None
    for fasta_path in fasta_paths:
        if fasta_path.exists():
            fasta = pyfaidx.Fasta(str(fasta_path))
            logger.info(f"Loaded reference: {fasta_path}")
            break
    
    if fasta is None:
        logger.error("Reference genome not found")
        return {'error': 'Reference genome not found'}
    
    # Load base models for target generation
    logger.info("\nLoading base models...")
    
    base_device = config.device
    if base_device == 'mps':
        base_device = 'cpu'  # Some models don't work on MPS
    
    try:
        from meta_spliceai.splice_engine.models.openspliceai import create_openspliceai_ensemble
        from meta_spliceai.system.config import get_model_path
        
        model_paths = get_model_path('openspliceai', 10000)
        if isinstance(model_paths, list):
            base_models = create_openspliceai_ensemble(model_paths, device=base_device)
        else:
            base_models = create_openspliceai_ensemble([model_paths], device=base_device)
        
        logger.info(f"Loaded {len(base_models)} base models")
    except Exception as e:
        logger.error(f"Failed to load base models: {e}")
        return {'error': f'Failed to load base models: {e}'}
    
    # Prepare training data
    logger.info(f"\nPreparing training data (max {config.max_train})...")
    start_time = time.time()
    
    train_samples = prepare_samples(
        train_variants,
        max_n=config.max_train,
        context_size=config.context_size,
        fasta=fasta,
        base_models=base_models,
        base_device=base_device,
        delta_threshold=config.delta_threshold,
        attention_sigma=config.attention_sigma,
        desc="Preparing train"
    )
    
    # Split into train/val
    val_size = int(len(train_samples) * 0.15)
    val_samples = train_samples[:val_size]
    train_samples = train_samples[val_size:]
    
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    
    # Prepare test data
    logger.info(f"\nPreparing test data (max {config.max_test})...")
    test_samples = prepare_samples(
        test_variants,
        max_n=config.max_test,
        context_size=config.context_size,
        fasta=fasta,
        base_models=base_models,
        base_device=base_device,
        delta_threshold=config.delta_threshold,
        attention_sigma=config.attention_sigma,
        desc="Preparing test"
    )
    
    data_prep_time = time.time() - start_time
    logger.info(f"Data preparation took {data_prep_time:.1f}s")
    
    # Create data loaders
    train_dataset = PositionLocalizationDataset(train_samples)
    val_dataset = PositionLocalizationDataset(val_samples)
    test_dataset = PositionLocalizationDataset(test_samples)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    
    # Create model
    logger.info(f"\nCreating {config.mode} position localizer...")
    model = create_position_localizer(
        mode=config.mode,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        use_effect_type=config.use_effect_type
    )
    model = model.to(config.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    best_metrics = {}
    patience_counter = 0
    train_start = time.time()
    
    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, config.device, config.mode
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, config.device, config.mode
        )
        
        scheduler.step()
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{config.epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"hit@1={val_metrics['hit_rate_top1']:.3f}, "
            f"hit@3={val_metrics['hit_rate_top3']:.3f}, "
            f"mean_dist={val_metrics['mean_distance']:.1f}"
        )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            patience_counter = 0
            
            # Save best model
            if config.save_checkpoint and config.output_dir:
                os.makedirs(config.output_dir, exist_ok=True)
                checkpoint_path = Path(config.output_dir) / 'best_position_localizer.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.__dict__,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    train_time = time.time() - train_start
    
    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on test set...")
    logger.info("=" * 60)
    
    test_loss, test_metrics = validate(
        model, test_loader, config.device, config.mode
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Hit Rate (top-1, ±5bp): {test_metrics['hit_rate_top1']:.3f}")
    logger.info(f"  Hit Rate (top-3, ±5bp): {test_metrics['hit_rate_top3']:.3f}")
    logger.info(f"  Mean Distance: {test_metrics['mean_distance']:.1f} bp")
    
    results = {
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'best_val_loss': best_val_loss,
        'best_val_metrics': best_metrics,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'data_prep_time': data_prep_time,
        'train_time': train_time,
        'epochs_trained': epoch + 1,
        'n_params': n_params
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Total time: {data_prep_time + train_time:.1f}s")
    logger.info("=" * 60)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train position localization model (Multi-Step Step 3)'
    )
    
    # Data args
    parser.add_argument('--max-train', type=int, default=5000,
                        help='Maximum training samples')
    parser.add_argument('--max-test', type=int, default=500,
                        help='Maximum test samples')
    parser.add_argument('--context-size', type=int, default=501,
                        help='Sequence context size')
    
    # Model args
    parser.add_argument('--mode', type=str, default='attention',
                        choices=['attention', 'segmentation'],
                        help='Localization mode')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--use-effect-type', action='store_true',
                        help='Use effect type conditioning')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    
    # Target generation
    parser.add_argument('--delta-threshold', type=float, default=0.1,
                        help='Threshold for significant delta')
    parser.add_argument('--attention-sigma', type=float, default=3.0,
                        help='Sigma for attention Gaussian')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, mps, cpu, auto)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    
    # Quick test mode
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with small dataset')
    
    args = parser.parse_args()
    
    # Quick test overrides
    if args.quick:
        args.max_train = 500
        args.max_test = 100
        args.epochs = 10
        args.batch_size = 16
        logger.info("Quick test mode: small dataset, few epochs")
    
    # Create config
    config = TrainingConfig(
        max_train=args.max_train,
        max_test=args.max_test,
        context_size=args.context_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        mode=args.mode,
        use_effect_type=args.use_effect_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        delta_threshold=args.delta_threshold,
        attention_sigma=args.attention_sigma,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Run experiment
    results = run_experiment(config)
    
    if 'error' not in results:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Test Hit Rate (top-1): {results['test_metrics']['hit_rate_top1']:.3f}")
        print(f"Test Hit Rate (top-3): {results['test_metrics']['hit_rate_top3']:.3f}")
        print(f"Test Mean Distance: {results['test_metrics']['mean_distance']:.1f} bp")


if __name__ == '__main__':
    main()

