"""
GPU Experiment Script for ValidatedDeltaPredictor with 50K samples + HyenaDNA.

This script extends test_validated_delta_experiments.py for GPU training:
1. Full SpliceVarDB dataset (~50K samples)
2. HyenaDNA-based single-pass ValidatedDelta variant
3. Proper GPU memory management

Usage:
    # Run on GPU (RunPods with A40):
    python -m meta_spliceai.splice_engine.meta_layer.tests.test_gpu_validated_delta_experiments \
        --exp full_dataset --device cuda
    
    # Run HyenaDNA experiment:
    python -m meta_spliceai.splice_engine.meta_layer.tests.test_gpu_validated_delta_experiments \
        --exp hyenadna --device cuda
    
    # Run all experiments:
    python -m meta_spliceai.splice_engine.meta_layer.tests.test_gpu_validated_delta_experiments \
        --exp all --device cuda

Expected Results (based on ROADMAP.md):
    - 8K samples: r=0.507 (current best)
    - 50K samples: r=0.55+ (expected +10-15% improvement)
    - HyenaDNA: r=0.55+ (expected from better sequence understanding)

GPU Requirements (from GPU_EXPERIMENTS.md):
    - 50K SimpleCNN: ~12-16GB VRAM
    - HyenaDNA-small: ~12GB VRAM
    - HyenaDNA-medium: ~24GB VRAM (A40 recommended)
"""

import argparse
import gc
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import logging
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

# Suppress warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Add project root to path."""
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_paths()


# ============================================================================
# HyenaDNA-based ValidatedDelta Model (Single-Pass)
# ============================================================================

class HyenaDNAValidatedDeltaPredictor(nn.Module):
    """
    HyenaDNA-based single-pass delta predictor.
    
    Uses HyenaDNA to encode the alternate sequence (with variant embedded),
    combined with variant information (ref_base, alt_base) to predict delta.
    
    This is the HyenaDNA version of ValidatedDeltaPredictor - single-pass
    inference without requiring both ref and alt sequences.
    
    Architecture:
        alt_seq ──→ [HyenaDNA Encoder] ──→ seq_features [B, H]
        ref_base ──┐                            ↓
        alt_base ──┴→ [Embed] ──→ var_features [B, H]
                                                ↓
                                    [Fusion + Delta Head]
                                                ↓
                                        Δ = [Δ_donor, Δ_acceptor, Δ_neither]
    """
    
    def __init__(
        self,
        model_name: str = 'hyenadna-small-32k',
        hidden_dim: int = 256,
        freeze_encoder: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
        
        # Load HyenaDNA encoder
        self.encoder, self.encoder_dim = self._load_hyenadna(model_name)
        
        if freeze_encoder and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"HyenaDNA encoder frozen (dim={self.encoder_dim})")
        
        # Projection from HyenaDNA dim to hidden_dim
        self.proj = nn.Linear(self.encoder_dim, hidden_dim)
        
        # Variant embedding: ref_base[4] + alt_base[4] → hidden_dim
        self.variant_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Delta prediction head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # [Δ_donor, Δ_acceptor, Δ_neither]
        )
        
        logger.info(f"HyenaDNAValidatedDeltaPredictor initialized with {model_name}")
    
    def _load_hyenadna(self, model_name: str) -> Tuple[Optional[nn.Module], int]:
        """Load HyenaDNA model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM
            
            hf_path = f"LongSafari/{model_name}"
            logger.info(f"Loading HyenaDNA from {hf_path}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                trust_remote_code=True,
                torch_dtype=torch.float16  # Use FP16 to save memory
            )
            
            # Get embedding dimension
            if hasattr(model.config, 'd_model'):
                embed_dim = model.config.d_model
            elif hasattr(model.config, 'hidden_size'):
                embed_dim = model.config.hidden_size
            else:
                embed_dim = 256
            
            logger.info(f"HyenaDNA loaded: {model_name}, dim={embed_dim}")
            return model, embed_dim
            
        except Exception as e:
            logger.warning(f"Could not load HyenaDNA: {e}. Using CNN fallback.")
            return self._create_fallback_encoder(), 256
    
    def _create_fallback_encoder(self) -> nn.Module:
        """Create a CNN fallback if HyenaDNA is not available."""
        return nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Encode DNA sequence using HyenaDNA.
        
        Parameters
        ----------
        seq : torch.Tensor
            One-hot encoded sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Global sequence features [B, encoder_dim]
        """
        if isinstance(self.encoder, nn.Sequential):
            # CNN fallback: [B, 4, L] -> [B, 256, 1] -> [B, 256]
            features = self.encoder(seq)
            return features.squeeze(-1)
        else:
            # HyenaDNA: needs token IDs
            # Convert one-hot to token IDs (A=0, C=1, G=2, T=3)
            token_ids = seq.argmax(dim=1)  # [B, L]
            
            with torch.set_grad_enabled(not self.freeze_encoder):
                outputs = self.encoder(token_ids, output_hidden_states=True)
                
                # Get last hidden state and pool
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden = outputs.hidden_states[-1]  # [B, L, H]
                    features = hidden.mean(dim=1)  # Global average pool [B, H]
                else:
                    # Fallback: use logits
                    features = outputs.logits.mean(dim=1)
            
            return features
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict delta scores.
        
        Parameters
        ----------
        alt_seq : torch.Tensor
            Alternate sequence (with variant) [B, 4, L]
        ref_base : torch.Tensor
            Reference base one-hot [B, 4]
        alt_base : torch.Tensor
            Alternate base one-hot [B, 4]
        
        Returns
        -------
        torch.Tensor
            Delta scores [B, 3] (Δ_donor, Δ_acceptor, Δ_neither)
        """
        # Encode sequence
        seq_features = self.encode_sequence(alt_seq)  # [B, encoder_dim]
        seq_features = self.proj(seq_features)  # [B, hidden_dim]
        
        # Encode variant info
        var_info = torch.cat([ref_base, alt_base], dim=-1)  # [B, 8]
        var_features = self.variant_embed(var_info)  # [B, hidden_dim]
        
        # Fuse and predict delta
        combined = torch.cat([seq_features, var_features], dim=-1)  # [B, 2H]
        delta = self.delta_head(combined)  # [B, 3]
        
        return delta


# ============================================================================
# Experiment Configuration
# ============================================================================

@dataclass
class GPUExperimentConfig:
    """Configuration for GPU experiments."""
    name: str
    max_train: int = 50000
    max_test: int = 2000
    context_size: int = 501
    epochs: int = 50
    batch_size: int = 128
    hidden_dim: int = 128
    n_layers: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    use_hyenadna: bool = False
    hyenadna_model: str = 'hyenadna-small-32k'
    freeze_encoder: bool = True
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_accumulation: int = 1
    description: str = ""
    device: str = "cuda"
    # Early stopping parameters
    val_split: float = 0.15  # Fraction of training data for validation
    patience: int = 5  # Early stopping patience (epochs without improvement)
    min_delta: float = 1e-4  # Minimum improvement to reset patience


@dataclass
class Sample:
    """Training/test sample."""
    alt_seq: str
    ref_base: str
    alt_base: str
    target_delta: np.ndarray
    classification: str


class SampleDataset(Dataset):
    """PyTorch dataset for samples."""
    
    def __init__(self, samples: List[Sample]):
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            'alt_seq': torch.tensor(one_hot_seq(s.alt_seq), dtype=torch.float32),
            'ref_base': torch.tensor(one_hot_base(s.ref_base), dtype=torch.float32),
            'alt_base': torch.tensor(one_hot_base(s.alt_base), dtype=torch.float32),
            'target_delta': torch.tensor(s.target_delta, dtype=torch.float32),
            'classification': s.classification
        }


# ============================================================================
# Helper Functions
# ============================================================================

def one_hot_base(base: str) -> np.ndarray:
    """Convert single base to one-hot encoding."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    idx = mapping.get(base.upper(), 0)
    onehot = np.zeros(4, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot


def one_hot_seq(seq: str) -> np.ndarray:
    """Convert sequence to one-hot encoding."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    oh = np.zeros((4, len(seq)), dtype=np.float32)
    for i, n in enumerate(seq.upper()):
        oh[mapping.get(n, 0), i] = 1.0
    return oh


def get_base_delta(
    ref_seq: str,
    alt_seq: str,
    models: List,
    device: str
) -> np.ndarray:
    """Compute base model delta scores."""
    
    def predict(seq: str) -> np.ndarray:
        x = torch.tensor(one_hot_seq(seq), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = [m(x).cpu() for m in models]
            avg = torch.mean(torch.stack(preds), dim=0)
            probs = F.softmax(avg.permute(0, 2, 1), dim=-1)
        return probs[0].numpy()
    
    ref_probs = predict(ref_seq)
    alt_probs = predict(alt_seq)
    delta = alt_probs - ref_probs
    
    # Get max delta in center region
    center = len(delta) // 2
    window = 50
    center_delta = delta[max(0, center-window):min(len(delta), center+window+1)]
    max_idx = np.abs(center_delta).sum(axis=1).argmax()
    
    return center_delta[max_idx]


def prepare_samples_gpu(
    variants: List,
    max_n: int,
    context_size: int,
    fasta,
    base_models: List,
    base_device: str,
    desc: str = "Preparing",
    num_workers: int = 0
) -> List[Sample]:
    """
    Prepare samples with validated delta targets.
    
    GPU-optimized version with progress tracking.
    """
    samples = []
    half = context_size // 2
    errors = 0
    
    pbar = tqdm(variants[:int(max_n * 1.5)], desc=desc, leave=True)
    
    for v in pbar:
        if len(samples) >= max_n:
            break
        
        try:
            # Normalize chromosome name
            chrom = str(v.chrom).replace('chr', '')
            if chrom not in fasta.keys():
                chrom = f'chr{chrom}' if f'chr{chrom}' in fasta.keys() else None
            if not chrom:
                continue
            
            pos = v.position
            
            # Get extended sequence for base model (needs 10k context)
            ext_start = max(0, pos - 6000)
            ext_end = pos + 6000
            ref_ext = str(fasta[chrom][ext_start:ext_end].seq)
            
            if len(ref_ext) < 11000:
                continue
            
            # Apply variant to extended sequence
            var_pos_in_ext = pos - ext_start - 1
            alt_ext = ref_ext[:var_pos_in_ext] + v.alt_allele + ref_ext[var_pos_in_ext + len(v.ref_allele):]
            
            # Ensure same length
            min_len = min(len(ref_ext), len(alt_ext))
            ref_ext, alt_ext = ref_ext[:min_len], alt_ext[:min_len]
            
            if len(ref_ext) < 11000:
                continue
            
            # --- VALIDATED DELTA TARGETS ---
            if v.classification == 'Splice-altering':
                target_delta = get_base_delta(ref_ext, alt_ext, base_models, base_device)
            elif v.classification == 'Normal':
                target_delta = np.zeros(3, dtype=np.float32)
            else:
                continue
            
            # Extract context window
            ctx_start = max(0, pos - half - 1)
            ctx_end = pos + half
            alt_ctx = str(fasta[chrom][ctx_start:ctx_end].seq)
            
            # Apply variant to context
            var_pos_in_ctx = pos - ctx_start - 1
            alt_ctx = alt_ctx[:var_pos_in_ctx] + v.alt_allele + alt_ctx[var_pos_in_ctx + len(v.ref_allele):]
            
            # Handle length mismatches
            if len(alt_ctx) != context_size:
                if len(alt_ctx) > context_size:
                    alt_ctx = alt_ctx[:context_size]
                else:
                    alt_ctx = alt_ctx.ljust(context_size, 'N')
            
            samples.append(Sample(
                alt_seq=alt_ctx,
                ref_base=v.ref_allele[0] if v.ref_allele else 'N',
                alt_base=v.alt_allele[0] if v.alt_allele else 'N',
                target_delta=target_delta.astype(np.float32),
                classification=v.classification
            ))
            
            pbar.set_postfix({'samples': len(samples), 'errors': errors})
            
        except Exception as e:
            errors += 1
            continue
    
    return samples


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler,
    config: GPUExperimentConfig
) -> float:
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(dataloader):
        alt_seq = batch['alt_seq'].to(device)
        ref_base = batch['ref_base'].to(device)
        alt_base = batch['alt_base'].to(device)
        target = batch['target_delta'].to(device)
        
        # Mixed precision forward
        with torch.amp.autocast('cuda', enabled=config.use_amp):
            pred = model(alt_seq, ref_base, alt_base)
            loss = F.mse_loss(pred, target)
            loss = loss / config.gradient_accumulation
        
        # Backward with scaler
        scaler.scale(loss).backward()
        
        if (i + 1) % config.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item() * config.gradient_accumulation
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    samples: List[Sample],
    device: torch.device,
    config: GPUExperimentConfig
) -> Dict:
    """Evaluate model on test samples."""
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for s in tqdm(samples, desc="Evaluating", leave=False):
            alt_seq = torch.tensor(one_hot_seq(s.alt_seq), dtype=torch.float32).unsqueeze(0).to(device)
            ref_base = torch.tensor(one_hot_base(s.ref_base), dtype=torch.float32).unsqueeze(0).to(device)
            alt_base = torch.tensor(one_hot_base(s.alt_base), dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                pred = model(alt_seq, ref_base, alt_base)
            
            pred = pred.float().cpu().numpy()[0]
            
            results.append({
                'classification': s.classification,
                'target_delta': s.target_delta,
                'pred_delta': pred,
                'target_max': np.abs(s.target_delta).max(),
                'pred_max': np.abs(pred).max()
            })
    
    # Compute metrics
    sa_results = [r for r in results if r['classification'] == 'Splice-altering']
    sa_target = [r['target_max'] for r in sa_results]
    sa_pred = [r['pred_max'] for r in sa_results]
    
    corr, pval = pearsonr(sa_target, sa_pred) if len(sa_target) > 1 else (0, 1)
    
    all_pred_max = [r['pred_max'] for r in results]
    all_is_sa = [1 if r['classification'] == 'Splice-altering' else 0 for r in results]
    
    auc = roc_auc_score(all_is_sa, all_pred_max) if len(set(all_is_sa)) > 1 else 0
    ap = average_precision_score(all_is_sa, all_pred_max) if len(set(all_is_sa)) > 1 else 0
    
    # Detection rates
    threshold = 0.1
    sa_det = sum(1 for r in results if r['classification'] == 'Splice-altering' and r['pred_max'] > threshold)
    sa_tot = sum(1 for r in results if r['classification'] == 'Splice-altering')
    fp = sum(1 for r in results if r['classification'] == 'Normal' and r['pred_max'] > threshold)
    fp_tot = sum(1 for r in results if r['classification'] == 'Normal')
    
    return {
        'correlation': corr,
        'pvalue': pval,
        'auc': auc,
        'ap': ap,
        'detection_rate': sa_det / max(1, sa_tot),
        'false_positive_rate': fp / max(1, fp_tot),
        'n_test': len(results)
    }


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(config: GPUExperimentConfig) -> Dict:
    """Run a single GPU experiment."""
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"EXPERIMENT: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"{'=' * 80}")
    
    start_time = time.time()
    
    # Device setup
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU (will be slow)")
    
    # Import after path setup
    from meta_spliceai.splice_engine.meta_layer.models.validated_delta_predictor import (
        ValidatedDeltaPredictor
    )
    from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb
    from meta_spliceai.splice_engine.base_models import load_base_model_ensemble
    from meta_spliceai.system.genomic_resources import Registry
    from meta_spliceai.splice_engine.meta_layer.core.path_manager import MetaLayerPathManager
    
    # Load resources
    logger.info("\n1. Loading resources...")
    base_models, metadata = load_base_model_ensemble('openspliceai', context=10000, verbosity=0)
    base_device = metadata['device']
    
    registry = Registry(build='GRCh38')
    from pyfaidx import Fasta
    fasta = Fasta(str(registry.get_fasta_path()), sequence_always_upper=True, rebuild=False)
    
    loader = load_splicevardb(genome_build='GRCh38')
    train_variants, test_variants = loader.get_train_test_split(test_chromosomes=['21', '22'])
    
    # Filter and balance
    train_sa = [v for v in train_variants if v.classification == 'Splice-altering']
    train_normal = [v for v in train_variants if v.classification == 'Normal']
    test_sa = [v for v in test_variants if v.classification == 'Splice-altering']
    test_normal = [v for v in test_variants if v.classification == 'Normal']
    
    logger.info(f"   Available: {len(train_sa)} SA, {len(train_normal)} Normal (train)")
    logger.info(f"   Available: {len(test_sa)} SA, {len(test_normal)} Normal (test)")
    
    random.shuffle(train_sa)
    random.shuffle(train_normal)
    n_each = min(len(train_sa), len(train_normal), config.max_train // 2)
    balanced_train = train_sa[:n_each] + train_normal[:n_each]
    random.shuffle(balanced_train)
    
    logger.info(f"   Using: {n_each} each = {len(balanced_train)} total training variants")
    
    # Prepare training data
    logger.info("\n2. Preparing training data...")
    train_samples = prepare_samples_gpu(
        balanced_train, config.max_train, config.context_size,
        fasta, base_models, base_device, "Train samples"
    )
    logger.info(f"   Prepared {len(train_samples)} training samples")
    
    # Free memory
    del base_models
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Split training data for validation (early stopping)
    if config.val_split > 0:
        val_size = int(len(train_samples) * config.val_split)
        random.shuffle(train_samples)
        val_samples = train_samples[:val_size]
        train_samples = train_samples[val_size:]
        logger.info(f"   Split: {len(train_samples)} train, {len(val_samples)} validation")
        
        val_loader = DataLoader(
            SampleDataset(val_samples),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None
        val_samples = []
    
    train_loader = DataLoader(
        SampleDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    logger.info("\n3. Creating model...")
    if config.use_hyenadna:
        model = HyenaDNAValidatedDeltaPredictor(
            model_name=config.hyenadna_model,
            hidden_dim=config.hidden_dim,
            freeze_encoder=config.freeze_encoder,
            dropout=0.1
        )
    else:
        model = ValidatedDeltaPredictor(
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=0.1
        )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   Device: {device}")
    logger.info(f"   Total params: {n_params:,}")
    logger.info(f"   Trainable params: {n_trainable:,}")
    
    if torch.cuda.is_available():
        logger.info(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Training setup
    logger.info("\n4. Training...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.learning_rate * 10, 
        epochs=config.epochs, 
        steps_per_epoch=len(train_loader) // config.gradient_accumulation
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    stopped_early = False
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, config)
        
        # Validation loss for early stopping
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    alt_seq = batch['alt_seq'].to(device)
                    ref_base = batch['ref_base'].to(device)
                    alt_base = batch['alt_base'].to(device)
                    target = batch['target_delta'].to(device)
                    
                    with torch.amp.autocast('cuda', enabled=config.use_amp):
                        pred = model(alt_seq, ref_base, alt_base)
                        loss = F.mse_loss(pred, target)
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            
            # Early stopping check
            if val_loss < best_val_loss - config.min_delta:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"   Epoch {epoch+1}/{config.epochs}: train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}, patience = {patience_counter}/{config.patience}")
            
            # Early stopping
            if patience_counter >= config.patience:
                logger.info(f"   Early stopping at epoch {epoch+1}! Best val_loss = {best_val_loss:.6f}")
                stopped_early = True
                break
        else:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"   Epoch {epoch+1}/{config.epochs}: loss = {train_loss:.6f}")
                if torch.cuda.is_available():
                    logger.info(f"      GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Restore best model if early stopping was used
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        logger.info(f"   Restored best model (val_loss = {best_val_loss:.6f})")
    
    # Prepare test data
    logger.info("\n5. Preparing test data...")
    
    # Reload base models for test evaluation
    base_models, metadata = load_base_model_ensemble('openspliceai', context=10000, verbosity=0)
    base_device = metadata['device']
    
    test_balanced = test_sa[:config.max_test//2] + test_normal[:config.max_test//2]
    random.shuffle(test_balanced)
    test_samples = prepare_samples_gpu(
        test_balanced, config.max_test, config.context_size,
        fasta, base_models, base_device, "Test samples"
    )
    logger.info(f"   Prepared {len(test_samples)} test samples")
    
    # Free memory again
    del base_models
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate
    logger.info("\n6. Evaluating...")
    metrics = evaluate(model, test_samples, device, config)
    
    elapsed = time.time() - start_time
    
    # Print results
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RESULTS: {config.name}")
    logger.info(f"{'=' * 80}")
    logger.info(f"   Pearson correlation: r = {metrics['correlation']:.4f} (p = {metrics['pvalue']:.2e})")
    logger.info(f"   ROC-AUC: {metrics['auc']:.4f}")
    logger.info(f"   PR-AUC:  {metrics['ap']:.4f}")
    logger.info(f"   Detection @ 0.1: {metrics['detection_rate']*100:.1f}%")
    logger.info(f"   False positive:  {metrics['false_positive_rate']*100:.1f}%")
    logger.info(f"   Time elapsed: {elapsed/60:.1f} minutes")
    
    # Save checkpoint
    pm = MetaLayerPathManager(base_model='openspliceai')
    checkpoint_name = config.name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    cp = pm.get_output_write_dir() / 'checkpoints' / f'gpu_{checkpoint_name}.pt'
    cp.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model': model.state_dict(),
        'config': {
            'name': config.name,
            'max_train': len(train_samples),
            'context_size': config.context_size,
            'epochs': config.epochs,
            'use_hyenadna': config.use_hyenadna,
            'hyenadna_model': config.hyenadna_model if config.use_hyenadna else None
        },
        'metrics': metrics,
        'elapsed_seconds': elapsed
    }
    
    torch.save(save_dict, cp)
    logger.info(f"\nSaved: {cp}")
    
    # Save results JSON for easy comparison
    results_json = cp.with_suffix('.json')
    with open(results_json, 'w') as f:
        json.dump({
            'config': save_dict['config'],
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in metrics.items()},
            'elapsed_minutes': elapsed / 60
        }, f, indent=2)
    
    return metrics


def main():
    """Run GPU experiments."""
    parser = argparse.ArgumentParser(description="GPU ValidatedDeltaPredictor experiments")
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'full_dataset', 'hyenadna', 'hyenadna_medium', 
                                 'long_context', 'quick_test', 'early_stopping',
                                 'early_stopping_regularized'],
                        help='Which experiment to run')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--samples', type=int, default=None,
                        help='Override max training samples')
    args = parser.parse_args()
    
    # Define experiment configurations
    experiments = {
        'quick_test': GPUExperimentConfig(
            name="Quick Test (1000 samples)",
            max_train=1000,
            max_test=200,
            epochs=10,
            batch_size=64,
            device=args.device,
            description="Quick sanity check before full experiments"
        ),
        'full_dataset': GPUExperimentConfig(
            name="Full Dataset (50K samples)",
            max_train=50000,
            max_test=2000,
            epochs=50,
            batch_size=128,
            hidden_dim=256,
            n_layers=8,
            device=args.device,
            description="Full SpliceVarDB dataset - expected r=0.55+"
        ),
        'hyenadna': GPUExperimentConfig(
            name="HyenaDNA-small ValidatedDelta",
            max_train=25000,
            max_test=1000,
            epochs=30,
            batch_size=32,
            hidden_dim=256,
            use_hyenadna=True,
            hyenadna_model='hyenadna-small-32k',
            freeze_encoder=True,
            gradient_accumulation=2,
            device=args.device,
            description="HyenaDNA-small encoder with validated delta targets"
        ),
        'hyenadna_medium': GPUExperimentConfig(
            name="HyenaDNA-medium ValidatedDelta",
            max_train=25000,
            max_test=1000,
            epochs=30,
            batch_size=16,
            hidden_dim=256,
            use_hyenadna=True,
            hyenadna_model='hyenadna-medium-160k',
            freeze_encoder=True,
            gradient_accumulation=4,
            device=args.device,
            description="HyenaDNA-medium encoder (requires A40+)"
        ),
        'long_context': GPUExperimentConfig(
            name="Long Context (1001nt)",
            max_train=25000,
            max_test=1000,
            context_size=1001,
            epochs=40,
            batch_size=64,
            hidden_dim=256,
            n_layers=8,
            device=args.device,
            description="Longer context window for distant regulatory elements"
        ),
        'early_stopping': GPUExperimentConfig(
            name="Full Dataset with Early Stopping",
            max_train=50000,
            max_test=2000,
            epochs=100,  # Max epochs (will stop early)
            batch_size=128,
            hidden_dim=256,
            n_layers=8,
            device=args.device,
            val_split=0.15,  # 15% validation
            patience=7,  # Stop after 7 epochs without improvement
            min_delta=1e-5,  # Minimum improvement threshold
            description="Full dataset with early stopping to prevent overfitting"
        ),
        'early_stopping_regularized': GPUExperimentConfig(
            name="Full Dataset with Early Stopping + Regularization",
            max_train=50000,
            max_test=2000,
            epochs=100,  # Max epochs (will stop early)
            batch_size=128,
            hidden_dim=128,  # Smaller model
            n_layers=6,  # Fewer layers
            learning_rate=5e-5,  # Lower learning rate
            weight_decay=0.05,  # More regularization
            device=args.device,
            val_split=0.15,  # 15% validation
            patience=10,  # More patience
            min_delta=1e-5,
            description="Regularized model with early stopping"
        ),
    }
    
    # Override samples if specified
    if args.samples:
        for exp in experiments.values():
            exp.max_train = args.samples
    
    # Run selected experiments
    if args.exp == 'all':
        all_results = {}
        # Run in order: quick test, then full experiments
        run_order = ['quick_test', 'full_dataset', 'hyenadna', 'long_context']
        
        for name in run_order:
            if name in experiments:
                try:
                    logger.info(f"\n{'#' * 80}")
                    logger.info(f"# Starting: {name}")
                    logger.info(f"{'#' * 80}")
                    metrics = run_experiment(experiments[name])
                    all_results[name] = metrics
                    
                    # Clear GPU memory between experiments
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Experiment {name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Summary
        logger.info(f"\n{'=' * 80}")
        logger.info("SUMMARY: All Experiments")
        logger.info(f"{'=' * 80}")
        logger.info(f"{'Experiment':<35} | {'Corr':>8} | {'AUC':>8} | {'AP':>8}")
        logger.info("-" * 70)
        for name, metrics in all_results.items():
            logger.info(f"{name:<35} | {metrics['correlation']:>8.4f} | "
                       f"{metrics['auc']:>8.4f} | {metrics['ap']:>8.4f}")
    else:
        config = experiments[args.exp]
        run_experiment(config)


if __name__ == '__main__':
    main()

