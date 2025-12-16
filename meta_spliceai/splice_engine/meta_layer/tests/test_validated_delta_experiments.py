"""
Test script for ValidatedDeltaPredictor experiments.

This script runs experiments with various configurations:
1. More training data (8000 samples)
2. Longer context (1001nt)
3. Attention variant for interpretability

Usage:
    # Run all experiments
    python -m meta_spliceai.splice_engine.meta_layer.tests.test_validated_delta_experiments
    
    # Run specific experiment
    python -m meta_spliceai.splice_engine.meta_layer.tests.test_validated_delta_experiments --exp more_data
    python -m meta_spliceai.splice_engine.meta_layer.tests.test_validated_delta_experiments --exp longer_context
    python -m meta_spliceai.splice_engine.meta_layer.tests.test_validated_delta_experiments --exp attention

Note: These experiments can be time-consuming on CPU/M1. 
      For faster iteration, consider running on GPU (RunPods).
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_paths():
    """Add project root to path."""
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_paths()

from meta_spliceai.splice_engine.meta_layer.models.validated_delta_predictor import (
    ValidatedDeltaPredictor,
    ValidatedDeltaPredictorWithAttention,
    one_hot_seq,
    one_hot_base
)
from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb
from meta_spliceai.splice_engine.base_models import load_base_model_ensemble
from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.splice_engine.meta_layer.core.path_manager import MetaLayerPathManager


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    max_train: int = 2000
    max_test: int = 300
    context_size: int = 501
    epochs: int = 40
    batch_size: int = 32
    hidden_dim: int = 128
    n_layers: int = 6
    use_attention: bool = False
    description: str = ""


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


def prepare_samples(
    variants: List,
    max_n: int,
    context_size: int,
    fasta,
    base_models: List,
    base_device: str,
    desc: str = "Preparing"
) -> List[Sample]:
    """
    Prepare samples with validated delta targets.
    
    Target derivation:
    - Splice-altering: Use base model delta (trusted because SpliceVarDB confirms)
    - Normal: Zero delta (override base model - we know there's no effect)
    - Low-frequency/Conflicting: Skip (uncertain)
    """
    samples = []
    half = context_size // 2
    
    for v in tqdm(variants[:int(max_n * 1.5)], desc=desc, leave=False):
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
            # This is the key innovation: use SpliceVarDB to validate targets
            if v.classification == 'Splice-altering':
                # SpliceVarDB confirms this is splice-altering
                # Trust the base model's delta prediction
                target_delta = get_base_delta(ref_ext, alt_ext, base_models, base_device)
            elif v.classification == 'Normal':
                # SpliceVarDB confirms this is NOT splice-altering
                # Target should be ZERO, regardless of what base model predicts
                target_delta = np.zeros(3, dtype=np.float32)
            else:
                # Low-frequency, Conflicting - uncertain labels, skip
                continue
            
            # Extract context window for model input
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
            
        except Exception as e:
            continue
    
    return samples


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run a single experiment."""
    
    logger.info(f"\n{'=' * 75}")
    logger.info(f"EXPERIMENT: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"{'=' * 75}")
    
    logger.info(f"Config: {config.max_train} train, {config.max_test} test, "
                f"{config.epochs} epochs, context={config.context_size}")
    
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
    
    logger.info(f"   Available: {len(train_sa)} SA, {len(train_normal)} Normal")
    
    random.shuffle(train_sa)
    random.shuffle(train_normal)
    n_each = min(len(train_sa), len(train_normal), config.max_train // 2)
    balanced_train = train_sa[:n_each] + train_normal[:n_each]
    random.shuffle(balanced_train)
    
    logger.info(f"   Using: {n_each} each = {len(balanced_train)} total")
    
    # Prepare data
    logger.info("\n2. Preparing training data...")
    train_samples = prepare_samples(
        balanced_train, config.max_train, config.context_size,
        fasta, base_models, base_device, "Train"
    )
    logger.info(f"   Prepared {len(train_samples)} training samples")
    
    train_loader = DataLoader(
        SampleDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Create model
    logger.info("\n3. Creating model...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    if config.use_attention:
        model = ValidatedDeltaPredictorWithAttention(
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=0.1
        )
    else:
        model = ValidatedDeltaPredictor(
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=0.1
        )
    
    model = model.to(device)
    logger.info(f"   Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    logger.info("\n4. Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, epochs=config.epochs, steps_per_epoch=len(train_loader)
    )
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            alt_seq = batch['alt_seq'].to(device)
            ref_base = batch['ref_base'].to(device)
            alt_base = batch['alt_base'].to(device)
            target = batch['target_delta'].to(device)
            
            optimizer.zero_grad()
            
            if config.use_attention:
                pred, _ = model(alt_seq, ref_base, alt_base)
            else:
                pred = model(alt_seq, ref_base, alt_base)
            
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"   Epoch {epoch+1}: loss = {total_loss/len(train_loader):.6f}")
    
    # Prepare test set
    logger.info("\n5. Preparing test set...")
    test_balanced = test_sa[:config.max_test//2] + test_normal[:config.max_test//2]
    random.shuffle(test_balanced)
    test_samples = prepare_samples(
        test_balanced, config.max_test, config.context_size,
        fasta, base_models, base_device, "Test"
    )
    logger.info(f"   Prepared {len(test_samples)} test samples")
    
    # Evaluate
    logger.info("\n6. Evaluating...")
    model.eval()
    
    results = []
    attention_weights = [] if config.use_attention else None
    
    with torch.no_grad():
        for s in test_samples:
            alt_seq = torch.tensor(one_hot_seq(s.alt_seq), dtype=torch.float32).unsqueeze(0).to(device)
            ref_base = torch.tensor(one_hot_base(s.ref_base), dtype=torch.float32).unsqueeze(0).to(device)
            alt_base = torch.tensor(one_hot_base(s.alt_base), dtype=torch.float32).unsqueeze(0).to(device)
            
            if config.use_attention:
                pred, attn = model(alt_seq, ref_base, alt_base)
                attention_weights.append(attn.cpu().numpy()[0])
            else:
                pred = model(alt_seq, ref_base, alt_base)
            
            pred = pred.cpu().numpy()[0]
            
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
    corr, pval = pearsonr(sa_target, sa_pred)
    
    all_pred_max = [r['pred_max'] for r in results]
    all_is_sa = [1 if r['classification'] == 'Splice-altering' else 0 for r in results]
    auc = roc_auc_score(all_is_sa, all_pred_max)
    ap = average_precision_score(all_is_sa, all_pred_max)
    
    # Detection rates
    threshold = 0.1
    sa_det = sum(1 for r in results if r['classification'] == 'Splice-altering' and r['pred_max'] > threshold)
    sa_tot = sum(1 for r in results if r['classification'] == 'Splice-altering')
    fp = sum(1 for r in results if r['classification'] == 'Normal' and r['pred_max'] > threshold)
    fp_tot = sum(1 for r in results if r['classification'] == 'Normal')
    
    # Print results
    logger.info(f"\n{'=' * 75}")
    logger.info(f"RESULTS: {config.name}")
    logger.info(f"{'=' * 75}")
    logger.info(f"   Pearson correlation: r = {corr:.4f} (p = {pval:.2e})")
    logger.info(f"   ROC-AUC: {auc:.4f}")
    logger.info(f"   PR-AUC:  {ap:.4f}")
    logger.info(f"   Detection @ 0.1: {sa_det}/{sa_tot} ({100*sa_det/max(1,sa_tot):.1f}%)")
    logger.info(f"   False positive:  {fp}/{fp_tot} ({100*fp/max(1,fp_tot):.1f}%)")
    
    # Save checkpoint
    pm = MetaLayerPathManager(base_model='openspliceai')
    checkpoint_name = config.name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    cp = pm.get_output_write_dir() / 'checkpoints' / f'validated_delta_{checkpoint_name}.pt'
    cp.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model': model.state_dict(),
        'config': {
            'name': config.name,
            'max_train': len(train_samples),
            'context_size': config.context_size,
            'epochs': config.epochs,
            'use_attention': config.use_attention
        },
        'metrics': {
            'correlation': corr,
            'auc': auc,
            'ap': ap,
            'detection_rate': sa_det / max(1, sa_tot),
            'false_positive_rate': fp / max(1, fp_tot)
        }
    }
    
    if attention_weights:
        save_dict['sample_attention'] = attention_weights[:10]  # Save a few examples
    
    torch.save(save_dict, cp)
    logger.info(f"\nSaved: {cp}")
    
    return save_dict['metrics']


def main():
    """Run experiments."""
    parser = argparse.ArgumentParser(description="ValidatedDeltaPredictor experiments")
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'more_data', 'longer_context', 'attention'],
                        help='Which experiment to run')
    args = parser.parse_args()
    
    # Define experiment configurations
    experiments = {
        'more_data': ExperimentConfig(
            name="More Data (8000)",
            max_train=8000,
            max_test=500,
            epochs=50,
            batch_size=64,
            description="4x more training data"
        ),
        'longer_context': ExperimentConfig(
            name="Longer Context (1001nt)",
            max_train=2000,
            max_test=300,
            context_size=1001,
            epochs=40,
            description="2x longer context window"
        ),
        'attention': ExperimentConfig(
            name="With Attention",
            max_train=2000,
            max_test=300,
            epochs=40,
            use_attention=True,
            description="Position attention for interpretability"
        ),
    }
    
    # Run selected experiments
    if args.exp == 'all':
        all_results = {}
        for name, config in experiments.items():
            try:
                metrics = run_experiment(config)
                all_results[name] = metrics
            except Exception as e:
                logger.error(f"Experiment {name} failed: {e}")
                continue
        
        # Summary
        logger.info(f"\n{'=' * 75}")
        logger.info("SUMMARY: All Experiments")
        logger.info(f"{'=' * 75}")
        logger.info(f"{'Experiment':<25} | {'Corr':>8} | {'AUC':>8} | {'AP':>8}")
        logger.info("-" * 60)
        for name, metrics in all_results.items():
            logger.info(f"{name:<25} | {metrics['correlation']:>8.4f} | "
                       f"{metrics['auc']:>8.4f} | {metrics['ap']:>8.4f}")
    else:
        config = experiments[args.exp]
        run_experiment(config)


if __name__ == '__main__':
    main()

