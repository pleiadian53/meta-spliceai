"""
HyenaDNA-based Validated Delta Predictor with Fine-tuning Support.

This module implements a single-pass delta predictor using HyenaDNA pre-trained
DNA language models, designed for the validated delta target strategy.

Key Features:
- Single-pass inference (no reference sequence needed)
- Gradual unfreezing for fine-tuning
- Discriminative learning rates
- Support for HyenaDNA small/medium/large variants

Architecture:
    alt_seq [B, 4, L] ──→ [HyenaDNA Encoder] ──→ seq_features [B, H]
    ref_base [B, 4]   ──┐
    alt_base [B, 4]   ──┴─→ [Variant Embed] ──→ var_features [B, H]
                                                        ↓
                                            concat [B, 2*H]
                                                        ↓
                                            [Delta Head]
                                                        ↓
                                        Δ = [Δ_donor, Δ_acceptor, Δ_neither]

Usage:
    from meta_spliceai.splice_engine.meta_layer.models.hyenadna_validated_delta import (
        HyenaDNAValidatedDelta,
        create_hyenadna_model
    )
    
    # Frozen encoder (transfer learning)
    model = create_hyenadna_model('hyenadna-small-32k', freeze_encoder=True)
    
    # Fine-tuning last 2 layers
    model = create_hyenadna_model(
        'hyenadna-medium-160k',
        freeze_encoder=False,
        unfreeze_last_n=2
    )
    
    # Get optimizer with discriminative learning rates
    optimizer = model.get_optimizer(base_lr=5e-5, encoder_lr_mult=0.1)
    
    # Forward pass
    delta = model(alt_seq, ref_base, alt_base)  # [B, 3]

References:
    - HyenaDNA: https://github.com/HazyResearch/hyena-dna
    - Paper: https://arxiv.org/abs/2306.15794
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HyenaDNAConfig:
    """Configuration for HyenaDNA model."""
    model_name: str = 'hyenadna-small-32k'
    hidden_dim: int = 256
    freeze_encoder: bool = True
    unfreeze_last_n: int = 0
    dropout: float = 0.1
    use_fp16: bool = True
    
    # Available models and their specifications
    MODELS = {
        'hyenadna-tiny-1k': {'layers': 2, 'dim': 128, 'context': 1024},
        'hyenadna-small-32k': {'layers': 4, 'dim': 256, 'context': 32768},
        'hyenadna-medium-160k': {'layers': 8, 'dim': 256, 'context': 160000},
        'hyenadna-medium-450k': {'layers': 8, 'dim': 256, 'context': 450000},
        'hyenadna-large-1m': {'layers': 12, 'dim': 512, 'context': 1000000},
    }


# =============================================================================
# Model Implementation
# =============================================================================

class HyenaDNAValidatedDelta(nn.Module):
    """
    HyenaDNA-based single-pass validated delta predictor.
    
    Predicts splice probability changes from the alternate sequence alone,
    using the validated delta target strategy:
    - Splice-altering variants: target = base_model(alt) - base_model(ref)
    - Normal variants: target = [0, 0, 0]
    
    Parameters
    ----------
    config : HyenaDNAConfig
        Model configuration
    """
    
    def __init__(self, config: Optional[HyenaDNAConfig] = None):
        super().__init__()
        
        self.config = config or HyenaDNAConfig()
        
        # Load HyenaDNA encoder
        self.encoder, self.encoder_dim, self.num_layers = self._load_encoder()
        
        # Apply freezing strategy
        self._apply_freeze_strategy()
        
        # Projection: encoder_dim → hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(self.encoder_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout)
        )
        
        # Variant embedding: ref_base[4] + alt_base[4] → hidden_dim
        self.variant_embed = nn.Sequential(
            nn.Linear(8, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )
        
        # Delta prediction head: 2*hidden_dim → 3
        self.delta_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 3)  # [Δ_donor, Δ_acceptor, Δ_neither]
        )
        
        self._log_model_info()
    
    def _load_encoder(self) -> Tuple[nn.Module, int, int]:
        """Load HyenaDNA encoder from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM
            
            hf_path = f"LongSafari/{self.config.model_name}"
            logger.info(f"Loading HyenaDNA from {hf_path}...")
            
            dtype = torch.float16 if self.config.use_fp16 else torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                trust_remote_code=True,
                torch_dtype=dtype
            )
            
            # Extract config
            if hasattr(model.config, 'd_model'):
                embed_dim = model.config.d_model
            elif hasattr(model.config, 'hidden_size'):
                embed_dim = model.config.hidden_size
            else:
                embed_dim = 256
            
            if hasattr(model.config, 'n_layer'):
                num_layers = model.config.n_layer
            elif hasattr(model.config, 'num_hidden_layers'):
                num_layers = model.config.num_hidden_layers
            else:
                num_layers = 4
            
            logger.info(f"Loaded: {self.config.model_name}, dim={embed_dim}, layers={num_layers}")
            return model, embed_dim, num_layers
            
        except Exception as e:
            logger.warning(f"Could not load HyenaDNA: {e}. Using CNN fallback.")
            return self._create_fallback_encoder(), 256, 6
    
    def _create_fallback_encoder(self) -> nn.Module:
        """Create CNN fallback if HyenaDNA unavailable."""
        logger.info("Creating CNN fallback encoder")
        return nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def _apply_freeze_strategy(self):
        """Apply freezing strategy to encoder."""
        if self.encoder is None or isinstance(self.encoder, nn.Sequential):
            return
        
        if self.config.freeze_encoder:
            # Freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Encoder fully frozen ({self.num_layers} layers)")
            
        elif self.config.unfreeze_last_n > 0:
            # First freeze all, then unfreeze last N
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            unfrozen = self._unfreeze_layers(self.config.unfreeze_last_n)
            logger.info(f"Unfroze last {unfrozen} layers (of {self.num_layers})")
        else:
            # Fully trainable
            logger.info(f"Encoder fully trainable ({self.num_layers} layers)")
    
    def _unfreeze_layers(self, n: int) -> int:
        """Unfreeze the last N transformer layers."""
        unfrozen = 0
        
        # HyenaDNA structure: backbone.layers
        if hasattr(self.encoder, 'backbone') and hasattr(self.encoder.backbone, 'layers'):
            layers = self.encoder.backbone.layers
            total = len(layers)
            start_idx = max(0, total - n)
            
            for i in range(start_idx, total):
                for param in layers[i].parameters():
                    param.requires_grad = True
                unfrozen += 1
            
            # Also unfreeze final layer norm
            if hasattr(self.encoder.backbone, 'norm_f'):
                for param in self.encoder.backbone.norm_f.parameters():
                    param.requires_grad = True
        
        return unfrozen
    
    def _log_model_info(self):
        """Log model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"HyenaDNAValidatedDelta initialized:")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Freeze mode: {'frozen' if self.config.freeze_encoder else f'unfreeze_last_{self.config.unfreeze_last_n}'}")
    
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Encode DNA sequence to features.
        
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
            # CNN fallback
            features = self.encoder(seq)  # [B, 256, 1]
            return features.squeeze(-1)  # [B, 256]
        else:
            # HyenaDNA: convert one-hot to token IDs
            token_ids = seq.argmax(dim=1)  # [B, L]
            
            # Determine gradient mode
            grad_enabled = not self.config.freeze_encoder or self.config.unfreeze_last_n > 0
            
            with torch.set_grad_enabled(grad_enabled):
                outputs = self.encoder(token_ids, output_hidden_states=True)
                
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden = outputs.hidden_states[-1]  # [B, L, H]
                    features = hidden.mean(dim=1)  # Global average pool [B, H]
                else:
                    features = outputs.logits.mean(dim=1)
            
            return features
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict splice delta scores.
        
        Parameters
        ----------
        alt_seq : torch.Tensor
            Alternate sequence with variant [B, 4, L]
        ref_base : torch.Tensor
            Reference base one-hot [B, 4]
        alt_base : torch.Tensor
            Alternate base one-hot [B, 4]
        
        Returns
        -------
        torch.Tensor
            Delta scores [B, 3] = [Δ_donor, Δ_acceptor, Δ_neither]
        """
        # Encode sequence
        seq_features = self.encode_sequence(alt_seq)  # [B, encoder_dim]
        seq_features = self.proj(seq_features)  # [B, hidden_dim]
        
        # Encode variant info
        var_info = torch.cat([ref_base, alt_base], dim=-1)  # [B, 8]
        var_features = self.variant_embed(var_info)  # [B, hidden_dim]
        
        # Fuse and predict
        combined = torch.cat([seq_features, var_features], dim=-1)  # [B, 2*H]
        delta = self.delta_head(combined)  # [B, 3]
        
        return delta
    
    def get_param_groups(
        self,
        base_lr: float,
        encoder_lr_mult: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups with discriminative learning rates.
        
        Parameters
        ----------
        base_lr : float
            Learning rate for new layers (projection, head)
        encoder_lr_mult : float
            Multiplier for encoder LR (e.g., 0.1 = 10x lower)
        
        Returns
        -------
        List[Dict]
            Parameter groups for optimizer
        """
        encoder_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        
        groups = []
        if encoder_params:
            groups.append({
                'params': encoder_params,
                'lr': base_lr * encoder_lr_mult,
                'name': 'encoder'
            })
        if head_params:
            groups.append({
                'params': head_params,
                'lr': base_lr,
                'name': 'head'
            })
        
        logger.info(f"Parameter groups: encoder={len(encoder_params)} (lr={base_lr * encoder_lr_mult:.2e}), "
                   f"head={len(head_params)} (lr={base_lr:.2e})")
        
        return groups
    
    def get_optimizer(
        self,
        base_lr: float = 5e-5,
        encoder_lr_mult: float = 0.1,
        weight_decay: float = 0.01
    ) -> torch.optim.Optimizer:
        """
        Create optimizer with discriminative learning rates.
        
        Parameters
        ----------
        base_lr : float
            Base learning rate for head layers
        encoder_lr_mult : float
            LR multiplier for encoder (default 0.1 = 10x lower)
        weight_decay : float
            Weight decay for AdamW
        
        Returns
        -------
        torch.optim.Optimizer
            Configured AdamW optimizer
        """
        param_groups = self.get_param_groups(base_lr, encoder_lr_mult)
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# =============================================================================
# Factory Functions
# =============================================================================

def create_hyenadna_model(
    model_name: str = 'hyenadna-small-32k',
    hidden_dim: int = 256,
    freeze_encoder: bool = True,
    unfreeze_last_n: int = 0,
    dropout: float = 0.1,
    use_fp16: bool = True
) -> HyenaDNAValidatedDelta:
    """
    Create a HyenaDNA validated delta model.
    
    Parameters
    ----------
    model_name : str
        HyenaDNA variant: 'hyenadna-small-32k', 'hyenadna-medium-160k', etc.
    hidden_dim : int
        Hidden dimension for projection and head
    freeze_encoder : bool
        If True, freeze all encoder weights (transfer learning)
    unfreeze_last_n : int
        If freeze_encoder=False, number of layers to unfreeze from the end
    dropout : float
        Dropout probability
    use_fp16 : bool
        Use FP16 for encoder (saves memory)
    
    Returns
    -------
    HyenaDNAValidatedDelta
        Configured model
    
    Examples
    --------
    # Transfer learning (frozen encoder)
    model = create_hyenadna_model('hyenadna-small-32k', freeze_encoder=True)
    
    # Fine-tuning last 2 layers
    model = create_hyenadna_model(
        'hyenadna-medium-160k',
        freeze_encoder=False,
        unfreeze_last_n=2
    )
    """
    config = HyenaDNAConfig(
        model_name=model_name,
        hidden_dim=hidden_dim,
        freeze_encoder=freeze_encoder,
        unfreeze_last_n=unfreeze_last_n,
        dropout=dropout,
        use_fp16=use_fp16
    )
    
    return HyenaDNAValidatedDelta(config)


def create_finetuned_model(
    unfreeze_last_n: int = 2,
    model_name: str = 'hyenadna-medium-160k',
    hidden_dim: int = 256
) -> HyenaDNAValidatedDelta:
    """
    Convenience function to create a fine-tuning-ready model.
    
    Parameters
    ----------
    unfreeze_last_n : int
        Number of encoder layers to unfreeze (2-4 recommended)
    model_name : str
        HyenaDNA variant (medium recommended for fine-tuning)
    hidden_dim : int
        Hidden dimension
    
    Returns
    -------
    HyenaDNAValidatedDelta
        Model configured for fine-tuning
    """
    return create_hyenadna_model(
        model_name=model_name,
        hidden_dim=hidden_dim,
        freeze_encoder=False,
        unfreeze_last_n=unfreeze_last_n,
        dropout=0.1,
        use_fp16=True
    )

