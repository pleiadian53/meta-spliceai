"""
Position Localizer: Multi-Step Framework Step 3.

This module implements models for localizing which positions are affected
by a splice-altering variant. Given a variant context, the model predicts
WHERE the splice effect occurs (which positions have significant delta).

Architecture Options
--------------------
1. AttentionPositionLocalizer: Soft attention distribution over positions
2. SegmentationPositionLocalizer: Binary mask of affected positions  
3. RegressionPositionLocalizer: Direct offset prediction from variant

Training Targets
----------------
Targets are derived from base model delta analysis (recommended) or HGVS parsing:
- See data/position_labels.py for target derivation utilities

Usage
-----
>>> from meta_spliceai.splice_engine.meta_layer.models.position_localizer import (
...     AttentionPositionLocalizer,
...     create_position_localizer
... )
>>> 
>>> model = create_position_localizer(mode='attention', hidden_dim=128)
>>> 
>>> # Forward pass
>>> attention = model(alt_seq, ref_base, alt_base)  # [B, L]
>>> 
>>> # Get top-K positions
>>> top_positions = model.get_top_k_positions(attention, k=3)

See Also
--------
- docs/methods/MULTI_STEP_FRAMEWORK.md: Multi-Step Framework documentation
- data/position_labels.py: Training target derivation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED ENCODER COMPONENTS
# =============================================================================

class GatedResidualBlock(nn.Module):
    """Gated residual block with dilated convolution."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            channels, channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Gated convolution
        out = self.conv(x)  # [B, 2*C, L]
        out, gate = out.chunk(2, dim=1)
        out = out * torch.sigmoid(gate)
        
        # Normalize (requires permute for LayerNorm)
        out = out.permute(0, 2, 1)  # [B, L, C]
        out = self.norm(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # [B, C, L]
        
        return out + residual


class PositionAwareEncoder(nn.Module):
    """
    Gated CNN encoder that preserves per-position features.
    
    Unlike the global pooling encoder in ValidatedDeltaPredictor,
    this encoder maintains spatial resolution for position localization.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        kernel_size: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Initial projection from one-hot [4] to hidden_dim
        self.embed = nn.Conv1d(4, hidden_dim, kernel_size=1)
        
        # Dilated residual blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)  # 1, 2, 4, 8, 1, 2, ...
            self.blocks.append(
                GatedResidualBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to per-position features.
        
        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Per-position features [B, L, hidden_dim]
        """
        x = self.embed(x)  # [B, H, L]
        
        for block in self.blocks:
            x = block(x)
        
        # Permute to [B, L, H] for downstream processing
        x = x.permute(0, 2, 1)
        
        return x
    
    def forward_with_global(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence to both per-position and global features.
        
        Returns
        -------
        tuple
            (per_position [B, L, H], global [B, H])
        """
        x = self.embed(x)  # [B, H, L]
        
        for block in self.blocks:
            x = block(x)
        
        # Global features via pooling
        global_features = x.mean(dim=-1)  # [B, H]
        
        # Per-position features
        position_features = x.permute(0, 2, 1)  # [B, L, H]
        
        return position_features, global_features


# =============================================================================
# ATTENTION-BASED POSITION LOCALIZER (RECOMMENDED)
# =============================================================================

class AttentionPositionLocalizer(nn.Module):
    """
    Attention-based position localizer.
    
    Outputs a soft attention distribution over positions, where high attention
    indicates positions likely affected by the splice-altering variant.
    
    Architecture
    ------------
    ```
    alt_seq ──→ [PositionAwareEncoder] ──→ position_features [B, L, H]
                                                │
    ref_base ──┐                                │
    alt_base ──┴→ [VariantEmbed] ──→ var_features [B, H]
                                                │
                              ┌─────────────────┴─────────────────┐
                              │  Combine: pos_features + var_features
                              │  (broadcast variant info to all positions)
                              └─────────────────┬─────────────────┘
                                                │
                                        [AttentionHead]
                                                │
                                        attention [B, L]
    ```
    
    Training
    --------
    - Target: Soft attention distribution from `create_position_attention_target()`
    - Loss: KL divergence or cross-entropy with soft targets
    
    Inference
    ---------
    - Get top-K positions: `model.get_top_k_positions(attention, k=3)`
    - Get positions above threshold: `model.get_positions_above_threshold(attention, threshold=0.1)`
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    n_layers : int
        Number of encoder layers
    kernel_size : int
        Convolution kernel size
    dropout : float
        Dropout probability
    use_effect_type : bool
        If True, accepts effect type embedding as additional input
    n_effect_types : int
        Number of effect types (for effect type embedding)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        kernel_size: int = 15,
        dropout: float = 0.1,
        use_effect_type: bool = False,
        n_effect_types: int = 5
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_effect_type = use_effect_type
        
        # Sequence encoder (per-position)
        self.encoder = PositionAwareEncoder(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Variant embedding (ref_base + alt_base → hidden_dim)
        self.variant_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Optional: Effect type embedding (from Step 2)
        if use_effect_type:
            self.effect_type_embed = nn.Embedding(n_effect_types, hidden_dim)
        
        # Attention head
        # Input: position_features + variant_features (concatenated or added)
        self.attention_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Temperature for attention sharpening (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor,
        effect_type: Optional[torch.Tensor] = None,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Predict attention distribution over positions.
        
        Parameters
        ----------
        alt_seq : torch.Tensor
            One-hot encoded alternate sequence [B, 4, L]
        ref_base : torch.Tensor
            Reference base one-hot [B, 4]
        alt_base : torch.Tensor
            Alternate base one-hot [B, 4]
        effect_type : torch.Tensor, optional
            Effect type index [B] (from Step 2), if use_effect_type=True
        return_logits : bool
            If True, return raw logits instead of softmax attention
        
        Returns
        -------
        torch.Tensor
            Attention weights [B, L] (sums to 1 along L dimension)
            Or logits [B, L] if return_logits=True
        """
        B, _, L = alt_seq.shape
        
        # Encode sequence to per-position features
        position_features = self.encoder(alt_seq)  # [B, L, H]
        
        # Encode variant info
        var_info = torch.cat([ref_base, alt_base], dim=-1)  # [B, 8]
        var_features = self.variant_embed(var_info)  # [B, H]
        
        # Optionally add effect type embedding
        if self.use_effect_type and effect_type is not None:
            effect_features = self.effect_type_embed(effect_type)  # [B, H]
            var_features = var_features + effect_features
        
        # Broadcast variant features to all positions
        var_features_expanded = var_features.unsqueeze(1).expand(-1, L, -1)  # [B, L, H]
        
        # Combine position and variant features
        combined = torch.cat([position_features, var_features_expanded], dim=-1)  # [B, L, 2H]
        
        # Compute attention logits
        logits = self.attention_head(combined).squeeze(-1)  # [B, L]
        
        if return_logits:
            return logits
        
        # Apply temperature and softmax
        attention = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        
        return attention
    
    def get_top_k_positions(
        self,
        attention: torch.Tensor,
        k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-K positions with highest attention.
        
        Parameters
        ----------
        attention : torch.Tensor
            Attention weights [B, L]
        k : int
            Number of top positions to return
        
        Returns
        -------
        tuple
            (positions [B, k], scores [B, k])
        """
        scores, positions = torch.topk(attention, k=k, dim=-1)
        return positions, scores
    
    def get_positions_above_threshold(
        self,
        attention: torch.Tensor,
        threshold: float = 0.1
    ) -> List[List[int]]:
        """
        Get positions with attention above threshold.
        
        Parameters
        ----------
        attention : torch.Tensor
            Attention weights [B, L]
        threshold : float
            Minimum attention to consider
        
        Returns
        -------
        List[List[int]]
            List of position lists for each batch item
        """
        results = []
        for b in range(attention.shape[0]):
            positions = torch.where(attention[b] >= threshold)[0].tolist()
            results.append(positions)
        return results
    
    def predict_with_interpretation(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor,
        top_k: int = 3,
        effect_type: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with interpretable outputs.
        
        Returns
        -------
        dict
            - 'attention': Full attention distribution [B, L]
            - 'top_positions': Top-K positions [B, k]
            - 'top_scores': Attention at top positions [B, k]
            - 'peak_position': Single most likely position [B]
            - 'peak_score': Attention at peak [B]
        """
        attention = self.forward(alt_seq, ref_base, alt_base, effect_type)
        
        top_positions, top_scores = self.get_top_k_positions(attention, k=top_k)
        
        peak_scores, peak_positions = attention.max(dim=-1)
        
        return {
            'attention': attention,
            'top_positions': top_positions,
            'top_scores': top_scores,
            'peak_position': peak_positions,
            'peak_score': peak_scores
        }


# =============================================================================
# SEGMENTATION-BASED POSITION LOCALIZER
# =============================================================================

class SegmentationPositionLocalizer(nn.Module):
    """
    Binary segmentation position localizer.
    
    Outputs a binary mask indicating which positions are affected by the variant.
    Unlike attention (soft, sums to 1), this can predict multiple independent positions.
    
    Training
    --------
    - Target: Binary mask from `create_binary_position_mask()`
    - Loss: Binary cross-entropy
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    n_layers : int
        Number of encoder layers
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        kernel_size: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Sequence encoder (per-position)
        self.encoder = PositionAwareEncoder(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Variant embedding
        self.variant_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Predict binary mask of affected positions.
        
        Parameters
        ----------
        alt_seq : torch.Tensor
            One-hot encoded alternate sequence [B, 4, L]
        ref_base : torch.Tensor
            Reference base one-hot [B, 4]
        alt_base : torch.Tensor
            Alternate base one-hot [B, 4]
        return_logits : bool
            If True, return raw logits instead of sigmoid
        
        Returns
        -------
        torch.Tensor
            Mask probabilities [B, L] (independent per position)
            Or logits [B, L] if return_logits=True
        """
        B, _, L = alt_seq.shape
        
        # Encode sequence
        position_features = self.encoder(alt_seq)  # [B, L, H]
        
        # Encode variant
        var_info = torch.cat([ref_base, alt_base], dim=-1)
        var_features = self.variant_embed(var_info)  # [B, H]
        var_features_expanded = var_features.unsqueeze(1).expand(-1, L, -1)
        
        # Combine and predict
        combined = torch.cat([position_features, var_features_expanded], dim=-1)
        logits = self.seg_head(combined).squeeze(-1)  # [B, L]
        
        if return_logits:
            return logits
        
        return torch.sigmoid(logits)
    
    def get_affected_positions(
        self,
        mask: torch.Tensor,
        threshold: float = 0.5
    ) -> List[List[int]]:
        """
        Get positions predicted as affected (above threshold).
        
        Returns
        -------
        List[List[int]]
            List of position lists for each batch item
        """
        results = []
        for b in range(mask.shape[0]):
            positions = torch.where(mask[b] >= threshold)[0].tolist()
            results.append(positions)
        return results


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def attention_kl_loss(
    pred_attention: torch.Tensor,
    target_attention: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    KL divergence loss for attention distributions.
    
    Parameters
    ----------
    pred_attention : torch.Tensor
        Predicted attention [B, L] (should sum to 1)
    target_attention : torch.Tensor
        Target attention [B, L] (should sum to 1)
    eps : float
        Small constant for numerical stability
    
    Returns
    -------
    torch.Tensor
        Scalar loss
    """
    # Ensure valid distributions
    pred = pred_attention.clamp(min=eps)
    target = target_attention.clamp(min=eps)
    
    # Normalize (in case not already)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    target = target / target.sum(dim=-1, keepdim=True)
    
    # KL(target || pred) = sum(target * log(target / pred))
    kl = target * (torch.log(target) - torch.log(pred))
    
    return kl.sum(dim=-1).mean()


def attention_cross_entropy_loss(
    pred_logits: torch.Tensor,
    target_attention: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Cross-entropy loss for attention with soft targets.
    
    Parameters
    ----------
    pred_logits : torch.Tensor
        Predicted logits [B, L] (before softmax)
    target_attention : torch.Tensor
        Target attention distribution [B, L]
    
    Returns
    -------
    torch.Tensor
        Scalar loss
    """
    # Log-softmax of predictions
    log_probs = F.log_softmax(pred_logits, dim=-1)
    
    # Normalize targets
    target = target_attention.clamp(min=eps)
    target = target / target.sum(dim=-1, keepdim=True)
    
    # Cross-entropy with soft targets
    loss = -(target * log_probs).sum(dim=-1)
    
    return loss.mean()


def focal_attention_loss(
    pred_attention: torch.Tensor,
    target_attention: torch.Tensor,
    gamma: float = 2.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Focal loss variant for attention, emphasizing hard examples.
    
    Parameters
    ----------
    pred_attention : torch.Tensor
        Predicted attention [B, L]
    target_attention : torch.Tensor
        Target attention [B, L]
    gamma : float
        Focusing parameter (higher = more focus on hard examples)
    
    Returns
    -------
    torch.Tensor
        Scalar loss
    """
    pred = pred_attention.clamp(min=eps, max=1-eps)
    target = target_attention.clamp(min=eps)
    target = target / target.sum(dim=-1, keepdim=True)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - pred).pow(gamma)
    
    # Cross-entropy with focal weight
    ce = -target * torch.log(pred)
    focal_loss = focal_weight * ce
    
    return focal_loss.sum(dim=-1).mean()


def segmentation_loss(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    pos_weight: float = 2.0
) -> torch.Tensor:
    """
    Binary cross-entropy loss for segmentation.
    
    Parameters
    ----------
    pred_logits : torch.Tensor
        Predicted logits [B, L]
    target_mask : torch.Tensor
        Target binary mask [B, L]
    pos_weight : float
        Weight for positive class (affected positions are rare)
    
    Returns
    -------
    torch.Tensor
        Scalar loss
    """
    weight = torch.ones_like(target_mask)
    weight[target_mask > 0.5] = pos_weight
    
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, target_mask, weight=weight
    )
    
    return loss


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_position_localizer(
    mode: str = 'attention',
    hidden_dim: int = 128,
    n_layers: int = 6,
    dropout: float = 0.1,
    use_effect_type: bool = False,
    **kwargs
) -> nn.Module:
    """
    Create a position localizer model.
    
    Parameters
    ----------
    mode : str
        'attention' or 'segmentation'
    hidden_dim : int
        Hidden dimension
    n_layers : int
        Number of encoder layers
    dropout : float
        Dropout probability
    use_effect_type : bool
        Whether to use effect type from Step 2 (attention mode only)
    
    Returns
    -------
    nn.Module
        Position localizer model
    
    Examples
    --------
    >>> model = create_position_localizer(mode='attention', hidden_dim=128)
    >>> attention = model(alt_seq, ref_base, alt_base)
    """
    if mode == 'attention':
        return AttentionPositionLocalizer(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            use_effect_type=use_effect_type,
            **kwargs
        )
    elif mode == 'segmentation':
        return SegmentationPositionLocalizer(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'attention' or 'segmentation'")


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_position_localization(
    model: nn.Module,
    dataloader,
    device: str = 'cpu',
    threshold: float = 0.1,
    top_k: int = 3
) -> Dict[str, float]:
    """
    Evaluate position localization model.
    
    Metrics
    -------
    - hit_rate_top1: Fraction where top-1 prediction is within target region
    - hit_rate_top3: Fraction where any top-3 prediction is within target region
    - mean_distance: Mean distance from predicted peak to target peak
    - recall_at_threshold: Fraction of target positions recalled at threshold
    
    Parameters
    ----------
    model : nn.Module
        Position localizer model
    dataloader : DataLoader
        Evaluation data
    device : str
        Device for inference
    threshold : float
        Threshold for "affected" in target
    top_k : int
        Number of top predictions to consider
    
    Returns
    -------
    Dict[str, float]
        Evaluation metrics
    """
    model.eval()
    model.to(device)
    
    total = 0
    hit_top1 = 0
    hit_top3 = 0
    total_distance = 0
    total_recall = 0
    recall_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            alt_seq = batch['alt_seq'].to(device)
            ref_base = batch['ref_base'].to(device)
            alt_base = batch['alt_base'].to(device)
            target = batch['position_target'].to(device)  # [B, L]
            
            # Get predictions
            attention = model(alt_seq, ref_base, alt_base)
            
            # Get top-K positions
            top_positions, _ = model.get_top_k_positions(attention, k=top_k)
            
            # Find target peak positions
            target_peaks = target.argmax(dim=-1)  # [B]
            
            # Check if top-1 is within ±5bp of target
            top1_positions = top_positions[:, 0]
            distance = (top1_positions - target_peaks).abs()
            hit_top1 += (distance <= 5).sum().item()
            total_distance += distance.float().sum().item()
            
            # Check if any top-3 is within ±5bp of target
            for i in range(top_positions.shape[0]):
                target_peak = target_peaks[i].item()
                any_hit = any(
                    abs(top_positions[i, k].item() - target_peak) <= 5
                    for k in range(min(top_k, top_positions.shape[1]))
                )
                if any_hit:
                    hit_top3 += 1
            
            # Recall: fraction of target positions (> threshold) found in predictions
            for i in range(attention.shape[0]):
                target_positions = torch.where(target[i] >= threshold)[0]
                if len(target_positions) > 0:
                    pred_positions = torch.where(attention[i] >= threshold)[0]
                    if len(pred_positions) > 0:
                        # Count how many target positions are within ±2bp of any prediction
                        recalled = 0
                        for tp in target_positions:
                            if any(abs(tp - pp) <= 2 for pp in pred_positions):
                                recalled += 1
                        total_recall += recalled / len(target_positions)
                    recall_count += 1
            
            total += alt_seq.shape[0]
    
    metrics = {
        'hit_rate_top1': hit_top1 / total if total > 0 else 0,
        'hit_rate_top3': hit_top3 / total if total > 0 else 0,
        'mean_distance': total_distance / total if total > 0 else 0,
        'recall_at_threshold': total_recall / recall_count if recall_count > 0 else 0
    }
    
    return metrics

