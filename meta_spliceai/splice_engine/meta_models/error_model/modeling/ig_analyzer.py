"""
Integrated Gradients analyzer for deep error models.

This module implements Integrated Gradients analysis for transformer-based
error classification models to provide interpretability insights.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.error_model.config import ErrorModelConfig, IGAnalysisConfig
from meta_spliceai.splice_engine.meta_models.error_model.dataset.data_utils import DNASequenceDataset, DataUtils
from meta_spliceai.splice_engine.meta_models.error_model.modeling.transformer_trainer import MultiModalTransformerModel


class IGAnalyzer:
    """
    Integrated Gradients analyzer for transformer error models.
    
    This class provides:
    1. IG computation for individual sequences
    2. Token-level attribution analysis
    3. Aggregation across error classes
    4. Visualization and interpretation utilities
    """
    
    def __init__(
        self,
        model: MultiModalTransformerModel,
        tokenizer: AutoTokenizer,
        config: ErrorModelConfig,
        ig_config: Optional[IGAnalysisConfig] = None
    ):
        """
        Initialize IG analyzer.
        
        Parameters
        ----------
        model : MultiModalTransformerModel
            Trained transformer model
        tokenizer : AutoTokenizer
            Tokenizer used for the model
        config : ErrorModelConfig
            Model configuration
        ig_config : IGAnalysisConfig, optional
            IG-specific configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.ig_config = ig_config or IGAnalysisConfig()
        
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Initialize IG
        self.ig = IntegratedGradients(self._forward_func)
        
        # Token mapping for DNA sequences
        self.token_to_nucleotide = self._create_token_mapping()
        
    def _create_token_mapping(self) -> Dict[int, str]:
        """Create mapping from token IDs to nucleotides/tokens."""
        vocab = self.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        
        # Filter for relevant DNA tokens
        dna_tokens = {}
        for token_id, token in id_to_token.items():
            # Keep DNA nucleotides and special tokens
            if any(char in token.upper() for char in 'ATCGN') or token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                dna_tokens[token_id] = token
        
        return dna_tokens
    
    def _forward_func(self, input_ids: torch.Tensor, additional_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward function for IG computation.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs
        additional_features : torch.Tensor, optional
            Additional features
            
        Returns
        -------
        torch.Tensor
            Logits for the target class
        """
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            additional_features=additional_features
        )
        
        return outputs['logits']
    
    def compute_attributions(
        self,
        sequences: List[str],
        labels: List[int],
        additional_features: Optional[np.ndarray] = None,
        target_class: int = 1,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute IG attributions for a list of sequences.
        
        Parameters
        ----------
        sequences : List[str]
            DNA sequences
        labels : List[int]
            True labels for sequences
        additional_features : np.ndarray, optional
            Additional features for sequences
        target_class : int, default 1
            Target class for attribution (0 or 1)
        batch_size : int, optional
            Batch size for processing. If None, uses config batch_size.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of attribution results for each sequence
        """
        self.logger.info(f"Computing IG attributions for {len(sequences)} sequences...")
        
        if batch_size is None:
            batch_size = self.ig_config.batch_size
        
        self.model.eval()
        
        all_attributions = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batch_features = additional_features[i:i+batch_size] if additional_features is not None else None
            
            batch_attributions = self._compute_batch_attributions(
                batch_sequences,
                batch_labels,
                batch_features,
                target_class
            )
            
            all_attributions.extend(batch_attributions)
            
            if (i // batch_size + 1) % 10 == 0:
                self.logger.info(f"Processed {i + len(batch_sequences)} / {len(sequences)} sequences")
        
        self.logger.info("IG attribution computation completed")
        return all_attributions
    
    def _compute_batch_attributions(
        self,
        sequences: List[str],
        labels: List[int],
        additional_features: Optional[np.ndarray],
        target_class: int
    ) -> List[Dict[str, Any]]:
        """Compute attributions for a batch of sequences."""
        batch_attributions = []
        
        for i, (sequence, label) in enumerate(zip(sequences, labels)):
            # Tokenize sequence
            encoding = self.tokenizer(
                sequence,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            
            # Prepare additional features
            features_tensor = None
            if additional_features is not None:
                features_tensor = torch.tensor(
                    additional_features[i:i+1], 
                    dtype=torch.float32,
                    device=self.device
                )
            
            # Create baseline (all padding tokens or zeros)
            if self.ig_config.baseline == "zero":
                baseline_ids = torch.zeros_like(input_ids)
            elif self.ig_config.baseline == "mask":
                baseline_ids = torch.full_like(input_ids, self.tokenizer.mask_token_id)
            else:  # random
                vocab_size = len(self.tokenizer.get_vocab())
                baseline_ids = torch.randint(0, vocab_size, input_ids.shape, device=self.device)
            
            # Compute attributions
            try:
                if features_tensor is not None:
                    # For multi-modal model
                    attributions = self.ig.attribute(
                        inputs=(input_ids, features_tensor),
                        baselines=(baseline_ids, torch.zeros_like(features_tensor)),
                        target=target_class,
                        n_steps=self.ig_config.steps,
                        return_convergence_delta=False
                    )
                    token_attributions = attributions[0].squeeze(0)
                    feature_attributions = attributions[1].squeeze(0)
                else:
                    # For sequence-only model
                    token_attributions = self.ig.attribute(
                        inputs=input_ids,
                        baselines=baseline_ids,
                        target=target_class,
                        n_steps=self.ig_config.steps,
                        return_convergence_delta=False
                    ).squeeze(0)
                    feature_attributions = None
                
                # Process token attributions
                attribution_data = self._process_token_attributions(
                    sequence,
                    input_ids.squeeze(0),
                    token_attributions,
                    label,
                    target_class
                )
                
                # Add feature attributions if available
                if feature_attributions is not None:
                    attribution_data['feature_attributions'] = feature_attributions.cpu().numpy()
                
                batch_attributions.append(attribution_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to compute attributions for sequence {i}: {e}")
                # Add empty attribution data
                batch_attributions.append({
                    'sequence': sequence,
                    'label': label,
                    'target_class': target_class,
                    'tokens': [],
                    'attributions': [],
                    'top_tokens': [],
                    'error': str(e)
                })
        
        return batch_attributions
    
    def _process_token_attributions(
        self,
        sequence: str,
        input_ids: torch.Tensor,
        attributions: torch.Tensor,
        label: int,
        target_class: int
    ) -> Dict[str, Any]:
        """Process token-level attributions."""
        # Convert to numpy
        input_ids_np = input_ids.cpu().numpy()
        attributions_np = attributions.cpu().numpy()
        
        # Get tokens and their attributions
        tokens = []
        token_attributions = []
        
        for token_id, attr in zip(input_ids_np, attributions_np):
            token = self.tokenizer.decode([token_id])
            tokens.append(token)
            token_attributions.append(float(attr))
        
        # Find top-k tokens by absolute attribution
        abs_attributions = np.abs(token_attributions)
        top_indices = np.argsort(abs_attributions)[-self.ig_config.top_k_tokens:][::-1]
        
        top_tokens = []
        for idx in top_indices:
            if abs_attributions[idx] > 1e-6:  # Avoid near-zero attributions
                top_tokens.append({
                    'token': tokens[idx],
                    'position': int(idx),
                    'attribution': token_attributions[idx],
                    'abs_attribution': abs_attributions[idx]
                })
        
        return {
            'sequence': sequence,
            'label': label,
            'target_class': target_class,
            'tokens': tokens,
            'attributions': token_attributions,
            'top_tokens': top_tokens,
            'total_attribution': float(np.sum(token_attributions)),
            'mean_attribution': float(np.mean(token_attributions)),
            'max_attribution': float(np.max(token_attributions)),
            'min_attribution': float(np.min(token_attributions))
        }
    
    def analyze_error_patterns(
        self,
        attributions: List[Dict[str, Any]],
        error_label: int = 1,
        correct_label: int = 0
    ) -> Dict[str, Any]:
        """
        Analyze attribution patterns across error and correct predictions.
        
        Parameters
        ----------
        attributions : List[Dict[str, Any]]
            Attribution results from compute_attributions
        error_label : int, default 1
            Label for error class
        correct_label : int, default 0
            Label for correct class
            
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        self.logger.info("Analyzing error patterns in attributions...")
        
        # Separate by label
        error_attributions = [attr for attr in attributions if attr['label'] == error_label]
        correct_attributions = [attr for attr in attributions if attr['label'] == correct_label]
        
        self.logger.info(f"Error samples: {len(error_attributions)}, Correct samples: {len(correct_attributions)}")
        
        # Analyze token frequencies
        error_token_freq = self._analyze_token_frequency(error_attributions)
        correct_token_freq = self._analyze_token_frequency(correct_attributions)
        
        # Compute token importance ratios
        token_ratios = self._compute_token_ratios(error_token_freq, correct_token_freq)
        
        # Analyze positional patterns
        error_position_stats = self._analyze_positional_patterns(error_attributions)
        correct_position_stats = self._analyze_positional_patterns(correct_attributions)
        
        # Overall statistics
        error_stats = self._compute_attribution_stats(error_attributions)
        correct_stats = self._compute_attribution_stats(correct_attributions)
        
        return {
            'summary': {
                'n_error_samples': len(error_attributions),
                'n_correct_samples': len(correct_attributions),
                'error_stats': error_stats,
                'correct_stats': correct_stats
            },
            'token_analysis': {
                'error_token_freq': error_token_freq,
                'correct_token_freq': correct_token_freq,
                'token_ratios': token_ratios
            },
            'positional_analysis': {
                'error_positions': error_position_stats,
                'correct_positions': correct_position_stats
            }
        }
    
    def _analyze_token_frequency(self, attributions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze frequency of top tokens."""
        token_counts = Counter()
        token_attributions = defaultdict(list)
        total_samples = len(attributions)
        
        for attr in attributions:
            for token_info in attr['top_tokens']:
                token = token_info['token']
                attribution = token_info['attribution']
                
                token_counts[token] += 1
                token_attributions[token].append(attribution)
        
        # Compute statistics for each token
        token_stats = {}
        for token, count in token_counts.items():
            attributions_list = token_attributions[token]
            token_stats[token] = {
                'frequency': count,
                'relative_frequency': count / total_samples,
                'mean_attribution': np.mean(attributions_list),
                'std_attribution': np.std(attributions_list),
                'total_attribution': np.sum(attributions_list)
            }
        
        # Sort by frequency
        return dict(sorted(token_stats.items(), key=lambda x: x[1]['frequency'], reverse=True))
    
    def _compute_token_ratios(
        self,
        error_freq: Dict[str, Dict[str, float]],
        correct_freq: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute token frequency ratios between error and correct classes."""
        all_tokens = set(error_freq.keys()) | set(correct_freq.keys())
        
        ratios = {}
        for token in all_tokens:
            error_rel_freq = error_freq.get(token, {}).get('relative_frequency', 0.0)
            correct_rel_freq = correct_freq.get(token, {}).get('relative_frequency', 0.0)
            
            # Compute ratio with smoothing
            if correct_rel_freq > 0:
                ratio = (error_rel_freq + 1e-6) / (correct_rel_freq + 1e-6)
            else:
                ratio = error_rel_freq + 1e-6
            
            ratios[token] = ratio
        
        # Sort by ratio (descending)
        return dict(sorted(ratios.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_positional_patterns(self, attributions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze positional patterns in attributions."""
        all_positions = []
        all_attributions = []
        
        for attr in attributions:
            for token_info in attr['top_tokens']:
                all_positions.append(token_info['position'])
                all_attributions.append(abs(token_info['attribution']))
        
        if not all_positions:
            return {}
        
        return {
            'mean_position': np.mean(all_positions),
            'std_position': np.std(all_positions),
            'mean_attribution': np.mean(all_attributions),
            'std_attribution': np.std(all_attributions),
            'position_range': [int(np.min(all_positions)), int(np.max(all_positions))]
        }
    
    def _compute_attribution_stats(self, attributions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall attribution statistics."""
        total_attrs = [attr['total_attribution'] for attr in attributions]
        mean_attrs = [attr['mean_attribution'] for attr in attributions]
        max_attrs = [attr['max_attribution'] for attr in attributions]
        min_attrs = [attr['min_attribution'] for attr in attributions]
        
        return {
            'mean_total_attribution': np.mean(total_attrs),
            'std_total_attribution': np.std(total_attrs),
            'mean_mean_attribution': np.mean(mean_attrs),
            'mean_max_attribution': np.mean(max_attrs),
            'mean_min_attribution': np.mean(min_attrs)
        }
    
    def save_results(
        self,
        attributions: List[Dict[str, Any]],
        analysis_results: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Save attribution results and analysis.
        
        Parameters
        ----------
        attributions : List[Dict[str, Any]]
            Attribution results
        analysis_results : Dict[str, Any]
            Analysis results
        output_dir : Path
            Output directory
            
        Returns
        -------
        Dict[str, Path]
            Paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save raw attributions
        attributions_df = pd.DataFrame(attributions)
        attributions_path = output_dir / "attributions.parquet"
        attributions_df.to_parquet(attributions_path, index=False)
        saved_files['attributions'] = attributions_path
        
        # Save analysis results
        analysis_path = output_dir / "analysis_results.json"
        import json
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        saved_files['analysis'] = analysis_path
        
        # Save token frequency tables
        token_freq_path = output_dir / "token_frequencies.csv"
        token_data = []
        
        for token, stats in analysis_results['token_analysis']['error_token_freq'].items():
            correct_stats = analysis_results['token_analysis']['correct_token_freq'].get(token, {})
            ratio = analysis_results['token_analysis']['token_ratios'].get(token, 0.0)
            
            token_data.append({
                'token': token,
                'error_frequency': stats['frequency'],
                'error_rel_frequency': stats['relative_frequency'],
                'error_mean_attribution': stats['mean_attribution'],
                'correct_frequency': correct_stats.get('frequency', 0),
                'correct_rel_frequency': correct_stats.get('relative_frequency', 0.0),
                'correct_mean_attribution': correct_stats.get('mean_attribution', 0.0),
                'frequency_ratio': ratio
            })
        
        token_df = pd.DataFrame(token_data)
        token_df.to_csv(token_freq_path, index=False)
        saved_files['token_frequencies'] = token_freq_path
        
        self.logger.info(f"Saved IG analysis results to {output_dir}")
        return saved_files
