"""
OpenSpliceAI prediction wrapper for variant scoring.

Provides a clean interface to OpenSpliceAI models for generating
wild-type and alternate predictions for splice site analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


class OpenSpliceAIPredictor:
    """
    Wrapper for OpenSpliceAI models to generate splice predictions.
    
    This class provides a simplified interface for:
    1. Loading OpenSpliceAI PyTorch models
    2. Generating predictions for wild-type and alternate sequences
    3. Computing delta scores for splice-altering variants
    
    Examples
    --------
    >>> predictor = OpenSpliceAIPredictor(model_dir="./data/models/openspliceai")
    >>> results = predictor.predict_variant(
    ...     chrom="7", pos=117199644, ref="C", alt="T",
    ...     sequence="ACGT..." * 1000  # 4kb context
    ... )
    >>> print(f"Donor gain: {results['donor_gain']:.3f}")
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        ensemble: bool = True,
        device: Optional[str] = None,
        context_size: int = 10000,
        verbose: int = 1
    ):
        """
        Initialize OpenSpliceAI predictor.
        
        Parameters
        ----------
        model_dir : str, optional
            Directory containing OpenSpliceAI models.
            If None, uses default location: data/models/openspliceai/
        ensemble : bool
            Whether to use ensemble of 5 models (recommended)
        device : str, optional
            Device for inference ('cpu', 'cuda', 'mps')
            If None, auto-detects best available
        context_size : int
            Context size for predictions (default: 10000 nt)
        verbose : int
            Verbosity level
        """
        self.verbose = verbose
        self.context_size = context_size
        self.ensemble = ensemble
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        
        if self.verbose >= 1:
            logger.info(f"Using device: {self.device}")
        
        # Find model directory
        if model_dir is None:
            model_dir = self._find_default_model_dir()
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"OpenSpliceAI model directory not found: {self.model_dir}\n"
                "Please download models using: "
                "./scripts/base_model/download_openspliceai_models.sh"
            )
        
        # Load models
        self.models = self._load_models()
        
        if self.verbose >= 1:
            logger.info(f"Loaded {len(self.models)} OpenSpliceAI models")
    
    def _find_default_model_dir(self) -> Path:
        """Find default OpenSpliceAI model directory."""
        # Try common locations
        candidates = [
            Path("data/models/openspliceai"),
            Path("./data/models/openspliceai"),
            Path("../data/models/openspliceai"),
            Path(__file__).parents[4] / "data/models/openspliceai",
        ]
        
        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("*.pt")):
                return candidate
        
        # Default to first candidate even if not exists (will raise error later)
        return candidates[0]
    
    def _load_models(self) -> List[torch.nn.Module]:
        """Load OpenSpliceAI PyTorch models."""
        # OpenSpliceAI uses 5 models trained with different random seeds
        model_patterns = [
            "model_10000nt_rs10.pt",
            "model_10000nt_rs11.pt",
            "model_10000nt_rs12.pt",
            "model_10000nt_rs13.pt",
            "model_10000nt_rs14.pt",
        ]
        
        models = []
        for pattern in model_patterns:
            model_path = self.model_dir / pattern
            
            if not model_path.exists():
                if self.ensemble:
                    logger.warning(f"Model not found: {model_path}")
                    continue
                else:
                    raise FileNotFoundError(f"Model not found: {model_path}")
            
            try:
                model = torch.load(model_path, map_location=self.device)
                model.eval()
                models.append(model)
                
                if self.verbose >= 2:
                    logger.debug(f"Loaded model: {pattern}")
                    
            except Exception as e:
                logger.warning(f"Failed to load {pattern}: {e}")
                continue
            
            # If not using ensemble, just load first model
            if not self.ensemble:
                break
        
        if not models:
            raise RuntimeError("No OpenSpliceAI models could be loaded")
        
        return models
    
    def predict_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        sequence: str,
        gene: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Predict splice impact for a single variant.
        
        Parameters
        ----------
        chrom : str
            Chromosome
        pos : int
            1-based genomic position
        ref : str
            Reference allele
        alt : str
            Alternate allele
        sequence : str
            Reference sequence context (must contain variant position)
        gene : str, optional
            Gene name (for metadata)
            
        Returns
        -------
        dict
            Dictionary with keys:
            - donor_gain, donor_loss: Max delta scores for donor sites
            - acceptor_gain, acceptor_loss: Max delta scores for acceptor sites
            - donor_gain_pos, donor_loss_pos: Positions of max deltas
            - acceptor_gain_pos, acceptor_loss_pos: Positions of max deltas
            - wt_scores, alt_scores: Full prediction arrays
        """
        # Generate WT prediction
        wt_scores = self._predict_sequence(sequence)
        
        # Generate ALT sequence
        alt_sequence = self._create_alt_sequence(sequence, pos, ref, alt)
        alt_scores = self._predict_sequence(alt_sequence)
        
        # Compute delta scores
        delta_donor = alt_scores["donor"] - wt_scores["donor"]
        delta_acceptor = alt_scores["acceptor"] - wt_scores["acceptor"]
        
        # Find max gains and losses
        donor_gain_idx = np.argmax(delta_donor)
        donor_loss_idx = np.argmin(delta_donor)
        acceptor_gain_idx = np.argmax(delta_acceptor)
        acceptor_loss_idx = np.argmin(delta_acceptor)
        
        return {
            # Delta scores
            "donor_gain": float(delta_donor[donor_gain_idx]),
            "donor_loss": float(delta_donor[donor_loss_idx]),
            "acceptor_gain": float(delta_acceptor[acceptor_gain_idx]),
            "acceptor_loss": float(delta_acceptor[acceptor_loss_idx]),
            
            # Positions
            "donor_gain_pos": int(donor_gain_idx),
            "donor_loss_pos": int(donor_loss_idx),
            "acceptor_gain_pos": int(acceptor_gain_idx),
            "acceptor_loss_pos": int(acceptor_loss_idx),
            
            # Full scores (for detailed analysis)
            "wt_donor": wt_scores["donor"],
            "wt_acceptor": wt_scores["acceptor"],
            "alt_donor": alt_scores["donor"],
            "alt_acceptor": alt_scores["acceptor"],
            
            # Metadata
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "gene": gene or "",
        }
    
    def _predict_sequence(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Generate OpenSpliceAI predictions for a sequence.
        
        Parameters
        ----------
        sequence : str
            DNA sequence
            
        Returns
        -------
        dict
            Dictionary with 'donor' and 'acceptor' prediction arrays
        """
        # Convert sequence to one-hot encoding
        seq_tensor = self._encode_sequence(sequence)
        
        # Predict with ensemble
        donor_preds = []
        acceptor_preds = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(seq_tensor)
                
                # OpenSpliceAI outputs 3 channels: [acceptor, donor, neither]
                # Extract donor (index 1) and acceptor (index 0)
                donor_preds.append(output[0, 1, :].cpu().numpy())
                acceptor_preds.append(output[0, 0, :].cpu().numpy())
        
        # Average ensemble predictions
        donor_avg = np.mean(donor_preds, axis=0)
        acceptor_avg = np.mean(acceptor_preds, axis=0)
        
        return {
            "donor": donor_avg,
            "acceptor": acceptor_avg
        }
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        One-hot encode DNA sequence for OpenSpliceAI.
        
        Parameters
        ----------
        sequence : str
            DNA sequence
            
        Returns
        -------
        torch.Tensor
            One-hot encoded tensor [1, 4, length]
        """
        # Mapping: A=0, C=1, G=2, T=3
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}  # N -> A
        
        sequence = sequence.upper()
        encoded = np.zeros((4, len(sequence)), dtype=np.float32)
        
        for i, base in enumerate(sequence):
            idx = mapping.get(base, 0)
            encoded[idx, i] = 1.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(encoded).unsqueeze(0)
        return tensor.to(self.device)
    
    def _create_alt_sequence(
        self,
        sequence: str,
        pos: int,
        ref: str,
        alt: str
    ) -> str:
        """
        Create alternate sequence by substituting variant.
        
        Parameters
        ----------
        sequence : str
            Reference sequence
        pos : int
            Variant position (must be within sequence)
        ref : str
            Reference allele
        alt : str
            Alternate allele
            
        Returns
        -------
        str
            Sequence with variant applied
        """
        # For now, handle simple SNVs
        # TODO: Handle indels properly
        if len(ref) == 1 and len(alt) == 1:
            # SNV - simple substitution
            # Assuming pos is 1-based and sequence starts at some offset
            # For simplicity, assume variant is at center
            center = len(sequence) // 2
            alt_seq = sequence[:center] + alt + sequence[center+1:]
            return alt_seq
        else:
            # Complex variant - not implemented yet
            raise NotImplementedError(
                f"Complex variants not yet supported: {ref}>{alt}"
            )
    
    def predict_batch(
        self,
        variants: List[Dict[str, Union[str, int]]],
        sequence_provider: Optional[callable] = None
    ) -> List[Dict[str, float]]:
        """
        Predict splice impact for multiple variants.
        
        Parameters
        ----------
        variants : list of dict
            List of variant dictionaries with keys:
            chrom, pos, ref, alt, gene (optional)
        sequence_provider : callable, optional
            Function that takes (chrom, pos) and returns sequence context
            
        Returns
        -------
        list of dict
            List of prediction results
        """
        results = []
        
        for i, variant in enumerate(variants):
            if self.verbose >= 1 and (i + 1) % 100 == 0:
                print(f"\rProcessed {i+1}/{len(variants)} variants...", end="", flush=True)
            
            try:
                # Get sequence if provider given
                if sequence_provider:
                    sequence = sequence_provider(
                        variant["chrom"],
                        variant["pos"]
                    )
                else:
                    sequence = variant.get("sequence")
                    if not sequence:
                        logger.warning(f"No sequence for variant {i}")
                        continue
                
                # Predict
                result = self.predict_variant(
                    chrom=variant["chrom"],
                    pos=variant["pos"],
                    ref=variant["ref"],
                    alt=variant["alt"],
                    sequence=sequence,
                    gene=variant.get("gene")
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to predict variant {i}: {e}")
                continue
        
        if self.verbose >= 1:
            print()  # New line
        
        return results


# Standalone test function
def test_predictor():
    """Test OpenSpliceAI predictor with demo sequence."""
    print("Testing OpenSpliceAI predictor...")
    
    # Create predictor
    predictor = OpenSpliceAIPredictor(verbose=2)
    
    # Demo sequence (10kb context)
    demo_sequence = "ACGT" * 2500  # Simple test sequence
    
    # Demo variant
    result = predictor.predict_variant(
        chrom="7",
        pos=5000,
        ref="C",
        alt="T",
        sequence=demo_sequence,
        gene="TEST"
    )
    
    print("\n=== Prediction Results ===")
    print(f"Donor gain: {result['donor_gain']:.3f}")
    print(f"Donor loss: {result['donor_loss']:.3f}")
    print(f"Acceptor gain: {result['acceptor_gain']:.3f}")
    print(f"Acceptor loss: {result['acceptor_loss']:.3f}")
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_predictor()





