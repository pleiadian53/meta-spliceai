"""
Configuration module for deep error models.

Provides user-configurable parameters for dataset preparation, model training,
and Integrated Gradients analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path


@dataclass
class ErrorModelConfig:
    """Configuration for deep error model training and analysis."""
    
    # Dataset Configuration
    context_length: int = 200  # Total context length (±100 nt around target position)
    error_label: str = "FP"  # Error class to analyze (FP, FN)
    correct_label: str = "TP"  # Correct class for comparison
    splice_type: str = "any"  # Splice type filter: "donor", "acceptor", "any"
    
    # Model Configuration
    model_name: str = "zhihan1996/DNABERT-2-117M"  # Pre-trained model
    tokenizer_name: str = "zhihan1996/DNABERT-2-117M"  # Tokenizer
    max_length: int = 512  # Maximum sequence length for tokenization
    
    # Training Configuration
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data Split Configuration
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    gene_level_split: bool = False  # Whether to split by genes or positions
    
    # Feature Configuration (Additional features beyond primary DNA sequence)
    # Primary feature: DNA sequence from analysis_sequences_* is ALWAYS included
    include_base_scores: bool = True  # Raw SpliceAI scores (acceptor_score, donor_score, neither_score, score)
    include_context_features: bool = True  # Context features (context_score_*, context_neighbor_mean, etc.)
    include_donor_features: bool = True  # Donor-specific features (donor_diff_*, donor_surge_ratio, etc.)
    include_acceptor_features: bool = True  # Acceptor-specific features (acceptor_diff_*, acceptor_surge_ratio, etc.)
    include_derived_features: bool = True  # General statistical features (entropy, variance, ratios, etc.)
    include_genomic_features: bool = False  # Gene-level features (exon_*, intron_*, gene_*, transcript_*)
    
    # IG Analysis Configuration
    ig_steps: int = 50  # Number of steps for IG computation
    ig_baseline: str = "zero"  # Baseline for IG: "zero", "random", "mask"
    top_k_tokens: int = 20  # Top k tokens to analyze
    
    # Output Configuration
    output_dir: Optional[Path] = None
    experiment_name: str = "deep_error_model"
    save_checkpoints: bool = True
    save_predictions: bool = True
    
    # Experiment Tracking Configuration
    enable_mlflow: bool = False  # Enable MLflow experiment tracking
    mlflow_tracking_uri: Optional[str] = None  # MLflow tracking server URI (None = local)
    mlflow_experiment_name: Optional[str] = None  # MLflow experiment name (None = use experiment_name)
    enable_wandb: bool = False  # Enable Weights & Biases tracking
    wandb_project: Optional[str] = None  # W&B project name
    enable_tensorboard: bool = False  # Enable TensorBoard logging
    
    # Computational Configuration
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu", "multi-gpu"
    num_workers: int = 4
    pin_memory: bool = True
    use_mixed_precision: bool = True  # Enable FP16 for better GPU utilization
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.context_length % 2 != 0:
            raise ValueError("context_length must be even for symmetric context")
        
        if not (0 < self.train_split < 1 and 0 < self.val_split < 1 and 0 < self.test_split < 1):
            raise ValueError("All split ratios must be between 0 and 1")
        
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if self.error_label == self.correct_label:
            raise ValueError("error_label and correct_label must be different")
    
    @property
    def context_radius(self) -> int:
        """Half of the context length (radius around target position)."""
        return self.context_length // 2
    
    @property
    def label_mapping(self) -> Dict[str, int]:
        """Mapping from prediction types to binary labels."""
        return {self.error_label: 1, self.correct_label: 0}
    
    @property
    def tracking_integrations(self) -> List[str]:
        """Get list of enabled tracking integrations for HuggingFace Trainer."""
        integrations = []
        if self.enable_mlflow:
            integrations.append("mlflow")
        if self.enable_wandb:
            integrations.append("wandb")
        if self.enable_tensorboard:
            integrations.append("tensorboard")
        return integrations
    
    def get_output_dir(self, base_dir: Optional[Path] = None) -> Path:
        """Get the output directory for this configuration."""
        if self.output_dir is not None:
            return self.output_dir
        
        if base_dir is None:
            base_dir = Path("results/error_models")
        
        dir_name = f"{self.experiment_name}_{self.error_label}_vs_{self.correct_label}"
        if self.splice_type != "any":
            dir_name += f"_{self.splice_type}"
        dir_name += f"_ctx{self.context_length}"
        
        return base_dir / dir_name
    
    def setup_experiment_tracking(self) -> Dict[str, Any]:
        """Setup experiment tracking configuration."""
        tracking_config = {}
        
        if self.enable_mlflow:
            import mlflow
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            experiment_name = self.mlflow_experiment_name or self.experiment_name
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
                mlflow.set_experiment(experiment_name)
                tracking_config['mlflow_experiment_id'] = experiment_id
            except Exception as e:
                print(f"Warning: MLflow setup failed: {e}")
                tracking_config['mlflow_error'] = str(e)
        
        if self.enable_wandb:
            try:
                import wandb
                project_name = self.wandb_project or self.experiment_name
                tracking_config['wandb_project'] = project_name
            except ImportError:
                print("Warning: wandb not installed but enable_wandb=True")
                tracking_config['wandb_error'] = "wandb not installed"
        
        return tracking_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ErrorModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass 
class IGAnalysisConfig:
    """Specific configuration for Integrated Gradients analysis."""
    
    steps: int = 50
    baseline: str = "zero"  # "zero", "random", "mask"
    top_k_tokens: int = 20
    batch_size: int = 8  # Smaller batch size for IG computation
    
    # Visualization options
    create_alignment_plots: bool = True
    create_frequency_plots: bool = True
    save_attributions: bool = True
    
    # Analysis options
    aggregate_by_gene: bool = True
    aggregate_by_error_type: bool = True
    compute_global_stats: bool = True
