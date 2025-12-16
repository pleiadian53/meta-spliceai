"""
Output Management for Inference Workflow

Implements standardized, clean directory structure for predictions and artifacts.

Directory Structure:
    predictions/
        <mode>/                      # base_only, hybrid, meta_only
            <gene_id>/              # Per-gene predictions
                combined_predictions.parquet
            tests/                  # Test outputs (separate from production)
                <gene_id>/
                    combined_predictions.parquet
        spliceai_eval/             # Artifacts (mode-independent)
            meta_models/
                analysis_sequences_*.tsv
                complete_base_predictions/
                    gene_sequence_*.parquet

Created: 2025-10-28
"""

from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass
import logging

InferenceMode = Literal['base_only', 'hybrid', 'meta_only']


@dataclass
class InferenceOutputPaths:
    """Container for all output paths for a gene."""
    
    # Main prediction output
    predictions_file: Path
    
    # Artifacts (mode-independent, shared)
    artifacts_dir: Path
    
    # Per-gene artifact files
    analysis_sequences_dir: Optional[Path] = None
    base_predictions_dir: Optional[Path] = None


class OutputManager:
    """
    Manages output directory structure for inference workflow.
    
    Principles:
    1. Flat structure: <mode>/<gene_id>/ (no extra nesting)
    2. Test separation: <mode>/tests/<gene_id>/
    3. Mode-independent artifacts: spliceai_eval/meta_models/
    4. Clean paths: predictions/{mode}/{gene_id}/combined_predictions.parquet
    """
    
    def __init__(
        self,
        base_dir: Path,
        mode: InferenceMode,
        is_test: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize output manager.
        
        Parameters
        ----------
        base_dir : Path
            Base predictions directory (e.g., /path/to/predictions)
        mode : InferenceMode
            Inference mode (base_only, hybrid, meta_only)
        is_test : bool
            If True, outputs go to <mode>/tests/ subdirectory
        logger : Optional[logging.Logger]
            Logger instance
        """
        self.base_dir = Path(base_dir)
        self.mode = mode
        self.is_test = is_test
        self.logger = logger or logging.getLogger(__name__)
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup base directory structure."""
        # Mode directory
        self.mode_dir = self.base_dir / self.mode
        
        # Test subdirectory if needed
        if self.is_test:
            self.mode_dir = self.mode_dir / "tests"
        
        # Artifacts directory (mode-independent)
        self.artifacts_base = self.base_dir / "spliceai_eval" / "meta_models"
        
        # Create directories
        self.mode_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_base.mkdir(parents=True, exist_ok=True)
    
    def get_gene_output_paths(self, gene_id: str) -> InferenceOutputPaths:
        """
        Get all output paths for a gene.
        
        Parameters
        ----------
        gene_id : str
            Gene ID (e.g., ENSG00000169239)
        
        Returns
        -------
        InferenceOutputPaths
            Container with all output paths
        
        Examples
        --------
        >>> manager = OutputManager(Path("predictions"), "hybrid", is_test=False)
        >>> paths = manager.get_gene_output_paths("ENSG00000169239")
        >>> paths.predictions_file
        Path('predictions/hybrid/ENSG00000169239/combined_predictions.parquet')
        """
        # Gene-specific directory
        gene_dir = self.mode_dir / gene_id
        gene_dir.mkdir(parents=True, exist_ok=True)
        
        # Main predictions file
        predictions_file = gene_dir / "combined_predictions.parquet"
        
        # Artifacts (shared across modes)
        analysis_sequences_dir = self.artifacts_base / "analysis_sequences"
        base_predictions_dir = self.artifacts_base / "complete_base_predictions"
        
        analysis_sequences_dir.mkdir(parents=True, exist_ok=True)
        base_predictions_dir.mkdir(parents=True, exist_ok=True)
        
        return InferenceOutputPaths(
            predictions_file=predictions_file,
            artifacts_dir=self.artifacts_base,
            analysis_sequences_dir=analysis_sequences_dir,
            base_predictions_dir=base_predictions_dir
        )
    
    def get_combined_output_path(self) -> Path:
        """
        Get path for combined predictions across all genes.
        
        Returns
        -------
        Path
            Path to combined predictions file
        
        Examples
        --------
        >>> manager = OutputManager(Path("predictions"), "hybrid")
        >>> manager.get_combined_output_path()
        Path('predictions/hybrid/all_genes_combined.parquet')
        """
        return self.mode_dir / "all_genes_combined.parquet"
    
    def cleanup_old_predictions(self, gene_id: str):
        """
        Clean up old predictions for a gene (for overwrite mode).
        
        Parameters
        ----------
        gene_id : str
            Gene ID to clean up
        """
        gene_dir = self.mode_dir / gene_id
        
        if gene_dir.exists():
            import shutil
            shutil.rmtree(gene_dir)
            self.logger.info(f"Cleaned up old predictions for {gene_id}")
    
    def get_artifact_path(
        self,
        artifact_type: Literal['analysis_sequences', 'base_predictions'],
        filename: str
    ) -> Path:
        """
        Get path for a specific artifact file.
        
        Parameters
        ----------
        artifact_type : Literal['analysis_sequences', 'base_predictions']
            Type of artifact
        filename : str
            Filename (e.g., 'analysis_sequences_1_chunk_1_500.tsv')
        
        Returns
        -------
        Path
            Full path to artifact file
        
        Examples
        --------
        >>> manager = OutputManager(Path("predictions"), "hybrid")
        >>> manager.get_artifact_path('analysis_sequences', 'chunk_1_500.tsv')
        Path('predictions/spliceai_eval/meta_models/analysis_sequences/chunk_1_500.tsv')
        """
        if artifact_type == 'analysis_sequences':
            return self.artifacts_base / "analysis_sequences" / filename
        elif artifact_type == 'base_predictions':
            return self.artifacts_base / "complete_base_predictions" / filename
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    def log_directory_structure(self):
        """Log the current directory structure."""
        self.logger.info("="*80)
        self.logger.info("OUTPUT DIRECTORY STRUCTURE")
        self.logger.info("="*80)
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Test mode: {self.is_test}")
        self.logger.info(f"")
        self.logger.info(f"Predictions: {self.mode_dir}/")
        self.logger.info(f"  ├── <gene_id>/")
        self.logger.info(f"  │   └── combined_predictions.parquet")
        self.logger.info(f"")
        self.logger.info(f"Artifacts: {self.artifacts_base}/")
        self.logger.info(f"  ├── analysis_sequences/")
        self.logger.info(f"  │   └── analysis_sequences_*.tsv")
        self.logger.info(f"  └── complete_base_predictions/")
        self.logger.info(f"      └── gene_sequence_*.parquet")
        self.logger.info("="*80)
    
    @staticmethod
    def create_from_config(
        config: 'EnhancedSelectiveInferenceConfig',
        logger: Optional[logging.Logger] = None
    ) -> 'OutputManager':
        """
        Create OutputManager from inference config.
        
        Parameters
        ----------
        config : EnhancedSelectiveInferenceConfig
            Inference configuration
        logger : Optional[logging.Logger]
            Logger instance
        
        Returns
        -------
        OutputManager
            Configured output manager
        """
        # Determine base directory
        if config.inference_base_dir:
            base_dir = Path(config.inference_base_dir)
        else:
            # Default to predictions/ in project root
            from meta_spliceai.system.genomic_resources import create_systematic_manager
            manager = create_systematic_manager()
            base_dir = Path(manager.cfg.data_root).parent / "predictions"
        
        # Determine if this is a test run
        is_test = config.output_name and 'test' in config.output_name.lower()
        
        return OutputManager(
            base_dir=base_dir,
            mode=config.inference_mode,
            is_test=is_test,
            logger=logger
        )


def migrate_old_predictions(old_base: Path, new_base: Path, dry_run: bool = True):
    """
    Migrate predictions from old structure to new structure.
    
    Old structure:
        predictions/diverse_test_{mode}/predictions/{gene_id}_{mode}/combined_predictions.parquet
    
    New structure:
        predictions/{mode}/{gene_id}/combined_predictions.parquet
        predictions/{mode}/tests/{gene_id}/combined_predictions.parquet (if test)
    
    Parameters
    ----------
    old_base : Path
        Old predictions base directory
    new_base : Path
        New predictions base directory
    dry_run : bool
        If True, only print what would be done
    """
    import shutil
    
    old_base = Path(old_base)
    new_base = Path(new_base)
    
    print("="*80)
    print("PREDICTIONS MIGRATION")
    print("="*80)
    print(f"From: {old_base}")
    print(f"To:   {new_base}")
    print(f"Dry run: {dry_run}")
    print("")
    
    # Find all prediction files in old structure
    pred_files = list(old_base.rglob("combined_predictions.parquet"))
    
    print(f"Found {len(pred_files)} prediction files to migrate")
    print("")
    
    for old_file in pred_files:
        # Parse old path structure
        # Example: predictions/diverse_test_hybrid/predictions/ENSG00000169239_hybrid/combined_predictions.parquet
        parts = old_file.parts
        
        # Extract mode and gene_id
        mode = None
        gene_id = None
        is_test = False
        
        for part in parts:
            if 'base_only' in part or 'hybrid' in part or 'meta_only' in part:
                if 'base_only' in part:
                    mode = 'base_only'
                elif 'hybrid' in part:
                    mode = 'hybrid'
                else:
                    mode = 'meta_only'
                    
                is_test = 'test' in part.lower()
            
            if part.startswith('ENSG'):
                gene_id = part.split('_')[0]  # Remove mode suffix
        
        if not mode or not gene_id:
            print(f"⚠️  Skipping (can't parse): {old_file}")
            continue
        
        # Construct new path
        if is_test:
            new_file = new_base / mode / "tests" / gene_id / "combined_predictions.parquet"
        else:
            new_file = new_base / mode / gene_id / "combined_predictions.parquet"
        
        print(f"{'[DRY RUN] ' if dry_run else ''}Migrating:")
        print(f"  From: {old_file}")
        print(f"  To:   {new_file}")
        
        if not dry_run:
            new_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_file, new_file)
            print(f"  ✅ Copied")
        
        print("")
    
    if dry_run:
        print("="*80)
        print("DRY RUN COMPLETE - No files were moved")
        print("Run with dry_run=False to perform actual migration")
        print("="*80)
    else:
        print("="*80)
        print("MIGRATION COMPLETE")
        print("="*80)

