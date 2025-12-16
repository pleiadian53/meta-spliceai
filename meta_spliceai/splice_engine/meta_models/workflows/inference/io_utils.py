"""
I/O utilities for selective meta-model inference.

This module contains all file I/O and data management functions extracted from selective_meta_inference.py.
"""

from __future__ import annotations

import json
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import polars as pl

from .config import SelectiveInferenceConfig


def get_test_data_directory(training_dataset_path: Optional[Path]) -> Path:
    """
    Get the test data directory path based on the training dataset path.
    
    If training_dataset_path is 'train_pc_1000_3mers', returns 'test_pc_1000_3mers'.
    
    Parameters
    ----------
    training_dataset_path : Optional[Path]
        Path to the training dataset
        
    Returns
    -------
    Path
        Path to the test data directory
    """
    if training_dataset_path is None:
        return Path("test_data")
    
    # Convert to string and get the last part of the path
    dataset_name = str(training_dataset_path.name)
    
    # Replace 'train' with 'test' in the dataset name
    if dataset_name.startswith("train_"):
        test_dataset_name = dataset_name.replace("train_", "test_", 1)
    else:
        test_dataset_name = f"test_{dataset_name}"
    
    # Return path relative to the parent directory
    if training_dataset_path.parent != Path("."):
        return training_dataset_path.parent / test_dataset_name
    else:
        return Path(test_dataset_name)


def setup_inference_directories(base_dir: Path) -> Dict[str, Path]:
    """
    Set up organized directory structure for selective inference.
    
    Parameters
    ----------
    base_dir : Path
        Base directory for inference outputs
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping directory names to paths
    """
    directories = {
        'base': base_dir,
        'artifacts': base_dir / "artifacts",
        'features': base_dir / "features",
        'predictions': base_dir / "predictions",
        'cache': base_dir / "cache",
        'manifests': base_dir / "cache" / "gene_manifests"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create README
    with open(directories['base'] / "README.md", "w") as f:
        f.write("""# Selective Meta-Model Inference Directory

This directory contains organized artifacts for selective meta-model inference.

## Strategy

- **Selective Featurization**: Generate features only for uncertain positions
- **Base Model Reuse**: Use confident base model predictions directly  
- **Complete Coverage**: Hybrid system provides prediction for every nucleotide
- **Structured Storage**: Organized artifacts with gene tracking

## Structure

- `artifacts/`: Base model predictions and intermediate files
- `features/`: Feature matrices (only for uncertain positions)
- `predictions/`: Hybrid prediction outputs
- `cache/manifests/`: Gene processing tracking
""")
    
    return directories


def create_gene_manifest(
    output_dir: Path,
    gene_ids: List[str],
    data_df: pd.DataFrame,
    verbose: bool = True
) -> Path:
    """
    Create a gene manifest file for test data similar to training data manifest.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save the manifest
    gene_ids : List[str]
        List of gene IDs
    data_df : pd.DataFrame
        DataFrame containing the test data
    verbose : bool
        Enable verbose output
        
    Returns
    -------
    Path
        Path to the created manifest file
    """
    # Get gene features file path using systematic path management
    try:
        from meta_spliceai.system.genomic_resources import create_systematic_manager
        manager = create_systematic_manager()
        gene_features_path = Path(manager.cfg.data_root) / "spliceai_analysis" / "gene_features.tsv"
    except:
        # Fallback to default path
        gene_features_path = Path('data/ensembl/spliceai_analysis/gene_features.tsv')
    
    # Load gene features if available
    gene_names = {}
    if gene_features_path.exists():
        try:
            gene_features = pd.read_csv(gene_features_path, sep='\t')
            gene_name_dict = dict(zip(gene_features['gene_id'], gene_features['gene_name']))
            gene_names = {gid: gene_name_dict.get(gid, 'Unknown') for gid in gene_ids}
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not load gene features: {e}")
            gene_names = {gid: 'Unknown' for gid in gene_ids}
    else:
        gene_names = {gid: 'Unknown' for gid in gene_ids}
    
    # Create manifest data
    manifest_data = []
    for idx, gene_id in enumerate(gene_ids):
        gene_data = data_df[data_df['gene_id'] == gene_id]
        batch_num = (idx // 100) + 1  # 100 genes per batch
        
        manifest_data.append({
            'global_index': idx,
            'gene_id': gene_id,
            'gene_name': gene_names.get(gene_id, 'Unknown'),
            'file_index': batch_num,
            'file_name': f'batch_{batch_num:05d}.parquet',
            'num_positions': len(gene_data)
        })
    
    # Save manifest
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = output_dir / 'gene_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    
    if verbose:
        print(f"   📋 Created gene manifest: {manifest_path}")
        print(f"      - Total genes: {len(gene_ids)}")
        print(f"      - Total batches: {manifest_df['file_index'].max()}")
    
    return manifest_path


def track_processed_genes(
    manifest_path: Path,
    gene_results: Dict[str, Dict],
    config: SelectiveInferenceConfig
) -> None:
    """
    Track gene processing in manifest file.
    
    Parameters
    ----------
    manifest_path : Path
        Path to the manifest file
    gene_results : Dict[str, Dict]
        Processing results for each gene
    config : SelectiveInferenceConfig
        Configuration used for processing
    """
    # Load existing manifest
    existing_data = []
    if manifest_path.exists():
        try:
            existing_df = pd.read_csv(manifest_path)
            existing_data = existing_df.to_dict('records')
        except Exception:
            pass
    
    # Create new records
    timestamp = datetime.datetime.now().isoformat()
    new_records = []
    
    for gene_id, stats in gene_results.items():
        record = {
            'gene_id': gene_id,
            'timestamp': timestamp,
            'total_positions': stats.get('total_positions', 0),
            'recalibrated_positions': stats.get('recalibrated_positions', 0),
            'reused_positions': stats.get('reused_positions', 0),
            'uncertainty_threshold_low': config.uncertainty_threshold_low,
            'uncertainty_threshold_high': config.uncertainty_threshold_high,
            'model_path': str(config.model_path),
            'workflow_version': 'selective_v1.0'
        }
        new_records.append(record)
    
    # Combine and deduplicate
    all_records = existing_data + new_records
    manifest_df = pd.DataFrame(all_records)
    manifest_df = manifest_df.sort_values('timestamp').drop_duplicates('gene_id', keep='last')
    
    manifest_df.to_csv(manifest_path, index=False)


def load_processed_genes(manifest_path: Path) -> Dict[str, Dict]:
    """
    Load gene processing manifest.
    
    Parameters
    ----------
    manifest_path : Path
        Path to the manifest file
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping gene IDs to processing statistics
    """
    if not manifest_path.exists():
        return {}
    
    try:
        manifest_df = pd.read_csv(manifest_path)
        return manifest_df.set_index('gene_id').to_dict('index')
    except Exception:
        return {}


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: Path,
    ensure_string_gene_id: bool = True
) -> None:
    """
    Save predictions to parquet file with proper data types.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions dataframe
    output_path : Path
        Output file path
    ensure_string_gene_id : bool
        Ensure gene_id column is string type
    """
    if ensure_string_gene_id and 'gene_id' in predictions_df.columns:
        predictions_df['gene_id'] = predictions_df['gene_id'].astype(str)
    
    predictions_df.to_parquet(output_path, index=False)


def save_inference_metadata(
    metadata_path: Path,
    config: SelectiveInferenceConfig,
    results: Dict[str, Any],
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save inference metadata to JSON file.
    
    Parameters
    ----------
    metadata_path : Path
        Path to save metadata
    config : SelectiveInferenceConfig
        Configuration used
    results : Dict[str, Any]
        Results dictionary
    additional_info : Dict[str, Any], optional
        Additional information to include
    """
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_path': str(config.model_path),
        'target_genes': config.target_genes,
        'uncertainty_thresholds': [config.uncertainty_threshold_low, config.uncertainty_threshold_high],
        'inference_mode': config.inference_mode,
        'ensure_complete_coverage': config.ensure_complete_coverage,
        'total_positions': results.get('total_positions', 0),
        'uncertain_positions': results.get('uncertain_positions', 0),
        'artifacts_preserved': config.keep_artifacts_dir is not None
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def preserve_artifacts(
    source_dir: Path,
    dest_dir: Path,
    patterns: List[str] = None,
    verbose: bool = True
) -> None:
    """
    Preserve artifacts from temporary directory to permanent location.
    
    Parameters
    ----------
    source_dir : Path
        Source directory containing artifacts
    dest_dir : Path
        Destination directory
    patterns : List[str], optional
        File patterns to preserve (default: all files)
    verbose : bool
        Enable verbose output
    """
    if not source_dir.exists():
        return
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if patterns is None:
        patterns = ['**/*.tsv', '**/*.parquet', '**/*.json', '**/*.csv']
    
    preserved_count = 0
    for pattern in patterns:
        for file_path in source_dir.glob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(source_dir)
                dest_path = dest_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                preserved_count += 1
                if verbose and preserved_count <= 10:  # Show first 10 files
                    print(f"   📄 Preserved: {rel_path}")
    
    if verbose and preserved_count > 10:
        print(f"   📄 ... and {preserved_count - 10} more files")
    
    if verbose:
        print(f"   ✅ Preserved {preserved_count} artifacts in: {dest_dir}")














