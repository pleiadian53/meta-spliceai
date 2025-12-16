"""
Main preprocessing pipeline that integrates OpenSpliceAI with MetaSpliceAI workflows.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

import polars as pl
import pandas as pd

# Import OpenSpliceAI modules
from meta_spliceai.openspliceai.create_data.create_datafile import get_sequences_and_labels, create_datafile
from meta_spliceai.openspliceai.create_data.utils import (
    split_chromosomes, create_or_load_db, one_hot_encode, create_datapoints
)
from meta_spliceai.openspliceai.create_data.create_dataset import create_dataset

# Import MetaSpliceAI modules
from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import (
    prepare_splice_site_annotations, prepare_genomic_sequences
)

# Import local modules
from .config import OpenSpliceAIAdapterConfig
from .data_converter import SpliceDataConverter

logger = logging.getLogger(__name__)

class OpenSpliceAIPreprocessor:
    """
    Main preprocessor that integrates OpenSpliceAI data processing with MetaSpliceAI workflows.
    
    This class provides a bridge between OpenSpliceAI's robust data preprocessing pipeline
    and MetaSpliceAI's meta-learning requirements, allowing reuse of OpenSpliceAI's
    standardized data processing while maintaining compatibility with existing workflows.
    """
    
    def __init__(
        self,
        config: Optional[OpenSpliceAIAdapterConfig] = None,
        gtf_file: Optional[str] = None,
        genome_fasta: Optional[str] = None,
        output_dir: Optional[str] = None,
        verbose: int = 1
    ):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        config : OpenSpliceAIAdapterConfig, optional
            Configuration object. If None, will create from other parameters.
        gtf_file : str, optional
            Path to GTF file (overrides config if provided)
        genome_fasta : str, optional
            Path to genome FASTA file (overrides config if provided)
        output_dir : str, optional
            Output directory (overrides config if provided)
        verbose : int, default=1
            Verbosity level
        """
        self.verbose = verbose
        
        # Initialize configuration
        if config is None:
            config_kwargs = {}
            if gtf_file:
                config_kwargs['gtf_file'] = gtf_file
            if genome_fasta:
                config_kwargs['genome_fasta'] = genome_fasta
            if output_dir:
                config_kwargs['output_dir'] = output_dir
            config = OpenSpliceAIAdapterConfig(**config_kwargs)
        
        self.config = config
        self.converter = SpliceDataConverter(verbose=verbose)
        
        # Validate input files
        self._validate_inputs()
        
        if self.verbose >= 1:
            print(f"[preprocessor] Initialized with GTF: {self.config.gtf_file}")
            print(f"[preprocessor] Genome FASTA: {self.config.genome_fasta}")
            print(f"[preprocessor] Output directory: {self.config.output_dir}")
    
    def _validate_inputs(self):
        """Validate that required input files exist."""
        if not os.path.exists(self.config.gtf_file):
            raise FileNotFoundError(f"GTF file not found: {self.config.gtf_file}")
        if not os.path.exists(self.config.genome_fasta):
            raise FileNotFoundError(f"Genome FASTA not found: {self.config.genome_fasta}")
    
    def create_openspliceai_datasets(
        self,
        train_test_split: bool = True,
        split_method: str = "random",
        split_ratio: float = 0.8
    ) -> Dict[str, str]:
        """
        Create OpenSpliceAI-format datasets using the standard pipeline.
        
        Parameters
        ----------
        train_test_split : bool, default=True
            Whether to create separate train/test datasets
        split_method : str, default="random"
            Method for chromosome splitting ("random" or "human")
        split_ratio : float, default=0.8
            Ratio for train/test split
            
        Returns
        -------
        Dict[str, str]
            Paths to created datasets
        """
        if self.verbose >= 1:
            print("[preprocessor] Creating OpenSpliceAI datasets...")
        
        # Create temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix="openspliceai_temp_")
        
        try:
            # Step 1: Create datafiles using OpenSpliceAI pipeline
            if self.verbose >= 1:
                print("[preprocessor] Step 1: Creating datafiles...")
            
            # Create database from GTF
            db_file = os.path.join(temp_dir, "annotations.db")
            db = create_or_load_db(self.config.gtf_file, db_file)
            
            # Load genome sequences
            from pyfaidx import Fasta
            seq_dict = Fasta(self.config.genome_fasta)
            
            # Split chromosomes for train/test
            if train_test_split:
                chrom_dict = {chrom: 0 for chrom in seq_dict.keys()}
                train_chroms, test_chroms = split_chromosomes(
                    seq_dict, method=split_method, split_ratio=split_ratio
                )
                
                if self.verbose >= 2:
                    print(f"[preprocessor] Train chromosomes: {sorted(train_chroms)}")
                    print(f"[preprocessor] Test chromosomes: {sorted(test_chroms)}")
            
            # Create output paths
            results = {}
            
            # Process training data
            if train_test_split:
                train_file = os.path.join(self.config.output_dir, "datafile_train.h5")
                results['train'] = self._create_single_dataset(
                    db, seq_dict, train_chroms, train_file, "train"
                )
                
                # Process test data
                test_file = os.path.join(self.config.output_dir, "datafile_test.h5")
                results['test'] = self._create_single_dataset(
                    db, seq_dict, test_chroms, test_file, "test"
                )
            else:
                # Create single dataset with all chromosomes
                all_chroms = list(seq_dict.keys())
                single_file = os.path.join(self.config.output_dir, "datafile_all.h5")
                results['all'] = self._create_single_dataset(
                    db, seq_dict, all_chroms, single_file, "all"
                )
            
            if self.verbose >= 1:
                print(f"[preprocessor] Created {len(results)} dataset(s)")
                for name, path in results.items():
                    print(f"[preprocessor]   {name}: {path}")
            
            return results
            
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _create_single_dataset(
        self,
        db: Any,
        seq_dict: Any,
        chromosomes: List[str],
        output_file: str,
        dataset_type: str
    ) -> str:
        """Create a single dataset file."""
        if self.verbose >= 2:
            print(f"[preprocessor] Creating {dataset_type} dataset: {output_file}")
        
        # Filter chromosomes based on target_genes if specified
        target_chroms = chromosomes
        if self.config.target_genes:
            # This would require gene-to-chromosome mapping
            # For now, use all specified chromosomes
            pass
        
        # Create chromosome dictionary for tracking
        chrom_dict = {chrom: 0 for chrom in target_chroms}
        
        # Extract sequences and labels
        NAME, CHROM, STRAND, TX_START, TX_END, SEQ, LABEL = get_sequences_and_labels(
            db=db,
            output_dir=os.path.dirname(output_file),
            seq_dict=seq_dict,
            chrom_dict=chrom_dict,
            train_or_test=dataset_type,
            parse_type=self.config.parse_type,
            biotype=self.config.biotype,
            canonical_only=self.config.canonical_only,
            write_fasta=self.config.write_fasta
        )
        
        # Create HDF5 file
        import h5py
        with h5py.File(output_file, 'w') as f:
            # Store data arrays
            f.create_dataset('NAME', data=[n.encode('utf-8') if isinstance(n, str) else n for n in NAME])
            f.create_dataset('CHROM', data=[c.encode('utf-8') if isinstance(c, str) else c for c in CHROM])
            f.create_dataset('STRAND', data=[s.encode('utf-8') if isinstance(s, str) else s for s in STRAND])
            f.create_dataset('TX_START', data=TX_START)
            f.create_dataset('TX_END', data=TX_END)
            f.create_dataset('SEQ', data=[s.encode('utf-8') if isinstance(s, str) else s for s in SEQ])
            
            # Handle variable-length labels
            dt = h5py.special_dtype(vlen=h5py.special_dtype(vlen=int))
            f.create_dataset('LABEL', data=LABEL, dtype=dt)
        
        return output_file
    
    def create_splicesurveyor_compatible_data(
        self,
        use_openspliceai_preprocessing: bool = True
    ) -> Dict[str, Any]:
        """
        Create data compatible with MetaSpliceAI workflows.
        
        Parameters
        ----------
        use_openspliceai_preprocessing : bool, default=True
            Whether to use OpenSpliceAI preprocessing or fall back to MetaSpliceAI methods
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data and file paths
        """
        if self.verbose >= 1:
            print("[preprocessor] Creating MetaSpliceAI-compatible data...")
        
        results = {}
        
        if use_openspliceai_preprocessing:
            # Use OpenSpliceAI preprocessing then convert
            openspliceai_datasets = self.create_openspliceai_datasets()
            
            # Convert to MetaSpliceAI format
            for dataset_type, h5_file in openspliceai_datasets.items():
                splice_sites_file = self.converter.extract_splice_sites_from_h5(
                    h5_file, output_format="tsv"
                )
                results[f'{dataset_type}_splice_sites'] = splice_sites_file
                results[f'{dataset_type}_h5'] = h5_file
        else:
            # Use existing MetaSpliceAI preprocessing
            if self.verbose >= 2:
                print("[preprocessor] Using MetaSpliceAI preprocessing methods...")
            
            # Prepare splice site annotations
            splice_result = prepare_splice_site_annotations(
                local_dir=self.config.output_dir,
                gtf_file=self.config.gtf_file,
                target_chromosomes=self.config.chromosomes,
                verbosity=self.verbose
            )
            results.update(splice_result)
            
            # Prepare genomic sequences
            seq_result = prepare_genomic_sequences(
                local_dir=self.config.output_dir,
                gtf_file=self.config.gtf_file,
                genome_fasta=self.config.genome_fasta,
                chromosomes=self.config.chromosomes,
                verbosity=self.verbose
            )
            results.update(seq_result)
        
        return results
    
    def create_training_datasets(
        self,
        flanking_size: Optional[int] = None,
        biotype: Optional[str] = None,
        target_genes: Optional[List[str]] = None,
        output_format: str = "hdf5"
    ) -> Dict[str, Any]:
        """
        Create training datasets with specified parameters.
        
        Parameters
        ----------
        flanking_size : int, optional
            Context window size (overrides config)
        biotype : str, optional
            Gene biotype filter (overrides config)
        target_genes : List[str], optional
            Specific genes to process (overrides config)
        output_format : str, default="hdf5"
            Output format ("hdf5", "parquet", "tsv")
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing dataset paths and metadata
        """
        # Update config with provided parameters
        if flanking_size is not None:
            self.config.flanking_size = flanking_size
        if biotype is not None:
            self.config.biotype = biotype
        if target_genes is not None:
            self.config.target_genes = target_genes
        
        if self.verbose >= 1:
            print(f"[preprocessor] Creating training datasets with:")
            print(f"[preprocessor]   flanking_size: {self.config.flanking_size}")
            print(f"[preprocessor]   biotype: {self.config.biotype}")
            print(f"[preprocessor]   target_genes: {len(target_genes) if target_genes else 'all'}")
            print(f"[preprocessor]   output_format: {output_format}")
        
        # Create datasets based on format
        if output_format == "hdf5":
            return self.create_openspliceai_datasets()
        else:
            # Create MetaSpliceAI-compatible format
            results = self.create_splicesurveyor_compatible_data()
            
            # Convert to requested format if needed
            if output_format in ["parquet", "tsv"]:
                converted_results = {}
                for key, path in results.items():
                    if key.endswith('_h5'):
                        # Convert H5 to requested format
                        converted_path = self.converter.extract_splice_sites_from_h5(
                            path, output_format=output_format
                        )
                        converted_results[key.replace('_h5', f'_{output_format}')] = converted_path
                    else:
                        converted_results[key] = path
                results.update(converted_results)
            
            return results
    
    def integrate_with_splicesurveyor_workflow(
        self,
        workflow_config: Any,
        enhance_with_openspliceai: bool = True
    ) -> Dict[str, Any]:
        """
        Integrate with existing MetaSpliceAI workflow.
        
        Parameters
        ----------
        workflow_config : Any
            MetaSpliceAI workflow configuration
        enhance_with_openspliceai : bool, default=True
            Whether to enhance with OpenSpliceAI preprocessing
            
        Returns
        -------
        Dict[str, Any]
            Enhanced workflow results
        """
        if self.verbose >= 1:
            print("[preprocessor] Integrating with MetaSpliceAI workflow...")
        
        # Create adapter config from MetaSpliceAI config
        if enhance_with_openspliceai:
            adapter_config = OpenSpliceAIAdapterConfig.from_splicesurveyor_config(workflow_config)
            enhanced_preprocessor = OpenSpliceAIPreprocessor(adapter_config, verbose=self.verbose)
            
            # Create enhanced datasets
            enhanced_data = enhanced_preprocessor.create_splicesurveyor_compatible_data()
            
            if self.verbose >= 1:
                print("[preprocessor] Enhanced data with OpenSpliceAI preprocessing")
            
            return enhanced_data
        else:
            # Use standard MetaSpliceAI workflow
            return self.create_splicesurveyor_compatible_data(use_openspliceai_preprocessing=False)
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """
        Get quality metrics for the processed datasets.
        
        Returns
        -------
        Dict[str, Any]
            Quality metrics and statistics
        """
        metrics = {
            'config': {
                'flanking_size': self.config.flanking_size,
                'biotype': self.config.biotype,
                'parse_type': self.config.parse_type,
                'canonical_only': self.config.canonical_only
            },
            'input_files': {
                'gtf_file': self.config.gtf_file,
                'genome_fasta': self.config.genome_fasta,
                'gtf_exists': os.path.exists(self.config.gtf_file),
                'fasta_exists': os.path.exists(self.config.genome_fasta)
            },
            'output_directory': self.config.output_dir
        }
        
        # Add file size information if files exist
        if metrics['input_files']['gtf_exists']:
            metrics['input_files']['gtf_size_mb'] = os.path.getsize(self.config.gtf_file) / (1024*1024)
        if metrics['input_files']['fasta_exists']:
            metrics['input_files']['fasta_size_mb'] = os.path.getsize(self.config.genome_fasta) / (1024*1024)
        
        return metrics
