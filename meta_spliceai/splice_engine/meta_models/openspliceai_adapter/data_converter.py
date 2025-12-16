"""
Data format conversion utilities between OpenSpliceAI and MetaSpliceAI formats.
"""

import os
import h5py
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

# Setup logging
logger = logging.getLogger(__name__)

class SpliceDataConverter:
    """Converts data between OpenSpliceAI and MetaSpliceAI formats."""
    
    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        
        # OpenSpliceAI label mapping
        self.openspliceai_labels = {
            0: "neither",
            1: "acceptor", 
            2: "donor"
        }
        
        # MetaSpliceAI label mapping (reverse for compatibility)
        self.splicesurveyor_labels = {
            "neither": 0,
            "acceptor": 1,
            "donor": 2
        }
        
        # One-hot encoding mappings from OpenSpliceAI
        self.nucleotide_map = {
            'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4
        }
        
        self.one_hot_map = np.array([
            [0, 0, 0, 0],  # N
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]   # T
        ])
    
    def convert_h5_to_dataframe(
        self, 
        h5_file: str, 
        output_format: str = "polars"
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Convert OpenSpliceAI HDF5 format to DataFrame.
        
        Parameters
        ----------
        h5_file : str
            Path to OpenSpliceAI HDF5 file
        output_format : str
            Output format ("pandas" or "polars")
            
        Returns
        -------
        DataFrame
            Converted data in requested format
        """
        if self.verbose >= 1:
            print(f"[converter] Converting H5 file: {h5_file}")
            
        data_records = []
        
        with h5py.File(h5_file, 'r') as f:
            # Get dataset keys
            gene_names = f['NAME'][:]
            sequences = f['SEQ'][:]
            labels = f['LABEL'][:]
            chromosomes = f['CHROM'][:]
            strands = f['STRAND'][:]
            tx_starts = f['TX_START'][:]
            tx_ends = f['TX_END'][:]
            
            # Convert each gene record
            for i in range(len(gene_names)):
                gene_name = gene_names[i].decode('utf-8') if isinstance(gene_names[i], bytes) else gene_names[i]
                sequence = sequences[i].decode('utf-8') if isinstance(sequences[i], bytes) else sequences[i]
                chromosome = chromosomes[i].decode('utf-8') if isinstance(chromosomes[i], bytes) else chromosomes[i]
                strand = strands[i].decode('utf-8') if isinstance(strands[i], bytes) else strands[i]
                
                # Process labels for each position in the sequence
                gene_labels = labels[i]
                
                for pos, label in enumerate(gene_labels):
                    if label > 0:  # Only include splice sites (skip "neither" positions for efficiency)
                        record = {
                            'gene_id': gene_name,
                            'chromosome': chromosome,
                            'position': tx_starts[i] + pos,
                            'strand': strand,
                            'splice_type': self.openspliceai_labels[label],
                            'sequence_context': sequence[max(0, pos-50):pos+50],  # 100bp context
                            'tx_start': tx_starts[i],
                            'tx_end': tx_ends[i]
                        }
                        data_records.append(record)
        
        # Convert to requested format
        if output_format == "polars":
            return pl.DataFrame(data_records)
        else:
            return pd.DataFrame(data_records)
    
    def convert_dataframe_to_h5(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        output_file: str,
        sequence_column: str = "sequence",
        label_column: str = "splice_type"
    ) -> str:
        """
        Convert DataFrame to OpenSpliceAI HDF5 format.
        
        Parameters
        ----------
        df : DataFrame
            Input dataframe with sequence and label data
        output_file : str
            Path for output HDF5 file
        sequence_column : str
            Column name containing sequences
        label_column : str
            Column name containing splice site labels
            
        Returns
        -------
        str
            Path to created HDF5 file
        """
        if self.verbose >= 1:
            print(f"[converter] Converting DataFrame to H5: {output_file}")
            
        # Convert polars to pandas if needed
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
            
        # Group by gene for HDF5 structure
        gene_groups = df.groupby('gene_id')
        
        # Prepare data arrays
        names, sequences, labels, chroms, strands, tx_starts, tx_ends = [], [], [], [], [], [], []
        
        for gene_id, group in gene_groups:
            names.append(gene_id)
            
            # Get gene sequence (assuming first row has the full sequence)
            gene_sequence = group[sequence_column].iloc[0]
            sequences.append(gene_sequence)
            
            # Create label array for the sequence
            seq_length = len(gene_sequence)
            gene_labels = np.zeros(seq_length, dtype=int)
            
            # Set labels for splice sites
            for _, row in group.iterrows():
                pos = row.get('position', 0)
                if 0 <= pos < seq_length:
                    splice_type = row[label_column]
                    if splice_type in self.splicesurveyor_labels:
                        gene_labels[pos] = self.splicesurveyor_labels[splice_type]
            
            labels.append(gene_labels)
            
            # Add other metadata
            chroms.append(group['chromosome'].iloc[0])
            strands.append(group['strand'].iloc[0])
            tx_starts.append(group.get('tx_start', [0]).iloc[0])
            tx_ends.append(group.get('tx_end', [len(gene_sequence)]).iloc[0])
        
        # Write to HDF5
        with h5py.File(output_file, 'w') as f:
            # Convert strings to bytes for HDF5 storage
            f.create_dataset('NAME', data=[n.encode('utf-8') for n in names])
            f.create_dataset('SEQ', data=[s.encode('utf-8') for s in sequences])
            f.create_dataset('CHROM', data=[c.encode('utf-8') for c in chroms])
            f.create_dataset('STRAND', data=[s.encode('utf-8') for s in strands])
            
            # Create variable-length datasets for labels
            dt = h5py.special_dtype(vlen=np.dtype('int32'))
            f.create_dataset('LABEL', data=labels, dtype=dt)
            
            # Numeric data
            f.create_dataset('TX_START', data=tx_starts)
            f.create_dataset('TX_END', data=tx_ends)
        
        return output_file
    
    def extract_splice_sites_from_h5(
        self,
        h5_file: str,
        output_format: str = "tsv"
    ) -> str:
        """
        Extract splice sites from OpenSpliceAI H5 file in MetaSpliceAI format.
        
        Parameters
        ----------
        h5_file : str
            Path to OpenSpliceAI HDF5 file
        output_format : str
            Output format ("tsv", "csv", "parquet")
            
        Returns
        -------
        str
            Path to output file with splice sites
        """
        # Convert to DataFrame first
        df = self.convert_h5_to_dataframe(h5_file, output_format="polars")
        
        # Generate output filename
        base_name = Path(h5_file).stem
        output_file = f"{base_name}_splice_sites.{output_format}"
        
        # Write in requested format
        if output_format == "tsv":
            df.write_csv(output_file, separator='\t')
        elif output_format == "csv":
            df.write_csv(output_file)
        elif output_format == "parquet":
            df.write_parquet(output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        if self.verbose >= 1:
            print(f"[converter] Extracted {len(df)} splice sites to: {output_file}")
            
        return output_file
    
    def create_openspliceai_compatible_dataset(
        self,
        splice_sites_df: Union[pd.DataFrame, pl.DataFrame],
        sequences_df: Union[pd.DataFrame, pl.DataFrame],
        output_dir: str,
        dataset_name: str = "splicesurveyor_dataset"
    ) -> Dict[str, str]:
        """
        Create OpenSpliceAI-compatible dataset from MetaSpliceAI data.
        
        Parameters
        ----------
        splice_sites_df : DataFrame
            Splice sites annotations
        sequences_df : DataFrame  
            Gene sequences
        output_dir : str
            Output directory for datasets
        dataset_name : str
            Name prefix for output files
            
        Returns
        -------
        Dict[str, str]
            Paths to created train/test datasets
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Merge splice sites with sequences
        if isinstance(splice_sites_df, pl.DataFrame) and isinstance(sequences_df, pl.DataFrame):
            merged_df = splice_sites_df.join(sequences_df, on='gene_id', how='inner')
        else:
            # Convert to pandas for merging
            if isinstance(splice_sites_df, pl.DataFrame):
                splice_sites_df = splice_sites_df.to_pandas()
            if isinstance(sequences_df, pl.DataFrame):
                sequences_df = sequences_df.to_pandas()
            merged_df = pd.merge(splice_sites_df, sequences_df, on='gene_id', how='inner')
        
        # Split into train/test (80/20 split by chromosome)
        if isinstance(merged_df, pl.DataFrame):
            unique_chroms = merged_df['chromosome'].unique().to_list()
        else:
            unique_chroms = merged_df['chromosome'].unique().tolist()
            
        # Simple chromosome-based split
        train_chroms = unique_chroms[:int(0.8 * len(unique_chroms))]
        test_chroms = unique_chroms[int(0.8 * len(unique_chroms)):]
        
        if isinstance(merged_df, pl.DataFrame):
            train_df = merged_df.filter(pl.col('chromosome').is_in(train_chroms))
            test_df = merged_df.filter(pl.col('chromosome').is_in(test_chroms))
        else:
            train_df = merged_df[merged_df['chromosome'].isin(train_chroms)]
            test_df = merged_df[merged_df['chromosome'].isin(test_chroms)]
        
        # Convert to HDF5 format
        train_file = os.path.join(output_dir, f"{dataset_name}_train.h5")
        test_file = os.path.join(output_dir, f"{dataset_name}_test.h5")
        
        self.convert_dataframe_to_h5(train_df, train_file)
        self.convert_dataframe_to_h5(test_df, test_file)
        
        if self.verbose >= 1:
            print(f"[converter] Created training dataset: {train_file}")
            print(f"[converter] Created testing dataset: {test_file}")
        
        return {
            'train': train_file,
            'test': test_file
        }


# Convenience functions
def convert_to_openspliceai_format(
    splice_sites_file: str,
    sequences_file: str,
    output_dir: str,
    dataset_name: str = "converted_dataset"
) -> Dict[str, str]:
    """
    Convert MetaSpliceAI format files to OpenSpliceAI format.
    
    Parameters
    ----------
    splice_sites_file : str
        Path to splice sites file (TSV/CSV/Parquet)
    sequences_file : str
        Path to sequences file (TSV/CSV/Parquet)
    output_dir : str
        Output directory
    dataset_name : str
        Name for the dataset
        
    Returns
    -------
    Dict[str, str]
        Paths to created train/test H5 files
    """
    converter = SpliceDataConverter()
    
    # Load data based on file extension
    if splice_sites_file.endswith('.parquet'):
        splice_sites_df = pl.read_parquet(splice_sites_file)
    else:
        sep = '\t' if splice_sites_file.endswith('.tsv') else ','
        splice_sites_df = pl.read_csv(splice_sites_file, separator=sep)
        
    if sequences_file.endswith('.parquet'):
        sequences_df = pl.read_parquet(sequences_file)
    else:
        sep = '\t' if sequences_file.endswith('.tsv') else ','
        sequences_df = pl.read_csv(sequences_file, separator=sep)
    
    return converter.create_openspliceai_compatible_dataset(
        splice_sites_df, sequences_df, output_dir, dataset_name
    )


def convert_from_openspliceai_format(
    h5_file: str,
    output_format: str = "tsv"
) -> str:
    """
    Convert OpenSpliceAI H5 file to MetaSpliceAI format.
    
    Parameters
    ----------
    h5_file : str
        Path to OpenSpliceAI HDF5 file
    output_format : str
        Output format ("tsv", "csv", "parquet")
        
    Returns
    -------
    str
        Path to converted file
    """
    converter = SpliceDataConverter()
    return converter.extract_splice_sites_from_h5(h5_file, output_format)
