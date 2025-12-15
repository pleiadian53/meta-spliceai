"""
Data processing for the splice site prediction foundation model.

This module handles the preparation of genomic data for training and
evaluating splice site prediction models, including sequence extraction,
labeling, and encoding.
"""

import os
import logging
import numpy as np
import pandas as pd
from Bio import SeqIO
from pybedtools import BedTool
from sklearn.model_selection import train_test_split
import tensorflow as tf

logger = logging.getLogger(__name__)

class GenomicDataProcessor:
    """Process genomic data for splice site prediction."""
    
    def __init__(self, context_length=10000, stride=1000):
        """
        Initialize the genomic data processor.
        
        Args:
            context_length (int): Length of sequence context to use (default: 10000)
            stride (int): Stride length when generating overlapping blocks (default: 1000)
        """
        self.context_length = context_length
        self.stride = stride
        
    def extract_splice_sites(self, gtf_file, donor_acceptor=True):
        """
        Extract splice site coordinates from a GTF file.
        
        Args:
            gtf_file (str): Path to GTF annotation file
            donor_acceptor (bool): If True, separate donor and acceptor sites
            
        Returns:
            pd.DataFrame: DataFrame containing splice site coordinates
        """
        logger.info(f"Extracting splice sites from {gtf_file}")
        gtf = BedTool(gtf_file)
        exons = gtf.filter(lambda x: x[2] == 'exon').saveas()
        
        splice_sites = []
        
        for exon in exons:
            chrom = exon.chrom
            strand = exon.strand
            exon_start = exon.start
            exon_end = exon.end
            gene_id = exon.attrs.get("gene_id", "")
            
            # Extract donor sites (5' SS, GT)
            if strand == '+':
                splice_sites.append({
                    "chrom": chrom,
                    "position": exon_end,
                    "strand": strand,
                    "type": "donor",
                    "gene_id": gene_id
                })
            else:  # '-' strand
                splice_sites.append({
                    "chrom": chrom,
                    "position": exon_start,
                    "strand": strand,
                    "type": "donor",
                    "gene_id": gene_id
                })
            
            # Extract acceptor sites (3' SS, AG)
            if strand == '+':
                splice_sites.append({
                    "chrom": chrom,
                    "position": exon_start,
                    "strand": strand,
                    "type": "acceptor",
                    "gene_id": gene_id
                })
            else:  # '-' strand
                splice_sites.append({
                    "chrom": chrom,
                    "position": exon_end,
                    "strand": strand,
                    "type": "acceptor",
                    "gene_id": gene_id
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(splice_sites)
        logger.info(f"Extracted {len(df)} splice sites ({df['type'].value_counts().to_dict()})")
        return df
    
    def extract_genomic_windows(self, fasta_file, splice_sites_df, flank_size=5000):
        """
        Extract genomic sequence windows around splice sites.
        
        Args:
            fasta_file (str): Path to genome FASTA file
            splice_sites_df (pd.DataFrame): DataFrame with splice site coordinates
            flank_size (int): Size of flanking sequence on each side
            
        Returns:
            pd.DataFrame: DataFrame with sequences and labels
        """
        logger.info(f"Extracting genomic windows from {fasta_file}")
        
        # Load genome sequences
        genome_dict = {}
        for record in SeqIO.parse(fasta_file, "fasta"):
            genome_dict[record.id] = str(record.seq)
        
        # Extract windows around splice sites
        windows = []
        for _, row in splice_sites_df.iterrows():
            chrom = row['chrom']
            pos = row['position']
            strand = row['strand']
            site_type = row['type']
            
            if chrom not in genome_dict:
                logger.warning(f"Chromosome {chrom} not found in FASTA file")
                continue
                
            # Extract sequence with flanking regions
            start = max(0, pos - flank_size)
            end = min(len(genome_dict[chrom]), pos + flank_size)
            
            sequence = genome_dict[chrom][start:end]
            center_pos = pos - start  # Relative position in the window
            
            # Reverse complement if on negative strand
            if strand == '-':
                sequence = self._reverse_complement(sequence)
                center_pos = len(sequence) - center_pos - 1
                
            windows.append({
                'chrom': chrom,
                'position': pos,
                'sequence': sequence,
                'center_pos': center_pos,
                'strand': strand,
                'type': site_type,
                'gene_id': row['gene_id']
            })
        
        results_df = pd.DataFrame(windows)
        logger.info(f"Extracted {len(results_df)} sequence windows")
        return results_df
    
    def generate_negative_examples(self, genome_dict, positive_sites, n_samples, flank_size=5000):
        """
        Generate negative examples (non-splice sites) for training.
        
        Args:
            genome_dict (dict): Dictionary of chromosome sequences
            positive_sites (pd.DataFrame): DataFrame with positive splice sites
            n_samples (int): Number of negative samples to generate
            flank_size (int): Size of flanking sequence
            
        Returns:
            pd.DataFrame: DataFrame with negative examples
        """
        # Create sets of positions to avoid (true splice sites)
        avoid_positions = {}
        for _, row in positive_sites.iterrows():
            chrom = row['chrom']
            pos = row['position']
            if chrom not in avoid_positions:
                avoid_positions[chrom] = set()
            
            # Avoid regions within 100bp of real splice sites
            for i in range(pos-100, pos+101):
                avoid_positions[chrom].add(i)
        
        # Generate random positions
        negative_samples = []
        chromosomes = list(genome_dict.keys())
        
        while len(negative_samples) < n_samples:
            # Randomly select chromosome
            chrom = np.random.choice(chromosomes)
            
            # Skip if chromosome is too short
            if len(genome_dict[chrom]) < 2*flank_size + 100:
                continue
                
            # Generate random position (avoiding ends)
            pos = np.random.randint(flank_size, len(genome_dict[chrom]) - flank_size)
            
            # Skip if this is close to a real splice site
            if chrom in avoid_positions and pos in avoid_positions[chrom]:
                continue
                
            # Randomly assign strand and site type for balanced dataset
            strand = np.random.choice(['+', '-'])
            site_type = np.random.choice(['donor', 'acceptor'])
            
            # Extract sequence
            start = max(0, pos - flank_size)
            end = min(len(genome_dict[chrom]), pos + flank_size)
            sequence = genome_dict[chrom][start:end]
            center_pos = pos - start
            
            # Reverse complement if on negative strand
            if strand == '-':
                sequence = self._reverse_complement(sequence)
                center_pos = len(sequence) - center_pos - 1
            
            negative_samples.append({
                'chrom': chrom,
                'position': pos,
                'sequence': sequence,
                'center_pos': center_pos,
                'strand': strand,
                'type': site_type,
                'gene_id': 'negative',
                'label': 0  # Negative class
            })
        
        return pd.DataFrame(negative_samples)
    
    def prepare_dataset(self, positive_windows, negative_windows=None, test_size=0.2, valid_size=0.1):
        """
        Prepare dataset splits for training, validation and testing.
        
        Args:
            positive_windows (pd.DataFrame): DataFrame with positive examples
            negative_windows (pd.DataFrame): DataFrame with negative examples
            test_size (float): Proportion for test set
            valid_size (float): Proportion for validation set
            
        Returns:
            dict: Dictionary with train, validation, and test datasets
        """
        # Add positive labels
        positive_windows = positive_windows.copy()
        positive_windows['label'] = 1
        
        # Combine positive and negative examples
        if negative_windows is not None:
            negative_windows = negative_windows.copy()
            negative_windows['label'] = 0
            all_data = pd.concat([positive_windows, negative_windows])
        else:
            all_data = positive_windows
        
        # Split data
        train_val, test = train_test_split(all_data, test_size=test_size, stratify=all_data['label'])
        train, val = train_test_split(train_val, test_size=valid_size/(1-test_size), stratify=train_val['label'])
        
        return {
            'train': train,
            'validation': val,
            'test': test
        }
    
    def one_hot_encode(self, sequences):
        """
        One-hot encode DNA sequences.
        
        Args:
            sequences (list): List of DNA sequences
            
        Returns:
            np.ndarray: One-hot encoded sequences (shape: [n_samples, seq_length, 4])
        """
        # Mapping of nucleotides to indices
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # Get maximum sequence length
        max_len = max(len(seq) for seq in sequences)
        
        # Initialize encoded array
        encoded = np.zeros((len(sequences), max_len, 5), dtype=np.float32)
        
        # Fill in encoded values
        for i, seq in enumerate(sequences):
            for j, nuc in enumerate(seq):
                if nuc in nuc_map:
                    encoded[i, j, nuc_map[nuc]] = 1.0
                else:
                    # For unknown nucleotides, use 'N' encoding
                    encoded[i, j, 4] = 1.0
        
        return encoded
    
    def prepare_windows_dataset(self, dataframe, batch_size=32, shuffle=True):
        """
        Prepare TensorFlow dataset from windows DataFrame.
        
        Args:
            dataframe (pd.DataFrame): DataFrame with sequences and labels
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Extract sequences and labels
        sequences = dataframe['sequence'].tolist()
        labels = dataframe['label'].values
        
        # One-hot encode sequences
        X = self.one_hot_encode(sequences)
        y = labels.astype(np.float32)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(dataframe))
            
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def generate_overlapping_blocks(self, sequence, context_length=None, stride=None):
        """
        Generate overlapping blocks from a long sequence.
        
        Args:
            sequence (str): Input DNA sequence
            context_length (int): Length of each block
            stride (int): Stride between blocks
            
        Returns:
            list: List of sequence blocks
        """
        if context_length is None:
            context_length = self.context_length
        
        if stride is None:
            stride = self.stride
            
        blocks = []
        for i in range(0, len(sequence) - context_length + 1, stride):
            blocks.append(sequence[i:i+context_length])
            
        return blocks
    
    def _reverse_complement(self, sequence):
        """
        Generate reverse complement of a DNA sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            str: Reverse complement
        """
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
                     'a': 't', 'c': 'g', 'g': 'c', 't': 'a',
                     'N': 'N', 'n': 'n'}
        return ''.join(complement.get(base, base) for base in reversed(sequence))
