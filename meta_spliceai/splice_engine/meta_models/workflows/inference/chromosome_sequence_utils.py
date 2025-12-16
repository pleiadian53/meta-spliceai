"""
Per-chromosome genomic sequence utilities for inference workflows.

This module provides efficient, targeted sequence extraction for inference scenarios
where we only need sequences from specific chromosomes containing target genes.
This avoids the overhead of processing all chromosomes when only a subset is needed.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import polars as pl
import pandas as pd

from meta_spliceai.splice_engine.utils_bio import (
    extract_genes_from_gtf,
    extract_gene_sequences,
    save_sequences_by_chromosome
)
from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, 
    print_with_indent
)


def extract_targeted_chromosome_sequences(
    target_genes: List[str],
    target_chromosomes: List[str],
    gtf_file: str,
    genome_fasta: str,
    output_dir: Union[str, Path],
    seq_type: str = 'full',
    seq_format: str = 'parquet',
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Extract genomic sequences for specific genes on target chromosomes only.
    
    This function provides targeted sequence extraction optimized for inference
    workflows where we only need sequences from specific chromosomes containing
    the target genes. This avoids processing all chromosomes unnecessarily.
    
    Parameters
    ----------
    target_genes : List[str]
        List of gene IDs to extract sequences for
    target_chromosomes : List[str]
        List of chromosomes containing the target genes
    gtf_file : str
        Path to GTF annotation file
    genome_fasta : str
        Path to genome FASTA file
    output_dir : Union[str, Path]
        Directory to store extracted sequence files
    seq_type : str, default='full'
        Type of sequence extraction ('full' or 'minmax')
    seq_format : str, default='parquet'
        Output format for sequence files
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with results including success status and file paths
    """
    result = {
        'success': False,
        'sequences_files': {},
        'error': None,
        'target_chromosomes': target_chromosomes,
        'target_genes': target_genes
    }
    
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if verbosity >= 1:
            print_emphasized(f"[chromosome-seq] Extracting sequences for {len(target_genes)} genes on chromosomes {target_chromosomes}")
            print_with_indent(f"Mode: {seq_type}, Format: {seq_format}", indent_level=1)
        
        # Step 1: Extract gene annotations from GTF
        if verbosity >= 2:
            print_with_indent("[1/3] Loading gene annotations from GTF...", indent_level=1)
        
        genes_df = extract_genes_from_gtf(gtf_file)
        
        # Step 2: Filter to target genes and chromosomes
        if verbosity >= 2:
            print_with_indent(f"[2/3] Filtering to target genes and chromosomes...", indent_level=1)
        
        # Convert to polars for efficient filtering
        if isinstance(genes_df, pd.DataFrame):
            genes_df = pl.from_pandas(genes_df)
        
        # Filter by target genes and chromosomes
        filtered_genes = genes_df.filter(
            pl.col('gene_id').is_in(target_genes) &
            pl.col('seqname').is_in(target_chromosomes)
        )
        
        if filtered_genes.height == 0:
            raise ValueError(f"No genes found for target genes {target_genes} on chromosomes {target_chromosomes}")
        
        if verbosity >= 2:
            genes_per_chrom = filtered_genes.group_by('seqname').agg(pl.count('gene_id').alias('gene_count'))
            print_with_indent(f"Genes per chromosome: {genes_per_chrom.to_pandas().to_dict('records')}", indent_level=2)
        
        # Step 3: Extract sequences chromosome by chromosome
        if verbosity >= 2:
            print_with_indent("[3/3] Extracting sequences per chromosome...", indent_level=1)
        
        sequences_files = {}
        
        for chrom in target_chromosomes:
            if verbosity >= 1:
                print_with_indent(f"Processing chromosome {chrom}...", indent_level=2)
            
            # Filter genes for this chromosome
            chrom_genes = filtered_genes.filter(pl.col('seqname') == chrom)
            
            if chrom_genes.height == 0:
                if verbosity >= 1:
                    print_with_indent(f"No target genes found on chromosome {chrom}, skipping", indent_level=3)
                continue
            
            # Convert back to pandas for extract_gene_sequences (legacy compatibility)
            chrom_genes_pd = chrom_genes.to_pandas()
            
            if verbosity >= 2:
                print_with_indent(f"Extracting {chrom_genes.height} genes from chromosome {chrom}", indent_level=3)
            
            # Extract sequences for this chromosome
            seq_df = extract_gene_sequences(
                chrom_genes_pd, 
                genome_fasta, 
                output_format='dataframe', 
                include_columns=None
            )
            
            # Save chromosome-specific sequence file
            chrom_filename = f"gene_sequence_{chrom}.{seq_format}"
            chrom_output_path = output_dir / chrom_filename
            
            if seq_format == 'parquet':
                # Convert to polars for efficient parquet writing
                if isinstance(seq_df, pd.DataFrame):
                    seq_df = pl.from_pandas(seq_df)
                seq_df.write_parquet(chrom_output_path)
            else:
                # Use pandas for TSV/CSV
                if isinstance(seq_df, pl.DataFrame):
                    seq_df = seq_df.to_pandas()
                seq_df.to_csv(chrom_output_path, sep='\t', index=False)
            
            sequences_files[chrom] = str(chrom_output_path)
            
            if verbosity >= 1:
                print_with_indent(f"✅ Saved {len(seq_df)} sequences to {chrom_filename}", indent_level=3)
        
        result['sequences_files'] = sequences_files
        result['success'] = True
        
        if verbosity >= 1:
            print_emphasized(f"[chromosome-seq] ✅ Successfully extracted sequences for {len(sequences_files)} chromosomes")
            total_files = len(sequences_files)
            print_with_indent(f"Created {total_files} chromosome-specific sequence files", indent_level=1)
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        if verbosity >= 1:
            print_emphasized(f"[chromosome-seq] ❌ Failed to extract targeted sequences: {e}")
            import traceback
            traceback.print_exc()
        
        return result


def create_per_chromosome_fasta_from_combined(
    combined_fasta: str,
    target_chromosomes: List[str],
    output_dir: Union[str, Path],
    verbosity: int = 1
) -> Dict[str, str]:
    """
    Create per-chromosome FASTA files from a combined genome FASTA file.
    
    This function splits a combined genome FASTA file into individual chromosome
    files for faster, targeted access during inference. Only creates files for
    the requested chromosomes.
    
    Parameters
    ----------
    combined_fasta : str
        Path to the combined genome FASTA file
    target_chromosomes : List[str]
        List of chromosomes to extract
    output_dir : Union[str, Path]
        Directory to store per-chromosome FASTA files
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, str]
        Mapping from chromosome name to output FASTA file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chromosome_files = {}
    
    if verbosity >= 1:
        print_emphasized(f"[fasta-split] Creating per-chromosome FASTA files for {target_chromosomes}")
    
    try:
        from Bio import SeqIO
        from Bio.SeqRecord import SeqRecord
        from Bio.Seq import Seq
        
        # Parse the combined FASTA file
        if verbosity >= 1:
            print_with_indent(f"Reading combined FASTA: {combined_fasta}", indent_level=1)
        
        sequences = SeqIO.parse(combined_fasta, "fasta")
        
        for record in sequences:
            # Normalize chromosome name (remove 'chr' prefix if present)
            chrom_name = record.id.replace('chr', '')
            
            if chrom_name in target_chromosomes:
                if verbosity >= 2:
                    print_with_indent(f"Processing chromosome {chrom_name}...", indent_level=2)
                
                # Create output file for this chromosome
                chrom_file = output_dir / f"chr{chrom_name}.fa"
                
                # Write single-chromosome FASTA
                with open(chrom_file, 'w') as f:
                    SeqIO.write(record, f, "fasta")
                
                chromosome_files[chrom_name] = str(chrom_file)
                
                if verbosity >= 1:
                    seq_length = len(record.seq)
                    print_with_indent(f"✅ Created {chrom_file.name} ({seq_length:,} bp)", indent_level=2)
        
        if verbosity >= 1:
            print_emphasized(f"[fasta-split] ✅ Created {len(chromosome_files)} chromosome FASTA files")
        
        return chromosome_files
        
    except ImportError:
        if verbosity >= 1:
            print_emphasized("[fasta-split] ❌ BioPython not available, cannot split FASTA files")
            print_with_indent("Install with: pip install biopython", indent_level=1)
        return {}
    
    except Exception as e:
        if verbosity >= 1:
            print_emphasized(f"[fasta-split] ❌ Failed to split FASTA file: {e}")
            import traceback
            traceback.print_exc()
        return {}


def verify_chromosome_sequences(
    sequences_files: Dict[str, str],
    target_genes: List[str],
    verbosity: int = 1
) -> bool:
    """
    Verify that the extracted chromosome sequence files contain the target genes.
    
    Parameters
    ----------
    sequences_files : Dict[str, str]
        Mapping from chromosome to sequence file path
    target_genes : List[str]
        List of target gene IDs to verify
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    bool
        True if all target genes are found in the sequence files
    """
    if verbosity >= 1:
        print_emphasized("[verify] Checking chromosome sequence files for target genes...")
    
    found_genes = set()
    
    try:
        for chrom, file_path in sequences_files.items():
            if not os.path.exists(file_path):
                if verbosity >= 1:
                    print_with_indent(f"❌ File not found: {file_path}", indent_level=1)
                return False
            
            if verbosity >= 2:
                print_with_indent(f"Checking {chrom}: {Path(file_path).name}", indent_level=1)
            
            # Load and check gene IDs
            if file_path.endswith('.parquet'):
                seq_df = pl.read_parquet(file_path)
            else:
                seq_df = pl.read_csv(file_path, separator='\t')
            
            if 'gene_id' in seq_df.columns:
                chrom_genes = set(seq_df['gene_id'].to_list())
                found_genes.update(chrom_genes)
                
                if verbosity >= 2:
                    target_in_chrom = chrom_genes.intersection(set(target_genes))
                    if target_in_chrom:
                        print_with_indent(f"Found target genes: {list(target_in_chrom)}", indent_level=2)
        
        missing_genes = set(target_genes) - found_genes
        
        if missing_genes:
            if verbosity >= 1:
                print_with_indent(f"❌ Missing genes: {list(missing_genes)}", indent_level=1)
            return False
        
        if verbosity >= 1:
            print_emphasized(f"[verify] ✅ All {len(target_genes)} target genes found in sequence files")
        
        return True
        
    except Exception as e:
        if verbosity >= 1:
            print_emphasized(f"[verify] ❌ Verification failed: {e}")
        return False


__all__ = [
    'extract_targeted_chromosome_sequences',
    'create_per_chromosome_fasta_from_combined', 
    'verify_chromosome_sequences'
]