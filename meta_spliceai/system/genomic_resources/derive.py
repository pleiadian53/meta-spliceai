"""Genomic data derivation utilities.

This module provides functions for deriving genomic datasets from GTF and FASTA files:
- Gene annotations extraction
- Splice site annotations
- Genomic sequences
- Overlapping gene analysis

These are refactored versions of the data preparation functions from the base model
workflow, designed for better reusability and maintainability.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import polars as pl
import pandas as pd

from .config import load_config, Config
from .registry import Registry


class GenomicDataDeriver:
    """Derives genomic datasets from GTF and FASTA files.
    
    This class provides a systematic interface for generating derived datasets
    needed for splice site prediction workflows:
    - Gene annotations (transcripts, exons)
    - Splice site positions
    - Genomic sequences
    - Overlapping gene metadata
    
    Parameters
    ----------
    data_dir : Path or str, optional
        Directory containing GTF/FASTA files and where derived data will be stored.
        If None, uses config default.
    config : Config, optional
        Genomic resources configuration. If None, loads default config.
    registry : Registry, optional
        Path registry for locating resources. If None, creates from config.
    verbosity : int, default=1
        Output verbosity level (0=silent, 1=normal, 2=detailed)
        
    Examples
    --------
    >>> deriver = GenomicDataDeriver()
    >>> result = deriver.derive_gene_annotations()
    >>> print(f"Extracted {len(result['annotations_df'])} annotations")
    
    >>> result = deriver.derive_splice_sites(consensus_window=2)
    >>> print(f"Found {len(result['splice_sites_df'])} splice sites")
    """
    
    def __init__(
        self,
        data_dir: Optional[Union[Path, str]] = None,
        config: Optional[Config] = None,
        registry: Optional[Registry] = None,
        verbosity: int = 1
    ):
        self.config = config or load_config()
        self.registry = registry or Registry()
        
        # CRITICAL: Use build-specific directory from Registry, not root data_root
        # This ensures derived datasets are stored per-build to avoid cross-contamination
        # between GRCh37 and GRCh38 datasets
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Use Registry's stash (build-specific directory) for derived datasets
            self.data_dir = self.registry.stash
        
        self.verbosity = verbosity
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def derive_gene_annotations(
        self,
        output_filename: str = "annotations_all_transcripts.tsv",
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False,
        use_shared_location: bool = True
    ) -> Dict[str, Any]:
        """Extract gene annotations from GTF file.
        
        Extracts transcript-level annotations including:
        - Chromosome, start, end positions
        - Strand orientation
        - Feature type (exon, CDS, etc.)
        - Gene ID and transcript ID
        
        Parameters
        ----------
        output_filename : str, default="annotations_all_transcripts.tsv"
            Name of output file for annotations
        target_chromosomes : List[str], optional
            Filter to specific chromosomes (e.g., ['1', '2', 'X'])
        force_overwrite : bool, default=False
            If True, re-extract even if file exists
        use_shared_location : bool, default=True
            If True, save to shared data directory for reuse across workflows
            
        Returns
        -------
        Dict[str, Any]
            - 'success': bool - Whether extraction succeeded
            - 'annotations_file': str - Path to annotations file
            - 'annotations_df': pl.DataFrame - Loaded annotations
            - 'error': str - Error message if failed
            
        Notes
        -----
        This is a refactored version of `prepare_gene_annotations()` from
        the original workflow, with improved error handling and path management.
        """
        result = {
            'success': False,
            'annotations_file': None,
            'annotations_df': None,
            'error': None
        }
        
        # Determine output path
        output_path = self.data_dir / output_filename
        
        # Check if file exists and we're not forcing overwrite
        if output_path.exists() and not force_overwrite:
            if self.verbosity >= 1:
                print(f"[derive] Using existing annotations: {output_path}")
            
            try:
                annotations_df = pl.read_csv(
                    output_path,
                    separator='\t',
                    schema_overrides={'chrom': pl.Utf8}
                )
                
                # Apply chromosome filtering if requested
                if target_chromosomes:
                    annotations_df = self._filter_by_chromosomes(
                        annotations_df, target_chromosomes
                    )
                
                result['success'] = True
                result['annotations_file'] = str(output_path)
                result['annotations_df'] = annotations_df
                return result
                
            except Exception as e:
                result['error'] = f"Failed to load existing annotations: {e}"
                if self.verbosity >= 1:
                    print(f"[warning] {result['error']}")
                # Fall through to extraction
        
        # Extract annotations from GTF
        if self.verbosity >= 1:
            print(f"[derive] Extracting gene annotations from GTF...")
        
        try:
            gtf_path = self.registry.get_gtf_path(validate=True)
            
            # Use existing extraction logic (import from original module)
            from meta_spliceai.splice_engine.extract_genomic_features import (
                extract_gene_annotations_from_gtf
            )
            
            annotations_df = extract_gene_annotations_from_gtf(
                str(gtf_path),
                output_file=str(output_path),
                target_chromosomes=target_chromosomes
            )
            
            result['success'] = True
            result['annotations_file'] = str(output_path)
            result['annotations_df'] = annotations_df
            
            if self.verbosity >= 1:
                print(f"[derive] ✓ Extracted {len(annotations_df)} annotations")
                print(f"[derive] ✓ Saved to: {output_path}")
            
        except Exception as e:
            result['error'] = f"Failed to extract annotations: {e}"
            if self.verbosity >= 0:
                print(f"[error] {result['error']}")
        
        return result
    
    def derive_splice_sites(
        self,
        output_filename: str = "splice_sites.tsv",
        consensus_window: int = 2,
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False,
        use_shared_location: bool = True
    ) -> Dict[str, Any]:
        """Extract splice site annotations from GTF file.
        
        Identifies canonical splice sites (donors and acceptors) from exon boundaries
        in the GTF annotation.
        
        Parameters
        ----------
        output_filename : str, default="splice_sites.tsv"
            Name of output file for splice sites
        consensus_window : int, default=2
            Window size around splice sites for consensus sequence
        target_chromosomes : List[str], optional
            Filter to specific chromosomes
        force_overwrite : bool, default=False
            If True, re-extract even if file exists
        use_shared_location : bool, default=True
            If True, save to shared data directory
            
        Returns
        -------
        Dict[str, Any]
            - 'success': bool
            - 'splice_sites_file': str - Path to splice sites file
            - 'splice_sites_df': pl.DataFrame - Splice site annotations
            - 'error': str - Error message if failed
            
        Notes
        -----
        Splice sites dataframe columns:
        - chrom: Chromosome
        - start, end: Genomic coordinates
        - position: Exact splice site position
        - strand: '+' or '-'
        - site_type: 'donor' or 'acceptor'
        - gene_id: Ensembl gene ID
        - transcript_id: Ensembl transcript ID
        """
        result = {
            'success': False,
            'splice_sites_file': None,
            'splice_sites_df': None,
            'error': None
        }
        
        output_path = self.data_dir / output_filename
        
        # Check if file exists
        if output_path.exists() and not force_overwrite:
            if self.verbosity >= 1:
                print(f"[derive] Using existing splice sites: {output_path}")
            
            try:
                splice_sites_df = pl.read_csv(
                    output_path,
                    separator='\t',
                    schema_overrides={'chrom': pl.Utf8}
                )
                
                if target_chromosomes:
                    splice_sites_df = self._filter_by_chromosomes(
                        splice_sites_df, target_chromosomes
                    )
                
                result['success'] = True
                result['splice_sites_file'] = str(output_path)
                result['splice_sites_df'] = splice_sites_df
                return result
                
            except Exception as e:
                result['error'] = f"Failed to load existing splice sites: {e}"
                if self.verbosity >= 1:
                    print(f"[warning] {result['error']}")
        
        # Extract splice sites from GTF
        if self.verbosity >= 1:
            print(f"[derive] Extracting splice sites from GTF...")
        
        try:
            gtf_path = self.registry.get_gtf_path(validate=True)
            
            # Use refactored extraction logic with enhanced metadata
            from .splice_sites import extract_splice_sites_from_gtf
            
            # extract_splice_sites_from_gtf returns the output file path
            output_file = extract_splice_sites_from_gtf(
                gtf_path=str(gtf_path),
                consensus_window=consensus_window,
                output_file=str(output_path),
                save=True,
                return_df=False,
                verbosity=self.verbosity,
                sep='\t'
            )
            
            # Load the generated splice sites file
            import polars as pl
            splice_sites_df = pl.read_csv(
                output_file,
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            
            # Filter by target chromosomes if specified
            if target_chromosomes:
                splice_sites_df = self._filter_by_chromosomes(
                    splice_sites_df, target_chromosomes
                )
            
            result['success'] = True
            result['splice_sites_file'] = output_file
            result['splice_sites_df'] = splice_sites_df
            
            if self.verbosity >= 1:
                print(f"[derive] ✓ Extracted {len(splice_sites_df)} splice sites")
                print(f"[derive] ✓ Saved to: {output_path}")
            
        except Exception as e:
            result['error'] = f"Failed to extract splice sites: {e}"
            if self.verbosity >= 0:
                print(f"[error] {result['error']}")
        
        return result
    
    def derive_genomic_sequences(
        self,
        mode: str = 'gene',
        seq_type: str = 'full',
        target_chromosomes: Optional[List[str]] = None,
        target_genes: Optional[List[str]] = None,
        seq_format: str = 'parquet',
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """Extract genomic sequences for genes or transcripts.
        
        Extracts DNA sequences from FASTA file based on coordinates in GTF.
        
        Parameters
        ----------
        mode : str, default='gene'
            Extraction mode: 'gene' or 'transcript'
        seq_type : str, default='full'
            Sequence type: 'full' (entire gene) or 'minmax' (min start to max end)
        target_chromosomes : List[str], optional
            Filter to specific chromosomes
        target_genes : List[str], optional
            Filter to specific genes
        seq_format : str, default='parquet'
            Output format: 'parquet', 'tsv', or 'csv'
        force_overwrite : bool, default=False
            If True, re-extract even if files exist
            
        Returns
        -------
        Dict[str, Any]
            - 'success': bool
            - 'sequences_file': str - Path to main sequences file
            - 'sequences_df': pl.DataFrame - Loaded sequences
            - 'chr_sequence_files': List[str] - Per-chromosome files
            - 'error': str - Error message if failed
            
        Notes
        -----
        Sequences are stored per-chromosome for efficient loading.
        Format: gene_sequence_{chrom}.{format}
        """
        result = {
            'success': False,
            'sequences_file': None,
            'sequences_df': None,
            'chr_sequence_files': [],
            'error': None
        }
        
        if self.verbosity >= 1:
            print(f"[derive] Extracting genomic sequences (mode={mode}, type={seq_type})...")
        
        try:
            gtf_path = self.registry.get_gtf_path(validate=True)
            fasta_path = self.registry.get_fasta_path(validate=True)
            
            # Use existing extraction logic
            from meta_spliceai.splice_engine.extract_gene_sequences import (
                extract_gene_sequences
            )
            
            sequences_result = extract_gene_sequences(
                gtf_file=str(gtf_path),
                fasta_file=str(fasta_path),
                output_dir=str(self.data_dir),
                mode=mode,
                seq_type=seq_type,
                chromosomes=target_chromosomes,
                genes=target_genes,
                output_format=seq_format,
                force_overwrite=force_overwrite
            )
            
            result['success'] = sequences_result.get('success', False)
            result['sequences_file'] = sequences_result.get('main_file')
            result['sequences_df'] = sequences_result.get('sequences_df')
            result['chr_sequence_files'] = sequences_result.get('chr_files', [])
            
            if self.verbosity >= 1 and result['success']:
                n_seqs = len(result['sequences_df']) if result['sequences_df'] is not None else 0
                print(f"[derive] ✓ Extracted {n_seqs} sequences")
                print(f"[derive] ✓ Created {len(result['chr_sequence_files'])} chromosome files")
            
        except Exception as e:
            result['error'] = f"Failed to extract sequences: {e}"
            if self.verbosity >= 0:
                print(f"[error] {result['error']}")
        
        return result
    
    def derive_overlapping_genes(
        self,
        output_filename: str = "overlapping_gene_counts.tsv",
        filter_valid_splice_sites: bool = True,
        min_exons: int = 2,
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """Identify and catalog overlapping genes.
        
        Finds genes whose genomic coordinates overlap, which is important for
        splice site prediction accuracy.
        
        Parameters
        ----------
        output_filename : str, default="overlapping_gene_counts.tsv"
            Name of output file
        filter_valid_splice_sites : bool, default=True
            Only include genes with valid splice sites (multi-exon genes)
        min_exons : int, default=2
            Minimum number of exons for a gene to be considered
        target_chromosomes : List[str], optional
            Filter to specific chromosomes
        force_overwrite : bool, default=False
            If True, re-compute even if file exists
            
        Returns
        -------
        Dict[str, Any]
            - 'success': bool
            - 'overlapping_genes_file': str
            - 'overlapping_genes_df': pl.DataFrame
            - 'error': str
            
        Notes
        -----
        Output dataframe columns:
        - gene_id: Ensembl gene ID
        - num_overlaps: Number of overlapping genes
        - overlapping_gene_ids: List of overlapping gene IDs
        """
        result = {
            'success': False,
            'overlapping_genes_file': None,
            'overlapping_genes_df': None,
            'error': None
        }
        
        output_path = self.data_dir / output_filename
        
        # Check if file exists
        if output_path.exists() and not force_overwrite:
            if self.verbosity >= 1:
                print(f"[derive] Using existing overlapping genes: {output_path}")
            
            try:
                overlapping_df = pl.read_csv(
                    output_path,
                    separator='\t'
                )
                
                result['success'] = True
                result['overlapping_genes_file'] = str(output_path)
                result['overlapping_genes_df'] = overlapping_df
                return result
                
            except Exception as e:
                result['error'] = f"Failed to load existing overlapping genes: {e}"
                if self.verbosity >= 1:
                    print(f"[warning] {result['error']}")
        
        # Compute overlapping genes
        if self.verbosity >= 1:
            print(f"[derive] Computing overlapping genes...")
        
        try:
            gtf_path = self.registry.get_gtf_path(validate=True)
            
            # Use existing logic
            from meta_spliceai.splice_engine.extract_genomic_features import (
                get_overlapping_gene_metadata
            )
            
            overlapping_df = get_overlapping_gene_metadata(
                gtf_file_path=str(gtf_path),
                filter_valid_splice_sites=filter_valid_splice_sites,
                min_exons=min_exons,
                output_format='dataframe',
                output_file=str(output_path)
            )
            
            result['success'] = True
            result['overlapping_genes_file'] = str(output_path)
            result['overlapping_genes_df'] = overlapping_df
            
            if self.verbosity >= 1:
                n_overlapping = len(overlapping_df.filter(pl.col('num_overlaps') > 0))
                print(f"[derive] ✓ Found {n_overlapping} genes with overlaps")
                print(f"[derive] ✓ Saved to: {output_path}")
            
        except Exception as e:
            result['error'] = f"Failed to compute overlapping genes: {e}"
            if self.verbosity >= 0:
                print(f"[error] {result['error']}")
        
        return result
    
    def derive_all(
        self,
        consensus_window: int = 2,
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Derive all genomic datasets in one call.
        
        Convenience method to generate all derived datasets:
        1. Gene annotations
        2. Gene features
        3. Transcript features
        4. Exon features
        5. Splice sites
        6. Junctions
        7. Genomic sequences
        8. Overlapping genes
        
        Parameters
        ----------
        consensus_window : int, default=2
            Window size for splice site consensus
        target_chromosomes : List[str], optional
            Filter to specific chromosomes
        force_overwrite : bool, default=False
            If True, regenerate all datasets
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with results for each derivation:
            - 'annotations': Gene annotations result
            - 'gene_features': Gene features result
            - 'transcript_features': Transcript features result
            - 'exon_features': Exon features result
            - 'splice_sites': Splice sites result
            - 'junctions': Junctions result
            - 'sequences': Genomic sequences result
            - 'overlapping_genes': Overlapping genes result
        """
        if self.verbosity >= 1:
            print("=" * 70)
            print("DERIVING ALL GENOMIC DATASETS")
            print("=" * 70)
        
        results = {}
        
        # 1. Gene annotations
        results['annotations'] = self.derive_gene_annotations(
            target_chromosomes=target_chromosomes,
            force_overwrite=force_overwrite
        )
        
        # 2. Gene features
        results['gene_features'] = self.derive_gene_features(
            target_chromosomes=target_chromosomes,
            force_overwrite=force_overwrite
        )
        
        # 3. Transcript features
        results['transcript_features'] = self.derive_transcript_features(
            target_chromosomes=target_chromosomes,
            force_overwrite=force_overwrite
        )
        
        # 4. Exon features
        results['exon_features'] = self.derive_exon_features(
            target_chromosomes=target_chromosomes,
            force_overwrite=force_overwrite
        )
        
        # 5. Splice sites
        results['splice_sites'] = self.derive_splice_sites(
            consensus_window=consensus_window,
            target_chromosomes=target_chromosomes,
            force_overwrite=force_overwrite
        )
        
        # 6. Junctions
        results['junctions'] = self.derive_junctions(
            target_chromosomes=target_chromosomes,
            force_overwrite=force_overwrite
        )
        
        # 7. Genomic sequences
        results['sequences'] = self.derive_genomic_sequences(
            target_chromosomes=target_chromosomes,
            force_overwrite=force_overwrite
        )
        
        # 8. Overlapping genes
        results['overlapping_genes'] = self.derive_overlapping_genes(
            target_chromosomes=target_chromosomes,
            force_overwrite=force_overwrite
        )
        
        # Summary
        if self.verbosity >= 1:
            print("\n" + "=" * 70)
            print("DERIVATION SUMMARY")
            print("=" * 70)
            for name, result in results.items():
                status = "✓" if result['success'] else "✗"
                print(f"{status} {name}: {result.get('error', 'Success')}")
            print("=" * 70)
        
        return results
    
    def _filter_by_chromosomes(
        self,
        df: pl.DataFrame,
        chromosomes: List[str]
    ) -> pl.DataFrame:
        """Filter dataframe to specific chromosomes.
        
        Handles both 'chr1' and '1' formats.
        """
        # Normalize chromosome names
        norm_chroms = set()
        for chrom in chromosomes:
            norm_chroms.add(str(chrom).replace('chr', ''))
            norm_chroms.add(f"chr{str(chrom).replace('chr', '')}")
        
        return df.filter(pl.col('chrom').is_in(list(norm_chroms)))
    
    def derive_gene_features(
        self,
        output_filename: str = "gene_features.tsv",
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """Derive gene-level features from GTF.
        
        Extracts gene-level metadata:
        - Chromosome, start, end, strand
        - Gene ID, gene name
        - Gene biotype (protein_coding, lncRNA, etc.)
        - Exon count
        
        Parameters
        ----------
        output_filename : str, default="gene_features.tsv"
            Name of output file
        target_chromosomes : List[str], optional
            Filter to specific chromosomes
        force_overwrite : bool, default=False
            If True, regenerate even if file exists
            
        Returns
        -------
        Dict[str, Any]
            - 'success': bool
            - 'gene_features_file': str
            - 'gene_features_df': pl.DataFrame
            - 'error': str (if failed)
        """
        result = {
            'success': False,
            'gene_features_file': None,
            'gene_features_df': None,
            'error': None
        }
        
        output_path = self.data_dir / output_filename
        
        # Check if exists and not forcing overwrite
        if output_path.exists() and not force_overwrite:
            if self.verbosity >= 1:
                print(f"[derive] Using existing gene features: {output_path}")
            try:
                gene_features_df = pl.read_csv(
                    output_path,
                    separator='\t',
                    schema_overrides={'chrom': pl.Utf8}
                )
                
                if target_chromosomes:
                    gene_features_df = self._filter_by_chromosomes(
                        gene_features_df, target_chromosomes
                    )
                
                result['success'] = True
                result['gene_features_file'] = str(output_path)
                result['gene_features_df'] = gene_features_df
                return result
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"[warning] Failed to load existing gene features: {e}")
        
        # Extract from GTF
        if self.verbosity >= 1:
            print(f"[derive] Extracting gene features from GTF...")
        
        try:
            from meta_spliceai.splice_engine.extract_genomic_features import extract_gene_features_from_gtf
            
            gtf_file = self.registry.get_gtf_path(validate=True)
            
            # Use the existing extraction function
            gene_features_df = extract_gene_features_from_gtf(
                gtf_file_path=str(gtf_file),
                verbose=self.verbosity,
                save=False  # We'll save manually with correct filename
            )
            
            # Apply chromosome filtering if requested
            if target_chromosomes:
                gene_features_df = self._filter_by_chromosomes(
                    gene_features_df, target_chromosomes
                )
            
            # Ensure we have essential columns
            required_cols = ['chrom', 'start', 'end', 'strand', 'gene_id']
            for col in required_cols:
                if col not in gene_features_df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Save to file
            gene_features_df.write_csv(output_path, separator='\t')
            
            if self.verbosity >= 1:
                print(f"[derive] ✅ Generated gene features: {len(gene_features_df)} genes")
                print(f"[derive] Saved to: {output_path}")
            
            result['success'] = True
            result['gene_features_file'] = str(output_path)
            result['gene_features_df'] = gene_features_df
            
        except Exception as e:
            result['error'] = f"Failed to derive gene features: {e}"
            if self.verbosity >= 1:
                print(f"[error] {result['error']}")
        
        return result
    
    def derive_transcript_features(
        self,
        output_filename: str = "transcript_features.tsv",
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """Derive transcript-level features from GTF.
        
        Extracts transcript-level metadata:
        - Chromosome, start, end, strand
        - Transcript ID, gene ID
        - Transcript biotype
        - Exon count
        - Transcript length (sum of exons)
        - CDS length (if available)
        
        Parameters
        ----------
        output_filename : str, default="transcript_features.tsv"
            Name of output file
        target_chromosomes : List[str], optional
            Filter to specific chromosomes
        force_overwrite : bool, default=False
            If True, regenerate even if file exists
            
        Returns
        -------
        Dict[str, Any]
            - 'success': bool
            - 'transcript_features_file': str
            - 'transcript_features_df': pl.DataFrame
            - 'error': str (if failed)
        """
        result = {
            'success': False,
            'transcript_features_file': None,
            'transcript_features_df': None,
            'error': None
        }
        
        output_path = self.data_dir / output_filename
        
        # Check if exists
        if output_path.exists() and not force_overwrite:
            if self.verbosity >= 1:
                print(f"[derive] Using existing transcript features: {output_path}")
            try:
                transcript_features_df = pl.read_csv(
                    output_path,
                    separator='\t',
                    schema_overrides={'chrom': pl.Utf8}
                )
                
                if target_chromosomes:
                    transcript_features_df = self._filter_by_chromosomes(
                        transcript_features_df, target_chromosomes
                    )
                
                result['success'] = True
                result['transcript_features_file'] = str(output_path)
                result['transcript_features_df'] = transcript_features_df
                return result
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"[warning] Failed to load existing transcript features: {e}")
        
        # Extract from GTF
        if self.verbosity >= 1:
            print(f"[derive] Extracting transcript features from GTF...")
        
        try:
            from meta_spliceai.splice_engine.extract_genomic_features import extract_transcript_features_from_gtf
            
            gtf_file = self.registry.get_gtf_path(validate=True)
            
            # Use the existing extraction function for transcripts
            transcript_features_df = extract_transcript_features_from_gtf(
                gtf_file_path=str(gtf_file),
                verbose=self.verbosity,
                save=False  # We'll save manually with correct filename
            )
            
            # Apply chromosome filtering if requested
            if target_chromosomes:
                transcript_features_df = self._filter_by_chromosomes(
                    transcript_features_df, target_chromosomes
                )
            
            # Save to file
            transcript_features_df.write_csv(output_path, separator='\t')
            
            if self.verbosity >= 1:
                print(f"[derive] ✅ Generated transcript features: {len(transcript_features_df)} transcripts")
                print(f"[derive] Saved to: {output_path}")
            
            result['success'] = True
            result['transcript_features_file'] = str(output_path)
            result['transcript_features_df'] = transcript_features_df
            
        except Exception as e:
            result['error'] = f"Failed to derive transcript features: {e}"
            if self.verbosity >= 1:
                print(f"[error] {result['error']}")
        
        return result
    
    def derive_exon_features(
        self,
        output_filename: str = "exon_features.tsv",
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """Derive exon-level features from GTF.
        
        Extracts exon-level metadata:
        - Chromosome, start, end, strand
        - Exon ID, exon number/rank
        - Transcript ID, gene ID
        - Exon length
        
        Parameters
        ----------
        output_filename : str, default="exon_features.tsv"
            Name of output file
        target_chromosomes : List[str], optional
            Filter to specific chromosomes
        force_overwrite : bool, default=False
            If True, regenerate even if file exists
            
        Returns
        -------
        Dict[str, Any]
            - 'success': bool
            - 'exon_features_file': str
            - 'exon_features_df': pl.DataFrame
            - 'error': str (if failed)
        """
        result = {
            'success': False,
            'exon_features_file': None,
            'exon_features_df': None,
            'error': None
        }
        
        output_path = self.data_dir / output_filename
        
        # Check if exists
        if output_path.exists() and not force_overwrite:
            if self.verbosity >= 1:
                print(f"[derive] Using existing exon features: {output_path}")
            try:
                exon_features_df = pl.read_csv(
                    output_path,
                    separator='\t',
                    schema_overrides={'chrom': pl.Utf8}
                )
                
                if target_chromosomes:
                    exon_features_df = self._filter_by_chromosomes(
                        exon_features_df, target_chromosomes
                    )
                
                result['success'] = True
                result['exon_features_file'] = str(output_path)
                result['exon_features_df'] = exon_features_df
                return result
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"[warning] Failed to load existing exon features: {e}")
        
        # Extract from GTF
        if self.verbosity >= 1:
            print(f"[derive] Extracting exon features from GTF...")
        
        try:
            from meta_spliceai.splice_engine.extract_genomic_features import extract_exon_features_from_gtf
            
            gtf_file = self.registry.get_gtf_path(validate=True)
            
            # Use the existing extraction function for exons
            exon_features_df = extract_exon_features_from_gtf(
                gtf_file_path=str(gtf_file),
                verbose=self.verbosity,
                save=False  # We'll save manually with correct filename
            )
            
            # Apply chromosome filtering if requested
            if target_chromosomes:
                exon_features_df = self._filter_by_chromosomes(
                    exon_features_df, target_chromosomes
                )
            
            # Save to file
            exon_features_df.write_csv(output_path, separator='\t')
            
            if self.verbosity >= 1:
                print(f"[derive] ✅ Generated exon features: {len(exon_features_df)} exons")
                print(f"[derive] Saved to: {output_path}")
            
            result['success'] = True
            result['exon_features_file'] = str(output_path)
            result['exon_features_df'] = exon_features_df
            
        except Exception as e:
            result['error'] = f"Failed to derive exon features: {e}"
            if self.verbosity >= 1:
                print(f"[error] {result['error']}")
        
        return result
    
    def derive_junctions(
        self,
        output_filename: str = "junctions.tsv",
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """Derive splice junctions from GTF.
        
        Extracts splice junctions (exon-exon boundaries):
        - Chromosome, junction start, junction end
        - Strand
        - Gene ID, transcript ID
        - Upstream exon ID, downstream exon ID
        - Junction type (annotated, canonical, non-canonical)
        
        Parameters
        ----------
        output_filename : str, default="junctions.tsv"
            Name of output file
        target_chromosomes : List[str], optional
            Filter to specific chromosomes
        force_overwrite : bool, default=False
            If True, regenerate even if file exists
            
        Returns
        -------
        Dict[str, Any]
            - 'success': bool
            - 'junctions_file': str
            - 'junctions_df': pl.DataFrame
            - 'error': str (if failed)
        """
        result = {
            'success': False,
            'junctions_file': None,
            'junctions_df': None,
            'error': None
        }
        
        output_path = self.data_dir / output_filename
        
        # Check if exists
        if output_path.exists() and not force_overwrite:
            if self.verbosity >= 1:
                print(f"[derive] Using existing junctions: {output_path}")
            try:
                junctions_df = pl.read_csv(
                    output_path,
                    separator='\t',
                    schema_overrides={'chrom': pl.Utf8}
                )
                
                if target_chromosomes:
                    junctions_df = self._filter_by_chromosomes(
                        junctions_df, target_chromosomes
                    )
                
                result['success'] = True
                result['junctions_file'] = str(output_path)
                result['junctions_df'] = junctions_df
                return result
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"[warning] Failed to load existing junctions: {e}")
        
        # Derive junctions from splice sites
        if self.verbosity >= 1:
            print(f"[derive] Deriving junctions from splice sites...")
        
        try:
            # Load splice sites using Registry (prefers enhanced version)
            splice_sites_path = self.registry.resolve('splice_sites')
            if not splice_sites_path:
                raise FileNotFoundError("Splice sites file not found. Run derive --splice-sites first.")
            
            if self.verbosity >= 1:
                print(f"[derive] Loading splice sites from: {Path(splice_sites_path).name}")
            
            splice_sites_df = pl.read_csv(
                splice_sites_path,
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            
            # Apply chromosome filtering if requested
            if target_chromosomes:
                splice_sites_df = self._filter_by_chromosomes(
                    splice_sites_df, target_chromosomes
                )
            
            # Create junctions by pairing donor and acceptor sites
            # Group by gene_id and transcript_id
            donors = splice_sites_df.filter(pl.col('site_type') == 'donor')
            acceptors = splice_sites_df.filter(pl.col('site_type') == 'acceptor')
            
            # Join donors and acceptors within the same transcript
            junctions_df = donors.join(
                acceptors,
                on=['gene_id', 'transcript_id', 'chrom', 'strand'],
                suffix='_acceptor'
            ).filter(
                # Donor position should be before acceptor position
                pl.col('position') < pl.col('position_acceptor')
            )
            
            # Select columns (include gene_name only if it exists)
            select_cols = [
                pl.col('chrom'),
                pl.col('position').alias('donor_pos'),
                pl.col('position_acceptor').alias('acceptor_pos'),
                pl.col('strand'),
                pl.col('gene_id'),
                pl.col('transcript_id'),
                (pl.col('position_acceptor') - pl.col('position')).alias('intron_length')
            ]
            
            # Add gene_name if available
            if 'gene_name' in junctions_df.columns:
                select_cols.insert(5, pl.col('gene_name'))  # Insert after transcript_id
            
            junctions_df = junctions_df.select(select_cols)
            
            # Save to file
            junctions_df.write_csv(output_path, separator='\t')
            
            if self.verbosity >= 1:
                print(f"[derive] ✅ Generated junctions: {len(junctions_df)} junctions")
                print(f"[derive] Saved to: {output_path}")
            
            result['success'] = True
            result['junctions_file'] = str(output_path)
            result['junctions_df'] = junctions_df
            
        except Exception as e:
            result['error'] = f"Failed to derive junctions: {e}"
            if self.verbosity >= 1:
                print(f"[error] {result['error']}")
        
        return result


# Convenience functions for backward compatibility
def derive_gene_annotations(data_dir: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Convenience function for deriving gene annotations."""
    deriver = GenomicDataDeriver(data_dir=data_dir, verbosity=kwargs.get('verbosity', 1))
    return deriver.derive_gene_annotations(**kwargs)


def derive_splice_sites(data_dir: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Convenience function for deriving splice sites."""
    deriver = GenomicDataDeriver(data_dir=data_dir, verbosity=kwargs.get('verbosity', 1))
    return deriver.derive_splice_sites(**kwargs)


def derive_genomic_sequences(data_dir: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Convenience function for deriving genomic sequences."""
    deriver = GenomicDataDeriver(data_dir=data_dir, verbosity=kwargs.get('verbosity', 1))
    return deriver.derive_genomic_sequences(**kwargs)


def derive_overlapping_genes(data_dir: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Convenience function for deriving overlapping genes."""
    deriver = GenomicDataDeriver(data_dir=data_dir, verbosity=kwargs.get('verbosity', 1))
    return deriver.derive_overlapping_genes(**kwargs)


def derive_gene_features(data_dir: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Convenience function for deriving gene features."""
    deriver = GenomicDataDeriver(data_dir=data_dir, verbosity=kwargs.get('verbosity', 1))
    return deriver.derive_gene_features(**kwargs)


def derive_transcript_features(data_dir: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Convenience function for deriving transcript features."""
    deriver = GenomicDataDeriver(data_dir=data_dir, verbosity=kwargs.get('verbosity', 1))
    return deriver.derive_transcript_features(**kwargs)


def derive_exon_features(data_dir: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Convenience function for deriving exon features."""
    deriver = GenomicDataDeriver(data_dir=data_dir, verbosity=kwargs.get('verbosity', 1))
    return deriver.derive_exon_features(**kwargs)


def derive_junctions(data_dir: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Convenience function for deriving splice junctions."""
    deriver = GenomicDataDeriver(data_dir=data_dir, verbosity=kwargs.get('verbosity', 1))
    return deriver.derive_junctions(**kwargs)
