"""
Utilities for enhancing data with additional features from various sources.
Useful for enriching splice site annotations with gene and transcript-level information.
"""

import os
import polars as pl
from typing import Optional, List, Union, Dict, Any, Tuple
from pathlib import Path


def enhance_splice_sites_with_features(
    splice_sites_path: str,
    gene_features_path: Optional[str] = None,
    transcript_features_path: Optional[str] = None,
    project_dir: Optional[str] = None,
    gene_types_to_keep: Optional[List[str]] = None,
    transcript_types_to_keep: Optional[List[str]] = None,
    verbose: int = 1
) -> pl.DataFrame:
    """
    Enhance splice site annotations with gene and transcript features.
    
    This function loads splice site annotations and joins them with gene and
    transcript features to provide additional filtering capabilities.
    
    Parameters
    ----------
    splice_sites_path : str
        Path to the splice sites TSV file
    gene_features_path : Optional[str], optional
        Path to the gene features TSV file. If None, will try to find in default location.
    transcript_features_path : Optional[str], optional
        Path to the transcript features TSV file. If None, will try to find in default location.
    project_dir : Optional[str], optional
        Project directory root, used to find default file locations if paths not provided.
    gene_types_to_keep : Optional[List[str]], optional
        List of gene types to include (e.g., ["protein_coding", "lncRNA"]). 
        If None, all gene types are kept.
    transcript_types_to_keep : Optional[List[str]], optional
        List of transcript types to include. If None, all transcript types are kept.
    verbose : int, optional
        Verbosity level (0=silent, 1=basic info, 2=detailed), by default 1
    
    Returns
    -------
    pl.DataFrame
        Enhanced dataframe with splice site annotations and additional features
    
    Examples
    --------
    >>> enhanced_splice_sites = enhance_splice_sites_with_features(
    ...     "/path/to/splice_sites.tsv",
    ...     gene_types_to_keep=["protein_coding"]
    ... )
    """
    # Resolve paths if not explicitly provided
    if project_dir is None:
        # Try to infer project directory - find the splice-surveyor project root
        if os.path.isabs(splice_sites_path):
            # Start from splice_sites_path and look for project markers
            current_dir = Path(splice_sites_path).parent
            while current_dir.name:
                if (current_dir / '.git').exists() or (current_dir / 'setup.py').exists() or current_dir.name == 'splice-surveyor':
                    project_dir = str(current_dir)
                    if verbose >= 1:
                        print(f"[info] Found project directory: {project_dir}")
                    break
                current_dir = current_dir.parent
        
        if project_dir is None:
            # Second approach: Try to detect from the current working directory
            current_dir = Path(os.getcwd())
            while current_dir.name:
                if (current_dir / '.git').exists() or (current_dir / 'setup.py').exists() or current_dir.name == 'splice-surveyor':
                    project_dir = str(current_dir)
                    if verbose >= 1:
                        print(f"[info] Found project directory: {project_dir}")
                    break
                current_dir = current_dir.parent
            
        if project_dir is None:
            # Fallback to hardcoded path if we can't detect
            project_dir = '/home/bchiu/work/splice-surveyor'
            if verbose >= 1:
                print(f"[warning] Could not detect project directory, using fallback: {project_dir}")
    
    # Derive default paths if not provided
    if gene_features_path is None:
        gene_features_path = os.path.join(
            project_dir, "data", "ensembl", "spliceai_analysis", "gene_features.tsv"
        )
        if verbose >= 1:
            print(f"[info] Using default gene features path: {gene_features_path}")
    
    if transcript_features_path is None and transcript_types_to_keep is not None:
        transcript_features_path = os.path.join(
            project_dir, "data", "ensembl", "spliceai_analysis", "transcript_features.tsv"
        )
        if verbose >= 1:
            print(f"[info] Using default transcript features path: {transcript_features_path}")
    
    # Load splice sites
    if verbose >= 1:
        print(f"[i/o] Loading splice sites from: {splice_sites_path}")
    
    splice_sites_df = pl.read_csv(
        splice_sites_path,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Check if files exist
    if not os.path.exists(gene_features_path):
        raise FileNotFoundError(f"Gene features file not found: {gene_features_path}")
    
    # Print basic stats before joining
    if verbose >= 1:
        print(f"[info] Loaded {len(splice_sites_df)} splice sites")
        print(f"[info] Number of unique genes in splice sites: {splice_sites_df['gene_id'].n_unique()}")
        print(f"[info] Number of unique transcripts in splice sites: {splice_sites_df['transcript_id'].n_unique()}")
    
    # Load gene features
    if verbose >= 1:
        print(f"[i/o] Loading gene features from: {gene_features_path}")
    
    gene_features_df = pl.read_csv(
        gene_features_path,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Print gene feature stats
    if verbose >= 2:
        print(f"[info] Gene features columns: {gene_features_df.columns}")
        if 'gene_type' in gene_features_df.columns:
            gene_type_counts = gene_features_df.group_by('gene_type').agg(
                pl.count('gene_id').alias('count')
            ).sort('count', descending=True)
            print("[info] Gene type distribution:")
            print(gene_type_counts.head(10))
    
    # Join splice sites with gene features
    enhanced_df = splice_sites_df.join(
        gene_features_df.select(['gene_id', 'gene_name', 'gene_type', 'gene_length']),
        on='gene_id',
        how='left'
    )
    
    # Check for missing gene feature information
    missing_gene_info = enhanced_df.filter(pl.col('gene_type').is_null()).select('gene_id').unique()
    if len(missing_gene_info) > 0 and verbose >= 1:
        pct_missing = len(missing_gene_info) / enhanced_df['gene_id'].n_unique() * 100
        print(f"[warning] {len(missing_gene_info)} genes ({pct_missing:.2f}%) in splice sites don't have gene feature information")
        if verbose >= 2:
            print(f"[info] Sample of genes without feature information: {missing_gene_info.head(5)}")
    
    # Optionally load transcript features
    if transcript_features_path is not None and os.path.exists(transcript_features_path):
        if verbose >= 1:
            print(f"[i/o] Loading transcript features from: {transcript_features_path}")
        
        transcript_features_df = pl.read_csv(
            transcript_features_path,
            separator='\t',
            schema_overrides={'chrom': pl.Utf8}
        )
        
        # Print transcript feature stats
        if verbose >= 2:
            print(f"[info] Transcript features columns: {transcript_features_df.columns}")
            if 'transcript_type' in transcript_features_df.columns:
                transcript_type_counts = transcript_features_df.group_by('transcript_type').agg(
                    pl.count('transcript_id').alias('count')
                ).sort('count', descending=True)
                print("[info] Transcript type distribution:")
                print(transcript_type_counts.head(10))
        
        # Check which columns to join on
        if 'transcript_id' in transcript_features_df.columns:
            # Determine which transcript columns to include (avoiding duplicates)
            # We want to keep all transcript-specific columns but avoid duplicates
            transcript_columns = []
            for col in transcript_features_df.columns:
                # Skip the join key and any columns already in enhanced_df
                if col == 'transcript_id' or col in enhanced_df.columns:
                    continue
                transcript_columns.append(col)
            
            if verbose >= 2:
                print(f"[info] Adding transcript columns: {transcript_columns}")
            
            # Join with transcript features
            enhanced_df = enhanced_df.join(
                transcript_features_df.select(
                    ['transcript_id'] + transcript_columns
                ),
                on='transcript_id',
                how='left'
            )
            
            # Check for missing transcript feature information
            missing_transcript_info = enhanced_df.filter(
                (pl.col('transcript_id').is_not_null()) & 
                (pl.col(transcript_columns[0]).is_null() if transcript_columns else False)
            ).select('transcript_id').unique() if transcript_columns else pl.DataFrame()
            
            if len(missing_transcript_info) > 0 and verbose >= 1:
                pct_missing = len(missing_transcript_info) / enhanced_df.filter(pl.col('transcript_id').is_not_null())['transcript_id'].n_unique() * 100
                print(f"[warning] {len(missing_transcript_info)} transcripts ({pct_missing:.2f}%) don't have transcript feature information")
    else:
        if transcript_features_path is not None and verbose >= 1:
            print(f"[warning] Transcript features file not found: {transcript_features_path}")
    
    # Apply gene type filtering if requested
    if gene_types_to_keep is not None and 'gene_type' in enhanced_df.columns:
        pre_filter_count = len(enhanced_df)
        enhanced_df = enhanced_df.filter(pl.col('gene_type').is_in(gene_types_to_keep))
        
        if verbose >= 1:
            filtered_count = len(enhanced_df)
            pct_kept = filtered_count / pre_filter_count * 100 if pre_filter_count > 0 else 0
            print(f"[info] Kept {filtered_count}/{pre_filter_count} splice sites ({pct_kept:.2f}%) after gene type filtering")
            print(f"[info] Gene types kept: {gene_types_to_keep}")
    
    # Apply transcript type filtering if requested
    if transcript_types_to_keep is not None and 'transcript_type' in enhanced_df.columns:
        pre_filter_count = len(enhanced_df)
        enhanced_df = enhanced_df.filter(pl.col('transcript_type').is_in(transcript_types_to_keep))
        
        if verbose >= 1:
            filtered_count = len(enhanced_df)
            pct_kept = filtered_count / pre_filter_count * 100 if pre_filter_count > 0 else 0
            print(f"[info] Kept {filtered_count}/{pre_filter_count} splice sites ({pct_kept:.2f}%) after transcript type filtering")
            print(f"[info] Transcript types kept: {transcript_types_to_keep}")
    
    # Final stats
    if verbose >= 1:
        print(f"[info] Final enhanced dataset has {len(enhanced_df)} splice sites")
        print(f"[info] Number of unique genes in final dataset: {enhanced_df['gene_id'].n_unique()}")
        print(f"[info] Number of unique transcripts in final dataset: {enhanced_df['transcript_id'].n_unique()}")
    
    return enhanced_df
