"""
Proposal for transcript-aware splice site position identification
"""

def get_position_grouping_columns(
    mode='genomic',  # 'genomic', 'transcript', 'hybrid'
    include_splice_type=False,
    custom_columns=None
):
    """
    Define grouping columns for position identification based on mode.
    
    Parameters
    ----------
    mode : str
        - 'genomic': Current approach - group by genomic coordinates only
        - 'transcript': Full transcript-aware grouping
        - 'hybrid': Group by genomic coords but preserve transcript info
    include_splice_type : bool
        Whether to include splice_type in grouping (usually False for deduplication)
    custom_columns : list
        Additional columns to include in grouping
        
    Returns
    -------
    list
        Column names to use for grouping
    """
    # Base genomic coordinates
    base_cols = ['gene_id', 'position', 'strand']
    
    if mode == 'genomic':
        # Current approach - genomic position only
        group_cols = base_cols
        
    elif mode == 'transcript':
        # Full transcript-aware approach
        group_cols = base_cols + ['transcript_id']
        
    elif mode == 'hybrid':
        # Hybrid: Group by genomic position but aggregate transcript info
        # This would require special handling to collect all transcript IDs
        # for each position into a list/set
        group_cols = base_cols
        # Note: Would need additional logic to preserve transcript_ids as a list
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Optionally include splice_type (usually not for deduplication)
    if include_splice_type:
        group_cols.append('splice_type')
        
    # Add any custom columns
    if custom_columns:
        group_cols.extend(custom_columns)
        
    return group_cols


def resolve_transcript_specific_conflicts(
    df,
    mode='genomic',
    preserve_transcript_info=True
):
    """
    Resolve conflicts with transcript awareness.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe with positions and predictions
    mode : str
        Grouping mode (see get_position_grouping_columns)
    preserve_transcript_info : bool
        Whether to preserve transcript information even when grouping genomically
        
    Returns
    -------
    DataFrame
        Deduplicated dataframe
    """
    import polars as pl
    
    # Define prediction type priorities
    pred_type_priority = {'TP': 0, 'FN': 1, 'FP': 2, 'TN': 3}
    
    # Get grouping columns
    group_cols = get_position_grouping_columns(mode=mode)
    
    if mode == 'genomic' and preserve_transcript_info:
        # Special handling: group by genomic position but preserve transcript info
        
        # First, aggregate transcript information for each position
        transcript_agg = df.group_by(group_cols).agg([
            pl.col('transcript_id').unique().alias('all_transcript_ids'),
            pl.col('transcript_id').count().alias('transcript_count'),
            # Collect all unique splice types seen at this position
            pl.col('splice_type').unique().alias('observed_splice_types')
        ])
        
        # Then proceed with priority-based deduplication
        # ... (rest of the deduplication logic)
        
    elif mode == 'transcript':
        # Each transcript-position combination is treated separately
        # No deduplication needed across transcripts
        # Only deduplicate within same transcript-position if there are conflicts
        
        # Apply priority-based resolution per transcript-position
        # ... (modified deduplication logic)
        
    else:
        # Standard genomic-only deduplication (current approach)
        # ... (existing logic)
        
    return df


# Additional columns that might be important for position identification:
POSITION_IDENTIFIER_COLUMNS = {
    'core': ['gene_id', 'position', 'strand'],
    'transcript_specific': ['gene_id', 'position', 'strand', 'transcript_id'],
    'clinical': ['gene_id', 'position', 'strand', 'transcript_id', 'hgvs_notation'],
    'structural': ['gene_id', 'position', 'strand', 'exon_number', 'intron_number'],
    'regulatory': ['gene_id', 'position', 'strand', 'regulatory_region_id'],
    'conservation': ['gene_id', 'position', 'strand', 'phylop_score_bin'],
}
