"""
Proposed modification to sequence_data_utils.py for transcript-aware position handling
"""

def extract_analysis_sequences_v2(
    sequence_df, 
    position_df, 
    window_size=250, 
    include_empty_entries=True,
    essential_columns_only=False, 
    additional_columns=None,
    add_transcript_count=True, 
    drop_transcript_id=False,  # Changed default to False
    resolve_prediction_conflicts=True,
    position_id_mode='genomic',  # NEW: 'genomic', 'transcript', 'hybrid'
    preserve_transcript_list=False,  # NEW: Keep list of all transcript IDs
    **kwargs
):
    """
    Enhanced version with transcript-aware position identification.
    
    New Parameters
    --------------
    position_id_mode : str
        - 'genomic': Current behavior - group by genomic position only
        - 'transcript': Each transcript-position is unique
        - 'hybrid': Group genomically but preserve transcript information
    preserve_transcript_list : bool
        If True and mode is 'genomic' or 'hybrid', keep a list of all
        transcript IDs that share each position
    """
    import polars as pl
    
    # ... (earlier code remains the same until conflict resolution) ...
    
    # Modified conflict resolution section (starting around line 549)
    if resolve_prediction_conflicts and 'pred_type' in output_df.columns:
        original_count = len(output_df)
        
        # Define prediction type priorities
        pred_type_priority = {'TP': 0, 'FN': 1, 'FP': 2, 'TN': 3}
        
        # Determine grouping columns based on mode
        if position_id_mode == 'transcript':
            # Full transcript-specific: include transcript_id in grouping
            group_cols = ['gene_id', 'position', 'strand', 'transcript_id']
            print(f"[info] Using transcript-specific position identification")
            
        elif position_id_mode == 'hybrid':
            # Hybrid: Group genomically but preserve transcript info
            group_cols = ['gene_id', 'position', 'strand']
            
            if preserve_transcript_list and 'transcript_id' in output_df.columns:
                # First collect all transcript IDs for each position
                transcript_groups = output_df.group_by(group_cols).agg([
                    pl.col('transcript_id').unique().alias('transcript_id_list'),
                    pl.col('transcript_id').count().alias('transcript_count_total'),
                    # Collect splice types seen across transcripts
                    pl.col('splice_type').unique().alias('splice_types_observed'),
                    # Track which transcripts have which splice types
                    pl.struct(['transcript_id', 'splice_type']).unique().alias('transcript_splice_map')
                ])
                print(f"[info] Hybrid mode: preserving transcript information for {len(transcript_groups)} positions")
                
        else:  # position_id_mode == 'genomic' (default/current behavior)
            group_cols = ['gene_id', 'position', 'strand']
            print(f"[info] Using genomic position identification (current behavior)")
        
        # Add priority column
        output_df = output_df.with_columns(
            pl.lit(99).alias('pred_type_priority')
        )
        
        # Update priority for each prediction type
        for pred_type, priority in pred_type_priority.items():
            output_df = output_df.with_columns(
                pl.when(pl.col('pred_type') == pred_type)
                .then(pl.lit(priority))
                .otherwise(pl.col('pred_type_priority'))
                .alias('pred_type_priority')
            )
        
        # Perform the deduplication based on the chosen mode
        if position_id_mode == 'hybrid' and preserve_transcript_list:
            # Special handling for hybrid mode with transcript preservation
            
            # First, get the best prediction for each genomic position
            best_predictions = output_df.sort('pred_type_priority').group_by(group_cols).first()
            
            # Then merge with the transcript information we collected
            output_df = best_predictions.join(
                transcript_groups,
                on=group_cols,
                how='left'
            )
            
            # Add a flag indicating this position has multiple transcript contexts
            output_df = output_df.with_columns(
                (pl.col('transcript_count_total') > 1).alias('is_multi_transcript')
            )
            
            print(f"[info] Preserved transcript information for {(output_df['is_multi_transcript']).sum()} multi-transcript positions")
            
        else:
            # Standard deduplication (works for both 'genomic' and 'transcript' modes)
            output_df = output_df.with_columns(
                pl.col('pred_type_priority').rank(method='dense', descending=False)
                .over(group_cols)
                .alias('group_rank')
            )
            
            output_df = output_df.filter(pl.col('group_rank') == 1)
            output_df = output_df.drop(['group_rank'])
        
        # Clean up temporary column
        output_df = output_df.drop('pred_type_priority')
        
        # Report on the deduplication
        if original_count > len(output_df):
            dedup_count = original_count - len(output_df)
            if position_id_mode == 'transcript':
                print(f"[info] Resolved {dedup_count} conflicting predictions within transcript-position pairs")
            else:
                print(f"[info] Resolved {dedup_count} conflicting prediction types through priority-based deduplication")
                
            # Additional reporting for biological insights
            if verbose and position_id_mode in ['hybrid', 'genomic'] and preserve_transcript_list:
                multi_role_positions = output_df.filter(
                    pl.col('splice_types_observed').list.len() > 1
                )
                if len(multi_role_positions) > 0:
                    print(f"[info] Found {len(multi_role_positions)} positions with different roles across transcripts:")
                    for row in multi_role_positions.head(5).iter_rows(named=True):
                        print(f"       Position {row['position']}: {row['splice_types_observed']}")
    
    return output_df


# Example usage showing the difference:
def demonstrate_transcript_awareness():
    """
    Show how different modes handle the same data
    """
    import polars as pl
    
    # Example data with a position that has different roles in different transcripts
    position_df = pl.DataFrame({
        'gene_id': ['GENE1'] * 4,
        'position': [1000, 1000, 1000, 2000],
        'strand': ['+'] * 4,
        'transcript_id': ['TX1', 'TX2', 'TX3', 'TX1'],
        'splice_type': ['donor', None, 'acceptor', 'donor'],  # Position 1000 has 3 different roles!
        'pred_type': ['TP', 'TN', 'FP', 'TP']
    })
    
    print("Original data:")
    print(position_df)
    print()
    
    # Mode 1: Genomic (current behavior)
    print("Mode: GENOMIC (current)")
    print("Result: One row per genomic position, highest priority prediction kept")
    print("Position 1000 would keep only the TP (donor) prediction")
    print()
    
    # Mode 2: Transcript-specific
    print("Mode: TRANSCRIPT")
    print("Result: Each transcript-position combination preserved")
    print("Position 1000 would have 3 separate entries (TX1:donor, TX2:neither, TX3:acceptor)")
    print()
    
    # Mode 3: Hybrid
    print("Mode: HYBRID")
    print("Result: One row per genomic position, but with transcript list")
    print("Position 1000 would have: best_prediction=TP, transcript_list=[TX1,TX2,TX3], splice_types=[donor,None,acceptor]")
    
if __name__ == "__main__":
    demonstrate_transcript_awareness()
