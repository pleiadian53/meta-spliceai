"""
Enhanced Transcript-Aware Splice Site Position Identification
============================================================

This module addresses the fundamental biological reality that splice sites are 
transcript-specific, not genomic-specific. The same nucleotide position can:

1. Be a donor site in one isoform
2. Be an acceptor site in another isoform (rare but biologically valid)
3. Not be a splice site at all in yet another isoform
4. Be exonic in one transcript but intronic in another

CRITICAL INSIGHT FOR META-LEARNING:
The goal is to enable meta-models to predict how variants affect splicing patterns.
This requires:
- Position identification: ['gene_id', 'position', 'strand', 'transcript_id']
- Prediction target: 'splice_type' (what we want to predict)
- NOT grouping by splice_type (that would be circular - we're trying to predict it!)

Current Limitation:
- All workflows use genomic-only grouping: ['gene_id', 'position', 'strand']
- This forces a single splice_type label per position across all transcripts
- Loses critical transcript-specific splicing information
- Prevents meta-learning from capturing variant effects on alternative splicing

Enhanced Solution:
- Implement transcript-aware position identification
- Enable meta-learning to capture how the same position can have different splice roles
- Maintain backward compatibility with existing workflows
- Support variant effect prediction and precision medicine applications
"""

def get_position_grouping_columns(
    mode='genomic',  # 'genomic', 'transcript', 'hybrid', 'splice_aware', 'splice_only'
    include_splice_type=False,
    custom_columns=None
):
    """
    Define grouping columns for position identification based on mode.
    
    Parameters
    ----------
    mode : str
        Position identification strategy:
        - 'genomic': Current approach - group by genomic coordinates only
                    Forces single splice_type per position (current limitation)
        - 'transcript': Full transcript-aware grouping - each transcript-position is unique
                       Preserves transcript-specific splice site roles
        - 'hybrid': Group by genomic coords but preserve transcript info as metadata
                   Maintains current ML efficiency while preserving biological context
        - 'splice_aware': Group by genomic coords + transcript_id (splice_type as prediction target)
                         Enables meta-learning to capture variant effects on splicing patterns
        - 'splice_only': Group by genomic coords + splice_type only
                        Groups by predicted splice role (for analysis, not training)
    include_splice_type : bool
        Whether to include splice_type in grouping (usually False for standard deduplication)
        Note: In 'splice_aware' mode, this is automatically True
    custom_columns : list
        Additional columns to include in grouping
        
    Returns
    -------
    list
        Column names to use for grouping
        
    Examples
    --------
    >>> get_position_grouping_columns('genomic')
    ['gene_id', 'position', 'strand']
    
    >>> get_position_grouping_columns('transcript') 
    ['gene_id', 'position', 'strand', 'transcript_id']
    
    >>> get_position_grouping_columns('splice_aware')
    ['gene_id', 'position', 'strand', 'transcript_id']
    
    >>> get_position_grouping_columns('splice_only')
    ['gene_id', 'position', 'strand', 'splice_type']
    """
    # Base genomic coordinates - universal foundation
    base_cols = ['gene_id', 'position', 'strand']
    
    if mode == 'genomic':
        # Current approach - genomic position only
        # LIMITATION: Forces single splice_type per position across all transcripts
        group_cols = base_cols
        
    elif mode in ['transcript', 'splice_aware']:
        # Both modes use identical grouping: ['gene_id', 'position', 'strand', 'transcript_id']
        # Each transcript-position combination treated separately
        
        # SEMANTIC DISTINCTIONS (same implementation):
        # 'transcript': Focus on biological reality - transcript-specific splice site roles
        #               Emphasizes that splice sites are inherently transcript-dependent
        #               
        # 'splice_aware': Focus on meta-learning capability - variant effect prediction
        #                 Emphasizes enabling meta-models to learn splice role plasticity
        #                 How the same position can have different splice roles due to:
        #                 - Genetic variants, sequence context, regulatory changes
        
        group_cols = base_cols + ['transcript_id']
        # splice_type remains as prediction TARGET, not part of position identification
        
    elif mode == 'hybrid':
        # Hybrid: Group by genomic position but aggregate transcript info as metadata
        # ADVANTAGE: Maintains current ML efficiency while preserving biological context
        # This requires special handling to collect all transcript IDs
        # for each position into a list/set during aggregation
        group_cols = base_cols
        # Note: Requires additional logic to preserve transcript_ids as metadata
        
    elif mode == 'splice_only':
        # Group by genomic position + splice_type only (no transcript_id)
        # TRADE-OFF: Preserves splice role diversity but loses transcript-specific prediction context
        # Use case: When you want splice role diversity but computational efficiency
        group_cols = base_cols + ['splice_type']
        include_splice_type = True  # Force inclusion for this mode
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: 'genomic', 'transcript', 'hybrid', 'splice_aware', 'splice_only'")
    
    # Include splice_type if requested (or forced by splice_aware mode)
    if include_splice_type and 'splice_type' not in group_cols:
        group_cols.append('splice_type')
        
    # Add any custom columns
    if custom_columns:
        group_cols.extend(custom_columns)
        
    return group_cols


def resolve_transcript_specific_conflicts(
    df,
    mode='genomic',
    preserve_transcript_info=True,
    conflict_resolution='priority'  # 'priority', 'majority_vote', 'max_confidence'
):
    """
    Resolve prediction conflicts with transcript awareness and biological context.
    
    This function addresses the core challenge: the same genomic position can have
    different splice site roles across transcripts. Current workflows force a single
    label per position, losing critical biological information.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe with positions and predictions
        Must contain: gene_id, position, strand, pred_type, splice_type
        May contain: transcript_id, splice_probability, etc.
    mode : str
        Position identification strategy:
        - 'genomic': Current behavior - group by genomic coords only
        - 'transcript': Each transcript-position treated separately  
        - 'hybrid': Group genomically but preserve transcript metadata
        - 'splice_aware': Group by genomic coords + splice_type
    preserve_transcript_info : bool
        Whether to preserve transcript information as metadata
    conflict_resolution : str
        Strategy for resolving conflicts within the same grouping unit:
        - 'priority': TP > FN > FP > TN (current approach)
        - 'majority_vote': Most common prediction across transcripts
        - 'max_confidence': Highest splice_probability wins
        
    Returns
    -------
    DataFrame
        Processed dataframe with conflicts resolved according to mode
        
    Examples
    --------
    Position chr1:100 in gene ENSG123:
    - Transcript A: donor site (TP)
    - Transcript B: not a splice site (TN)
    - Transcript C: acceptor site (FP)
    
    mode='genomic': Forces single label → TP (by priority)
    mode='transcript': Preserves all three → 3 separate entries
    mode='splice_aware': Groups by transcript+splice_type → donor_T1(TP), donor_T2(FP), acceptor_T3(FP), neither_T2(TN)
    mode='hybrid': Single entry with transcript metadata → TP + transcript_list
    """
    import polars as pl
    
    # Validate required columns
    required_cols = ['gene_id', 'position', 'strand', 'pred_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Define prediction type priorities (current system)
    pred_type_priority = {'TP': 0, 'FN': 1, 'FP': 2, 'TN': 3}
    
    # Get grouping columns based on mode
    group_cols = get_position_grouping_columns(mode=mode)
    
    if mode == 'genomic' and preserve_transcript_info:
        # HYBRID APPROACH: Group by genomic position but preserve transcript context
        
        # First, collect transcript-specific information for each position
        transcript_metadata = df.group_by(['gene_id', 'position', 'strand']).agg([
            pl.col('transcript_id').unique().alias('transcript_ids'),
            pl.col('transcript_id').count().alias('transcript_count'),
            pl.col('splice_type').unique().alias('observed_splice_types'),
            # Count occurrences of each splice type at this position
            pl.col('splice_type').value_counts().alias('splice_type_counts'),
            # Collect all pred_types seen at this position
            pl.col('pred_type').unique().alias('observed_pred_types')
        ])
        
        # Apply standard priority-based deduplication
        df_deduplicated = _apply_priority_deduplication(df, group_cols, pred_type_priority)
        
        # Join back the transcript metadata
        df_final = df_deduplicated.join(
            transcript_metadata, 
            on=['gene_id', 'position', 'strand'], 
            how='left'
        )
        
        return df_final
        
    elif mode == 'transcript':
        # TRANSCRIPT-SPECIFIC APPROACH: Each transcript-position is unique
        # No cross-transcript deduplication needed
        # Only resolve conflicts within the same transcript-position
        
        # Apply priority-based resolution per transcript-position
        return _apply_priority_deduplication(df, group_cols, pred_type_priority)
        
    elif mode == 'splice_aware':
        # SPLICE-AWARE APPROACH: Allow same position to have different splice roles
        # Group by genomic position + splice_type
        
        # This naturally preserves different splice site roles at the same position
        return _apply_priority_deduplication(df, group_cols, pred_type_priority)
        
    else:  # mode == 'genomic' (standard current behavior)
        # GENOMIC-ONLY APPROACH: Current system - single label per position
        # LIMITATION: Loses transcript-specific splice site information
        
        return _apply_priority_deduplication(df, group_cols, pred_type_priority)


def _apply_priority_deduplication(df, group_cols, pred_type_priority):
    """Apply priority-based deduplication within groups."""
    import polars as pl
    
    # Add priority column
    df = df.with_columns(
        pl.col('pred_type').replace(pred_type_priority).alias('pred_type_priority')
    )
    
    # Keep only the highest priority entry per group
    df_deduplicated = df.with_columns(
        pl.col('pred_type_priority').rank(method='dense', descending=False)
        .over(group_cols)
        .alias('group_rank')
    ).filter(
        pl.col('group_rank') == 1
    ).drop(['pred_type_priority', 'group_rank'])
    
    return df_deduplicated


# Enhanced position identification strategies for different use cases
POSITION_IDENTIFIER_COLUMNS = {
    # Current system (genomic-only)
    'core': ['gene_id', 'position', 'strand'],
    
    # Biological reality - transcript-specific splice sites
    'transcript_specific': ['gene_id', 'position', 'strand', 'transcript_id'],
    
    # Splice-aware - identical to transcript_specific (splice_type is prediction target)
    'splice_aware': ['gene_id', 'position', 'strand', 'transcript_id'],
    
    # Clinical applications - precise isoform identification
    'clinical': ['gene_id', 'position', 'strand', 'transcript_id', 'hgvs_notation'],
    
    # Structural context - exon/intron boundaries
    'structural': ['gene_id', 'position', 'strand', 'exon_number', 'intron_number'],
    
    # Regulatory elements - enhancers, silencers
    'regulatory': ['gene_id', 'position', 'strand', 'regulatory_region_type'],
    
    # Conservation-based grouping
    'conservation': ['gene_id', 'position', 'strand', 'phylop_score_bin'],
    
    # Disease-specific analysis
    'pathogenic': ['gene_id', 'position', 'strand', 'transcript_id', 'pathogenicity_class'],
}


def analyze_position_complexity(df, verbose=True):
    """
    Analyze the complexity of position identification in a dataset.
    
    This function reveals how much biological information is lost by using
    genomic-only position identification vs. transcript-aware approaches.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe with position data
    verbose : bool
        Whether to print detailed analysis
        
    Returns
    -------
    dict
        Analysis results showing complexity metrics
    """
    import polars as pl
    
    results = {}
    
    # Count unique positions by different grouping strategies
    genomic_positions = df.select(['gene_id', 'position', 'strand']).n_unique()
    
    if 'transcript_id' in df.columns:
        transcript_positions = df.select(['gene_id', 'position', 'strand', 'transcript_id']).n_unique()
        transcript_complexity = transcript_positions / genomic_positions
        results['transcript_expansion_factor'] = transcript_complexity
    else:
        transcript_positions = None
        transcript_complexity = None
    
    if 'splice_type' in df.columns:
        splice_aware_positions = df.select(['gene_id', 'position', 'strand', 'splice_type']).n_unique()
        splice_complexity = splice_aware_positions / genomic_positions
        results['splice_aware_expansion_factor'] = splice_complexity
        
        # Analyze positions with multiple splice types
        multi_splice_positions = df.group_by(['gene_id', 'position', 'strand']).agg([
            pl.col('splice_type').n_unique().alias('splice_type_count')
        ]).filter(pl.col('splice_type_count') > 1)
        
        results['positions_with_multiple_splice_types'] = len(multi_splice_positions)
        results['percent_complex_positions'] = (len(multi_splice_positions) / genomic_positions) * 100
    
    results.update({
        'genomic_positions': genomic_positions,
        'transcript_positions': transcript_positions,
    })
    
    if verbose:
        print("=== Position Identification Complexity Analysis ===")
        print(f"Genomic positions (current): {genomic_positions:,}")
        if transcript_positions:
            print(f"Transcript-specific positions: {transcript_positions:,}")
            print(f"Transcript expansion factor: {transcript_complexity:.2f}x")
        
        if 'splice_aware_expansion_factor' in results:
            print(f"Splice-aware positions: {splice_aware_positions:,}")
            print(f"Splice-aware expansion factor: {splice_complexity:.2f}x")
            print(f"Positions with multiple splice types: {results['positions_with_multiple_splice_types']:,}")
            print(f"Percent complex positions: {results['percent_complex_positions']:.1f}%")
    
    return results


def demonstrate_biological_reality():
    """
    Demonstrate the biological reality that motivates transcript-aware position identification.
    
    Returns
    -------
    str
        Educational example showing the limitation of genomic-only grouping
    """
    example = """
    === BIOLOGICAL REALITY: Why Transcript-Aware Position ID Matters ===
    
    Consider position chr1:12345 in gene BRCA1:
    
    Transcript BRCA1-001 (canonical):
    - Position is a DONOR site (exon 5 → intron 5)
    - Prediction: TP (correctly identified donor)
    
    Transcript BRCA1-002 (alternative):  
    - Position is NEITHER (within exon 6)
    - Prediction: TN (correctly identified non-splice site)
    
    Transcript BRCA1-003 (rare isoform):
    - Position is an ACCEPTOR site (intron 4 → exon 5)
    - Prediction: FP (incorrectly called acceptor)
    
    CURRENT SYSTEM (genomic-only grouping):
    - Forces single label per position
    - Result: TP (by priority: TP > TN > FP)
    - LOST INFORMATION: Transcript-specific splice site roles
    - IMPACT: Training data quality, biological interpretation
    
    TRANSCRIPT-AWARE SYSTEM:
    - Preserves all three entries
    - Enables isoform-specific predictions
    - Supports precision medicine applications
    - Better training data for meta-learning
    
    SPLICE-AWARE SYSTEM (compromise):
    - Groups by position + splice_type
    - Result: donor(TP), neither(TN), acceptor(FP)
    - Preserves splice site role diversity
    - Maintains computational efficiency
    
    === Impact on 5000-Gene Meta Model ===
    
    Current 1000-gene model may fail to generalize because:
    1. Training data oversimplifies biological complexity
    2. Same position forced to single label across transcripts
    3. Meta-learning misses isoform-specific patterns
    4. Cannot learn variant effects on alternative splicing
    
    Solution: Train 5000-gene model with transcript-aware position ID
    - Position ID: ['gene_id', 'position', 'strand', 'transcript_id']
    - Prediction target: 'splice_type' (not part of position ID!)
    - Enables meta-learning to capture how variants affect splice site roles
    - Better captures alternative splicing complexity
    - Improves meta model generalization
    - Enables disease-specific adaptation
    """
    
    return example


# Configuration class for easy integration into existing workflows
class TranscriptAwareConfig:
    """Configuration for transcript-aware position identification."""
    
    def __init__(
        self,
        mode='genomic',  # 'genomic', 'transcript', 'hybrid', 'splice_aware'
        preserve_transcript_info=True,
        conflict_resolution='priority',
        enable_complexity_analysis=False,
        backward_compatible=True
    ):
        self.mode = mode
        self.preserve_transcript_info = preserve_transcript_info
        self.conflict_resolution = conflict_resolution
        self.enable_complexity_analysis = enable_complexity_analysis
        self.backward_compatible = backward_compatible
        
        # Validate configuration
        valid_modes = ['genomic', 'transcript', 'hybrid', 'splice_aware', 'splice_only']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Choose from: {valid_modes}")
    
    def get_grouping_columns(self, include_splice_type=False, custom_columns=None):
        """Get grouping columns for this configuration."""
        return get_position_grouping_columns(
            mode=self.mode,
            include_splice_type=include_splice_type,
            custom_columns=custom_columns
        )
    
    def resolve_conflicts(self, df):
        """Apply conflict resolution for this configuration."""
        return resolve_transcript_specific_conflicts(
            df=df,
            mode=self.mode,
            preserve_transcript_info=self.preserve_transcript_info,
            conflict_resolution=self.conflict_resolution
        )
