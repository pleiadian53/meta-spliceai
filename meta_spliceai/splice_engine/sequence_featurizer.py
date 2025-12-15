import pandas as pd
import polars as pl 
import numpy as np
from collections import Counter
from itertools import product

# Pandas-based featurization
# from meta_spliceai.sequence_model.featurize_sequence import (
#     get_kmer_counts, 
#     get_gc_content, 
#     get_sequence_length, 
#     get_marker_counts, 
#     get_transition_counts, 
#     get_sequence_complexity
#     # SequenceMarkers
# )

# from meta_spliceai.sequence_model.data_model import (
#     Sequence, 
#     SequenceMarkers
# )

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display, 
    display_dataframe_in_chunks, 
    display_dataframe
)


# Define the function to compute sequence length
def get_sequence_length(sequence):
    # Handle non-string inputs gracefully
    if not isinstance(sequence, str):
        if pd.isna(sequence) or sequence is None:
            return 0
        try:
            sequence = str(sequence)
        except:
            return 0
    
    # Handle empty strings
    if not sequence or len(sequence) == 0:
        return 0
    
    try:
        return len(sequence)
    except Exception:
        return 0


def get_kmer_counts_v0(sequence, k):
    """Compute k-mer counts for a given sequence."""
    # Handle non-string inputs gracefully
    if not isinstance(sequence, str):
        if pd.isna(sequence) or sequence is None:
            return {}
        try:
            sequence = str(sequence)
        except:
            return {}
    
    # Handle empty strings or sequences too short for k-mers
    if not sequence or len(sequence) < k:
        return {}
    
    try:
        return dict(Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)]))
    except Exception:
        return {}


def get_gc_content(sequence):
    """Compute GC content of a sequence."""
    # Handle non-string inputs gracefully
    if not isinstance(sequence, str):
        if pd.isna(sequence) or sequence is None:
            return 0.0
        try:
            sequence = str(sequence)
        except:
            return 0.0
    
    # Handle empty strings
    if not sequence or len(sequence) == 0:
        return 0.0
    
    try:
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0
    except Exception:
        return 0.0


def get_sequence_complexity(sequence):
    """Compute sequence complexity based on nucleotide frequencies."""
    # Handle non-string inputs gracefully
    if not isinstance(sequence, str):
        if pd.isna(sequence) or sequence is None:
            return 0.0
        try:
            sequence = str(sequence)
        except:
            return 0.0
    
    # Handle empty strings
    if not sequence or len(sequence) == 0:
        return 0.0
    
    try:
        frequency_list = [sequence.count(nucleotide) for nucleotide in set(sequence)]
        return sum(-f/len(sequence) * np.log2(f/len(sequence)) for f in frequency_list if f > 0)
    except Exception:
        return 0.0


def featurize_gene_sequences_v0(df, **kwargs):
    """
    Featurize a DataFrame containing gene sequences and return the derived features.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'gene_id' and 'sequence' columns.
    - kmer_sizes (list): List of k-mer sizes to compute. Default: [2, 3].
    - markers (dict): Marker-to-symbol dictionary for marker counts. Default: None.
    - drop_source_columns (bool): Whether to drop original sequence/marker columns. Default: True.
    - return_feature_set (bool): Whether to return the set of derived feature columns. Default: False.
    - verbose (int): Level of verbosity. Default: 0.

    Returns:
    - pd.DataFrame: DataFrame augmented with derived features.
    - feature_columns (list): List of derived feature column names (optional).
    """
    is_polars = False
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
        is_polars = True

    # Configurable parameters
    kmer_sizes = kwargs.get('kmer_sizes', 6)  # Default: Single k-mer value (6)
    if isinstance(kmer_sizes, int):  
        kmer_sizes = [kmer_sizes]  # Convert single integer to list
    markers = kwargs.get('markers', None)  # Marker symbols (optional)
    drop_source_columns = kwargs.get('drop_source_columns', True)
    return_feature_set = kwargs.get('return_feature_set', False)
    verbose = kwargs.get('verbose', 0)

    # Keep track of original columns
    original_columns = set(df.columns)

    # Verify required columns
    if 'gene_id' not in df.columns or 'sequence' not in df.columns:
        raise ValueError("Input DataFrame must contain at least 'gene_id' and 'sequence' columns.")

    # Feature extraction
    if verbose:
        print("[action] Featurizing gene sequences...")

    # Initialize feature columns
    kmer_features = []
    for k in kmer_sizes:
        feature_name = f'{k}mer_counts'
        df[feature_name] = df['sequence'].apply(lambda x: get_kmer_counts(x, k))
        kmer_features.append(feature_name)

    df['gc_content'] = df['sequence'].apply(get_gc_content)
    df['sequence_length'] = df['sequence'].apply(get_sequence_length)
    df['sequence_complexity'] = df['sequence'].apply(get_sequence_complexity)

    # Expand k-mer counts into separate columns
    for feature_name in kmer_features:
        kmer_df = pd.json_normalize(df[feature_name]).add_prefix(feature_name.split('_')[0] + '_')
        df = pd.concat([df.drop(feature_name, axis=1), kmer_df], axis=1)

    # Replace NaN values with 0
    df = df.fillna(0)

    # Determine derived feature columns
    all_columns = set(df.columns)
    derived_feature_columns = list(all_columns - original_columns)

    # Optional diagnostic: Identify derived features
    if verbose:
        print(f"[diagnostics] Derived feature columns: {derived_feature_columns[:100]}")

    # Optionally drop original sequence and marker columns
    if drop_source_columns:
        df = df.drop(columns=['sequence'] + (['marker'] if 'marker' in df.columns else []))

    if verbose:
        print(f"[featurize] Completed featurization. Shape: {df.shape}")

    if is_polars:
        df = pl.DataFrame(df)

    if return_feature_set:
        return df, derived_feature_columns
    return df


def get_kmer_counts(sequence, k, filter_invalid_kmers):
    """Compute k-mer counts for a given sequence with optional filtering (only for k >= 6)."""
    # Handle None or invalid sequences
    if sequence is None:
        return {}
    
    # Handle non-string inputs gracefully
    if not isinstance(sequence, str):
        if pd.isna(sequence):
            return {}
        try:
            sequence = str(sequence)
        except:
            return {}
    
    # Handle empty strings or sequences too short for k-mers
    if not sequence or len(sequence) < k:
        return {}
    
    try:
        kmer_counts = Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

        # Apply filtering for all k-mer sizes to ensure consistency
        if filter_invalid_kmers:
            return {kmer: count for kmer, count in kmer_counts.items() if is_valid_kmer(kmer)}
        
        return kmer_counts
    except Exception:
        return {}


def is_valid_kmer(kmer):
    """Checks whether a k-mer should be kept based on biological relevance."""
    # Remove k-mers with any N's (ambiguous nucleotides)
    if "N" in kmer:
        return False
    
    # For longer k-mers (6+), apply additional filtering
    if len(kmer) >= 6:
        # Remove repetitive GGGGGG or CCCCCC (but allow AAAAAA and TTTTTT)
        if kmer in {"GGGGGG", "CCCCCC"}:
            return False

        # Remove excessive GC-repeats (e.g., "GCGCGC", "CGCGCG")
        if all(kmer[i] == kmer[i+2] for i in range(len(kmer) - 2)):
            return False  

    return True


def featurize_gene_sequences_v0(df, **kwargs):
    """
    Featurize a DataFrame containing gene sequences, filtering out non-informative motifs.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'gene_id' and 'sequence' columns.
    - kmer_sizes (list or int): List of k-mer sizes to compute. Can also be a single integer (e.g., 6). Default: [6].
    - filter_invalid_kmers (bool): Whether to filter out non-sensible k-mers (e.g., N-rich or repetitive G/C). Default: True.
    - drop_source_columns (bool): Whether to drop original sequence column. Default: True.
    - return_feature_set (bool): Whether to return the set of derived feature columns. Default: False.
    - verbose (int): Level of verbosity.

    Returns:
    - pd.DataFrame: Augmented with derived features.
    - feature_columns (list): List of derived feature column names (optional).
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    # Default settings
    kmer_sizes = kwargs.get('kmer_sizes', 6)  
    if isinstance(kmer_sizes, int):  
        kmer_sizes = [kmer_sizes]  
    filter_invalid_kmers = kwargs.get('filter_invalid_kmers', True)
    drop_source_columns = kwargs.get('drop_source_columns', True)
    return_feature_set = kwargs.get('return_feature_set', False)
    verbose = kwargs.get('verbose', 0)

    # Keep track of original columns
    original_columns = set(df.columns)

    # Verify required columns
    if 'gene_id' not in df.columns or 'sequence' not in df.columns:
        raise ValueError("Input DataFrame must contain 'gene_id' and 'sequence' columns.")

    # Feature extraction
    if verbose:
        print(f"[action] Featurizing gene sequences with k={kmer_sizes}...")

    # Initialize feature columns
    kmer_features = []
    for k in kmer_sizes:
        feature_name = f'{k}mer_counts'
        df[feature_name] = df['sequence'].apply(lambda x: get_kmer_counts(x, k, filter_invalid_kmers))
        kmer_features.append(feature_name)

    # Convert k-mer dictionaries to separate columns
    for feature_name in kmer_features:
        kmer_df = pd.json_normalize(df[feature_name]).add_prefix(feature_name.split('_')[0] + '_')
        df = pd.concat([df.drop(feature_name, axis=1), kmer_df], axis=1)

    # Compute additional sequence properties
    df['gc_content'] = df['sequence'].apply(get_gc_content)
    df['sequence_length'] = df['sequence'].apply(get_sequence_length)
    df['sequence_complexity'] = df['sequence'].apply(get_sequence_complexity)

    # Replace NaN values with 0
    df = df.fillna(0)

    # Determine derived feature columns
    all_columns = set(df.columns)
    derived_feature_columns = list(all_columns - original_columns)

    # Optional diagnostic: Identify derived features
    if verbose:
        print(f"[diagnostics] Derived feature columns: {derived_feature_columns[:100]}")

    # Optionally drop original sequence and marker columns
    if drop_source_columns:
        df = df.drop(columns=['sequence'])

    if verbose:
        print(f"[featurize] Completed featurization with k={kmer_sizes}. Shape: {df.shape}")

    if is_polars:
        df = pl.DataFrame(df)

    if return_feature_set:
        return df, derived_feature_columns
    return df


def featurize_gene_sequences(df, **kwargs):
    """
    Featurize a DataFrame containing gene sequences, optionally pruning rare k-mers.

    Parameters:
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame containing at least 'gene_id' and 'sequence'.
    kmer_sizes : list or int, optional
        List of k-mer sizes or a single integer. Default: [6].
    filter_invalid_kmers : bool, optional
        Whether to filter out non-sensible k-mers (e.g., N-rich) in get_kmer_counts(). Default: True.
    drop_source_columns : bool, optional
        Whether to drop the original 'sequence' column. Default: True.
    return_feature_set : bool, optional
        Whether to return the set of derived feature columns. Default: False.
    min_kmer_count : int, optional
        Only keep k-mers that appear at least this many times across the entire dataset.
        Default 1 means no pruning.
    verbose : int, optional
        Level of verbosity.

    Returns:
    -------
    pd.DataFrame or pl.DataFrame
        DataFrame augmented with derived features (k-mer columns, GC content, etc.).
    list of str, optional
        List of derived feature columns (if return_feature_set=True).

    Updates:
    --------
    - Added new parameter for pruning rare k-mers globally.

    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    # Default settings
    kmer_sizes = kwargs.get('kmer_sizes', 6)
    if isinstance(kmer_sizes, int):
        kmer_sizes = [kmer_sizes]

    filter_invalid_kmers = kwargs.get('filter_invalid_kmers', True)
    drop_source_columns = kwargs.get('drop_source_columns', True)
    return_feature_set = kwargs.get('return_feature_set', False)
    min_kmer_count = kwargs.get('min_kmer_count', 1)  # <--- new parameter for pruning
    verbose = kwargs.get('verbose', 0)

    original_columns = set(df.columns)

    # Verify required columns
    if 'gene_id' not in df.columns or 'sequence' not in df.columns:
        raise ValueError("Input DataFrame must contain 'gene_id' and 'sequence' columns.")

    # Start featurization
    if verbose:
        print(f"[action] Featurizing gene sequences with k={kmer_sizes} (min_kmer_count={min_kmer_count})...")

    # For each k, we create a column with a dictionary of {k-mer: count} for each sequence
    kmer_dict_cols = []
    for k in kmer_sizes:
        dict_col_name = f"{k}mer_counts"
        # step 1: create dictionary col
        df[dict_col_name] = df['sequence'].apply(
            lambda x: get_kmer_counts(x, k, filter_invalid_kmers)
        )
        kmer_dict_cols.append(dict_col_name)

    # If min_kmer_count > 1, we prune rare k-mers globally
    if min_kmer_count > 1:
        for dict_col_name in kmer_dict_cols:
            # Convert to list of dictionaries
            kmer_dicts = df[dict_col_name].tolist()

            # 1) Compute global sums for each k-mer
            global_kmer_counts = {}
            for km_dict in kmer_dicts:
                for km, cnt in km_dict.items():
                    global_kmer_counts[km] = global_kmer_counts.get(km, 0) + cnt

            # 2) Identify which k-mers pass the threshold
            allowed_kmers = {km for km, total_cnt in global_kmer_counts.items()
                             if total_cnt >= min_kmer_count}

            if verbose:
                print(f"[prune] {dict_col_name}: #all_kmers={len(global_kmer_counts)}, "
                      f"#allowed_kmers={len(allowed_kmers)} (prune={len(global_kmer_counts)-len(allowed_kmers)})")

            # 3) For each row's dictionary, remove keys not in allowed_kmers
            def prune_kmer_dict(km_dict):
                return {km: c for km, c in km_dict.items() if km in allowed_kmers}

            df[dict_col_name] = df[dict_col_name].apply(prune_kmer_dict)

    # Now each dict_col_name is pruned. Next, we expand them into columns.
    # Convert the pruned dictionaries into individual columns
    for dict_col_name in kmer_dict_cols:
        # pd.json_normalize => wide format
        # e.g. "6mer_counts" => columns "6mer_AAAAAA", "6mer_CCGTGA", etc.
        kmer_df = pd.json_normalize(df[dict_col_name])  # shape: (num_rows, #unique_kmers)
        # prefix e.g. "6mer_" but we can also do dict_col_name.split('_')[0] + '_'
        # If dict_col_name = '6mer_counts', splitted[0] = '6mer'
        prefix = dict_col_name.split('_')[0] + '_'
        kmer_df = kmer_df.add_prefix(prefix)

        # Drop the dictionary column from df
        df = pd.concat([df.drop(dict_col_name, axis=1), kmer_df], axis=1)

    # Compute additional properties
    df['gc_content'] = df['sequence'].apply(get_gc_content)
    df['sequence_length'] = df['sequence'].apply(get_sequence_length)
    df['sequence_complexity'] = df['sequence'].apply(get_sequence_complexity)

    # Replace NaN with 0
    df = df.fillna(0)

    # Determine derived feature columns
    all_columns = set(df.columns)
    derived_feature_columns = list(all_columns - original_columns)

    # Optionally drop original sequence column
    if drop_source_columns and 'sequence' in df.columns:
        df.drop(columns=['sequence'], inplace=True)

    if verbose:
        print(f"[featurize] Completed featurization with k={kmer_sizes}. "
              f"Shape={df.shape}, min_kmer_count={min_kmer_count}")
        if min_kmer_count > 1:
            print(f"[featurize] Rare k-mers (count < {min_kmer_count}) have been pruned.")

    # Optionally convert back to Polars
    if is_polars:
        df = pl.DataFrame(df)

    if return_feature_set:
        return df, derived_feature_columns
    return df



def harmonize_features(dfs, feature_sets, default_value=0):
    """
    Harmonize the feature sets across multiple dataframes (supports pandas and Polars).

    Parameters:
    - dfs (list): List of dataframes (pandas or Polars) to harmonize.
    - feature_sets (list): List of feature column sets for each dataframe.
    - default_value (int/float): Default value for missing features.

    Returns:
    - harmonized_dfs (list): List of dataframes with consistent feature columns.
    """
    # Determine the union of all features
    all_features = set().union(*feature_sets)

    harmonized_dfs = []
    for df, feature_set in zip(dfs, feature_sets):
        missing_features = all_features - set(feature_set)
        
        # Check if the dataframe is Polars
        if isinstance(df, pl.DataFrame):
            # Add missing features to Polars dataframe
            for feature in missing_features:
                df = df.with_columns(pl.lit(default_value).alias(feature))
        else:
            # Add missing features to pandas dataframe
            for feature in missing_features:
                df[feature] = default_value
        
        harmonized_dfs.append(df)

    return harmonized_dfs


# Featurize gene sequences
def featurize_gene_sequences_polars(sequence_df, kmer_sizes=(2, 3, 4), drop_sequence_column=True, **kwargs):
    """
    Featurize gene sequences in a Polars DataFrame.

    Parameters:
    - sequence_df (pl.DataFrame): Polars DataFrame with 'gene_id' and 'sequence' columns.
    - kmer_sizes (tuple): Tuple of k-mer sizes to compute (default: (2, 3, 4)).
    - drop_sequence_column (bool): If True, drop the 'sequence' column after featurization.
    - **kwargs: Additional options, e.g., verbose output.

    Returns:
    - pl.DataFrame: Polars DataFrame with featurized sequence columns.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    verbose = kwargs.get("verbose", 1)

    # Ensure required columns
    if not {"gene_id", "sequence"}.issubset(sequence_df.columns):
        raise ValueError("Input DataFrame must contain 'gene_id' and 'sequence' columns.")

    # Helper to compute k-mer counts
    def get_kmer_counts(sequence, k):
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
        kmers = vectorizer.fit_transform([sequence])
        kmer_counts = kmers.toarray().sum(axis=0)
        kmer_names = vectorizer.get_feature_names_out()
        return dict(zip(kmer_names, kmer_counts))

    # Step 1: Add k-mer features as dictionaries
    feature_df = sequence_df
    for k in kmer_sizes:
        feature_df = feature_df.with_columns(
            pl.col("sequence").map_elements(
                lambda seq: get_kmer_counts(seq, k), return_dtype=pl.Object
            ).alias(f"{k}mer_counts")
        )

    # Step 2: Expand k-mer dictionaries into separate columns
    for k in kmer_sizes:
        kmer_col = f"{k}mer_counts"
        # Expand dictionaries into columns
        feature_df = feature_df.with_columns(
            pl.col(kmer_col).cast(pl.Object)
        ).unnest(kmer_col, eager=True).fill_null(0)  # eager=True

    # Step 3: Add GC Content and Sequence Length
    def get_gc_content(sequence):
        # Handle non-string inputs gracefully
        if not isinstance(sequence, str):
            if pd.isna(sequence) or sequence is None:
                return 0.0
            try:
                sequence = str(sequence)
            except:
                return 0.0
        
        # Handle empty strings
        if not sequence or len(sequence) == 0:
            return 0.0
        
        try:
            return (sequence.count("G") + sequence.count("C")) / len(sequence) if len(sequence) > 0 else 0
        except Exception:
            return 0.0

    feature_df = feature_df.with_columns([
        pl.col("sequence").map_elements(get_gc_content, return_dtype=pl.Float32).alias("gc_content"),
        pl.col("sequence").map_elements(len, return_dtype=pl.Int32).alias("sequence_length")
    ])

    # Step 4: Drop sequence column if requested
    if drop_sequence_column:
        feature_df = feature_df.drop("sequence")

    # Step 5: Replace nulls with 0
    feature_df = feature_df.fill_null(0)

    if verbose:
        print(f"[info] Featurized gene sequences with k-mer sizes: {kmer_sizes}")
        print(f"[info] Output DataFrame shape: {feature_df.shape}")

    return feature_df


def display_feature_set(df, max_kmers=100, verbose=0): 
    return display_columns_with_limited_kmers(df, max_kmers, verbose)

def display_columns_with_limited_kmers(df, max_kmers=100, verbose=0):
    """
    Display all non-k-mer columns and a subset of k-mer columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - max_kmers (int): Maximum number of k-mer columns to display. Default is 100.
    - verbose (bool): If True, prints the selected columns.

    Returns:
    - List of selected columns.
    """
    # Identify k-mer columns (assumes they follow format '{k}mer_*')
    kmer_columns = [col for col in df.columns if 'mer_' in col]
    non_kmer_columns = [col for col in df.columns if col not in kmer_columns]

    # Select a subset of k-mers
    selected_kmers = kmer_columns[:max_kmers] if len(kmer_columns) > max_kmers else kmer_columns

    # Combine non-k-mer columns with selected k-mers
    final_columns = non_kmer_columns + selected_kmers

    if verbose:
        print(f"Total Columns: {len(df.columns)} | Showing {len(final_columns)} Columns (Non-k-mer: {len(non_kmer_columns)}, k-mers: {len(selected_kmers)}/{len(kmer_columns)})")
        print(final_columns)

    return final_columns


# def generate_marker_seq(**kargs): 
#     return SequenceMarkers.generate_marker_seq(**kargs)


# Pandas-based featurization
def featurize_sequence_dataframe(transcript_df, **kargs): 
    """
    Featurizes a DataFrame containing transcript sequences.

    This function applies various feature extraction methods to the sequences and markers in the input DataFrame.
    It adds new columns to the DataFrame with the extracted features and optionally drops the source columns after featurization.

    Parameters:
    -----------
    transcript_df : pandas.DataFrame
        A DataFrame containing transcript sequences and markers. The DataFrame must contain columns specified in `source_columns`.
    
    **kargs : dict
        Additional keyword arguments:
        - markers : dict, optional
            A dictionary of markers to be used for marker count extraction. Default is `SequenceMarkers.markers`.
        - source_columns : list, optional
            A list of column names in `transcript_df` that contain the sequence and marker data. Default is ['seq', 'marker'].
        - drop_source_columns_after : bool, optional
            Whether to drop the source columns after featurization. Default is True.

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with additional columns for the extracted features. 
        If `drop_source_columns_after` is True, the source columns are dropped.

    Raises:
    -------
    AssertionError
        If any of the columns specified in `source_columns` are not present in `transcript_df`.

    Notes:
    ------
    The following features are extracted and added to the DataFrame:
    - 2-mer counts
    - 3-mer counts
    - GC content
    - Sequence length
    - Sequence complexity
    - Marker counts
    - Transition counts

    The extracted features are unpacked from dictionaries into separate columns, and NaN values are replaced with 0.
    """
    import meta_spliceai.sequence_model.featurize_sequence as fs

    markers = kargs.get("markers", SequenceMarkers.get_markers(active_only=True))  # SequenceMarkers.markers

    # --- Test ---
    source_columns = kargs.get("source_columns", Sequence.source_columns)  # ['sequence', 'marker', ]
    drop_source_columns_after = kargs.get("drop_source_columns_after", True)  # Drop source columns after featurization
    for col in source_columns: 
        assert col in transcript_df.columns

    col_seq = Sequence.col_seq  # 'sequence'
    col_marker = Sequence.col_marker  # 'marker'

    # Apply the feature extraction functions to the DataFrame
    print("[action] Featurizing sequence data: 2-mer counts, 3-mer counts, GC content, sequence complexity, etc.")
    
    # Extract 2-mer counts from the sequences
    transcript_df['2mer_counts'] = transcript_df[col_seq].apply(lambda x: fs.get_kmer_counts(x, 2))
    
    # Extract 3-mer counts from the sequences
    transcript_df['3mer_counts'] = transcript_df[col_seq].apply(lambda x: fs.get_kmer_counts(x, 3))

    # Extract 4-mer counts from the sequences
    transcript_df['4mer_counts'] = transcript_df[col_seq].apply(lambda x: fs.get_kmer_counts(x, 4))

    # Extract GC content from the sequences
    transcript_df['gc_content'] = transcript_df[col_seq].apply(fs.get_gc_content)

    # Compute sequence stats
    transcript_df['sequence_length'] = transcript_df[col_seq].apply(fs.get_sequence_length)
    transcript_df['sequence_complexity'] = transcript_df[col_seq].apply(fs.get_sequence_complexity)

    transcript_df['marker_counts'] = transcript_df[col_marker].apply(lambda x: fs.get_marker_counts(x, markers))
    # NOTE: This leads to features like 'exon', 'intron', ... etc. 
    #       exon count, for instance, is essentially size_exon

    transcript_df['transition_counts'] = transcript_df[col_marker].apply(fs.get_transition_counts)
    
    # Unpack the dictionaries in the k-mer counts, marker counts, and transition counts columns into separate columns
    transcript_df = pd.concat([transcript_df.drop(['2mer_counts', '3mer_counts', 'marker_counts', 'transition_counts'], axis=1), 
                                transcript_df['2mer_counts'].apply(pd.Series), 
                                transcript_df['3mer_counts'].apply(pd.Series), 
                                transcript_df['marker_counts'].apply(pd.Series), 
                                transcript_df['transition_counts'].apply(pd.Series)], axis=1)

    # Replace NaNs with 0
    transcript_df = transcript_df.fillna(0)

    # Show the shape and first few rows of the updated DataFrame
    print(transcript_df.head())
    print(f"[featurize] shape of featurized sequence df: {transcript_df.shape}")

    # Drop source columns after featurization
    if drop_source_columns_after: 
        transcript_df = transcript_df.drop(source_columns, axis=1)

    return transcript_df


def verify_consensus_sequences_via_analysis_data(
    analysis_sequence_df, splice_type, window_radius=2, adjust_position=False, **kwargs
):
    """
    Verify consensus sequences near splice sites using analysis sequence segments centered around splice sites.

    Parameters:
    - analysis_sequence_df (pd.DataFrame): DataFrame with sequences and metadata.
      Required columns: ['gene_id', 'pred_type', 'sequence', 'splice_type', 'strand'].
      The sequence is centered around the splice site.
    - splice_type (str): 'donor' or 'acceptor' to specify the splice site type.
    - window_radius (int): Radius around the splice site (center of the sequence) to check for consensus sequences.
    - adjust_position (bool): Whether to adjust consensus sequence logic for systematic discrepancies.
    - verbose (int): Verbosity level.

    Returns:
    - pd.DataFrame: DataFrame with an additional column 'has_consensus' indicating consensus sequence presence.
    """
    from Bio.Seq import Seq  # For reverse complements if needed

    # Define consensus sequences
    consensus_sequences = {
        "acceptor": {"AG", "CAG", "TAG", "AAG"},
        "donor": {"GT", "GC"}
    }

    # Validate splice type
    if splice_type not in consensus_sequences:
        raise ValueError(f"Invalid splice_type '{splice_type}'. Must be 'donor' or 'acceptor'.")

    verbose = kwargs.get("verbose", 1)
    col_seq = kwargs.get("col_seq", "sequence")
    col_pred_type = kwargs.get("col_pred_type", "pred_type")
    col_strand = kwargs.get("col_strand", "strand")

    # Convert Polars to Pandas if necessary
    is_polars = False
    if isinstance(analysis_sequence_df, pl.DataFrame):
        is_polars = True
        analysis_sequence_df = analysis_sequence_df.to_pandas()

    # Check required columns
    required_columns = {"gene_id", col_pred_type, "sequence", "splice_type", col_strand}
    if not required_columns.issubset(analysis_sequence_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Adjust splice site positions if needed
    def adjust_position_logic(midpoint, strand):
        if not adjust_position:
            return midpoint
        if splice_type == "donor":
            return midpoint + 2 if strand == '+' else midpoint + 1
        elif splice_type == "acceptor":
            return midpoint if strand == '+' else midpoint - 1
        else:
            raise ValueError(f"Invalid splice_type '{splice_type}'.")

    # Function to check for consensus sequences within a sequence
    def check_consensus_in_centered_sequence(sequence, strand):
        """Check for consensus sequences at or near the center of the sequence."""
        if sequence is None or len(sequence) == 0:
            return False

        midpoint = len(sequence) // 2  # Center of the sequence
        adjusted_midpoint = adjust_position_logic(midpoint, strand)
        start_idx = max(0, adjusted_midpoint - window_radius)
        end_idx = min(len(sequence), adjusted_midpoint + window_radius + 1)

        neighborhood = sequence[start_idx:end_idx]
        return any(consensus in neighborhood for consensus in consensus_sequences[splice_type])

    # Apply consensus check for each row
    analysis_sequence_df["has_consensus"] = analysis_sequence_df.apply(
        lambda row: check_consensus_in_centered_sequence(row[col_seq], row[col_strand]), axis=1
    )

    # Summary DataFrame
    summary = (
        analysis_sequence_df.groupby(col_pred_type)["has_consensus"]
        .agg(["sum", "count"])
        .reset_index()
    )
    summary["ratio"] = summary["sum"] / summary["count"]
    if verbose:
        print("[Summary] Consensus Sequence Analysis:")
        print(summary)
        # NOTE: "sum" indicates the total number of rows where `has_consensus` is True.
        #       "count" reflects the total number of rows for each `pred_type`.

    if is_polars:
        analysis_sequence_df = pl.DataFrame(analysis_sequence_df)

    return analysis_sequence_df, summary


def verify_consensus_sequences(
    analysis_sequence_df, splice_type, window_radius=2, adjust_position=False, **kwargs
):
    """
    Verify consensus sequences near splice site positions, adjusting for strand-dependent relative positions.

    Parameters:
    - analysis_sequence_df (pd.DataFrame): DataFrame with sequences and splice site metadata.
      Required columns: ['gene_id', 'pred_type', 'position', 'sequence', 'strand', 'gene_start', 'gene_end'].
    - splice_type (str): 'donor' or 'acceptor' to specify the splice site type.
    - window_radius (int): Radius around the splice site position to check for consensus sequences.
    - adjust_position (bool): Whether to adjust splice site positions for systematic discrepancies.
    - verbose (int): Verbosity level.

    Returns:
    - pd.DataFrame: DataFrame with an additional column 'has_consensus' indicating consensus sequence presence.
    """
    from Bio.Seq import Seq  # To verify reverse complements if needed

    # Define consensus sequences
    consensus_sequences = {
        "acceptor": {"AG", "CAG", "TAG", "AAG"},
        "donor": {"GT", "GC"}
    }

    # Configuration
    verbose = kwargs.get("verbose", 1)
    col_seq = kwargs.get("col_seq", "sequence")
    col_pred_type = kwargs.get("col_pred_type", "pred_type")
    col_strand = kwargs.get("col_strand", "strand")
    col_gene_start = kwargs.get("col_gene_start", "gene_start")
    col_gene_end = kwargs.get("col_gene_end", "gene_end")

    # Convert Polars DataFrame to Pandas if needed
    if isinstance(analysis_sequence_df, pl.DataFrame):
        analysis_sequence_df = analysis_sequence_df.to_pandas()

    # Input validation
    if splice_type not in consensus_sequences:
        raise ValueError(f"Invalid splice_type '{splice_type}'. Must be 'donor' or 'acceptor'.")

    required_columns = {"gene_id", col_pred_type, "position", col_seq, col_strand, col_gene_start, col_gene_end}
    if not required_columns.issubset(analysis_sequence_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Adjust splice site positions to absolute coordinates based on strand
    def compute_absolute_position(row):
        """Convert relative positions to absolute genomic coordinates."""
        if row[col_strand] == '+':
            return row[col_gene_start] + row["position"]
        elif row[col_strand] == '-':
            return row[col_gene_end] - row["position"]
        else:
            raise ValueError(f"Invalid strand value: {row[col_strand]}.")

    analysis_sequence_df["absolute_position"] = analysis_sequence_df.apply(compute_absolute_position, axis=1)

    # Adjust splice site positions for systematic discrepancies
    def adjust_position_logic(position, strand):
        if not adjust_position:
            return position
        if splice_type == "donor":
            return position + 2 if strand == '+' else position + 1
        elif splice_type == "acceptor":
            return position if strand == '+' else position - 1
        else:
            raise ValueError(f"Invalid splice_type '{splice_type}'.")

    # Function to check for consensus sequences near a position
    def check_consensus(sequence, absolute_position, strand):
        """Search for consensus sequences within a neighborhood of ±window_radius."""
        if absolute_position is None or absolute_position < 0 or absolute_position >= len(sequence):
            return False

        # Adjust position based on strand-specific discrepancy
        adjusted_position = adjust_position_logic(absolute_position, strand)

        start_idx = max(0, adjusted_position - window_radius)
        end_idx = min(len(sequence), adjusted_position + window_radius + 1)

        neighborhood = sequence[start_idx:end_idx]
        return any(consensus in neighborhood for consensus in consensus_sequences[splice_type])

    # Apply consensus checking
    analysis_sequence_df["has_consensus"] = analysis_sequence_df.apply(
        lambda row: check_consensus(row[col_seq], row["absolute_position"], row[col_strand]), axis=1
    )

    # Verbose Summary
    if verbose:
        summary = analysis_sequence_df.groupby(col_pred_type)["has_consensus"].agg(["sum", "count"])
        print("[Summary] Consensus Sequence Analysis:")
        print(summary)

    return analysis_sequence_df


def compute_local_splice_index(window_start, window_end, position, strand):
    """
    Return the 0-based index of the predicted splice site within the extracted
    5'→3' oriented substring.

    window_start, window_end: the slice in the original reference,
                             (extracted_seq length = window_end - window_start).
    position: the gene-based splice site coordinate (in the gene's local orientation).
    strand: '+' or '-'

    Returns an integer local index in [0..(window_end-window_start-1)].
    """
    seq_len = window_end - window_start
    if strand == '+':
        # local left is window_start
        return position - window_start
    else:  # strand == '-'
        # local left is originally (window_end - 1) in the reference
        return (window_end - 1) - position


def extract_analysis_sequences(sequence_df, position_df, window_size=500, include_empty_entries=True, **kargs):
    """
    Extract sequences from genes based on positions for analysis.

    Parameters:
    - sequence_df (pl.DataFrame): Polars DataFrame with columns ['gene_id', 'sequence', 'strand'].
    - position_df (pl.DataFrame): Polars DataFrame with columns ['gene_id', 'position', ...].
                                  All other columns in position_df will be retained in the output.
    - window_size (int): Size of the flanking sequence window around the position.
    - include_empty_entries (bool): If True, include entries for missing or invalid gene IDs/windows.
    - verbose (int): Verbosity level for progress messages.

    Returns:
    - pl.DataFrame: Polars DataFrame with extracted sequences and metadata.

      Columns: ['gene_id', 'transcript_id', 'position', 'pred_type', 'score', 'strand', 
                'splice_type', 'chrom', 'window_start', 'window_end', 'sequence']
    """
    from Bio.Seq import Seq

    col_tid = kargs.get("col_tid", "transcript_id")
    col_gid = kargs.get("col_gid", "gene_id")
    col_pos = kargs.get("col_pos", "position")
    verbose = kargs.get("verbose", 1)

    # Fast lookups for gene sequences and strand
    gene_seq_lookup = dict(zip(sequence_df[col_gid], sequence_df["sequence"]))
    strand_lookup = dict(zip(sequence_df[col_gid], sequence_df["strand"]))

    if verbose:
        print(f"[info] Number of genes in sequence_df: {sequence_df.shape[0]}")
        print(f"[info] Number of positions in position_df: {position_df.shape[0]}")

    # Check required columns
    if "gene_id" not in position_df.columns or "position" not in position_df.columns:
        raise ValueError("position_df must contain at least 'gene_id' and 'position' columns.")

    # List to collect extracted sequences
    extracted_sequences = []

    # Iterate over position_df rows
    for row in position_df.iter_rows(named=True):
        gene_id = row[col_gid]
        position = row[col_pos]

        # Check if gene exists in sequence_df
        if gene_id not in gene_seq_lookup:
            if include_empty_entries:
                extracted_sequences.append({**row, 'window_start': None, 'window_end': None, 'sequence': None})
            continue

        sequence = gene_seq_lookup[gene_id]
        strand = strand_lookup[gene_id]

        # Calculate window boundaries, ensuring they don't exceed sequence limits
        window_start = max(position - window_size, 0)
        window_end = min(position + window_size, len(sequence))

        # Extract the sequence segment
        extracted_seq = sequence[window_start:window_end]

        # Reverse complement if on negative strand
        # if strand == '-':
        #     extracted_seq = str(Seq(extracted_seq).reverse_complement())
        # NOTE: No reverse complementing needed, since 'sequence' is pre-oriented !!!

        # Append results: retain all original columns, plus extracted sequence and window info
        result_row = row.copy()
        result_row.update({
            'window_start': window_start,
            'window_end': window_end,
            'sequence': extracted_seq
        })
        extracted_sequences.append(result_row)

    # Convert to Polars DataFrame
    output_df = pl.DataFrame(extracted_sequences)

    # Reorder columns
    columns = output_df.columns
    first_columns = [col_gid, col_tid]
    last_columns = ['sequence']
    middle_columns = [col for col in columns if col not in first_columns + last_columns]
    ordered_columns = first_columns + middle_columns + last_columns
    output_df = output_df.select(ordered_columns)

    if verbose:
        print(f"[info] Extracted sequences for {len(output_df)} positions.")

    return output_df


def extract_analysis_sequences_with_probabilities(
    sequence_df: pl.DataFrame,
    position_df: pl.DataFrame,
    predictions_df: pl.DataFrame,
    window_size: int = 500,
    include_empty_entries: bool = True,
    **kargs
) -> pl.DataFrame:
    raise NotImplementedError("This function is implemented in seq_model_explainer")


def extract_error_sequences_v0(sequence_df, error_df, verbose=1):
    """
    Extract sequences from genes based on error window positions for FP/FN analysis.

    Parameters:
    - sequence_df (pd.DataFrame): DataFrame with columns ['gene_id', 'sequence', 'strand'].
    - error_df (pd.DataFrame): DataFrame with columns ['gene_id', 'error_type', 'window_start', 'window_end'].
    - verbose (int): If > 0, print progress messages.

    Returns:
    - pd.DataFrame: DataFrame with extracted sequences and metadata for training.
    """
    from Bio.Seq import Seq
    
    extracted_sequences = []

    # Convert sequence dataframe to a dictionary for fast lookup
    gene_sequences = sequence_df.set_index('gene_id').to_dict('index')

    if verbose:
        print(f"Number of genes in sequence_df: {len(sequence_df)}")
        print(f"Number of error entries in error_df: {len(error_df)}")

    for _, row in error_df.iterrows():
        gene_id = row['gene_id']
        error_type = row['error_type']
        window_start = row['window_start']
        window_end = row['window_end']

        if gene_id not in gene_sequences:
            if verbose:
                print(f"Gene ID {gene_id} not found in sequence_df. Skipping...")
            continue

        # Retrieve the DNA sequence and strand information
        seq_info = gene_sequences[gene_id]
        sequence = seq_info['sequence']
        strand = seq_info['strand']

        # Extract the subsequence within the error window
        extracted_seq = sequence[window_start:window_end]

        # Reverse-complement the sequence if on the negative strand
        # if strand == '-':
        #     extracted_seq = str(Seq(extracted_seq).reverse_complement())

        # Append to results
        extracted_sequences.append({
            'gene_id': gene_id,
            'error_type': error_type,
            'window_start': window_start,
            'window_end': window_end,
            'sequence': extracted_seq
        })

        if verbose > 1:
            print(f"Extracted sequence for gene {gene_id} ({error_type}): {extracted_seq[:30]}...")

    return pd.DataFrame(extracted_sequences)


def extract_error_sequences(sequence_df, error_df, include_empty_entries=True, **kargs):
    """
    Extract sequences from genes based on error window positions for FP/FN analysis.

    Parameters:
    - sequence_df (pl.DataFrame): Polars DataFrame with columns ['gene_id', 'sequence', 'strand'].
    - error_df (pl.DataFrame): Polars DataFrame with columns ['gene_id', 'error_type', 'window_start', 'window_end'].
    - include_empty_entries (bool): If True, include genes without valid error windows in the output.
    - verbose (int): If > 0, print progress messages.

    Returns:
    - pl.DataFrame: Polars DataFrame with extracted sequences and metadata for training.
    """
    from Bio.Seq import Seq
    
    verbose = kargs.get("verbose", 1)

    extracted_sequences = []

    # Create a dictionary for fast lookup of sequences and strand info
    gene_sequences = sequence_df.to_dict(as_series=False)
    # NOTE: When sequence_df.to_dict(as_series=False) is called, 
    #       gene_sequences becomes a dictionary with keys corresponding to the column names 
    #       in the dataframe (gene_id, sequence, strand) and their values are lists of data for each column.

    if verbose:
        print(f"Number of genes in sequence_df: {sequence_df.shape[0]}")
        print(f"Number of error entries in error_df: {error_df.shape[0]}")

    num_genes = error_df['gene_id'].n_unique()
    genes_processed = set()
    n_genes_with_defined_windows = 0

    required_columns = {"gene_id", "error_type", "position", "window_start", "window_end"}
    if not required_columns.issubset(error_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns} but got {list(error_df.columns)}")

    for row in error_df.iter_rows(named=True):
        gene_id = row['gene_id']
        transcript_id = row.get('transcript_id', None)
        error_type = row['error_type']
        position = row['position']
        window_start = row['window_start']
        window_end = row['window_end']

        # Check if the gene ID exists in the sequence data
        if gene_id not in gene_sequences['gene_id']:
            if verbose:
                print(f"Gene ID {gene_id} not found in sequence_df. Skipping...")
            continue

        # Handle cases where window_start or window_end is None
        if window_start is None or window_end is None:
            if include_empty_entries:
                extracted_sequences.append({
                    'gene_id': gene_id,
                    'transcript_id': transcript_id,
                    'error_type': None,
                    'position': None,
                    'window_start': None,
                    'window_end': None,
                    'sequence': None
                })
            continue

        # Retrieve the DNA sequence and strand information
        idx = gene_sequences['gene_id'].index(gene_id)
        sequence = gene_sequences['sequence'][idx]
        strand = gene_sequences['strand'][idx]

        # Extract the subsequence within the error window
        extracted_seq = sequence[window_start:window_end]

        # Reverse-complement the sequence if it's on the negative strand
        # if strand == '-':
        #     extracted_seq = str(Seq(extracted_seq).reverse_complement())
        # NOTE: No reverse complementing needed, since 'sequence' is pre-oriented.

        # Append the result to the list
        extracted_sequences.append({
            'gene_id': gene_id,
            'transcript_id': transcript_id,
            'error_type': error_type,
            'position': position,
            'window_start': window_start,
            'window_end': window_end,
            'sequence': extracted_seq
        })

        # Update the count of genes with defined windows
        if gene_id not in genes_processed:
            n_genes_with_defined_windows +=1 
            genes_processed.add(gene_id)  # same gene can have multiple errors

        # assert len(extracted_seq) == window_end-window_start, \
        #     f"len(extracted_seq)={len(extracted_seq)} <> {window_end-window_start}"

        if verbose > 1:
            print(f"Extracted sequence for gene {gene_id} ({error_type}): {extracted_seq[:30]}...")

    if verbose:
        print(f"[info] Number of genes with defined windows: {n_genes_with_defined_windows} / {num_genes}")

    # Convert the list of results back to a Polars DataFrame
    return pl.DataFrame(extracted_sequences)


################################################################################
# Featur Engineering Utilities
################################################################################
# from collections import Counter
# from itertools import product

# Function to get all possible k-mers for DNA
def get_all_kmers(k):
    return [''.join(p) for p in product('ATGC', repeat=k)]

# Function to extract sliding window k-mer frequencies from a DNA sequence
def extract_sliding_window_kmer_features(seq, window_size=10, k=3):
    """
    Parameters:
        seq: DNA sequence (str)
        window_size: Size of sliding window (int)
        k: length of k-mers
    Returns:
        features: pd.Series, indexed by 'windowX_kmerY'
    """
    # from itertools import product
    seq_len = len(seq)
    num_windows = seq_len // window_size
    kmers = get_all_kmers(k)

    features = {}
    for w in range(num_windows):
        start = w * window_size
        end = start + window_size
        window_seq = seq[start:end]  # assign window_seq first!

        # Corrected line: separated assignment and iteration clearly
        kmer_counts = Counter([window_seq[i:i+k] 
                               for i in range(len(window_seq) - k + 1)])

        for kmer in kmers:
            feature_name = f'window{w+1}_{kmer}'
            features[feature_name] = kmer_counts.get(kmer, 0)

    return pd.Series(features)



def demo_extract_analysis_sequences():

    # Example sequence DataFrame
    sequence_df = pl.DataFrame({
        'gene_id': ['ENSG000001', 'ENSG000002'],
        'sequence': ['ATGCGTACGTAGCTAGCTAGCGTAGC', 'CGTACGTAGCTAGCTAGCTAGCTAGC'],
        'strand': ['+', '-']
    })

    # Example position DataFrame
    position_df = pl.DataFrame({
        'gene_id': ['ENSG000001', 'ENSG000002'],
        'position': [10, 15],
        'splice_type': ['donor', 'acceptor'],
        'pred_type': ['TP', 'FP'],
        'score': [0.95, 0.85]
    })

    # Extract flanking sequences with a window size of 5
    output_sequences = extract_analysis_sequences(sequence_df, position_df, window_size=5, verbose=1)
    print(output_sequences)


def demo_featurize_gene_sequences(): 
    # from meta_spliceai.sequence_model.data_model import Sequence
    # from meta_spliceai.sequence_model.data_model import SequenceMarkers

    # Sample input
    data = {
        "gene_id": ["gene1", "gene2"],
        "sequence": ["ATCGATCG", "GGCCAA"]
    }
    df = pl.DataFrame(data)

    # Featurize sequences
    # featurized_df = featurize_gene_sequences_polars(df, kmer_sizes=(2, 3, 4), verbose=1) # Todo
    featurized_df, feature_set = \
        featurize_gene_sequences(df, kmer_sizes=[2, 3, 4], return_feature_set=True, verbose=1)

    # Display results
    display_dataframe_in_chunks(featurized_df)


def demo_verify_consensus_sequences():
    import polars as pl

    # Realistic input data where splice sites align with biological motifs
    data = {
        "gene_id": ["gene1", "gene1", "gene2", "gene2", "gene3", "gene4"],
        "pred_type": ["TP", "FP", "FN", "TP", "FP", "TP"],
        "position": [5, 8, 10, 7, 8, 4],  # Position of the splice site
        "sequence": [
            "TTCAGGTGCTAG",  # Position 5 -> Donor: 'GT' at position 5
            "AGGGTAGCCTAG",  # Position 8 -> Acceptor: 'AG' at position 8
            "AAGTAGGCTCAG",  # Position 10 -> Acceptor: 'CAG' at position 10
            "GCTGGTGTAGCT",  # Position 7 -> Donor: 'GT' at position 7
            "CTGGTAGCCTAG",  # Position 8 -> Acceptor: 'AG' at position 8
            "AGGTGTAGCCTA"   # Position 4 -> Donor: 'GT' at position 4
        ],
        "splice_type": ["donor", "acceptor", "acceptor", "donor", "acceptor", "donor"]
    }

    # Convert to Polars DataFrame
    df = pl.DataFrame(data)

    # Verify consensus sequences for acceptor sites
    print("### Acceptor Sites ###")
    acceptor_result = verify_consensus_sequences(df, splice_type="acceptor", verbose=1)
    print(acceptor_result)

    # Verify consensus sequences for donor sites
    print("\n### Donor Sites ###")
    donor_result = verify_consensus_sequences(df, splice_type="donor", verbose=1)
    print(donor_result)


def demo_extract_sliding_window_kmer_features():
    # Example usage
    seq = "ATGCGTACGTAGCTAGCTAGCGTATGCGATCGTAGCTATCGTAGCTAGCTAGCTAGCTAGCTGATGC"
    window_size = 10
    k = 3
    features = extract_sliding_window_kmer_features(seq, window_size=window_size, k=k)
    print(features.head(10))



def test(): 

    # demo_featurize_gene_sequences()
    # demo_extract_analysis_sequences()

    # demo_verify_consensus_sequences()

    # Extract sliding window k-mer features
    demo_extract_sliding_window_kmer_features()


if __name__ == "__main__": 
    test()