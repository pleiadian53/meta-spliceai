import polars as pl
import pandas as pd
import psutil

# Define ANSI escape codes for colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


def align_and_append(df1, df2, **kargs):
    """
    Align two Polars DataFrames by adding missing columns with None values and append df2 to df1.

    Parameters:
    - df1 (pl.DataFrame): The first DataFrame.
    - df2 (pl.DataFrame): The second DataFrame to append to the first DataFrame.
    - **kargs: Optional arguments, such as descriptions for df1 and df2:
      - 'descriptions': Dictionary with keys 'df1' and 'df2' providing descriptions.

    Returns:
    - pl.DataFrame: The combined DataFrame with aligned columns.
    """
    # Get optional descriptions
    descriptions = kargs.get('descriptions', {'df1': 'first DataFrame', 'df2': 'second DataFrame'})

    # Handle empty DataFrame cases
    if df1.is_empty() and df2.is_empty():
        print("[warning] Both df1 and df2 are empty. Returning an empty DataFrame.")
        return pl.DataFrame()

    if df1.is_empty():
        print(f"[info] {BLUE}{descriptions.get('df1', 'df1')}{RESET} is empty. Initializing schema from {BLUE}{descriptions.get('df2', 'df2')}{RESET}")
        # NOTE: The {RESET} is used to reset the text color back to the default terminal color after applying the ANSI escape codes for color formatting.
        
        # Initialize df1 with the schema from df2
        df1 = pl.DataFrame(schema=df2.schema)

    # Compute all columns and sort them upfront
    all_columns = sorted(set(df1.columns).union(set(df2.columns)))

    # Add missing columns with an appropriate default type
    for col in all_columns:
        if col not in df1.columns:
            # Infer type from df2 if possible, else default to Utf8
            dtype = df2.schema.get(col, pl.Utf8)
            df1 = df1.with_columns(pl.lit(None, dtype=dtype).alias(col))
            print(
                f"[warning] Column '{col}' is missing in {descriptions.get('df1', 'df1')}. "
                f"Filling with None (type: {dtype})."
            )
        if col not in df2.columns:
            # Infer type from df1 if possible, else default to Utf8
            dtype = df1.schema.get(col, pl.Utf8)
            df2 = df2.with_columns(pl.lit(None, dtype=dtype).alias(col))
            print(
                f"[warning] Column '{col}' is missing in {descriptions.get('df2', 'df2')}. "
                f"Filling with None (type: {dtype})."
            )

    # Align column order using the pre-sorted column list
    df1 = df1.select(all_columns)
    df2 = df2.select(all_columns)

    # Ensure consistent data types across both DataFrames
    for col in all_columns:
        dtype1 = df1.schema.get(col)
        dtype2 = df2.schema.get(col)
        if dtype1 != dtype2:
            common_dtype = pl.Float64 if dtype1 == pl.Float64 or dtype2 == pl.Float64 else pl.Utf8
            df1 = df1.with_columns(pl.col(col).cast(common_dtype))
            df2 = df2.with_columns(pl.col(col).cast(common_dtype))

    # Append df2 to df1
    combined_df = df1.vstack(df2)

    return combined_df


def long_to_gene_dict(
    df, 
    gene_col="gene_id", 
    sort_col="position", 
    prefer_pandas=True
):
    """
    Convert a long "tidy" DataFrame of splice site probabilities 
    into a dictionary of per-gene DataFrames.

    Each sub-DataFrame contains only the rows for one gene, 
    preserving columns like [position, donor_prob, acceptor_prob, ...].

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        The tall/long DataFrame with at least:
            - gene_id (or gene_col) identifying the gene
            - position
            - donor_prob, acceptor_prob, neither_prob, ...
    gene_col : str
        Column name identifying genes (default: "gene_id").
    sort_col : str
        Which column to sort by within each gene’s sub-DataFrame (default: "position").
    prefer_pandas : bool
        If True, each sub-DataFrame is returned as a pandas DataFrame; 
        otherwise, as a Polars DataFrame.

    Returns
    -------
    dict
        { gene_identifier -> sub-DataFrame },
        e.g.  { "ENSG00000104435" -> pd.DataFrame, ... }
    """
    # 1) Ensure we have a Polars DataFrame to use Polars grouping/filter
    if isinstance(df, pd.DataFrame):
        df = pl.DataFrame(df)

    # 2) Check that gene_col exists
    if gene_col not in df.columns:
        raise ValueError(f"[error] Column '{gene_col}' not found in df columns: {df.columns}")

    # 3) Collect unique gene identifiers
    unique_genes = df.select(gene_col).unique().to_series().to_list()

    gene_dict = {}
    for gene_val in unique_genes:
        # Filter rows for this gene
        sub = df.filter(pl.col(gene_col) == gene_val)
        # Sort rows
        if sort_col in sub.columns:
            sub = sub.sort(sort_col)

        # Convert to pandas if requested
        if prefer_pandas:
            sub_df = sub.to_pandas()
        else:
            sub_df = sub

        gene_dict[gene_val] = sub_df

    return gene_dict


def gene_long_to_single_row(
    df: pl.DataFrame,
    gene_id_col: str = "gene_id",
    sort_col: str = "position",
    extra_cols: list = ("seqname","strand","gene_name"),
    prob_cols: list = ("donor_prob","acceptor_prob","neither_prob"),
    position_col: str = "position"
) -> pl.DataFrame:
    """
    Collapse a tall/long DataFrame (one row per nucleotide in each gene)
    into a "wide" one-row-per-gene format using Polars list columns.

    Example:
      Input columns: [gene_id, gene_name, position, donor_prob, acceptor_prob, strand, ...]
      Output columns: 
         gene_id, gene_name, strand,  (and so on) 
         positions (list), donor_prob_list (list), acceptor_prob_list (list), ...

    Parameters
    ----------
    df : pl.DataFrame
        Tall data with at least:
          - gene_id_col (e.g. "gene_id")
          - position_col (e.g. "position")
          - probability columns (e.g. "donor_prob","acceptor_prob", "neither_prob").
    gene_id_col : str
        Column indicating the gene identifier (default: "gene_id").
    sort_col : str
        Column used to sort the nucleotides within each gene (default: "position").
    extra_cols : list
        Additional per-gene metadata columns to carry over (taken from the first row for each gene).
    prob_cols : list
        Names of probability columns to be turned into list columns.
    position_col : str
        Name of the column that indicates the per-base position.

    Returns
    -------
    pl.DataFrame
        A DataFrame with one row per gene.  The row has:
          - The gene_id_col
          - Each of the extra_cols (one value for the entire gene)
          - A list of positions (sorted by `sort_col`)
          - A list of probabilities for each prob_col, matching the position list.

    Memo
    ----
    This function is useful for converting a "long" DataFrame with one row per nucleotide 
    into a "wide" DataFrame with one row per gene and lists of positions and probabilities for each gene. 
    This is a common format for storing per-gene probabilities for splice sites.
    
    Pros:
        - Great if you truly want one object per gene and you accept each gene’s probabilities as an array.
        - Polars can handle list columns well; you can do group or window calculations with .explode("donor_prob_list") if needed.
    Cons:
        - Many standard machine-learning or statistical tools expect a tall DataFrame format where each row is one data point (or at least something numeric in each column).
        - If a gene is extremely large (e.g. 1+ Mb), you store an extremely large array in a single cell, which can be memory-intensive.
    """

    # Check that columns exist
    required_cols = [gene_id_col, position_col] + list(prob_cols)
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # We'll define an aggregation for each column we want in the final wide table.
    # We gather metadata columns using `.first()` (so we take a single value per gene),
    # and we gather the "per-nucleotide" columns using `.list().sort_by()`.

    # Start building the groupby aggregation instructions:
    agg_instructions = []

    # Gene ID remains as one column, so we can select it directly
    # We'll rename it just in case, but typically we'd keep the same name
    agg_instructions.append(pl.col(gene_id_col).first().alias(gene_id_col))

    # For other single-value metadata columns
    for c in extra_cols:
        if c in df.columns:
            agg_instructions.append(pl.col(c).first().alias(c))

    # For the position column (we want a list of positions)
    if position_col in df.columns:
        agg_instructions.append(
            pl.col(position_col)
            .sort_by(sort_col)         # Ensure the list is in ascending order
            .alias("positions")        # we can rename if we like
        )

    # For the probability columns, also sorted by position
    for pc in prob_cols:
        if pc in df.columns:
            # e.g. "donor_prob" -> "donor_prob_list"
            new_col_name = pc + "_list"
            agg_instructions.append(
                pl.col(pc)
                .sort_by(sort_col)
                .alias(new_col_name)
            )

    # Now we group by gene_id_col and apply these aggregates
    df_wide = (
        df.groupby(gene_id_col, maintain_order=True)
          .agg(agg_instructions)
    )

    return df_wide


def check_available_memory():
    """
    Check the available memory and print it out.
    
    Returns:
    - available_memory (float): Available memory in GB.
    """
    memory_info = psutil.virtual_memory()
    available_memory = memory_info.available / (1024 ** 3)  # Convert bytes to GB
    total_memory = memory_info.total / (1024 ** 3)  # Convert total memory to GB
    print(f"[memory] Available memory: {available_memory:.2f} GB out of {total_memory:.2f} GB")
    return available_memory


def adjust_chunk_size(chunk_size, seq_len_avg):
    """
    Adjust the chunk size based on available memory and sequence length.
    
    Parameters:
    - chunk_size (int): Initial chunk size.
    - seq_len_avg (int): Average length of sequences in the dataset.

    Returns:
    - adjusted_chunk_size (int): Adjusted chunk size to fit into available memory.
    """
    available_memory = check_available_memory()

    # Rough estimate of memory needed per sequence, assuming each sequence is about 4 bytes per nucleotide.
    memory_per_sequence = seq_len_avg * 4 / (1024 ** 3)  # Memory per sequence in GB

    # Estimate how many sequences can fit into half of the available memory
    target_memory_usage = available_memory / 2
    estimated_chunk_size = int(target_memory_usage / memory_per_sequence)

    adjusted_chunk_size = min(chunk_size, estimated_chunk_size)
    print(f"[adjust_chunk_size] Adjusted chunk size: {adjusted_chunk_size}, based on available memory.")
    
    return adjusted_chunk_size


def demo_align_and_append():
    examples = [
        # Create two DataFrames with different columns
        (
            pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            pl.DataFrame({"b": [7, 8, 9], "c": [10, 11, 12]})
        ),

        (
            pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}),
            pl.DataFrame({"b": [7.0, 8.0, 9.0], "c": ["x", "y", "z"]})
        )
    ]

    # Iterate through the examples
    for i, (df1, df2) in enumerate(examples):
        print(f"Example {i + 1}:")
        combined_df = align_and_append(df1, df2)
        print(combined_df)


    # Align and append the DataFrames
    combined_df = align_and_append(df1, df2)

    print(combined_df)


def test_compare_sequences(): 
    import polars as pl

    # Define file paths
    path_als = '/path/to/meta-spliceai/data/ensembl/spliceai_analysis/ALS/gene_sequence.tsv'
    path_seq = '/path/to/meta-spliceai/data/ensembl/gene_sequence_19.parquet'

    # Read data into DataFrames
    seq_df_als = pl.read_csv(path_als, separator='\t')
    seq_df = pl.read_parquet(path_seq)

    # Define the gene ID to compare
    gene_id_to_compare = 'ENSG00000130477'

    # Retrieve the sequence for the given gene ID from both DataFrames
    sequence_als = seq_df_als.filter(pl.col('gene_id') == gene_id_to_compare)['sequence'].to_list()
    sequence_seq = seq_df.filter(pl.col('gene_id') == gene_id_to_compare)['sequence'].to_list()

    # Ensure that we have exactly one sequence for the given gene ID in each DataFrame
    if len(sequence_als) != 1 or len(sequence_seq) != 1:
        print(f"[error] Expected exactly one sequence for gene {gene_id_to_compare} in each DataFrame.")
    else:
        sequence_als = sequence_als[0]
        sequence_seq = sequence_seq[0]

        # Compare the sequences
        if sequence_als == sequence_seq:
            print(f"[info] The sequences for gene {gene_id_to_compare} are identical.")
        elif sequence_als == sequence_seq[::-1]:
            print(f"[info] The sequences for gene {gene_id_to_compare} are reverse complements of each other.")
        else:
            print(f"[info] The sequences for gene {gene_id_to_compare} are different.")


def demo(): 
    # demo_align_and_append()

    # Misc tests
    test_compare_sequences()


if __name__ == "__main__":
    demo()