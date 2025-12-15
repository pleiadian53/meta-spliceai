import polars as pl
import pandas as pd


def pandas_to_polars(pandas_df):
    """
    Convert a Pandas DataFrame to a Polars DataFrame.
    
    Parameters:
    - pandas_df (pd.DataFrame): The Pandas DataFrame to convert.
    
    Returns:
    - polars_df (pl.DataFrame): The converted Polars DataFrame.
    """
    return pl.from_pandas(pandas_df)

def polars_to_pandas(polars_df):
    """
    Convert a Polars DataFrame to a Pandas DataFrame.
    
    Parameters:
    - polars_df (pl.DataFrame): The Polars DataFrame to convert.
    
    Returns:
    - pandas_df (pd.DataFrame): The converted Pandas DataFrame.
    """
    return polars_df.to_pandas()


def truncate_sequences(df, max_length=200):
    """
    Truncate DNA sequences in a DataFrame for display purposes.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): DataFrame containing gene names and sequences with columns 'gene_name' and 'sequence'.
    - max_length (int): Maximum length of the sequence to display. Default is 200.

    Returns:
    - pd.DataFrame or pl.DataFrame: DataFrame with truncated sequences.

    Example usage:
        sequences_df = pd.DataFrame({
            'gene_name': ['gene1', 'gene2'],
            'sequence': [
                'ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC',
                'CGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA']
        })
        truncated_df = truncate_sequences(sequences_df, max_length=10)
        print(truncated_df)
    """
    if isinstance(df, pd.DataFrame):
        truncated_df = df.copy()
        truncated_df['sequence'] = truncated_df['sequence'].apply(
            lambda seq: seq[:max_length] + '...' if len(seq) > max_length else seq
        )
        return truncated_df
    elif isinstance(df, pl.DataFrame):
        truncated_df = df.clone()
        truncated_df = truncated_df.with_columns(
            pl.when(pl.col('sequence').str.len() > max_length)
            .then(pl.col('sequence').str.slice(0, max_length) + '...')
            .otherwise(pl.col('sequence'))
            .alias('sequence')
        )
        return truncated_df
    else:
        raise TypeError("Unsupported DataFrame type. Please provide a Pandas or Polars DataFrame.")



def demo_truncated_df(): 
    # Example usage with Pandas DataFrame
    sequences_df_pandas = pd.DataFrame({
        'gene_name': ['gene1', 'gene2'],
        'sequence': [
            'ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC',
            'CGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA']
    })
    truncated_df_pandas = truncate_sequences(sequences_df_pandas, max_length=10)
    print(truncated_df_pandas)

    # Example usage with Polars DataFrame
    sequences_df_polars = pl.DataFrame({
        'gene_name': ['gene1', 'gene2'],
        'sequence': [
            'ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC',
            'CGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA']
    })
    truncated_df_polars = truncate_sequences(sequences_df_polars, max_length=10)
    print(truncated_df_polars)

    return


def demo_with_columns(): 
    """

    The .with_columns method in polars is used to add or modify columns in a DataFrame. 
    It allows you to perform various operations on existing columns or create new columns based on expressions.

    Usage of .with_columns: 
      - The .with_columns method takes one or more expressions that define the transformations or 
        additions you want to apply to the DataFrame. Each expression can be a column operation, 
        such as casting a column to a different data type, applying a function to a column, or 
        creating a new column based on existing ones. 

      - pl.col('seqname'): This selects the column named 'seqname' from the DataFrame.
      - .cast(pl.Utf8): This casts the selected column to the Utf8 data type, which is equivalent to a string type in polars.
      - .with_columns: 
            This method applies the transformation to the DataFrame and returns a new DataFrame with the modified column.

    """


    # Sample data
    data = {
        'seqname': [1, 2, 3],
        'value': [10, 20, 30]
    }

    # Create a DataFrame
    df = pl.DataFrame(data)

    # Print the original DataFrame
    print("Original DataFrame:")
    print(df)

    # Apply the transformation
    df = df.with_columns(pl.col('seqname').cast(pl.Utf8))

    # Print the modified DataFrame
    print("\nModified DataFrame:")
    print(df)


def demo_group_by_agg():
    """
    In the context of Polars, a "struct" is a complex data type that allows you to group multiple columns together 
    into a single column. This is similar to a struct or record in other programming languages, 
    where a struct can contain multiple fields of different types.

    Explanation
        Struct Column: 
            A struct column in Polars is a column that contains multiple fields, 
            each of which can be of a different data type. This allows you to encapsulate related data together 
            in a single column.
    Usage: 
        Struct columns are useful for organizing and manipulating complex data structures within a DataFrame. 
        They can be used to group related columns together, making it easier to perform operations on 
        them as a single unit.

    """

    # Sample data
    data = {
        'gene_id': ['gene1', 'gene1', 'gene2', 'gene2'],
        'start': [100, 200, 300, 400],
        'end': [110, 210, 310, 410],
        'strand': ['+', '+', '-', '-'],
        'site_type': ['donor', 'acceptor', 'donor', 'acceptor']
    }

    # Create a DataFrame
    df = pl.DataFrame(data)

    # Group by 'gene_id' and aggregate into a struct column
    grouped_annotations = df.group_by('gene_id').agg(
        pl.struct(['start', 'end', 'strand', 'site_type']).alias('annotations')
    )
    # The resulting DataFrame has a gene_id column and an annotations column, where 
    # each entry in the annotations column is a struct containing the aggregated fields.

    print(grouped_annotations)
    


def generate_label_arrays(annotations_df, gene_sequences):
    """
    Generate label arrays for splice site prediction based on gene sequences and annotations.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing annotations for splice sites (donor, acceptor).
                                     Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - gene_sequences (pl.DataFrame): DataFrame containing gene sequences with positions and gene information.
                                     Columns: ['gene_id', 'seqname', 'gene_name', 'start', 'end', 'strand', 'sequence']

    Returns:
    - labels_dict (dict): A dictionary where each key is a gene_id, and the value is a dict with:
        - 'donor_labels': Array of 0s and 1s (1 at donor sites).
        - 'acceptor_labels': Array of 0s and 1s (1 at acceptor sites).
    """
    
    # Pre-filter annotations_df for the relevant genes in gene_sequences
    relevant_gene_ids = gene_sequences.select('gene_id').unique().to_series().to_list()
    filtered_annotations_df = annotations_df.filter(pl.col('gene_id').is_in(relevant_gene_ids))

    # Method 1: Group annotations by gene_id for efficient lookups
    # grouped_annotations = filtered_annotations_df.group_by('gene_id').agg(
    #     pl.struct(['start', 'end', 'strand', 'site_type']).alias('annotations')
    # ).to_dict(as_series=False)
    # NOTE: group_by('gene_id').agg(pl.struct(['start', 'end', 'strand', 'site_type']).alias('annotations')): 
    #   This groups the annotations by gene_id and aggregates the columns 'start', 'end', 'strand', and 'site_type'
    #   into a struct column named 'annotations'. This allows for efficient lookup of annotations for each gene.

    # Method 2: Group annotations by gene_id and aggregate them
    grouped_annotations = filtered_annotations_df.group_by('gene_id').agg(
        pl.struct(['start', 'end', 'strand', 'site_type']).alias('annotations')
    )

    # Convert the result to a dictionary where keys are gene_id, and values are annotations
    annotations_lookup = {row['gene_id']: row['annotations'] for row in grouped_annotations.iter_rows(named=True)}


    labels_dict = {}

    # Iterate over gene sequences in the current chunk
    for row in gene_sequences.iter_rows(named=True):
        gene_id = row['gene_id']
        gene_seq = row['sequence']
        gene_len = len(gene_seq)
        gene_start = row['start']
        gene_end = row['end']
        strand = row['strand']

        # Initialize label arrays of length equal to the gene sequence
        donor_labels = [0] * gene_len
        acceptor_labels = [0] * gene_len

        # Get annotations for this gene from the pre-grouped dictionary
        if gene_id in annotations_lookup:
            annotations = grouped_annotations[gene_id]

            # Iterate over the splice site annotations for this gene
            for ann_row in annotations:
                site_type = ann_row['site_type']
                site_start = ann_row['start']
                site_end = ann_row['end']
                site_strand = ann_row['strand']

                # Ensure the annotation matches the strand of the gene
                if site_strand != strand:
                    continue

                # Convert absolute positions (start, end) to relative positions in the gene sequence
                relative_start = site_start - gene_start
                relative_end = site_end - gene_start

                if relative_start < 0 or relative_end >= gene_len:
                    print("[debug] Splice site outside gene sequence range: gene_id =", gene_id)
                    print("... relative_start =", relative_start, "relative_end =", relative_end)
                    continue  # Ignore splice sites that are outside the gene sequence range

                # Mark the labels based on site_type (donor or acceptor)
                if site_type == 'donor':
                    for pos in range(relative_start, relative_end + 1):
                        donor_labels[pos] = 1
                elif site_type == 'acceptor':
                    for pos in range(relative_start, relative_end + 1):
                        acceptor_labels[pos] = 1

        # Store the label arrays in the dictionary
        labels_dict[gene_id] = {
            'donor_labels': donor_labels,
            'acceptor_labels': acceptor_labels
        }

    return labels_dict


def demo_group_by_agg_polars_vs_pandas(): 

    # Sample DataFrame
    data = {
        'gene_id': ['gene1', 'gene1', 'gene2', 'gene2', 'gene3'],
        'site_type': ['donor', 'acceptor', 'donor', 'donor', 'acceptor'],
        'start': [100, 200, 300, 400, 500],
        'end': [150, 250, 350, 450, 550]
    }

    # Polars equivalent
    annotations_df = pl.DataFrame(data)

    # Group by 'gene_id' and aggregate 'start' and 'end' into a struct
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'donor').group_by('gene_id').agg(
        pl.struct(['start', 'end']).alias('donor_sites')
    ).to_dict(as_series=False)

    print(grouped_annotations)

    # .group_by().agg() in Polars: 
    # - In Polars, the group_by method returns a GroupBy object, similar to Pandas. 
    #   However, unlike Pandas, Polars does not directly support iterating over the GroupBy object using a for loop. 
    #   Instead, you typically perform aggregation operations on the GroupBy object and 
    #   then iterate over the resulting DataFrame.

    # Group by 'gene_id' and aggregate 'start' and 'end' into a struct
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'donor').group_by('gene_id').agg(
        pl.struct(['start', 'end']).alias('donor_sites')
    )
    # The group_by method returns a GroupBy object, but you cannot directly iterate over it. 
    # Instead, you perform an aggregation and then iterate over the resulting DataFrame.
    # - the agg method performs the aggregation, creating a new DataFrame with the aggregated results.

    # Q: What Happens if .alias() is Removed?
    # - If you remove the .alias('donor_sites') part, the resulting DataFrame will still be created, 
    #   but the column containing the aggregated struct will have a default name generated by Polars. 
    #   The default name is typically based on the names of the aggregated columns.
    # - In this case, the column containing the aggregated struct would be named 'struct[start, end]'.
    # shape: (2, 2)
    # ┌─────────┬────────────────────┐
    # │ gene_id │ struct[start, end] │
    # │ ---     │ ---                │
    # │ str     │ struct[2]          │
    # ├─────────┼────────────────────┤
    # │ gene1   │ {100,150}          │
    # │ gene2   │ {300,350}          │
    # └─────────┴────────────────────┘

    # With the .alias('donor_sites') part, the column containing the aggregated struct is named 'donor_sites',
    # as specified in the alias method. This provides a more descriptive name for the column.
    # 
    # shape: (2, 2)
    # ┌─────────┬────────────────────┐
    # │ gene_id │ donor_sites        │
    # │ ---     │ ---                │
    # │ str     │ struct[2]          │
    # ├─────────┼────────────────────┤
    # │ gene1   │ {100,150}          │
    # │ gene2   │ {300,350}          │
    # └─────────┴────────────────────┘


    # Iterate over the resulting DataFrame
    for row in grouped_annotations.iter_rows(named=True):
        print(row)
    # What's in row? 
    # - The row variable contains a dictionary-like object that allows you to access the values of each column by name.
    # - the named=True argument in the iter_rows method of Polars specifies that the rows should be 
    #   returned as dictionaries, not named tuples. When named=True is used, each row is represented 
    #   as a dictionary where the keys are the column names and the values are the corresponding data for that row.
    #     - named=True: Rows are returned as dictionaries.
    #     - named=False (default): Rows are returned as tuples.
    #   
    # - When iterating over the rows with iter_rows(named=True), each row will be a dictionary with the following structure:
    # 
    # {
    #     'gene_id': 'gene1',
    #     'donor_sites': {'start': 100, 'end': 150}
    # }

    # --------------------------------------------------------

    # Pandas equivalent
    annotations_df = pd.DataFrame(data)

    # Filter for 'donor' site_type and group by 'gene_id'
    grouped_annotations = annotations_df[annotations_df['site_type'] == 'donor'].groupby('gene_id').agg(
        list
    ).reset_index()

    # Combine 'start' and 'end' into a list of dictionaries
    grouped_annotations['donor_sites'] = grouped_annotations.apply(
        lambda row: [{'start': s, 'end': e} for s, e in zip(row['start'], row['end'])],
        axis=1
    )

    # Drop the original 'start' and 'end' columns
    grouped_annotations = grouped_annotations[['gene_id', 'donor_sites']]

    print(grouped_annotations.to_dict(orient='list'))

    # Iterating over the GroupBy object in Pandas:
    
    # Filter for 'donor' site_type and group by 'gene_id'
    grouped_annotations = annotations_df[annotations_df['site_type'] == 'donor'].groupby('gene_id')

    # Iterate over the GroupBy object
    for gene_id, group in grouped_annotations:
        print(f"Gene ID: {gene_id}")
        print(group)




def demo_generate_label_arrays(): 

    # Mock annotations DataFrame
    annotations_data = {
        'chrom': ['chr1', 'chr1', 'chr1', 'chr2', 'chr2'],
        'start': [2581648, 2583368, 2583493, 3069294, 3069300],
        'end': [2581651, 2583371, 2583496, 3069297, 3069303],
        'strand': ['+', '+', '+', '+', '-'],
        'site_type': ['donor', 'acceptor', 'donor', 'donor', 'acceptor'],
        'gene_id': ['ENSG00000228037', 'ENSG00000228037', 'ENSG00000228037', 'ENSG00000142611', 'ENSG00000142611'],
        'transcript_id': ['ENST00000424215', 'ENST00000424215', 'ENST00000424215', 'ENST00000511072', 'ENST00000511072']
    }
    annotations_df = pl.DataFrame(annotations_data)

    # Mock gene sequences DataFrame
    gene_sequences_data = {
        'gene_id': ['ENSG00000228037', 'ENSG00000142611'],
        'seqname': ['chr1', 'chr2'],
        'gene_name': ['Gene1', 'Gene2'],
        'start': [2581600, 3069000],
        'end': [2584000, 3070000],
        'strand': ['+', '+'],
        'sequence': ['A' * 3000, 'T' * 1000]
    }
    gene_sequences_df = pl.DataFrame(gene_sequences_data)

    # Validate the function with mock data
    labels_dict = generate_label_arrays(annotations_df, gene_sequences_df)

    # Print the positions of 1s in the donor and acceptor labels
    for gene_id, labels in labels_dict.items():
        donor_positions = [i for i, label in enumerate(labels['donor_labels']) if label == 1]
        acceptor_positions = [i for i, label in enumerate(labels['acceptor_labels']) if label == 1]
        print(f"Gene ID: {gene_id}")
        print(f"Donor positions: {donor_positions}")
        print(f"Acceptor positions: {acceptor_positions}")
        print()

    return

def demo_dataframe_conversion(): 

    # Create a mock Pandas DataFrame
    data = {
        'A': [1, 2, 3, 4],
        'B': ['a', 'b', 'c', 'd'],
        'C': [1.1, 2.2, 3.3, 4.4]
    }
    pandas_df = pd.DataFrame(data)
    print("Original Pandas DataFrame:")
    print(pandas_df)

    # Convert Pandas DataFrame to Polars DataFrame
    polars_df = pandas_to_polars(pandas_df)
    print("\nConverted to Polars DataFrame:")
    print(polars_df)

    # Convert Polars DataFrame back to Pandas DataFrame
    converted_pandas_df = polars_to_pandas(polars_df)
    print("\nConverted back to Pandas DataFrame:")
    print(converted_pandas_df)

    # Validate the conversion
    assert pandas_df.equals(converted_pandas_df), "Conversion failed: DataFrames are not equal"
    print("\nConversion successful: DataFrames are equal")


def demo_extract_gene_sequences(): 
    import polars as pl

    # Example DataFrame
    data = {
        "sequence": ["ATGCGTACGTAGCTAGCTAG", "CGTAGCTAGCTAGCTAGCTA", "GCTAGCTAGCTAGCTAGCTA"],
        "window_start": [2, 3, 4],
        "window_end": [8, 10, 12]
    }
    merged_df = pl.DataFrame(data)

    # Extract relevant gene sequences
    extracted_sequences = merged_df.with_columns([
        pl.col("sequence").str.slice(
            pl.col("window_start"), 
            pl.col("window_end") - pl.col("window_start")).alias("extracted_sequence")
    ])

    print(extracted_sequences)


def demo_apply_function(): 
    """

    Memo
    ----
    args=(pl.col("position"),):
    
    This specifies additional arguments to pass to the lambda function. Here, it passes the position column 
    as an argument to the lambda function. The apply method will apply the lambda function to each element 
    of the extracted_sequence column, and for each element, it will also pass the corresponding value 
    from the position column.
    """
    import polars as pl

    # Example DataFrame
    data = {
        "extracted_sequence": ["ATCGTAGCTAGCTA", "CGATCGTAGCTAGC", "TGCATGCATGCATG"],
        "position": [3, 2, 5],
        "gene_id": ["ENSG00000034693", "ENSG00000034694", "ENSG00000034695"],
        "error_type": ["FP", "FP", "FP"]
    }
    error_sequence_df = pl.DataFrame(data)

    # Define the has_consensus function
    def has_consensus(sequence, position, splice_type, window_radius):
        consensus_sequences = {
            "acceptor": {"AG", "CAG", "TAG", "AAG"},
            "donor": {"GT", "GC"}
        }
        if splice_type not in consensus_sequences:
            raise ValueError(f"Invalid splice_type '{splice_type}'. Must be 'donor' or 'acceptor'.")
        
        # Error handling for out-of-bounds positions
        if position < 0 or position >= len(sequence):
            return False
        start_idx = max(0, position - window_radius)
        end_idx = min(len(sequence), position + window_radius + 1)
        neighborhood = sequence[start_idx:end_idx]
        return any(consensus in neighborhood for consensus in consensus_sequences[splice_type])

    # Apply the logic using apply on the DataFrame
    splice_type = "donor"
    window_radius = 2

    # Use apply on the DataFrame to create the has_consensus column
    error_sequence_df = error_sequence_df.with_columns([
        pl.apply(
            lambda row: has_consensus(row["extracted_sequence"], row["position"], splice_type, window_radius),
            return_dtype=pl.Boolean
        ).alias("has_consensus")
    ])

    print(error_sequence_df)

    return


def test(): 

    # Test the demo functions
    # demo_with_columns()
    # demo_group_by_agg()
    # demo_generate_label_arrays()

    # demo_dataframe_conversion()

    # demo_truncated_df()

    # demo_extract_gene_sequences()

    demo_apply_function()

    return



if __name__ == "__main__": 
    test()