import os
import pandas as pd
import polars as pl
from .utils_bio import (
    extract_genes_from_gtf, 
    extract_exons_from_gtf, 
)

from .utils_doc import (
    print_emphasized,
    print_with_indent,
    print_section_separator, 
    display
)

def extract_gene_boundaries_v0(gtf_file):
    """
    Extract gene boundaries from a GTF file.

    Parameters:
    - gtf_file (str): Path to the GTF file.

    Returns:
    - pl.DataFrame: A Polars DataFrame with columns ['gene_id', 'chrom', 'start', 'end', 'strand', 'gene_name'].
    """
    # Extract annotations
    gtf_df = extract_genes_from_gtf(gtf_file)

    # Group by gene_id to find boundaries
    gene_boundaries = (
        gtf_df
        .group_by("gene_id")
        .agg([
            pl.col("seqname").first().alias("chrom"),
            pl.col("start").min().alias("start"),
            pl.col("end").max().alias("end"),
            pl.col("strand").first().alias("strand"),
            pl.col("gene_name").first().alias("gene_name"),
        ])
    )

    return gene_boundaries


def extract_gene_boundaries(gtf_file, filter_valid_splice_sites=True, min_exons=2):
    """
    Extract gene boundaries from a GTF file, optionally filtering for genes with valid splice sites.

    Parameters:
    - gtf_file (str): Path to the GTF file.
    - filter_valid_splice_sites (bool): Whether to filter genes based on valid splice sites (default: True).
    - min_exons (int): Minimum number of exons for a gene to be considered valid (default: 2).

    Returns:
    - pl.DataFrame: A Polars DataFrame with columns ['gene_id', 'chrom', 'start', 'end', 'strand', 'gene_name'].
    """
    # Extract gene annotations
    gtf_df = extract_genes_from_gtf(gtf_file)  # Extract gene features

    if filter_valid_splice_sites:
        # Extract exon annotations for exon counts
        exon_df = extract_exons_from_gtf(gtf_file)  # Extract exon features
        exon_counts = (
            exon_df.group_by("gene_id").agg(pl.len().alias("exon_count"))
        )
        # pl.count() deprecated 

        valid_genes = exon_counts.filter(pl.col("exon_count") >= min_exons).select("gene_id")
        gtf_df = gtf_df.filter(pl.col("gene_id").is_in(valid_genes["gene_id"]))

    # Group by gene_id to find boundaries
    gene_boundaries = (
        gtf_df
        .group_by("gene_id")
        .agg([
            pl.col("seqname").first().alias("chrom"),
            pl.col("start").min().alias("start"),
            pl.col("end").max().alias("end"),
            pl.col("strand").first().alias("strand"),
            pl.col("gene_name").first().alias("gene_name"),
        ])
    )

    return gene_boundaries


def find_overlapping_genes_with_boundaries(gene_boundaries, output_file=None):
    """
    Identify overlapping genes based on their boundaries and add overlap counts comprehensively.

    Parameters:
    - gene_boundaries (pl.DataFrame): DataFrame with gene boundary information.

    Returns:
    - pl.DataFrame: DataFrame with overlapping genes and their details, including comprehensive overlap counts.
    """
    overlapping_list = []

    # Iterate through chromosomes to find overlapping genes
    for chrom in gene_boundaries["chrom"].unique():
        chrom_genes = gene_boundaries.filter(pl.col("chrom") == chrom).sort("start")

        for gene1 in chrom_genes.iter_rows(named=True):
            gene1_id, gene1_start, gene1_end, gene1_strand = (
                gene1["gene_id"], gene1["start"], gene1["end"], gene1["strand"]
            )

            # Find overlapping genes
            overlaps = chrom_genes.filter(
                (pl.col("start") <= gene1_end) &
                (pl.col("end") >= gene1_start) &
                (pl.col("strand") == gene1_strand) &
                (pl.col("gene_id") != gene1_id)
            )

            for gene2 in overlaps.iter_rows(named=True):
                # Add consistent gene pair ordering to avoid duplicates
                gene_id_1 = min(gene1_id, gene2["gene_id"])
                gene_id_2 = max(gene1_id, gene2["gene_id"])
                overlapping_list.append({
                    "gene_id_1": gene_id_1,
                    "gene_id_2": gene_id_2,
                    "chrom": gene1["chrom"],
                    "start_1": gene1["start"],
                    "end_1": gene1["end"],
                    "start_2": gene2["start"],
                    "end_2": gene2["end"],
                    "strand_1": gene1["strand"],
                    "strand_2": gene2["strand"],
                    "gene_name_1": gene1.get("gene_name", gene1_id),
                    "gene_name_2": gene2.get("gene_name", gene2["gene_id"]),
                })

    # Create a Polars DataFrame
    overlapping_genes_df = pl.DataFrame(overlapping_list)

    # Eliminate handshake duplicates
    overlapping_genes_df = overlapping_genes_df.unique(subset=["gene_id_1", "gene_id_2"])

    # Compute comprehensive overlap counts
    # Combine gene_id_1 and gene_id_2 into a single column and count unique overlaps for each gene
    all_overlaps = (
        overlapping_genes_df
        .melt(id_vars=["chrom", "start_1", "end_1", "start_2", "end_2", "strand_1", "strand_2"], 
              value_vars=["gene_id_1", "gene_id_2"], 
              variable_name="gene_role", 
              value_name="gene_id")
        .group_by("gene_id")
        .agg(pl.count().alias("num_overlaps"))
    )

    # Join the overlap counts back into the original DataFrame
    overlapping_genes_df = overlapping_genes_df.join(all_overlaps, left_on="gene_id_1", right_on="gene_id", how="left")
    overlapping_genes_df = overlapping_genes_df.with_columns(pl.col("num_overlaps").fill_null(0).cast(int))

    if output_file:
        file_extension = output_file.split('.')[-1]
        if file_extension == 'csv':
            overlapping_genes_df.write_csv(output_file)
        elif file_extension == 'tsv':
            overlapping_genes_df.write_csv(output_file, separator='\t')
        else:
            raise ValueError("Unsupported file format. Please use 'csv' or 'tsv'.")

    return overlapping_genes_df


def find_overlapping_genes_with_boundaries_v0(gene_boundaries):
    """
    Identify overlapping genes based on their boundaries and add overlap counts.

    Parameters:
    - gene_boundaries (pl.DataFrame): DataFrame with gene boundary information.

    Returns:
    - pl.DataFrame: DataFrame with overlapping genes and their details, including unique overlap counts.

    - The output DataFrame will now have consistent column names like: 
      ['gene_id_1', 'gene_id_2', 'chrom', 'start_1', 'end_1', 'start_2', 'end_2', 
       'strand_1', 'strand_2', 'gene_name_1', 'gene_name_2', 'num_overlaps']

    """
    overlapping_list = []

    for chrom in gene_boundaries["chrom"].unique():
        chrom_genes = gene_boundaries.filter(pl.col("chrom") == chrom).sort("start")

        for gene1 in chrom_genes.iter_rows(named=True):
            gene1_id, gene1_start, gene1_end, gene1_strand = (
                gene1["gene_id"], gene1["start"], gene1["end"], gene1["strand"]
            )

            # Find overlapping genes
            overlaps = chrom_genes.filter(
                (pl.col("start") <= gene1_end) &
                (pl.col("end") >= gene1_start) &
                (pl.col("strand") == gene1_strand) &
                (pl.col("gene_id") != gene1_id)
            )

            for gene2 in overlaps.iter_rows(named=True):
                # Add consistent gene pair ordering to avoid duplicates
                gene_id_1 = min(gene1_id, gene2["gene_id"])  # Standardized naming convention and following lexicographic order
                gene_id_2 = max(gene1_id, gene2["gene_id"])
                overlapping_list.append({
                    "gene_id_1": gene_id_1,  
                    "gene_id_2": gene_id_2,
                    "chrom": gene1["chrom"],
                    "start_1": gene1["start"],
                    "end_1": gene1["end"],
                    "start_2": gene2["start"],
                    "end_2": gene2["end"],
                    "strand_1": gene1["strand"],
                    "strand_2": gene2["strand"],
                    "gene_name_1": gene1.get("gene_name", gene1_id),
                    "gene_name_2": gene2.get("gene_name", gene2["gene_id"]),
                })

    # Create a Polars DataFrame
    overlapping_genes_df = pl.DataFrame(overlapping_list)

    # Eliminate handshake duplicates
    overlapping_genes_df = overlapping_genes_df.unique(subset=["gene_id_1", "gene_id_2"])
    # Defensive Programming: Adding unique() acts as a safeguard against any unexpected duplicates 
    # But this is not strictly necessary if overalpping gene pairs are already in lexigraphic order

    # Calculate overlap counts
    if not overlapping_genes_df.is_empty():
        overlap_counts = overlapping_genes_df.group_by("gene_id_1").agg(
            pl.count("gene_id_2").alias("num_overlaps")
        )
    else:
        overlap_counts = pl.DataFrame({"gene_id_1": [], "num_overlaps": []})

    # Join overlap counts into the overlapping genes dataframe
    overlapping_genes_df = overlapping_genes_df.join(overlap_counts, on="gene_id_1", how="left")
    overlapping_genes_df = overlapping_genes_df.with_columns(pl.col("num_overlaps").fill_null(0).cast(int))

    return overlapping_genes_df



def save_as_bed(df, output_file):
    """
    Save a DataFrame of overlapping genes as a BED file for UCSC visualization.

    Parameters:
    - df (pl.DataFrame): DataFrame containing overlapping gene information.
    - output_file (str): Path to the output BED file.
    """
    # Prepare BED format: chrom, start, end, name, score, strand
    bed_df = df.select([
        pl.col("chrom").alias("chrom"),
        pl.col("start_1").alias("start"),
        pl.col("end_1").alias("end"),
        pl.concat_str(
            [pl.col("gene_name_1"), pl.lit("|"), pl.col("gene_name_2")], separator=""
        ).alias("name"),
        pl.lit("0").alias("score"),
        pl.col("strand_1").alias("strand"),
    ])

    # Save as a BED file
    bed_df.write_csv(output_file, separator="\t")
    print(f"[info] BED file saved to {output_file}")


def preprocess_for_bed_v0(overlapping_genes_df):
    """
    Preprocess the overlapping genes dataframe for BED file compatibility.
    
    Parameters:
    - overlapping_genes_df (pl.DataFrame): Output of find_overlapping_genes_with_boundaries().
    
    Returns:
    - pl.DataFrame: Preprocessed dataframe with unified start and end columns.
    """
    # Create unified start and end columns
    return overlapping_genes_df.with_columns([
        pl.min_horizontal(["start_1", "start_2"]).alias("start"),
        pl.max_horizontal(["end_1", "end_2"]).alias("end"),
        # pl.concat_str(["gene_id_1", "gene_id_2"], separator="|").alias("name"),
        pl.concat_str(
            [pl.col("gene_name_1"), pl.lit("|"), pl.col("gene_name_2")], separator=""
        ).alias("name"),
        pl.col("strand_1").alias("strand")  
    ]).select(["chrom", "start", "end", "name", "strand"])


def preprocess_for_bed(overlapping_genes_df, error_type_column=None):
    """
    Preprocess the overlapping genes DataFrame for BED file compatibility.

    Parameters:
    - overlapping_genes_df (pl.DataFrame): Output of find_overlapping_genes_with_boundaries().
    - error_type_column (str, optional): Name of the error type column (e.g., 'FP', 'FN') to include if present.

    Returns:
    - pl.DataFrame: Preprocessed DataFrame with unified start and end columns for BED file generation.
    """
    # Ensure required columns exist
    required_columns = [
        "gene_id_1", "gene_id_2", "start_1", "start_2", 
        "end_1", "end_2", "gene_name_1", "gene_name_2", "chrom", "strand_1"
    ]
    missing_columns = [col for col in required_columns if col not in overlapping_genes_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in input DataFrame: {missing_columns}")

    # Create unified start, end, and name columns for BED format
    preprocessed_df = overlapping_genes_df.with_columns([
        pl.min_horizontal(["start_1", "start_2"]).alias("start"),
        pl.max_horizontal(["end_1", "end_2"]).alias("end"),
        pl.concat_str([
            pl.when(pl.col("gene_name_1").is_not_null()).then(pl.col("gene_name_1")).otherwise(pl.col("gene_id_1")),
            pl.lit("|"),
            pl.when(pl.col("gene_name_2").is_not_null()).then(pl.col("gene_name_2")).otherwise(pl.col("gene_id_2"))
        ], separator="").alias("name"),
        pl.col("strand_1").alias("strand")
    ])

    # Base columns for BED
    selected_columns = ["chrom", "start", "end", "name", "strand"]

    # Optionally include num_overlaps if present
    if "num_overlaps" in overlapping_genes_df.columns:
        selected_columns.append("num_overlaps")

    # Optionally include error_type_column if provided and present
    if error_type_column and error_type_column in overlapping_genes_df.columns:
        selected_columns.append(error_type_column)

    # Select only the relevant columns dynamically
    preprocessed_df = preprocessed_df.select(selected_columns)

    return preprocessed_df


def save_as_bed_with_colors(df, output_file, error_type_column=None, verbose=1):
    """
    Save overlapping gene information to a BED file format with optional color coding for different error types.

    Parameters:
    - df (pl.DataFrame): Input Polars DataFrame with columns ['chrom', 'start', 'end', 'name', 'strand'].
      Optionally includes an 'error_type' column.
    - output_file (str): Path to save the BED file.
    - error_type_column (str, optional): Name of the column indicating error types ('FP', 'FN'). If None, no color coding.
    - verbose (int): Verbosity level (default=1).

    Returns:
    - None
    """
    # import pandas as pd

    # Convert to Pandas for ease of manipulation and writing
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Define BED format columns
    required_columns = ['chrom', 'start', 'end', 'name', 'strand']
    if error_type_column:
        required_columns.append(error_type_column)
    if 'num_overlaps' not in df.columns:
        df['num_overlaps'] = 1  # Default to 1 overlap if missing

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Columns in dataframe: {list(df.columns)}")

    # Add score and itemRgb columns as defaults
    df['score'] = 0  # Placeholder score column for BED format
    
    df['itemRgb'] = '240,240,240'  # '238,238,228'  # Default to very light gray for overlapping regions

    # Dynamic color coding
    # df['itemRgb'] = df['num_overlaps'].apply(
    #     lambda x: '144,238,144' if x <= 5 else '255,255,153' if x <= 10 else '255,160,122'
    # )

    # Apply color coding if error_type_column is provided
    if error_type_column:
        color_map = {
            'FP': '255,0,0',  # Red for False Positives
            # 'FP': '220,50,47',  # Soft red for False Positives

            'FN': '0,0,255',  # Blue for False Negatives
            # 'FN': '38,139,210',  # Soft blue for False Negatives
        }
        df['itemRgb'] = df[error_type_column].map(color_map).fillna('0,0,0')

    # Append overlap count to name
    # df['name'] = df['name'] + ' | Overlaps: ' + df['num_overlaps'].astype(str)

    # Select and reorder columns for BED format
    bed_columns = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'itemRgb']
    bed_df = df[bed_columns]

    # Sort chromosomes naturally (e.g., chr1, chr2, ..., chrX, chrY, chrMT, etc.)
    # bed_df['chrom'] = pd.Categorical(
    #     bed_df['chrom'],
    #     categories=sorted(
    #         bed_df['chrom'].unique(),
    #         key=lambda x: (
    #             int(x[3:]) if x[3:].isdigit() else float('inf') if x[3:] == 'MT' else ord(x[3:]),
    #             x,
    #         )
    #     ),
    #     ordered=True
    # )
    bed_df = bed_df.sort_values(['chrom', 'start'])

    # Save to BED file with | as the separator
    # bed_df.to_csv(output_file, sep='\t', header=False, index=False)

    # Add track line
    track_line = 'track name="Overlapping Genes" description="Overlapping genes visualized" visibility=2 itemRgb="On"\n'

    # Save to BED file
    with open(output_file, 'w') as f:
        f.write(track_line)
        bed_df.to_csv(f, sep='\t', header=False, index=False)

    if verbose:
        print(f"[info] BED file saved to {output_file}")


def generate_bed_files_for_overlapping_genes(gtf_file=None, output_bed_file="overlapping_genes.bed", **kargs):
    """
    Generate a BED file with overlapping genes from a GTF file.

    Parameters:
    - gtf_file (str): Path to the GTF file.
    - output_bed_file (str): Path to the output BED file.
    """
    # gtf_file = "Homo_sapiens.GRCh38.112.gtf"
    if gtf_file is None: 
        gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your default GTF file path

    verbose = kargs.get("verbose", 1)
    filter_valid_splice_sites = kargs.get("filter_valid_splice_sites", True)
    min_exons = kargs.get("min_exons", 2)

    # Extract gene boundaries
    print_emphasized("[info] Extracting gene boundaries ...")
    gene_boundaries = \
        extract_gene_boundaries(
            gtf_file, 
            filter_valid_splice_sites=filter_valid_splice_sites, 
            min_exons=min_exons)
    print(f"[info] column(gene_boundaries) from extract_gene_boundaries:\n{gene_boundaries.columns}\n")

    # Find overlapping genes
    print_emphasized("[info] Finding overlapping genes given gene boundaries ...")
    gene_boundaries = find_overlapping_genes_with_boundaries(gene_boundaries)
    assert not gene_boundaries.is_empty()
    print(f"[info] column(gene_boundaries) from find_overlapping_genes_with_boundaries:\n{gene_boundaries.columns}\n")

    gene_boundaries = gene_boundaries.sort("num_overlaps", descending=True)
    columns = ["gene_id_1", "gene_id_2", "chrom", "gene_name_1", "gene_name_2", "num_overlaps"]

    print(f"[info] Top 100 overlapping genes:\n{gene_boundaries.select(columns).head(100)}\n")
    print(gene_boundaries.select(columns).head(100))
    print(f"[info] Bottom 100 overlapping genes:\n{gene_boundaries.select(columns).tail(100)}\n")
    print(gene_boundaries.select(columns).tail(100))

    overlapping_genes_df = preprocess_for_bed(gene_boundaries)

    # Save overlaps as a BED file
    print_emphasized("[i/o] Saving overlaps as a BED file ...")
    # save_as_bed(overlapping_genes_df, output_bed_file)
    save_as_bed_with_colors(overlapping_genes_df, output_bed_file)


def demo_find_overlapping_genes_with_boundaries(**kargs):

    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    gtf_file_path = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"
    output_bed_file_path = os.path.join(local_dir, "overlapping_genes.bed")

    # Mock dataset of genes with boundaries
    mock_gene_boundaries = pl.DataFrame({
        "gene_id": ["A", "B", "C", "D"],
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "start": [100, 150, 240, 100],
        "end": [200, 250, 300, 200],
        "strand": ["+", "+", "+", "+"],
        "gene_name": ["GeneA", "GeneB", "GeneC", "GeneD"]
    })
    # Gene A overlaps with Gene B.
    # Gene B overlaps with Gene C.

    # Run the function with the mock dataset
    result = find_overlapping_genes_with_boundaries(mock_gene_boundaries)

    # Inspect the result
    print(result) 

    ############################################################
    gtf_file = kargs.get("gtf_file", gtf_file_path)
    verbose = kargs.get("verbose", 1)
    filter_valid_splice_sites = kargs.get("filter_valid_splice_sites", True)
    min_exons = kargs.get("min_exons", 2)
    output_file = os.path.join(local_dir, "overlapping_gene_counts.tsv")

    # Extract gene boundaries
    print_emphasized("[info] Extracting gene boundaries ...")
    gene_boundaries = \
        extract_gene_boundaries(
            gtf_file, 
            filter_valid_splice_sites=filter_valid_splice_sites, 
            min_exons=min_exons)
    print(f"[info] column(gene_boundaries) from extract_gene_boundaries:\n{gene_boundaries.columns}\n")

    # Find overlapping genes
    print_emphasized("[info] Finding overlapping genes given gene boundaries ...")
    gene_boundaries = find_overlapping_genes_with_boundaries(gene_boundaries)
    assert not gene_boundaries.is_empty()
    print(f"[info] column(gene_boundaries) from find_overlapping_genes_with_boundaries:\n{gene_boundaries.columns}\n")

    gene_boundaries = gene_boundaries.sort("num_overlaps", descending=True)
    columns = ["gene_id_1", "gene_id_2", "chrom", "gene_name_1", "gene_name_2", "num_overlaps"]

    print(f"[info] Top 100 overlapping genes:\n{gene_boundaries.select(columns).head(100)}\n")
    print(gene_boundaries.select(columns).head(100))
    print(f"[info] Bottom 100 overlapping genes:\n{gene_boundaries.select(columns).tail(100)}\n")
    print(gene_boundaries.select(columns).tail(100))

    # Optionally save the output DataFrame to a file
    if output_file:
        file_extension = output_file.split('.')[-1]
        if file_extension == 'csv':
            gene_boundaries.write_csv(output_file)
        elif file_extension == 'tsv':
            gene_boundaries.write_csv(output_file, separator='\t')
        else:
            raise ValueError("Unsupported file format. Please use 'csv' or 'tsv'.")

    return gene_boundaries

    
def demo(): 

    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    
    gtf_file_path = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"
    output_bed_file_path = os.path.join(local_dir, "overlapping_genes.bed")

    demo_find_overlapping_genes_with_boundaries()

    # generate_bed_files_for_overlapping_genes(
    #     gtf_file=gtf_file_path, 
    #     output_bed_file=output_bed_file_path)    


    


if __name__ == "__main__": 
    demo()
