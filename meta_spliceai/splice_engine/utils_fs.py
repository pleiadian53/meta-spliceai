import os
import gffutils

from tqdm import tqdm
import pandas as pd
import polars as pl


def get_id_columns(column_list):
    """
    Get the list of columns ending with '_id' from the given comma-separated list.

    Parameters:
    - column_list (str): Comma-separated list of column names.

    Returns:
    - list: List of columns ending with '_id'.
    """
    columns = column_list.split(',')
    id_columns = [column.strip() for column in columns if column.strip().endswith('_id')]
    return id_columns


def build_gtf_database(gtf_file, db_file='annotations.db', overwrite=False):
    """
    Build a gffutils database from a GTF file for efficient querying of genomic features.

    Parameters:
    - gtf_file (str): Path to the GTF file containing gene annotations.
    - db_file (str): Path to save the created database (default: 'annotations.db').
    - overwrite (bool): Whether to overwrite the database if it already exists (default: False).

    Returns:
    - db (gffutils.FeatureDB): gffutils database object for querying annotations.

    Memo: 
    - force=True: Overwrites the database if it already exists.
    - keep_order=True: Preserves the feature order as in the original GTF file.
    - disable_infer_genes and disable_infer_transcripts: Prevent gffutils from inferring genes or transcripts not explicitly annotated.

    Example usage:
    gtf_file = 'path/to/your/gtf_file.gtf'
    db_file = 'path/to/your/db_file.db'
    db = build_gtf_database(gtf_file, db_file)

    """
    if os.path.exists(db_file) and not overwrite:
        # Load the existing database
        db = gffutils.FeatureDB(db_file)
        print(f"[info] Database loaded from {db_file}")
    else:
        # Create the database from the GTF file
        db = gffutils.create_db(
            gtf_file,
            dbfn=db_file,
            force=True,  # Overwrite if the database already exists
            keep_order=True,
            disable_infer_genes=True,
            disable_infer_transcripts=True
        )
        print(f"[info] Database created and saved to {db_file}")
    
    return db


def extract_all_gtf_annotations(db):
    """
    Extract exon, CDS, 5'UTR, and 3'UTR annotations for all transcripts from a gffutils database.

    Parameters:
    - db (gffutils.FeatureDB): gffutils database object.

    Returns:
    - annotations_df (pd.DataFrame): DataFrame containing annotations for all transcripts.
    """
    annotations_list = []
    n_test = 10
    tx_cnt = 0

    # Get the total number of transcripts for the progress bar
    total_transcripts = db.count_features_of_type('transcript')

    # Iterate over all transcripts
    for transcript in tqdm(db.features_of_type('transcript', order_by='start'), total=total_transcripts, desc="Extracting annotations"):
        transcript_id = transcript.id
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        chrom = transcript.chrom
        strand = transcript.strand

        # Extract exons
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Debugging: Print the number of exons found
        if tx_cnt < n_test:
            print(f"Transcript {transcript_id}: Found {len(exons)} exons")

        # Skip if no exons are found
        if not exons:
            continue

        # Extract CDS
        cds_features = list(db.children(transcript, featuretype='CDS', order_by='start'))

        # Debugging: Print the number of CDS features found
        if tx_cnt < n_test:
            print(f"Transcript {transcript_id}: Found {len(cds_features)} CDS features")

        # Collect exon annotations
        for exon in exons:
            annotations_list.append({
                'chrom': exon.chrom,
                'start': exon.start,
                'end': exon.end,
                'strand': exon.strand,
                'feature': 'exon',
                'gene_id': gene_id,
                'transcript_id': transcript_id
            })

        # Collect CDS annotations
        for cds in cds_features:
            annotations_list.append({
                'chrom': cds.chrom,
                'start': cds.start,
                'end': cds.end,
                'strand': cds.strand,
                'feature': 'CDS',
                'gene_id': gene_id,
                'transcript_id': transcript_id
            })

        # Infer 5'UTR and 3'UTR if CDS is present
        if cds_features:
            # Transcript boundaries
            transcript_start = min(exon.start for exon in exons)
            transcript_end = max(exon.end for exon in exons)

            cds_start = min(cds.start for cds in cds_features)
            cds_end = max(cds.end for cds in cds_features)

            # For positive strand
            if strand == '+':
                # 5'UTR: from transcript start to CDS start - 1
                if transcript_start < cds_start:
                    annotations_list.append({
                        'chrom': chrom,
                        'start': transcript_start,
                        'end': cds_start - 1,
                        'strand': strand,
                        'feature': "5'UTR",
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })
                # 3'UTR: from CDS end + 1 to transcript end
                if cds_end < transcript_end:
                    annotations_list.append({
                        'chrom': chrom,
                        'start': cds_end + 1,
                        'end': transcript_end,
                        'strand': strand,
                        'feature': "3'UTR",
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })
            # For negative strand
            else:
                # 5'UTR: from CDS end + 1 to transcript end
                if cds_end < transcript_end:
                    annotations_list.append({
                        'chrom': chrom,
                        'start': cds_end + 1,
                        'end': transcript_end,
                        'strand': strand,
                        'feature': "5'UTR",
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })
                # 3'UTR: from transcript start to CDS start - 1
                if transcript_start < cds_start:
                    annotations_list.append({
                        'chrom': chrom,
                        'start': transcript_start,
                        'end': cds_start - 1,
                        'strand': strand,
                        'feature': "3'UTR",
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

        tx_cnt += 1
        
    ### End of loop over transcripts

    # Debugging: Check the contents of annotations_list
    print(f"Total annotations collected: {len(annotations_list)}")
    if len(annotations_list) > 0:
        print("Sample annotation:", annotations_list[0])

    # Convert to DataFrame if annotations_list is not empty
    if annotations_list:
        annotations_df = pd.DataFrame(annotations_list)
    else:
        print("Error: No annotations were collected.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Debugging: Print the columns of the DataFrame
    print("Columns in annotations_df:", annotations_df.columns)

    # Ensure that 'start' and 'end' columns exist before filtering
    if 'start' in annotations_df.columns and 'end' in annotations_df.columns:
        annotations_df = annotations_df[annotations_df['start'] <= annotations_df['end']]
    else:
        print("Error: 'start' or 'end' column not found in annotations_df")

    return annotations_df


def read_annotations(file_path, sep='\t', dtypes=None):
    annotations_df = pd.read_csv(file_path, sep=sep)
    # NOTE: columns = ['chrom', 'start', 'end', 'strand', 'feature', 'gene_id', 'transcript_id']

    return annotations_df


def read_splice_sites(file_path, separator='\t', dtypes=None, **kargs):
    """
    Reads a CSV/TSV file into a Polars DataFrame with specified data types and separator.

    Parameters:
    - file_path (str): The path to the CSV/TSV file.
    - separator (str): The delimiter used in the file (default is '\t').
    - dtypes (dict): A dictionary specifying the data types for the columns (default is None).

    Returns:
    - pl.DataFrame: The Polars DataFrame containing the data from the file.

    Memo: 
    - The default columns for a splice site file are:
      columns = ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    """
    verbose = kargs.get('verbose', 1)

    if dtypes is None:
        # Define the default column data types
        dtypes = {
            'chrom': pl.Utf8,
            'start': pl.Int64,
            'end': pl.Int64,
            'strand': pl.Utf8,
            'site_type': pl.Utf8,
            'gene_id': pl.Utf8,
            'transcript_id': pl.Utf8
        }

    if verbose:
        print(f"[i/o] Reading annotations from {file_path}")

    # Determine the file format based on the file extension
    file_extension = file_path.split('.')[-1].lower()

    # Read the DataFrame using Polars
    if file_extension in ['csv', 'tsv']:
        # Read the DataFrame using Polars for CSV/TSV files
        annotations_df = pl.read_csv(file_path, separator=separator, schema_overrides=dtypes)
    elif file_extension == 'parquet':
        # Read the DataFrame using Polars for Parquet files
        annotations_df = pl.read_parquet(file_path, schema_overrides=dtypes)
    else:
        raise ValueError("Unsupported file format. Use 'csv', 'tsv', or 'parquet'.")
    
    return annotations_df


def demo_extract_annotations(gtf_file, db_file='annotations.db', output_file='annotations_all_transcripts.csv'):
    # Step 1: Build the database
    db = build_gtf_database(gtf_file, db_file=db_file, overwrite=False)

    # Step 2: Extract exon, CDS, 5'UTR, and 3'UTR annotations for all transcripts from the DB
    annotations_df = extract_all_gtf_annotations(db)
    # NOTE: columns = ['chrom', 'start', 'end', 'strand', 'feature', 'gene_id', 'transcript_id']

    # Step 3: Save annotations to a file
    annotations_df.to_csv(output_file, index=False, sep='\t')
    print(f"[info] Annotations extracted and saved to {output_file}")


def demo_extract_sequences(gtf_file, genome_fasta, gene_tx_map=None, output_file="tx_sequence.parquet"):
    from .utils_bio import (
        extract_transcripts_from_gtf,
        extract_transcript_sequences,
        save_sequences, 
        save_sequences_by_chromosome, 
        load_sequences, 
        load_sequences_by_chromosome
    )

    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    local_dir = "/path/to/meta-spliceai/data/ensembl"
    genome_annot = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf") 

    genome_fasta = os.path.join(
        data_prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") # "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    tx_ids = gene_tx_map  # Set to None to extract all transcripts
    # NOTE: To focus on specific genes and transcripts
    #    E.g. gene_names = {"STMN2", "UNC13A"}
    # 
    # Set tx_ids as a dictionary: 
    #
    # tx_ids = {'STMN2': ['ENST00000220876', ], 
    #           'UNC13A': ['ENST00000519716', ]}
    # or
    # 
    # tx_ids = {'ENSG00000104435': ['ENST00000220876'], 'ENSG00000130477': ['ENST00000519716']}  # Example with gene IDs

    print("[info] Extract transcript sequences")

    # Extract transcripts and their features
    gtf_file = genome_annot
    transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids=tx_ids, ignore_version=True)
    print(transcripts_df.head())  # Display the extracted transcript information

    additional_columns = ['seqname', 'start', 'end', 'strand', ]
    seq_df = extract_transcript_sequences(transcripts_df, genome_fasta, output_format='dataframe', include_columns=additional_columns)
    
    format = 'parquet'
    output_path = output_file  # os.path.join(local_dir, f"tx_sequence.{format}")
    assert os.path.exists( os.path.dirname(output_path))

    print(f"[i/o] Saving transcript sequences to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    
    
    # NOTE: Columns: ['transcript_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']
    # save_sequences(seq_df, output_path, format=format)
    save_sequences_by_chromosome(seq_df, output_path, format=format)

    return 


def test(): 

    prefix = '/path/to/meta-spliceai/data/ensembl/'
    gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path
    assert os.path.exists(gtf_file)
    db_file = os.path.join(prefix, "annotations.db")
    
    output_file = os.path.join(prefix, "annotations_all_transcripts.csv")
    demo_extract_annotations(gtf_file, db_file=db_file, output_file=output_file)

    # Load the annotations from the saved file
    annotations_df = pd.read_csv(output_file)
    print(annotations_df.head())

    prefix = "/path/to/meta-spliceai/data/ensembl/"
    genome_fasta = os.path.join(prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") # "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

    demo_extract_sequences(gtf_file, genome_fasta)




if __name__ == '__main__':
    test()
