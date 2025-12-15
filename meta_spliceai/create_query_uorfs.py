import os, sys
import shutil
import random, csv
import pandas as pd
import numpy as np
import meta_spliceai.system.sys_config as config

from meta_spliceai import TranscriptIO
from meta_spliceai.utils.utils_doc import (
    print_emphasized
)

from tqdm import tqdm
import gffutils

# Set a seed for reproducibility
# np.random.seed(0)

def parse_attributes_simple(attribute_field):
    """Parse the 'attribute' field of a GTF file."""
    attributes = {}
    for attribute in attribute_field.split(';'):
        if attribute:
            key, value = attribute.strip().split(' ')
            # Remove the double quotes around the value
            value = value.strip('"')
            attributes[key] = value
    return attributes

def parse_attributes(attribute_str):
    """
    Parse the attribute column from a GTF file into a dictionary of key-value pairs.

    Similar to parse_attributes_simple(), but remove keys do not have values. 

    Parameters:
    attribute_str (str): A string containing the attributes from a GTF file.

    Returns:
    attributes (dict): A dictionary with attribute keys and their corresponding values.
    """
    attributes = {}
    for attribute in attribute_str.split(';'):
        if attribute.strip():
            parts = attribute.strip().split(' ', 1)
            if len(parts) == 2:  # Ensure there is a value
                key, value = parts
                attributes[key] = value.strip('"')
    return attributes


def parse_attributes_gff3(attributes_str):
    """
    Parse the 'attributes' column of a GFF3 file.

    Parameters:
    attributes_str (str): The 'attributes' column of a GFF3 file.

    Returns:
    dict: A dictionary where the keys are attribute names and the values are the corresponding attribute values.
    """
    attributes = {}

    # Split the attributes string into a list of 'tag=value' strings
    tag_value_pairs = attributes_str.split(';')

    # For each 'tag=value' string
    for tag_value_pair in tag_value_pairs:
        # Split the 'tag=value' string into a tag and a value
        tag, value = tag_value_pair.split('=')

        # Add the tag and value to the dictionary
        attributes[tag] = value

    return attributes


# Function to create GTF entry
def create_gtf_entry(seqname, source, feature, start, end, score, strand, frame, attributes):
    attr_str = ' '.join([f'{key} "{value}";' for key, value in attributes.items()])
    return pd.Series([seqname, source, feature, start, end, score, strand, frame, attr_str], index=columns)

def get_transcript_biotype(row, default_biotype="unknown"):
    if pd.notnull(row['tx_type_ref']):
        return row['tx_type_ref']
    elif pd.notnull(row['tx_type_pred']):
        return row['tx_type_pred']
    else:
        return default_biotype

def create_query_uorfs_gtf_by_init_candidates(): 

    # To create the initial query_uorfs.gtf
    # analyze_gtex_uorf_connected_tx()
    # make_query_uorfs_gtf(use_src_id=False)

    # --- Load the provided query_uorfs.gtf file ---
    file_path = os.path.join("uORF_explorer/pub_results", "query_uorfs.gtf")
    columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    query_uorfs_df = pd.read_csv(file_path, sep='\t', comment='#', header=None, names=columns)

    # --- Load from the experimental data ---
    # Read the CSV files gtex_uORFconnected_txs.csv and gtex_uORFconnected_txs.w_utrs.csv
    # columns = ['transcript_id', 'reference_id', 'orf_id']
    # filepath = os.path.join("uORF_explorer/pub_results", "gtex_uORFconnected_txs.csv")
    # df_original = pd.read_csv(filepath)
    # print("> Columns(df_original): {}".format(list(df_original.columns)))
    # print("> shape(df_original): {}".format(df_original.shape))
    # print_unique_counts(df_original, columns, dataset="gtex_uORFconnected_txs")


    # Apply the parsing function to the 'attribute' column
    query_uorfs_df['parsed_attributes'] = query_uorfs_df['attribute'].apply(parse_attributes)
    query_uorfs_df['transcript_id'] = query_uorfs_df['parsed_attributes'].apply(lambda x: x.get('transcript_id'))
    query_uorfs_df['gene_id'] = query_uorfs_df['parsed_attributes'].apply(lambda x: x.get('gene_id'))
    query_uorfs_df['associated_gene'] = query_uorfs_df['parsed_attributes'].apply(lambda x: x.get('associated_gene'))

    # Create a new DataFrame to hold the augmented GTF entries
    augmented_gtf_df = pd.DataFrame(columns=columns)

    # Add entries for genes, transcripts, exons, and CDS
    for _, row in query_uorfs_df.iterrows():

        # Extract the existing transcript feature
        if row['feature'] == 'transcript':
            transcript_id = row['transcript_id']
            gene_id = row['gene_id']
            associated_gene = row['associated_gene']
            seqname = row['seqname']
            start = row['start']
            end = row['end']
            strand = row['strand']
            score = '.'
            frame = '.'
            source = 'uORFExplorer'
            
            # Add gene entry
            gene_attributes = {
                'gene_id': gene_id,
                'gene_name': associated_gene,
                'gene_biotype': 'protein_coding'  # Example biotype, adjust as needed
            }
            gene_entry = create_gtf_entry(seqname, source, 'gene', start, end, score, strand, frame, gene_attributes)
            augmented_gtf_df = augmented_gtf_df.append(gene_entry, ignore_index=True)
            
            # Add transcript entry
            transcript_attributes = row['parsed_attributes']  # Preserve all original attributes
            transcript_entry = create_gtf_entry(seqname, source, 'transcript', start, end, score, strand, frame, transcript_attributes)
            augmented_gtf_df = augmented_gtf_df.append(transcript_entry, ignore_index=True)
            
            # Add exon entry
            exon_attributes = {
                'transcript_id': transcript_id,
                'gene_id': gene_id,
                'exon_number': '1'  # Example exon number, adjust as needed
            }
            exon_entry = create_gtf_entry(seqname, source, 'exon', start, end, score, strand, frame, exon_attributes)
            augmented_gtf_df = augmented_gtf_df.append(exon_entry, ignore_index=True)
            
            # Add CDS entry
            cds_attributes = {
                'transcript_id': transcript_id,
                'gene_id': gene_id,
                'protein_id': transcript_id  # Example protein ID, adjust as needed
            }
            cds_entry = create_gtf_entry(seqname, source, 'CDS', start, end, score, strand, '0', cds_attributes)
            augmented_gtf_df = augmented_gtf_df.append(cds_entry, ignore_index=True)

    # Save the augmented GTF file
    augmented_gtf_df.to_csv('augmented_query_uorfs.gtf', sep='\t', header=False, index=False)

    return augmented_gtf_df

def df_to_gff3(df, output_file):
    """
    Write a DataFrame to a GFF3 file.

    Parameters:
    df (pandas.DataFrame): DataFrame to write.
    output_file (str): Path to the output GFF3 file.

    Returns:
    None
    """
    # Define the GFF3 header
    gff3_header = "##gff-version 3\n"

    # Open the output file
    with open(output_file, 'w') as f:
        # Write the GFF3 header
        f.write(gff3_header)

        # Write the DataFrame to the file
        for _, row in df.iterrows():
            # Convert the 'attributes' dictionary back into a string
            attributes_str = ';'.join(f'{k}={v}' for k, v in row['parsed_attributes'].items())

            # Write the row to the file
            f.write('\t'.join(str(x) for x in row[:8]) + '\t' + attributes_str + '\n')


def filter_gff3_by_transcript_ids(file_path, transcript_ids, output_file):
    """
    Filter a GFF3 file to only include entries with certain transcript IDs.

    Parameters:
    file_path (str): Path to the input GFF3 file.
    transcript_ids (list of str): List of transcript IDs to include.
    output_file (str): Path to the output GFF3 file.

    Returns:
    None

    Example: 

        transcript_ids_of_interest = [...]  # Replace with your list of transcript IDs
        filter_gff3_by_transcript_ids("gencode.v43.annotation.gff3", 
                                      transcript_ids_of_interest, 'filtered_gencode.v43.annotation.gff3')
    """
    # Define the column names for a GFF3 file
    columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]

    # Load the GFF3 file
    gencode_df = pd.read_csv(file_path, sep='\t', comment='#', header=None, names=columns)

    # Parse the 'attribute' column to extract 'transcript_id'
    gencode_df['parsed_attributes'] = gencode_df['attribute'].apply(parse_attributes_gff3)
    gencode_df['transcript_id'] = gencode_df['parsed_attributes'].apply(lambda x: x.get('transcript_id'))

    # Take a sample subset of gencode_df['transcript_id']
    num_samples = min(10, len(gencode_df['transcript_id']))
    sample_transcript_ids = gencode_df['transcript_id'].sample(n=num_samples)
    print("> Example transcript IDs:\n{}\n".format(sample_transcript_ids))
    print("> Input transcript IDs (n={}):\n{}\n".format(len(transcript_ids), transcript_ids[:10]))

    # Filter the DataFrame to only include rows where 'transcript_id' is in the list of transcript IDs
    filtered_gencode_df = gencode_df[gencode_df['transcript_id'].isin(transcript_ids)]

    assert not filtered_gencode_df.empty, f"No entries found for the specified transcript IDs (n={len(transcript_ids)})"

    # Todo: Save the filtered DataFrame as a new GFF3 file
    # filtered_gencode_df.to_csv(output_file, sep='\t', header=False, index=False)
    df_to_gff3(filtered_gencode_df, output_file)

    return 


def subset_gff3_by_transcript_ids(gff3_file, transcript_ids, gene_ids=None, remove_comments=True, return_ids=False):
    """
    Subset a GFF3 file to include only entries for specified transcript IDs and their associated gene IDs.

    Parameters:
    gff3_file (str): Path to the input GFF3 file.
    transcript_ids (set): A set of transcript IDs to include in the output.
    remove_comments (bool): Whether to remove comment lines.
    
    Returns:
    list: A list of filtered GFF3 entries.
    """
    filtered_entries = []
    associated_gene_ids = set()
    associated_tx_ids = set()

    # First, count the total number of lines in the file
    total_lines = sum(1 for _ in open(gff3_file, 'r'))

    # First pass to collect gene_ids for the specified transcript_ids
    if gene_ids is None: 
        with open(gff3_file, 'r') as infile:
            lines = infile.readlines()
            for line in tqdm(lines, desc="Collecting gene IDs", total=total_lines):
                if line.startswith("#"):
                    continue
                
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue
                
                attributes_str = fields[8]
                attributes = dict(attr.strip().split('=') for attr in attributes_str.split(';') if '=' in attr)
                
                transcript_id = attributes.get("transcript_id") # or attributes.get("ID")
                gene_id = attributes.get("gene_id")
                
                if transcript_id in transcript_ids:
                    if gene_id:
                        associated_gene_ids.add(gene_id)
    else: 
        associated_gene_ids = set(gene_ids)

    print("[subset] Find n={} associated gene IDs for the specified transcript IDs.".format(len(associated_gene_ids)))
    
    # Second pass to filter entries
    with open(gff3_file, 'r') as infile:
        msg = f"Filtering entries by n={len(associated_gene_ids)} gene IDs and m={len(transcript_ids)} tx IDs"
        for line in tqdm(infile, desc=msg, total=total_lines):
            if line.startswith("#"):
                if remove_comments:
                    continue
                else:
                    filtered_entries.append(line)
                    continue
            
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            
            attributes_str = fields[8]
            attributes = dict(attr.strip().split('=') for attr in attributes_str.split(';') if '=' in attr)
            
            transcript_id = attributes.get("transcript_id") # or attributes.get("ID")
            gene_id = attributes.get("gene_id")
            
            if gene_id in associated_gene_ids and fields[2] == 'gene':  # Assuming a standard annotation file has gene feature
                filtered_entries.append(line)

            elif transcript_id in transcript_ids:
                filtered_entries.append(line)
                associated_tx_ids.add(transcript_id)

    print("[subset] Find n={} associated transcript IDs for the specified transcript IDs.".format(len(associated_tx_ids)))
    
    if return_ids:
        return filtered_entries, associated_gene_ids, associated_tx_ids
    return filtered_entries


def augment_gff3_for_query_uorfs(entries, return_ids=False):
    """
    Augment GFF3 entries to include necessary attributes for query_uorfs.gtf.

    Parameters:
    entries (list): List of GFF3 entries to be augmented.
    
    Returns:
    list: A list of augmented GFF3 entries.
    """
    augmented_entries = []
    gene_ids = set()
    transcript_ids = set()

    # Initialize a dictionary to store the counts
    feature_counts = {'gene': 0, 'CDS': 0, 'exon': 0, 'transcript': 0}
    # transcript_count = Counter()

    for line in tqdm(entries, desc="Processing entries"):
        if line.startswith("#"):
            augmented_entries.append(line)
            continue
        
        fields = line.strip().split("\t")
        if len(fields) < 9:
            continue
        
        feature = fields[2]
        attributes_str = fields[8]
        attributes = dict(attr.strip().split('=') for attr in attributes_str.split(';') if '=' in attr)

        # Increment the count for the feature
        if feature in feature_counts:
            feature_counts[feature] += 1
        
        if feature == 'transcript':
            attributes['associated_gene'] = attributes.get('gene_name', 'unknown')
            attributes['transcript_biotype'] = attributes.get('transcript_type', 'unknown')
            attributes['reference_id'] = attributes['transcript_id']
            transcript_ids.add(attributes['transcript_id'])

        if feature == 'gene':
            attributes['gene_biotype'] = attributes.get('gene_type', 'unknown')
            gene_ids.add(attributes['gene_id'])

        if feature in ['CDS', 'exon', ]: 
            attributes['reference_id'] = attributes['transcript_id']    
        
        attributes_str = ';'.join([f'{key}={value}' for key, value in attributes.items()])
        fields[8] = attributes_str
        augmented_entries.append('\t'.join(fields) + '\n')

    # Print the counts
    for feature, count in feature_counts.items():
        print(f"(augment_gff3) Number of rows associated with feature '{feature}': {count}")
    
    if return_ids: 
        return augmented_entries, gene_ids, transcript_ids

    return augmented_entries


def write_gff_lines_as_gtf(entries, output_file):
    """
    Save GFF3 entries as a GTF file.

    Parameters:
    entries (list): List of GFF3 entries to be saved.
    output_file (str): Path to the output GTF file.
    """
    with open(output_file, 'w') as outfile:
        for line in entries:
            if line.startswith("#"):
                outfile.write(line)
                continue
            
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            
            attributes_str = fields[8]
            attributes = dict(attr.strip().split('=') for attr in attributes_str.split(';') if '=' in attr)
            
            attributes_str = ' '.join([f'{key} "{value}";' for key, value in attributes.items()])
            fields[8] = attributes_str
            outfile.write('\t'.join(fields) + '\n')
    print("> Saved the augmented GFF3 entries as a GTF file: {}\n".format(output_file))
    return

def create_gffutils_db(gtf_file, db_file):
    """
    Create a gffutils database from a GTF file.

    Parameters:
    gtf_file (str): Path to the input GTF file.
    db_file (str): Path to the output gffutils database file.
    """
    gffutils.create_db(gtf_file, db_file, id_spec="transcript_id", merge_strategy="create_unique", keep_order=True)

def verify_gtf_file(gtf_file_path):
    """
    Verify that a GTF file is valid and contains the required features.

    Parameters:
    gtf_file_path (str): Path to the GTF file.

    Returns:
    None
    """
    import re

    # Create a GTF database
    db = gffutils.create_db(gtf_file_path, ':memory:')

    # Define the required features
    required_features = ['gene', 'transcript', 'CDS', 'exon']

    # Check if the GTF file contains the required features
    for feature in required_features:
        try:
            # Get the first two features of the current type
            features = [next(db.features_of_type(feature)) for _ in range(2)]
            print_emphasized(f"\nThe GTF file contains '{feature}' features. Here are a couple of examples:")
            for feat in features:
                print(feat)

                # Check the format of the coordinates
                if not isinstance(feat.start, int) or not isinstance(feat.end, int):
                    print(f"Warning: Invalid coordinates in feature {feat.id}.")

                # Check the strand
                if feat.strand not in ['+', '-']:
                    print(f"Warning: Invalid strand in feature {feat.id}.")

                # Check the frame
                if feat.frame not in ['0', '1', '2', '.']:
                    print(f"Warning: Invalid frame in feature {feat.id}.")

                # Check the format of the attributes
                for key, values in feat.attributes.items():
                    for value in values:
                        if not re.match(r'^\w+ ".*"$', f"{key} \"{value}\""):
                            print(f"Warning: Invalid attribute format in feature {feat.id}.")

                # Check required attributes
                required_attributes = ['gene_id', 'transcript_id'] if feature != 'gene' else ['gene_id']
                for attr in required_attributes:
                    if attr not in feat.attributes:
                        print(f"Warning: Missing required attribute {attr} in feature {feat.id}.")
                # NOTE: The transcript_id attribute is required for transcript features, but not for gene features.

        except StopIteration:
            print(f"The GTF file does not contain any '{feature}' features.")


def verify_gtf_file_v0(gtf_file_path):
    """
    Verify that a GTF file is valid and contains the required features.

    Parameters:
    gtf_file_path (str): Path to the GTF file.

    Returns:
    None
    """
    # Create a GTF database
    db = gffutils.create_db(gtf_file_path, ':memory:')

    # Define the required features
    required_features = ['gene', 'transcript', 'CDS', 'exon']

    # Check if the GTF file contains the required features
    for feature in required_features:
        try:
            # Get the first two features of the current type
            features = [next(db.features_of_type(feature)) for _ in range(2)]
            print(f"The GTF file contains '{feature}' features. Here are a couple of examples:")
            for feat in features:
                print(feat)
        except StopIteration:
            print(f"The GTF file does not contain any '{feature}' features.")


def verify_gtf_file_v1(gtf_file_path):
    import re 
    
    # Create a GTF database
    db = gffutils.create_db(gtf_file_path, ':memory:')

    # Define the required features
    required_features = ['gene', 'transcript', 'CDS', 'exon']

    # Check if the GTF file contains the required features
    for feature in required_features:
        try:
            # Get the first two features of the current type
            features = [next(db.features_of_type(feature)) for _ in range(2)]
            print(f"The GTF file contains '{feature}' features. Here are a couple of examples:")
            for feat in features:
                print(feat)
                # Check the values of each attribute
                for key, values in feat.attributes.items():
                    for value in values:
                        # Check if the value is double quoted
                        if not re.match(r'^".*"$', value):
                            print(f"Warning: The value of the '{key}' attribute is not double quoted.")
                        # Check if the value is over-quoted
                        elif re.match(r'^".*".*"$', value):
                            print(f"Warning: The value of the '{key}' attribute is over-quoted.")
        except StopIteration:
            print(f"The GTF file does not contain any '{feature}' features.")


def update_gtf_attributes_with_dict(input_gtf, output_gtf, update_dict):
    """
    Update the GTF attributes to ensure compatibility with the specified attributes in the update_dict.

    Parameters:
    input_gtf (str): Path to the input GTF file.
    output_gtf (str): Path to the output GTF file.
    update_dict (dict): Dictionary with attribute names as keys and desired values as values.

    Returns:
    list: List of new attributes that were added.
    """
    new_attributes = set()

    with open(input_gtf, 'r') as infile, tempfile.NamedTemporaryFile('w', delete=False) as outfile:
        # Get the total number of lines in the file for the progress bar
        total_lines = sum(1 for line in infile)
        infile.seek(0)  # Reset the file pointer to the start of the file

        for line in tqdm(infile, total=total_lines, desc="Processing lines"):
            if line.startswith("#"):
                outfile.write(line)
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                outfile.write(line)
                continue

            attributes_str = fields[8]
            attributes = dict(attr.strip().split(' ', 1) for attr in attributes_str.strip(';').split(';') if ' ' in attr)

            # Update or add attributes based on the update_dict
            for key, new_value_key in update_dict.items():
                if new_value_key in attributes:
                    value = attributes[new_value_key]
                    if key not in attributes:
                        new_attributes.add(key)
                    attributes[key] = value

            # Reconstruct the attributes string without over-quoting and trailing semicolons
            attributes_list = []
            for key, value in attributes.items():
                if value:
                    value = value.replace('\"', '')
                    attribute_str = f'{key} "{value}";'
                    attributes_list.append(attribute_str)
            attributes_str = ' '.join(attributes_list).rstrip(' ;')
            fields[8] = attributes_str

            outfile.write('\t'.join(fields) + '\n')

    # Replace the original file with the temporary file
    shutil.move(outfile.name, output_gtf)

    return list(new_attributes)


def read_gtf_into_dataframe_v0(gtf_file_path):
    """
    Read a GTF file into a DataFrame and parse the 'attribute' column into separate columns.

    Parameters:
    gtf_file_path (str): Path to the GTF file.

    Returns:
    df (DataFrame): DataFrame with the GTF data, where keys in the 'attribute' column have become new columns.
    """
    # Load the GTF file into a DataFrame
    df = pd.read_csv(gtf_file_path, sep='\t', header=None, comment='#')

    # Define the reserved GTF columns
    gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    df.columns = gtf_columns

    # Parse the 'attribute' column into a DataFrame
    attributes_df = df['attribute'].str.split('; ', expand=True).apply(lambda s: s.str.split(' ', 1, expand=True))

    # Display the first few rows of the DataFrame
    print(attributes_df.head())
    
    # Remove any keys that do not have corresponding values
    attributes_df = attributes_df[attributes_df[1].notna()]
    
    attributes_df[0] = attributes_df[0].str.strip()
    attributes_df[1] = attributes_df[1].str.strip('"')

    # Rename the columns
    attributes_df.columns = ['key', 'value']

    # Pivot the DataFrame to have keys as columns
    attributes_df = attributes_df.pivot(columns='key', values='value')

    # Concatenate the original DataFrame with the attributes DataFrame
    df = pd.concat([df, attributes_df], axis=1)

    return df

def subsample_by_attribute(df, transcript_of_interest, key='reference_id', return_values=False):

    for r, row in df.iterrows():
        pass

    # Create a boolean mask where True indicates the row's attribute contains a key with a value in transcript_of_interest
    mask = df['attribute'].apply(lambda attr: attr.get(key, None) in transcript_of_interest)
    
    # Subset the DataFrame based on the mask
    subsampled_df = df[mask]
    
    if return_values:
        # If requested, return the unique values of the key within the subsampled DataFrame
        unique_values = subsampled_df['attributes'].apply(lambda attr: attr.get(key, None)).unique()
        return subsampled_df, unique_values
    
    return subsampled_df


def subset_rows_by_ids(df, id_set, key='reference_id', feature_type='transcript', return_values=False):
    if feature_type is not None:
        df = df[df['feature'] == feature_type]

    if df.empty: 
        print(f"No '{feature_type}' features found in the DataFrame.")
        return df
    
    assert key in ['reference_id', 'gene_id', 'transcript_id', 'orf_id', 'ID'] 

    # Optimized step to extract both reference_id and gene_id in one go
    def extract_ids(attr):
        return attr.get('reference_id', None), attr.get('gene_id', None)

    df[['reference_id', 'gene_id']] = df['attribute'].apply(lambda attr: extract_ids(attr)).apply(pd.Series)

    df_subset = df[df[key].isin(id_set)]
    return df_subset


def subset_gtf_by_ids_v0(df, transcript_ids_of_interest, gene_ids_of_interest):

    df_gene = subset_rows_by_ids(df, gene_ids_of_interest, key='gene_id', feature_type='gene', return_values=False)
    df_tx = subset_rows_by_ids(df, transcript_ids_of_interest, key='reference_id', feature_type='transcript', return_values=False)
    df_cds = subset_rows_by_ids(df, transcript_ids_of_interest, key='reference_id', feature_type='CDS', return_values=False)
    df_exon = subset_rows_by_ids(df, transcript_ids_of_interest, key='reference_id', feature_type='exon', return_values=False)

    return pd.concat([df_gene, df_tx, df_cds, df_exon])


def subsample_by_reference_ids(input_gtf, n=50, id_spec=None, return_ids=True, transcript_ids_as_reference=False):

    if not isinstance(input_gtf, pd.DataFrame):
        df = read_gtf_into_dataframe(input_gtf)  # attribute_as_type='dict' by default
    else:
        df = input_gtf

    if id_spec is None:
        id_spec = {'gene': 'gene_id', 
                   'transcript': 'reference_id', 'CDS': 'reference_id', 'exon': 'reference_id'}

    if transcript_ids_as_reference:
        df = add_reference_id_customized(df)  # Todo: Re-usability

    # Optimized step to extract both reference_id and gene_id in one go
    def extract_ids(attr):
        return attr.get('reference_id', None), attr.get('gene_id', None)

    # Factor out the extraction of reference_id and gene_id to serve as columns in the DataFrame
    df[['reference_id', 'gene_id']] = df['attribute'].apply(lambda attr: extract_ids(attr)).apply(pd.Series)
    # NOTE: The output of the apply(lambda attr: extract_ids(attr)) is a Series where each element is a tuple
    #       Applying pd.Series to this Series converts it into a DataFrame where each tuple is split into 
    #       its components, with each component becoming a column. 
    #       This results in a temporary DataFrame where the first column contains reference_ids and 
    #       the second column contains gene_ids extracted from the

    ref_ids = df['reference_id'].unique()
    sampled_tx_ids = np.random.choice(ref_ids, min(n, len(ref_ids)), replace=False)
    
    subsampled_df = df[df['reference_id'].isin(sampled_tx_ids)]
    sampled_gene_ids = subsampled_df['gene_id'].unique()

    sampled_ids = {'gene': sampled_gene_ids, 
                   'transcript': sampled_tx_ids, 
                   'CDS': sampled_tx_ids, 
                   'exon': sampled_tx_ids}

    final_df, used_ids = subset_gtf_by_ids(df, id_spec, sampled_ids, return_ids=True)

    # If you want to return the IDs as well, you can return them along with the DataFrame
    if return_ids:
        return final_df, sampled_tx_ids, sampled_gene_ids
    else:
        return final_df

def subset_gtf_by_ids(input_gtf, id_spec, sampled_ids, return_ids=False):
    """
    Subsets a DataFrame based on given IDs for specified features.

    Parameters:
    - df (DataFrame): The DataFrame to filter.
    - id_spec (dict): A dictionary where keys are features and values are the ID attributes to filter by.
    - sampled_ids (dict): A dictionary of lists where keys match the id_spec keys and values are lists of IDs to keep.
    - return_ids (bool): Whether to return the used IDs along with the DataFrame.

    Returns:
    - DataFrame: The filtered DataFrame.
    - (optional) dict: The IDs used for filtering, if return_ids is True.
    """

    if not isinstance(input_gtf, pd.DataFrame):
        df = read_gtf_into_dataframe(input_gtf)
    else:
        df = input_gtf

    filtered_dfs = []
    for feature, id_attr in id_spec.items():
        if feature in sampled_ids:
            # Filter the DataFrame based on the feature and corresponding ID attribute
            filtered_df = df[(df['feature'] == feature) & (df[id_attr].isin(sampled_ids[feature]))]
            filtered_dfs.append(filtered_df)

    # Concatenate the filtered DataFrames
    final_subsampled_df = pd.concat(filtered_dfs)

    # Sort the concatenated DataFrame by index to maintain original row order
    final_subsampled_df = final_subsampled_df.sort_index()

    if return_ids:
        return final_subsampled_df, sampled_ids
    else:
        return final_subsampled_df
    

def count_attribute(df, key, feature_type='transcript', return_values=False):
    """
    Count the number of occurrences of each key for a given attribute represented as dictionaries.

    Parameters:
    df (DataFrame): DataFrame with the GTF data.
    attribute_name (str): Name of the attribute to count.

    Returns:
    Series: Series with the attribute values as the index and the counts as the values.
    """
    df = df[df['feature'] == feature_type]

    unique_count = 0
    unique_values = np.array([])
    if df.empty:
        print(f"No '{feature_type}' features found in the DataFrame.")
        unique_count = 0
    else: 
        if 'attribute' not in df.columns:
            msg = f"No 'attribute' column found in the DataFrame." 
            raise ValueError(msg)

        unique_values = df['attribute'].apply(lambda x: x.get(key))
        unique_count = unique_values.nunique()
        
    return unique_count if not return_values else list(unique_values.unique()) 

def convert_attribute(df, dtype='dict'):
    if dtype.startswith('dict'):
        # Convert from string to dictionary
        def parse_attributes(attribute_str):
            attributes = {}
            for attribute in attribute_str.split(';'):
                if attribute.strip():
                    parts = attribute.strip().split(' ', 1)
                    if len(parts) == 2:  # Ensure there is a value
                        key, value = parts
                        attributes[key] = value.strip('"')
            return attributes

        df['attribute'] = df['attribute'].apply(parse_attributes)
    else:
        # Convert from dictionary to string
        def assemble_attributes(attributes):
            attribute_strs = []
            for key, value in attributes.items():
                attribute_strs.append(f'{key} "{value}"')
            return '; '.join(attribute_strs)

        df['attribute'] = df['attribute'].apply(assemble_attributes)

    return df

def read_gtf_into_dataframe(gtf_file_path, attribute_as_type='dict', feature_types=None):
    """
    Read a GTF file into a DataFrame and parse the 'attribute' column into separate columns.

    Parameters:
    gtf_file_path (str): Path to the GTF file.
    feature_types (list of str, optional): List of feature types to include in the DataFrame.
    attribute_as_type (str, optional): Whether to parse the 'attribute' column as a dictionary, 
        as a string, or as separate columns.

    Returns:
    df (DataFrame): DataFrame with the GTF data, where keys in the 'attribute' column have become new columns.
    """
    # Load the GTF file into a DataFrame
    df = pd.read_csv(gtf_file_path, sep='\t', header=None, comment='#')

    # Define the reserved GTF columns
    gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    df.columns = gtf_columns

    # If feature types are specified, filter the DataFrame to include only these feature types

    if isinstance(feature_types, str):
        feature_types = [feature_types]

    if feature_types is not None:
        df = df[df['feature'].isin(feature_types)]

    # Parse the 'attribute' column into a list of dictionaries
    def parse_attributes(attribute_str):
        attributes = {}
        for attribute in attribute_str.split(';'):
            if attribute.strip():
                parts = attribute.strip().split(' ', 1)
                if len(parts) == 2:  # Ensure there is a value
                    key, value = parts
                    attributes[key] = value.strip('"')
        return attributes

    if attribute_as_type.startswith('dict'): 
        # Apply the parsing function to the 'attribute' column
        df['attribute'] = df['attribute'].apply(parse_attributes)

    elif attribute_as_type.startswith('str'):
        pass
        # pass 
    else: 
        # Todo: Some how the cancatnation results in null values

        if feature_types is not None and len(feature_types) == 1:
            
            # Apply the parsing function to the 'attribute' column
            attributes_list = df['attribute'].apply(parse_attributes)  # List of dictionaries in the perform of Series
            print("> Example attributes: {}".format(attributes_list.iloc[0]))

            # Convert the list of dictionaries into a DataFrame
            attributes_df = pd.DataFrame(attributes_list.tolist())

            # Check if 'transcript_id' column contains NaN values
            if 'transcript_id' in attributes_df.columns:
                contains_nan = attributes_df['transcript_id'].isna().any()
                print("[test] Does 'transcript_id' contain NaN values? ", contains_nan)
                print("... Number of NaN values in 'transcript_id': {}".format(attributes_df['transcript_id'].isna().sum()))

            print("[test] columns(df): {}".format(list(df.drop(columns=['attribute']))))

            # Concatenate the original DataFrame with the attributes DataFrame
            df = pd.concat([df.drop(columns=['attribute']), attributes_df], axis=1)

            # Check if each column contains NaN values
            for column in df.columns:
                if df[column].isnull().any():
                    print(f"> Column '{column}' contains NaN values. Examples:")
                    print(df[df[column].isnull()].head())

        else: 
            msg = f"Please specify a single feature type (given {feature_types})"
            raise ValueError(msg)

    return df

def dict_to_attributes(attributes_dict):
    """
    Convert attributes dictionary to a GTF/GFF3 attributes string.
    """
    return '; '.join(f'{key} "{value}"' for key, value in attributes_dict.items()) + ';'

def write_dataframe_into_gtf_v0(df, gtf_file_path):
    """
    Write a DataFrame with attributes represented as dictionaries into a GTF file.
    """
    gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    df['attribute'] = df['attribute'].apply(dict_to_attributes)
    df.to_csv(gtf_file_path, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)


def write_dataframe_into_gtf(df, gtf_file_path, attribute_columns=None, attributes_as_dict=True):
    """
    Write a DataFrame into a GTF file, using the specified columns as attributes.

    Parameters:
    df (DataFrame): DataFrame with the GTF data.
    gtf_file_path (str): Path to the GTF file to write.
    attribute_columns (list of str, optional): List of column names to include in the 'attribute' field of the GTF file.
    attributes_as_dict (bool, optional): Whether the 'attribute' column is represented as a dictionary.
    """
    # Define the reserved GTF columns
    gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    if attributes_as_dict:
        df['attribute'] = df['attribute'].apply(dict_to_attributes)
    else:
        # If attribute columns are not specified, infer them from the DataFrame
        if attribute_columns is None:
            attribute_columns = [col for col in df.columns if col not in gtf_columns]

        # If the 'attribute' column exists in the DataFrame, remove it
        if 'attribute' in df.columns: 
            df = df.drop(columns='attribute')

        # Create the 'attribute' column by joining the attribute columns with ' ' and '; '
        def format_attributes(row):
            attributes = []
            for col in attribute_columns:
                if pd.notna(row[col]):
                    attributes.append(f'{col} "{row[col]}"')
            return '; '.join(attributes) + ';'

        df['attribute'] = df.apply(format_attributes, axis=1)

    # Ensure the DataFrame has all the necessary columns in the correct order
    for col in gtf_columns:
        if col not in df.columns:
            df[col] = '.'  # Add missing columns with default GTF value

    df = df[gtf_columns]

    # Write the DataFrame to a GTF file
    df.to_csv(gtf_file_path, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)
    # NOTE: The quoting=csv.QUOTE_NONE argument to to_csv ensures that the output is not quoted,
    #       since the attribute values are already quoted.

def test_features_presence(df):
    """
    Test that each unique combination of reference_id, transcript_id, orf_id in the DataFrame
    has rows for transcript, gene, CDS, and exon features.

    Parameters:
    df (DataFrame): DataFrame with the GTF data.
    
    Returns:
    bool: True if all combinations have the required features, False otherwise.
    missing_features (dict): Dictionary of missing features for each combination.
    """
    # Define the genomic features of interest
    required_features = ['transcript', 'CDS', 'exon']
    
    # Group the DataFrame by 'reference_id', 'transcript_id', and 'orf_id'
    grouped = df.groupby(['reference_id', 'transcript_id', 'orf_id'])
    
    # Initialize a dictionary to track missing features
    missing_features = {}

    # Iterate through each group and check for required features
    for (reference_id, transcript_id, orf_id), group in grouped:
        features_present = group['feature'].unique()
        missing = [feature for feature in required_features if feature not in features_present]
        
        if missing:
            missing_features[(reference_id, transcript_id, orf_id)] = missing

    if missing_features:
        print("Missing features detected for some combinations:")
        for key, value in missing_features.items():
            print(f"Combination {key} is missing features: {value}")
        return False, missing_features
    else:
        print("All combinations have the required features.")
        return True, missing_features

def display_example_rows(gtf_df, columns_to_display=None):
    """
    Display example rows associated with each genomic feature (transcript, gene, CDS, exon) in a DataFrame derived from a GTF file.

    Parameters:
    gtf_df (DataFrame): DataFrame with the GTF data.
    columns_to_display (list of str, optional): List of columns to display. Defaults to displaying all columns.
    """
    # Define the genomic features of interest
    genomic_features = ['transcript', 'gene', 'CDS', 'exon']
    
    # If no specific columns are provided, default to displaying all columns
    if columns_to_display is None:
        columns_to_display = gtf_df.columns
    
    # Adjust pandas display settings to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    # Group the DataFrame by 'reference_id', 'transcript_id', and 'orf_id'
    grouped = gtf_df.groupby(['reference_id', 'transcript_id', 'orf_id'])
    
    # Initialize a dictionary to store example rows for each feature
    example_rows = {feature: [] for feature in genomic_features}

    # Iterate through each group
    for (reference_id, transcript_id, orf_id), group in grouped:
        for feature in genomic_features:
            # Filter the group for the current feature and take the first example row
            feature_row = group[group['feature'] == feature].head(1)
            if not feature_row.empty:
                example_rows[feature].append(feature_row[columns_to_display])

    # Concatenate example rows for each feature into a DataFrame and display them
    for feature, rows in example_rows.items():
        if rows:
            feature_df = pd.concat(rows)
            print(f"Example rows for {feature} feature:")
            print(feature_df)
            print("\n")


def get_unique_attribute_keys(df):
    """
    Get the unique keys that appear in the 'attribute' column of a DataFrame.

    Parameters:
    df (DataFrame): DataFrame with a column named 'attribute' that contains dictionaries.

    Returns:
    unique_keys (array): Array of unique keys that appear in the 'attribute' column.
    """
    # Get the keys of each dictionary in the 'attribute' column
    keys = df['attribute'].apply(lambda x: list(x.keys()))

    # Flatten the list of keys
    keys = [key for sublist in keys for key in sublist]

    # Get the unique keys
    unique_keys = pd.unique(keys)

    return unique_keys


def check_keys_in_attributes(df, keys):
    """
    Check if specific keys are in each dictionary in the 'attribute' column of a DataFrame.

    Parameters:
    df (DataFrame): DataFrame with a column named 'attribute' that contains dictionaries.
    keys (list): The keys to check for.

    Returns:
    result (dict): Dictionary with each key and a boolean value indicating if the key is in each dictionary.
    """
    if isinstance(keys, str):
        keys = [keys]
    assert isinstance(keys, list), "keys must be a list of strings."
    result = {key: df['attribute'].apply(lambda x: key in x).all() for key in keys}

    return result


def add_reference_id(df, id_spec=None):
    """
    Add a 'reference_id' key to each dictionary in the 'attribute' column of a DataFrame if it doesn't already exist.

    Parameters:
    df (DataFrame): DataFrame with a column named 'attribute' that contains dictionaries.

    Returns:
    df (DataFrame): DataFrame with the potentially updated 'attribute' column.
    """
    df.loc[:, 'attribute'] = df['attribute'].apply(lambda x: 
        {**x, 'reference_id': x['transcript_id']} 
            if 'reference_id' not in x else x)
    # This checks if 'reference_id' is not in the dictionary x, and if so, it adds it with the value 
    # of x['transcript_id']. If 'reference_id' is already present, it leaves x unchanged.

    # NOTE: **x is using the dictionary unpacking operator to include all the key-value pairs 
    #       from x in the new dictionary
    #       Included additionally in this new dictionary is a new key-value pair 
    #       with 'reference_id' as the key and x['transcript_id'] as the value.

    return df

def add_reference_id_customized(df, id_spec=None):

    if id_spec is None:
        id_spec = {
            'gene': 'gene_id',
            'transcript': 'reference_id', 
            'CDS': 'reference_id', 
            'exon': 'reference_id'
            }

    # Ensure df has a unique index
    df = df.reset_index(drop=True)

    dfs = split_dataframe_by_feature(df)

    dfx = []
    for feature in ['transcript',  'CDS', 'exon', 'gene' ]: 

        if feature == 'gene':
            # gene feature is not associated with a reference ID (with transcript-level identity)
            dfx.append(dfs[feature])
        else: 
            dfi = add_reference_id(dfs[feature])  # using 'transcript_id' as the reference_id
            dfx.append(dfi)

    # Concatenate and then sort by index to maintain the original order
    result_df = pd.concat(dfx).sort_index()

    return result_df

def print_unique_values(df, feature_type, keys=['reference_id', 'transcript_id', 'orf_id']):
    # Filter for the given feature type
    df_filtered = df[df['feature'] == feature_type]

    # --- Test --- 
    # Check if 'attribute' exist in the DataFrame, if not, what columns does it have? 
    if 'attribute' not in df.columns:
        print(f"Columns in the DataFrame: {list(df.columns)}")

    # Check unique values
    for key in keys:
        unique_values = df_filtered['attribute'].apply(lambda x: x.get(key))
        unique_count = unique_values[unique_values.notnull()].nunique()
        print(f"Number of unique {key} in '{feature_type}': {unique_count}")


def split_dataframe_by_feature(df, feature_types=None):
    """
    Split a DataFrame into sub-dataframes based on feature types.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    feature_types (list): A list of feature types to split the DataFrame by.

    Returns:
    dict: A dictionary where the keys are the feature types and the values are the corresponding sub-dataframes.
    """
    if feature_types is None:
        feature_types = df['feature'].unique()
    return {feature: df[df['feature'] == feature] for feature in feature_types}


def update_reference_by_matching_ids(df, df_ref, 
        feature_type='transcript', key='reference_id', raise_exception=False):   
    """
    Update attributes in specific rows in reference (df_ref) using attributes in the input (df). 
    Input dataframe represents a standard gene annotation whereas 
    the reference dataframe represents a custom, domain-specific annotation.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    df_ref (pandas.DataFrame): The reference DataFrame.
    feature (str): The feature to update.
    key (str): The key that connects df and df_ref.

    Returns:
    pandas.DataFrame: The updated DataFrame.
    """
    df = df[df['feature'] == feature_type]
    df_ref = df_ref[df_ref['feature'] == feature_type]

    # df['key_exists'] = df['attribute'].apply(lambda x: key in x)
    df.loc[:, 'key_exists'] = df['attribute'].apply(lambda x: key in x)
    all_have_key = df['key_exists'].all()

    if not all_have_key: 
        M = (~df['attribute'].apply(lambda x: key in x)).sum()
        print(f"[info] Number of rows without the key '{key}': {M}/({df.shape[0]})")

        msg = f"[info] Foreign key ({key}) does not exist in the input DataFrame. No operation performed."
        if raise_exception: 
            raise ValueError(msg)
        else: 
            print(msg)

        return df_ref

    # Find all reference_ids in df
    print(f"[info] Usnig key={key} to match rows in the input and reference DataFrames.")
    # df[key] = df['attribute'].apply(lambda x: x.get(key)) # assuming that the key is present in the attribute
    # df_ref[key] = df_ref['attribute'].apply(lambda x: x.get(key))
    df.loc[:, key] = df['attribute'].apply(lambda x: x.get(key))  # For the original df DataFrame
    df_ref.loc[:, key] = df_ref['attribute'].apply(lambda x: x.get(key))  # For the df_ref DataFrame

    # Ensure no key has Null values
    assert df[key].notnull().all(), f"Null values found in '{key}' column in the input DataFrame."
    assert df_ref[key].notnull().all(), f"Null values found in '{key}' column in the reference DataFrame."

    if set(df[key]) != set(df_ref[key]): 
       print("[warning] Foreign key mismatch between the input and reference DataFrames.")

    # Filter rows in df where 'reference_id' is in the set of 'reference_id's from df
    # df = df_ref[df_ref[key].isin(reference_ids)]

    # Update the 'attribute' column of each row in df_ref_filtered with the 'attribute' of the matching row in df
    n_unmatched = 0
    n_multiple_matches = 0
    for r, row in df_ref.iterrows():  # Todo: Optimize by using merge operation
        ref_id = row['reference_id']

        matching_row = df[df['reference_id'] == ref_id]

        if not matching_row.empty:
            # Check for multiple matches
            if len(matching_row) > 1:
                n_multiple_matches += 1
                if n_multiple_matches < 5: 
                    print(f"? Multiple matches (n={matching_row.shape[0]}) found for reference_id={ref_id} in the input DataFrame.")
                    print(matching_row.head(3))

            # df_ref_filtered.at[r, 'attribute'].update(matching_row.iloc[0]['attribute'])

            if feature_type == 'gene': # uORF explorer's experimental results do not have gene features
                pass 
            elif feature_type == 'transcript': 

                keys_to_update = ['ID', 'Parent', 'gene_type', 'gene_name', 'transcript_type', ]
                custom_keys = {'associated_gene': 'gene_name' , 'transcript_biotype': 'transcript_type', }
                
                source_attribute = matching_row.iloc[0]['attribute']  # Attribute dictionary from the source row
                target_attribute = row['attribute']  # Attribute dictionary of the target row to be updated
                
                for key in keys_to_update: 
                    target_attribute[key] = source_attribute[key]

                for key, key_standard in custom_keys.items():

                    # The key in df_ref corresponds to the standard key in df 
                    target_attribute[key] = source_attribute[key_standard]

                # Assign the updated dictionary back to the DataFrame
                df_ref.at[r, 'attribute'] = target_attribute
            elif feature_type == 'CDS':
                pass 
            elif feature_type == 'exon':
                pass
            else: 
                print(f"[warning] Unknown feature type: {feature_type}")
        else: 
            n_unmatched += 1
    print("[info] Number of unmatched rows (ref_id found in reference but not in input): ", n_unmatched)
    
    return df_ref

def update_by_matching_ids(df, df_ref, feature=None, key='reference_id'):
    """
    Update attributes in specific rows in df using attributes in another reference dataframe df_ref.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    df_ref (pandas.DataFrame): The reference DataFrame.
    feature (str): The feature to update.
    key (str): The key that connects df and df_ref.

    Returns:
    pandas.DataFrame: The updated DataFrame.
    """
    if feature is not None: 
        df = df[df['feature'] == feature]
        df_ref = df_ref[df_ref['feature'] == feature]

    # Find all reference_ids in df
    df[key] = df['attribute'].apply(lambda x: x.get(key))
    df_ref[key] = df_ref['attribute'].apply(lambda x: x.get(key))

    # Ensure no key has Null values
    assert df[key].notnull().all(), f"Null values found in '{key}' column in the input DataFrame."
    assert df_ref[key].notnull().all(), f"Null values found in '{key}' column in the reference DataFrame."

    ref_ids = df[key].unique()

    # Filter rows in df_ref where 'reference_id' is in the set of 'reference_id's from df
    df_ref_filtered = df_ref[df_ref[key].isin(ref_ids)]

    # Update the 'attribute' column of each row in df_ref_filtered with the 'attribute' of the matching row in df
    # n_unmatched = 0
    # for r, row in df_ref_filtered.iterrows():
    #     ref_id = row['reference_id']
    #     feature = row['feature']
    #     matching_row = df[(df['reference_id'] == ref_id) & (df['feature'] == feature)]
    #     if not matching_row.empty:
    #         df_ref_filtered.at[r, 'attribute'].update(matching_row.iloc[0]['attribute'])
    #     else: 
    #         n_unmatched += 1
    # print("> Number of unmatched rows (ref_id found in reference but not in input): ", n_unmatched)

    # Merge df_ref_filtered with df on 'reference_id' and 'feature'
    merged_df = pd.merge(df_ref_filtered, df[['reference_id', 'feature', 'attribute']], 
                        on=['reference_id', 'feature'], 
                        how='left', 
                        suffixes=('', '_y'))

    # Update 'attribute' where not null in df
    merged_df.loc[merged_df['attribute_y'].notnull(), 'attribute'] = merged_df.loc[merged_df['attribute_y'].notnull()].apply(lambda row: {**row['attribute'], **row['attribute_y']}, axis=1)

    # Drop the extra column
    merged_df.drop(columns=['attribute_y'], inplace=True)

    # Replace df_ref_filtered with the updated DataFrame
    df_ref_filtered = merged_df
    
    return df_ref_filtered


def update_attributes(df, df_ref, feature=None, key='reference_id'):
    """
    Update attributes in specific rows in df using attributes in another reference dataframe df_ref.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    df_ref (pandas.DataFrame): The reference DataFrame.
    feature (str): The feature to update.
    key (str): The key that connects df and df_ref.

    Returns:
    pandas.DataFrame: The updated DataFrame.
    """
    if feature is not None: 
        df = df[df['feature'] == feature]
        df_ref = df_ref[df_ref['feature'] == feature]

    # Create the key column in both DataFrames so that the two dataframes can be merged on the key
    df[key] = df['attribute'].apply(lambda x: x.get(key))
    df_ref[key] = df_ref['attribute'].apply(lambda x: x.get(key))

    # Ensure no key has Null values
    assert df[key].notnull().all(), f"Null values found in '{key}' column in the input DataFrame."
    assert df_ref[key].notnull().all(), f"Null values found in '{key}' column in the reference DataFrame."

    # Merge df and df_ref on the key
    df_merged = pd.merge(df, df_ref, on=key, suffixes=('', '_ref'))

    # Update the attribute dictionaries in df with those in df_ref
    df_merged['attribute'] = df_merged.apply(lambda row: {**row['attribute'], **row['attribute_ref']}, axis=1)

    # Drop the key column and the attribute column from df_ref
    columns_to_drop = df_merged.filter(regex='_ref$', axis=1).columns
    columns_to_drop = columns_to_drop.union([key])
    df_merged.drop(columns=columns_to_drop, inplace=True)

    return df_merged

def update_reference_gtf_attributes(input_gtf, reference_gtf, output_gtf):
    """
    Update the GTF attributes of the reference GTF file based on the input GTF file.
    input_gtf represents the standard gene annotation whereas 
    reference_gtf represents a custom, domain-specific annotation.

    Parameters:
    input_gtf (str): Path to the input GTF file.
    reference_gtf (str): Path to the reference GTF file.
    output_gtf (str): Path to the output GTF file.
    """
    # Define the reserved GTF columns
    gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load the input GTF file into a DataFrame
    df = read_gtf_into_dataframe(input_gtf)  # attribute_as_type defaults to 'dictionary'
    dfs = split_dataframe_by_feature(df)  # Split the DataFrame into different feature types
    feature_types= df['feature'].unique()
    print("[info] Found n={} feature types from the input GTF file.".format(len(feature_types)))

    # Load the reference GTF file into a DataFrame
    df_ref = read_gtf_into_dataframe(reference_gtf)  # By defeault, attributes_as_dict=True, attributes are stored as dictionaries
    dfs_ref = split_dataframe_by_feature(df_ref)
    feature_types_ref = df_ref['feature'].unique()
    print("[info] Found n={} feature types from the reference GTF file.".format(len(feature_types_ref)))



def update_gtf_attributes(input_gtf, reference_gtf, output_gtf):
    """
    Update the GTF attributes of the input GTF file based on the reference GTF file.

    Parameters:
    input_gtf (str): Path to the input GTF file.
    reference_gtf (str): Path to the reference GTF file.
    output_gtf (str): Path to the output GTF file.
    """
    import sys

    def print_unique_counts_flattened(df, column_names, show_examples=True):
        for column_name in column_names:
            print(f"... Number of unique {column_name}: {df[column_name].nunique()}")
            if show_examples:
                print(f"... Example values of {column_name}: {df[column_name].unique()[:5]}")

    def check_and_convert(value):
        if isinstance(value, dict):
            return dict_to_attributes(value)
        return value

    def check_duplicates(df, header=None): 
        # df = pd.DataFrame(entries, columns=['entry'])  # Convert filtered_entries into a DataFrame
        if header is not None: print_emphasized(header)
        df_copy = df.copy()
        df_copy['attribute'] = df_copy['attribute'].apply(check_and_convert)

        duplicate_rows = df_copy.duplicated()  # Check for duplicate rows
        num_duplicate_rows = duplicate_rows.sum()  # Count the number of duplicate rows
        print("... Number of duplicate rows: ", num_duplicate_rows)  # Group by all columns, count the size of each group, and sort in descending order
        print("... Duplicate sizes:")
        duplicate_sizes = df_copy.groupby(df_copy.columns.tolist()).size().sort_values(ascending=False)  
        duplicate_sizes = duplicate_sizes[duplicate_sizes > 1]  # Filter out groups with size 1
        print(duplicate_sizes)
        return num_duplicate_rows

    # Define the reserved GTF columns
    gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load the input GTF file into a DataFrame
    df = read_gtf_into_dataframe(input_gtf)
    feature_types = df['feature'].unique()
    dfs = split_dataframe_by_feature(df)  # Split the DataFrame into different feature types

    # Test: Check if given feature types have the 'transcript_id' key
    for feature, dfi in dfs.items():
        if feature == 'gene': 
            pass
        if feature in ['transcript', 'exon', 'CDS', ]:
            if not check_keys_in_attributes(dfs[feature], keys=['transcript_id', ]): 
                print("... feature={} contains a subset of rows without 'transcript_id' key.".format(feature))

        assert check_keys_in_attributes(dfs['transcript'], keys=['transcript_id', ]), "Missing 'transcript_id' in the input GTF file."
    
    df_tx = dfs['transcript']
    # df_tx = add_reference_id(df_tx)
    for feature in ['transcript', 'CDS', 'exon', ]:
        assert check_keys_in_attributes(dfs[feature], keys=['transcript_id','reference_id' ]), "Missing 'transcript_id' in the input GTF file."

    # Load the reference GTF file into a DataFrame
    df_ref = read_gtf_into_dataframe(reference_gtf)  # attributes_as_dict=True
    dfs_ref = split_dataframe_by_feature(df_ref)

    df_tx_ref = dfs_ref['transcript']
    for feature in ['transcript', 'CDS', 'exon', ]:
        assert check_keys_in_attributes(dfs_ref[feature], keys=['transcript_id', 'reference_id' ]), "Missing 'transcript_id' in the input GTF file."

    # Data structure:
    # gene feature: df_gene, df_ref_gene
    # transcript feature: df_tx, df_ref_tx

    # Now update df_tx using df_ref_tx
    # 1. Extract 'reference_id' from the attribute dictionaries in both df_tx and df_ref_tx and add them as new columns.
    # 2. Merge df_tx and df_ref_tx on 'reference_id'.
    # 3. Update the attribute dictionaries in df_tx with those in df_ref_tx. 
    #    Preserve all keys in df_tx. If a key exists in both df_tx and df_ref_tx, 
    #    overwrite df_tx's values with df_ref_tx's.
    # 4. Drop the 'reference_id' column as it's no longer needed.

    print_emphasized("[update] Update transcript feature")
    # df_tx = update_attributes(df, df_ref, feature='transcript',  key='reference_id')
    df_tx = update_by_matching_ids(df, df_ref, feature='transcript', key='reference_id')
    assert not df_tx.isnull().any().any(), "Null values found in the updated df_tx: n={}".format(df_tx.isnull().sum().sum())

    print("> Columns of df_tx after updates: {}".format(df_tx.columns.tolist()))
    assert df_tx.columns.tolist() == gtf_columns, "Mismatch in the columns of the updated df_tx."
    check_duplicates(df_tx, header="df_tx (updated)")
    print_unique_values(df_tx, 'transcript', keys=['reference_id', ])
    
    print_emphasized("[update] Update CDS feature")
    # df_cds = dfs['CDS']
    # df_cds = add_reference_id(df_cds)  # reference_id is now the transcript_id
    # df_cds_ref = dfs_ref['CDS']
    # df_cds = update_attributes(df_cds, df_cds_ref, key='reference_id')
    df_cds = update_by_matching_ids(df, df_ref, feature='CDS', key='reference_id')
    assert df_cds.columns.tolist() == gtf_columns, f"Mismatch in the columns of the updated df_cds)"
    assert not df_cds.isnull().any().any(), "Null values found in the updated df_cds: n={}".format(df_cds.isnull().sum().sum())
    check_duplicates(df_cds, header="df_cds (updated)")
    print_unique_values(df_cds, 'CDS', keys=['reference_id', ])

    print_emphasized("[update] Update exon feature")
    # df_exon = dfs['exon']
    # df_exon = add_reference_id(df_exon)  # reference_id is now the transcript_id
    # df_exon_ref = dfs_ref['exon']
    # df_exon = update_attributes(df_exon, df_exon_ref, key='reference_id')  
    df_exon = update_by_matching_ids(df, df_ref, feature='exon', key='reference_id')  
    assert not df_exon.isnull().any().any(), "Null values found in the updated df_exon: n={}".format(df_exon.isnull().sum().sum())
    assert df_exon.columns.tolist() == gtf_columns, "Mismatch in the columns of the updated df_exon."
    check_duplicates(df_exon, header="df_exon (updated)")
    print_unique_values(df_exon, 'exon', keys=['reference_id', ])

    print_emphasized("[update] Update gene feature")
    df_gene = dfs['gene']
    # No updates for gene feature 
    assert df_gene.columns.tolist() == gtf_columns, "Mismatch in the columns of the updated df_gene."
    print_unique_values(df_gene, 'gene', keys=['gene_id', ])

    # --- Test --- 
    # Check if all DataFrames have the same columns
    msg = "The DataFrames do not have the same columns."
    assert set(df_gene.columns) == set(df_tx.columns) == set(df_cds.columns) == set(df_exon.columns), msg

    # Concatenate df_gene and df_tx to form the final DataFrame
    df = pd.concat([df_gene, df_tx, df_cds, df_exon], ignore_index=True)
    assert df.columns.tolist() == gtf_columns, "Mismatch in the columns of the final DataFrame."

    # Now df should have all feature types 
    df = df.sort_values(by=['start', 'end'])
    shape0 = df.shape[0]

    # Todo: Structure feature rows in the final DataFrame


    

    # --- Check updated attribute ---
    print_emphasized("Checking unique values after merging the DataFrames (df)")
    for feature in ['transcript', 'gene', 'CDS', 'exon']:
        print_unique_values(df, feature, keys=['reference_id', 'transcript_id', 'orf_id'])

    print(f"> Writing the updated GTF file to: {output_gtf}")
    write_dataframe_into_gtf(df, output_gtf)

    #############################################################################

    # --- Test ---
    for feature in ['transcript', 'gene', 'CDS', 'exon']:
        df_prime = read_gtf_into_dataframe(output_gtf, feature_types=feature)
        shape1 = df_prime.shape[0]
        print("> shape0={}, shape1={}".format(shape0, shape1))
    
    return df


def update_gtf_attributes_v1(input_gtf, reference_gtf, output_gtf):
    """
    Update the GTF attributes of the input GTF file based on the reference GTF file.

    Parameters:
    input_gtf (str): Path to the input GTF file.
    reference_gtf (str): Path to the reference GTF file.
    output_gtf (str): Path to the output GTF file.
    """
    # Define the reserved GTF columns
    gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load the reference GTF file into a DataFrame
    df_ref = read_gtf_into_dataframe(reference_gtf)

    # Load the input GTF file into a DataFrame
    df = read_gtf_into_dataframe(input_gtf)

    # Create a new attribute 'reference_id' in df and assign the values of 'transcript_id' to it
    print("> Updating the input GTF file based on the reference GTF file using the 'reference_id' column.")
    df['reference_id'] = df['transcript_id']
    # NOTE: This is used to link the input GTF file with the reference GTF file

    # List of genomic features to include in the final query_uorfs.gtf
    # features = ['transcript', 'gene', 'CDS', 'exon']

    # ---- Update the input GTF file based on the reference GTF file ----

    # Filter df and df_ref for the 'transcript' feature
    df_transcript = df[df['feature'] == 'transcript']
    df_ref_transcript = df_ref[df_ref['feature'] == 'transcript']

    # Merge df_transcript and df_ref_transcript on 'reference_id'
    df_merged = pd.merge(df_transcript, df_ref_transcript, on='reference_id', suffixes=('', '_ref'))

    # List of columns to update/add from df_ref_transcript
    columns_to_update = [col for col in df_ref_transcript.columns if col not in gtf_columns]

    # Update/add the columns in df_merged based on the values in df_ref_transcript
    for col in columns_to_update:
        if f'{col}_ref' in df_merged.columns:
            df_merged[col] = df_merged[f'{col}_ref']
        else:
            df_merged[col] = df_ref_transcript[col]

    # Remove the '_ref' columns
    df_merged = df_merged.drop(columns=[f'{col}_ref' for col in columns_to_update if f'{col}_ref' in df_merged.columns])

    # Filter df for the 'gene', 'CDS', and 'exon' features
    df_other_features = df[df['feature'].isin(['gene', 'CDS', 'exon'])]

    # Concatenate df_merged and df_other_features
    df = pd.concat([df_merged, df_other_features])

    # Group df by 'reference_id', 'transcript_id', and 'orf_id'
    grouped = df.groupby(['reference_id', 'transcript_id', 'orf_id'], as_index=False)

    # Define a function to propagate the gene, CDS, and exon information to all rows within each group
    def propagate_info(group):
        gene_info = group[group['feature'] == 'gene'].iloc[0]
        cds_info = group[group['feature'] == 'CDS'].iloc[0]
        exon_info = group[group['feature'] == 'exon'].iloc[0]

        group.loc[:, 'gene'] = gene_info['attribute']
        group.loc[:, 'CDS'] = cds_info['attribute']
        group.loc[:, 'exon'] = exon_info['attribute']

        return group

    # Apply the function to each group
    df = grouped.apply(propagate_info)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Write the updated GTF file
    write_dataframe_into_gtf(df, output_gtf)

    return df

def update_gtf_attributes_v0(input_gtf, output_gtf, transcript_id_df=None):
    """
    Update the GTF attributes to ensure compatibility with the load_orfs function in uORFinder.py.

    Parameters:
    input_gtf (str): Path to the input GTF file.
    output_gtf (str): Path to the output GTF file.
    transcript_id_df (pd.DataFrame): DataFrame with 'reference_id' and 'new_transcript_id' columns.
    """
    import tempfile
    # new_attributes = set()

    # Convert the DataFrame to a dictionary for faster lookup
    # if transcript_id_df is not None:
    #     transcript_id_dict = transcript_id_df.set_index('reference_id')['new_transcript_id'].to_dict()

    with open(input_gtf, 'r') as infile, tempfile.NamedTemporaryFile('w', delete=False) as outfile:
        for line in infile:
            if line.startswith("#"):
                outfile.write(line)
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                outfile.write(line)
                continue

            feature = fields[2]
            attributes_str = fields[8]
            attributes = dict(attr.strip().split(' ') for attr in attributes_str.strip(';').split(';') if ' ' in attr)

            # Custom attribute updates to work with uORFinder
            if feature == "transcript":
                # Update transcript_biotype with transcript_type
                if "transcript_type" in attributes:
                    attributes["transcript_biotype"] = attributes["transcript_type"]

                # Update associated_gene with gene_name
                if "gene_name" in attributes:
                    attributes["associated_gene"] = attributes["gene_name"]

                # Add reference_id attribute
                if "reference_id" not in attributes:
                    attributes["reference_id"] = attributes.get("transcript_id")    

                # # Update transcript_id with new values from the DataFrame
                # if transcript_id_dict and attributes["reference_id"] in transcript_id_dict:
                #     attributes["transcript_id"] = transcript_id_dict[attributes["reference_id"]]

            # Reconstruct the attributes string without over-quoting and trailing semicolons
            attributes_list = []
            for key, value in attributes.items():
                if value:
                    value = value.replace('\"', '')
                    attribute_str = f'{key} "{value}";'
                    attributes_list.append(attribute_str)
            attributes_str = ' '.join(attributes_list).rstrip(' ;')

            fields[8] = attributes_str

            outfile.write('\t'.join(fields) + '\n')

    # Replace the original file with the temporary file
    shutil.move(outfile.name, output_gtf)

    return


def test_unique_transcript_entry(df):
    # Parse the 'attribute' column into a DataFrame
    attributes_df = df['attribute'].str.split('; ', expand=True).apply(lambda s: s.str.split(' ', expand=True))

    # Rename the columns to the attribute names
    attributes_df.columns = attributes_df.apply(lambda s: s[0].str.strip())

    # Remove the attribute names from the cells
    attributes_df = attributes_df.apply(lambda s: s[1].str.strip('"'))

    # Add the attributes DataFrame to the original DataFrame
    df = pd.concat([df, attributes_df], axis=1)

    # Group by the attributes of interest and count the number of unique rows in each group
    grouped = df[df['feature'] == 'transcript'].groupby(['transcript_id', 'reference_id', 'orf_id']).nunique()

    # Check if the count is 1 for all groups
    print((grouped == 1).all())


def read_uorf_data(file_path):
    """Read the TSV file containing uORF data."""
    return pd.read_csv(file_path, sep='\t')


def create_gtf_line(chrom, source, feature, start, end, score, strand, frame, attributes):
    attr_str = ' '.join([f'{key} "{value}";' for key, value in attributes.items()])
    return f'{chrom}\t{source}\t{feature}\t{start}\t{end}\t{score}\t{strand}\t{frame}\t{attr_str}\n'

def generate_query_uorfs_gtf(experimental_tsv, output_gtf, source):
    """
    Generate a query_uorfs.gtf file from the experimental data TSV.

    Parameters:
    experimental_tsv (str): Path to the experimental TSV file containing replicated Ribo-seq ORFs data.
    output_gtf (str): Path to the output GTF file to be generated.
    source (str): The source to be included in the GTF file.

    Returns:
    None

    Example Usage: 

    # Path to the experimental data TSV file (Table S2 from the paper)
    experimental_tsv_path = '/mnt/data/S2. PHASE I Ribo-seq ORFs.tsv'
    
    # Output GTF file path (query_uorfs.gtf)
    output_gtf_path = '/mnt/data/query_uorfs.gtf'

    # Source for the GTF file
    source_name = 'uORF_explorer'

    # Generate the query_uorfs.gtf file
    generate_query_uorfs_gtf(experimental_tsv_path, output_gtf_path, source_name)

    print("GTF file generated successfully.")
    """
    # Read the experimental data TSV file
    df = pd.read_csv(experimental_tsv, sep='\t')
    print("(generate_query_uorfs_gtf) Columns:\n", list(df.columns), "\n")

    # Open the output GTF file for writing
    with open(output_gtf, 'w') as out_fh:
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            chrom = f'chr{row["chrm"]}'
            strand = row['strand']
            transcript_id = row['orf_name']

            # Find the column name that starts with 'gene_id' (its actual name may vary e.g. gene_id (GENCODE v39))
            gene_id_col = next(col for col in row.keys() if col.startswith('gene_id'))
            # Use the found column name to get the value
            gene_id = row[gene_id_col]

            # Optional: Get the transcript ID from which uORF is derived
            reference_id = row.get('transcript', row.get('transcript_id'))

            gene_name = row['gene_name']
            transcript_biotype = row['orf_biotype']
            starts = list(map(int, str(row['starts']).split(';')))
            ends = list(map(int, str(row['ends']).split(';')))
            
            attributes = {
                'gene_id': gene_id,
                'transcript_id': transcript_id,
                'reference_id': reference_id,
                'associated_gene': gene_name,
                'transcript_biotype': transcript_biotype
            }
            
            # Create transcript line
            transcript_start = min(starts)
            transcript_end = max(ends)
            out_fh.write(create_gtf_line(chrom, source, 'transcript', transcript_start, transcript_end, '.', strand, '.', attributes))
            
            # Create exon and CDS lines for each exon
            for start, end in zip(starts, ends):
                out_fh.write(create_gtf_line(chrom, source, 'exon', start, end, '.', strand, '.', attributes))
                out_fh.write(create_gtf_line(chrom, source, 'CDS', start, end, '.', strand, '.', attributes))


def create_query_uorfs_gtf(input_path=None, output_path=None, verbose=1, **kargs):
    from meta_spliceai.utils.utils_excel import load_excel_sheets_into_dataframes

    # Path to the experimental data sheets
    if input_path is None: 
        input_dir = "/path/to/meta-spliceai/data/ORF"
        input_path = os.path.join(input_dir, "NIHMS1854551-supplement-supplementary_tables.xlsx")

    dfs, paths = load_excel_sheets_into_dataframes(input_path, return_paths=True, verbose=verbose)

    ############################################################

    # Path to the experimental data TSV file (E.g. Table S2 from the paper)
    target_sheet = kargs.get("target_sheet", 'S2. PHASE I Ribo-seq ORFs')   
    source_tsv = paths[target_sheet]

    # Path to the experimental data TSV file (Table S2 from the paper)
    experimental_tsv_path = source_tsv  # '/mnt/data/S2. PHASE I Ribo-seq ORFs.tsv'
    print("[info] Path to the experimental data TSV file: ", experimental_tsv_path)
    
    # Output GTF file path (query_uorfs.gtf)
    if output_path is None:
        output_path = os.path.join(config.get_proj_dir(), "output/query_uorfs.gtf")   # '/mnt/data/query_uorfs.gtf'
    print("[info] Output GTF file path: ", output_path)

    # Source for the GTF file
    source_name = 'uORF_explorer'

    # Generate the query_uorfs.gtf file
    generate_query_uorfs_gtf(experimental_tsv_path, output_path, source_name)

    print("GTF file generated successfully.")

    return

def create_query_uorfs_gtf_from_pub_results(**kargs): 

    # Load the initial query_uorfs.gtf file
    data_prefix = os.path.join(os.getenv('HOME'), "work/meta-spliceai/data")
    gff3_file_path = os.path.join(data_prefix, "gencode/gencode.v43.annotation.gff3")   

    # output_dir = os.path.join(config.get_proj_dir(), 'output')
    if 'output_dir' in kargs: 
        output_dir = kargs['output_dir']
    else: 
        output_dir = os.path.join(os.getenv('HOME'), "work/meta-spliceai/output")
    # NOTE: /path/to/meta-spliceai/output may have permission issues

    subsetted_gff3_file_path = os.path.join(output_dir, 'query_uorfs.gff3')
    init_gtf_file_path = os.path.join(output_dir, 'query_uorfs_init.gtf')   
    output_gtf_file_path = os.path.join(output_dir, 'query_uorfs.gtf')
    gffutils_db_file_path = os.path.join(output_dir, 'query_uorfs.db' )

    # This is the template GTF file that will be used as the initial query_uorfs.gtf file
    # - gtex_uORFconnected_txs.w_utrs.gtf
    # - gtex_uORFconnected_txs.gtf
    reference_gtf_path = os.path.join("uORF_explorer/pub_results", "gtex_uORFconnected_txs.w_utrs.gtf")   # 'path/to/gtex_uORFconnected_txs.gtf'

    def verify_duplicates_and_unique_values(df, desc=None): 

        if desc is not None:
            print_emphasized(desc)

        # Check for duplicate rows
        df = convert_attribute(df, dtype='str')
        num_duplicate_rows_ref = df.duplicated().sum()
        print("[verify] Number of duplicate rows in df: ", num_duplicate_rows_ref)
        if num_duplicate_rows_ref > 0:
            n = 10  # Replace with the number of rows you want to display
            duplicated_rows = df[df.duplicated(keep=False)]
            print(duplicated_rows.head(n))

        # Check unique values for ID keys
        df = convert_attribute(df, dtype='dict')
        # print(df_ref[ ['feature', 'attribute']].head(5))
        for feature in ['transcript', 'gene', 'CDS', 'exon']:
            print_unique_values(df, feature)  # keys=['reference_id', 'transcript_id', 'orf_id']

        # Check null values
        has_null = df.isnull().any().any()
        assert not has_null, "Null values found in the reference GTF file."

        print('-' * 80)
        return

    def verify_unique_ids(df, desc=None):  # assuming attribute column is a dictionary

        if desc is not None:
            print_emphasized(desc)

        # Extract the unique gene and transcript IDs from the DataFrame
        gene_ids_of_interest = \
            count_attribute(df, 'gene_id',
                feature_type='transcript', return_values=True)  # uORF Explorer's experimental results do not have gene features
        assert len(gene_ids_of_interest) > 0, "No gene IDs found in the reference GTF file."

        transcript_ids_of_interest = \
            count_attribute(df, 'reference_id',
                feature_type='transcript', return_values=True)    # ['ENST00000237247.10', 'ENST00000237247.10', ...]
        print("... Found m={} unique gene IDs from gtex_uORFconnected_txs".format(len(gene_ids_of_interest)))
        print("... Example gene IDs:\n{}\n".format(gene_ids_of_interest[:10]))
        print("... Found n={} unique transcript IDs from gtex_uORFconnected_txs".format(len(transcript_ids_of_interest)))
        print("... Example reference IDs:\n{}\n".format(transcript_ids_of_interest[:10]))

        assert len(gene_ids_of_interest) > 0, "No gene IDs found in the reference GTF file."

        return gene_ids_of_interest, transcript_ids_of_interest

    ############################################################
    print_emphasized("Step 0 (Initialization): Load the initial experiment-specific GFF3 file ...")

    df_ref = read_gtf_into_dataframe(reference_gtf_path)
    
    # --- Test --- 
    header = "[verify] Reference GTF file: {}\n".format(reference_gtf_path)
    verify_duplicates_and_unique_values(df_ref, header)

    # Extract the unique gene and transcript IDs from the DataFrame
    gene_ids_of_interest, transcript_ids_of_interest = verify_unique_ids(df_ref)

    # Sample a subset of the transcript IDs
    print_emphasized("Step 0 (Initialization): Sample a subset of the target reference IDs ...")
    n_transcripts = kargs.get('n_transcripts', 50)
    if n_transcripts is not None: 
        n_transcripts = min(n_transcripts, len(transcript_ids_of_interest))
        df_ref_sampled, sampled_transcript_ids, sampled_gene_ids = \
            subsample_by_reference_ids(df_ref, n=n_transcripts, id_spec=None, return_ids=True)
        transcript_ids_of_interest = sampled_transcript_ids
        gene_ids_of_interest = sampled_gene_ids
        print("> Sampled {} transcript IDs:\n{}\n".format(len(sampled_transcript_ids), sampled_transcript_ids[:10]))
        print("> Sampled {} gene IDs:\n{}\n".format(len(sampled_gene_ids), sampled_gene_ids[:10]))

        verify_unique_ids(df_ref_sampled, desc="[verify] Sampled reference GTF file")

        df_ref = df_ref_sampled

    # -----------------------------------

    # Step 1: Subset the GFF3 file by transcript IDs
    print_emphasized("Step 1: Subset the GFF3 file by transcript IDs")
    subsetted_entries, gene_ids, reference_ids = \
        subset_gff3_by_transcript_ids(gff3_file_path,  # path to the standard GFF3 annotation file
            transcript_ids=transcript_ids_of_interest,
            gene_ids=gene_ids_of_interest,
            return_ids=True)

    # --- Test ---
    print("> n(gene_ids_of_interest): {} =?= n(gene_ids) found in reference: {}".format(
        len(gene_ids_of_interest), len(gene_ids)))
    print("> n(transcript_ids_of_interest): {} =?= n(reference_ids) found in reference: {}".format(
        len(transcript_ids_of_interest), len(reference_ids)))
    # Find the difference
    diff = set(transcript_ids_of_interest).difference(set(reference_ids))
    print("... Difference: {}".format(diff))

    # -----------------------------------

    # Step 2: Save the augmented entries as a GTF file
    print_emphasized(f"Step 2: Save initial query_uorfs.gtf to:\n{init_gtf_file_path}\n")
    write_gff_lines_as_gtf(subsetted_entries, init_gtf_file_path)

    print_emphasized("Step 2a: Load the initial query_uorfs.gtf file ...")    
    df_query_init = read_gtf_into_dataframe(init_gtf_file_path)
    dfs = split_dataframe_by_feature(df_query_init)
    
    print("Step 2b: Add reference IDs to the initial query_uorfs ...")
    df_query_init = add_reference_id_customized(df_query_init)  # Todo: reusability 

    # --- Test ---
    print("[info] Initial query_uorfs.gtf: nrow={}".format(df_query_init.shape[0]))
    print(df_query_init.head(5)) 
    verify_unique_ids(df_query_init, desc="[verify] Sampled input/standard GTF file")

    # ref_ids = count_attribute(df_ref, 'reference_id', feature_type='transcript', return_values=True)
    verify_unique_ids(df_ref, desc="[verify] Sampled reference/template GTF file")
    # -----------------------------------

    # Step 3: Update the attributes of the reference GTF file based on the augmented GTF file
    
    print_emphasized("Step 3: Update the attributes of the reference GTF file with standard annotation file")
    dfs = []
    for feature in ['gene', 'transcript', 'CDS', 'exon',]:
        if feature == 'gene':
            dfi = df_query_init[df_query_init['feature']==feature] 
            dfs.append(dfi)

            print("[test] Example gene attribute:")
            print(dfi['attribute'].head(5))
        else:  # If feature type is in transcript, CDS, exon ... 
            print("[action] Updating feature={} using info in the standard GTF file ...".format(feature))
            dfi = update_reference_by_matching_ids(df_query_init, df_ref,
                    feature_type=feature, raise_exception=False)
        print_unique_values(dfi, feature)
        dfs.append(dfi)
    df_query = pd.concat(dfs, ignore_index=True)

    verify_unique_ids(df_query, desc="[verify] query_urof GTF file")
    
    # Step 4. Reorder rows to make it easier to read 
    print_emphasized("Step 4: Reorder rows to make it easier to read")
    # - A common way to order gene annotation files is by 'seqname', 'start', 'end', and 'feature' 
    #   in that order. Within 'feature', the rows are often ordered as 'gene', 'transcript', 'CDS', and 'exon'.

    # Define the order of the features
    feature_order = ['gene', 'transcript', 'CDS', 'exon']

    # Create a categorical column with the defined order
    df_query['feature'] = pd.Categorical(df_query['feature'], categories=feature_order, ordered=True)

    # Sort the DataFrame by 'seqname', 'start', 'end', and 'feature'
    df_query = df_query.sort_values(by=['seqname', 'start', 'end', 'feature'])

    # Reset the index of the DataFrame
    df_query = df_query.reset_index(drop=True)

    # Step 5: Save the updated query_uorfs.gtf
    print(f"[output] Step 5: Writing the updated GTF file to: {output_gtf_file_path}")
    write_dataframe_into_gtf(df_query, output_gtf_file_path)

    return df_query


def create_query_uorfs_gtf_v0(**kargs): 

    # Load the initial query_uorfs.gtf file
    data_prefix = os.path.join(os.getenv('HOME'), "work/meta-spliceai/data")

    # Example usage
    gff3_file_path = os.path.join(data_prefix, "gencode/gencode.v43.annotation.gff3")     # 'path/to/gencode_annotation.gff3'
    
    print_emphasized("Step 0 (Initialization): Load the initial experiment-specific GFF3 file ...")
    # Read the CSV files gtex_uORFconnected_txs.csv and gtex_uORFconnected_txs.w_utrs.csv (derived from the GTEx data)
    filepath = os.path.join("uORF_explorer/pub_results", "gtex_uORFconnected_txs.csv")
    df_original = pd.read_csv(filepath)

    # Extract the unique transcript IDs from the DataFrame
    transcript_ids_of_interest = df_original['reference_id'].unique()  # ['ENST00000237247.10', 'ENST00000237247.10', ...]
    print("> Found {} unique transcript IDs from gtex_uORFconnected_txs".format(len(transcript_ids_of_interest)))
    print("> Example reference IDs:\n{}\n".format(transcript_ids_of_interest[:10]))

    # Sample a subset of the transcript IDs
    n_transcripts = kargs.get('n_transcripts', 50)
    n_transcripts = min(n_transcripts, len(transcript_ids_of_interest))
    sampled_transcript_ids = np.random.choice(transcript_ids_of_interest, size=n_transcripts, replace=False)
    print("> Sampled {} transcript IDs:\n{}\n".format(len(sampled_transcript_ids), sampled_transcript_ids[:10]))
    transcript_ids_of_interest = sampled_transcript_ids

    # output_dir = os.path.join(config.get_proj_dir(), 'output')
    if 'output_dir' in kargs: 
        output_dir = kargs['output_dir']
    else: 
        output_dir = os.path.join(os.getenv('HOME'), "work/meta-spliceai/output")
    # NOTE: /path/to/meta-spliceai/output may have permission issues

    subsetted_gff3_file_path = os.path.join(output_dir, 'query_uorfs.gff3')
    augmented_gtf_file_path = os.path.join(output_dir, 'temp_augmented.gtf')   # 'augmented_query_uorfs.gtf'
    output_gtf_file_path = os.path.join(output_dir, 'query_uorfs.gtf')
    gffutils_db_file_path = os.path.join(output_dir, 'query_uorfs.db' )

    # Step 1: Subset the GFF3 file by transcript IDs
    print_emphasized("Step 1: Subset the GFF3 file by transcript IDs")
    subsetted_entries = subset_gff3_by_transcript_ids(gff3_file_path, transcript_ids_of_interest)

    # --- Test ---
    def check_duplicates(entries, header=None): 
        if header:
            print(header); print('-'*80)
        df = pd.DataFrame(entries, columns=['entry'])  # Convert filtered_entries into a DataFrame
        duplicate_rows = df.duplicated()  # Check for duplicate rows
        num_duplicate_rows = duplicate_rows.sum()  # Count the number of duplicate rows
        print("... Number of duplicate rows: ", num_duplicate_rows)  # Group by all columns, count the size of each group, and sort in descending order
        print("... Duplicate sizes:")
        duplicate_sizes = df.groupby(df.columns.tolist()).size().sort_values(ascending=False)  
        duplicate_sizes = duplicate_sizes[duplicate_sizes > 1]  # Filter out groups with size 1
        print(duplicate_sizes)

    print("> Given entries associated with n={} transcripts from gtex_uORFconnected_txs".format(n_transcripts))
    check_duplicates(subsetted_entries, "@ Step 1")
    # -----------------------------------

    # Step 2: Augment the subsetted GFF3 entries to include necessary attributes
    print_emphasized("Step 2: Augment the subsetted GFF3 entries to include necessary attributes")
    augmented_entries, gene_set, tx_set = \
        augment_gff3_for_query_uorfs(subsetted_entries, return_ids=True)
    print("[2] Number of unique transcript IDs: {}".format(len(tx_set)))
    check_duplicates(augmented_entries, "@ Step 2")

    # Step 3: Save the augmented entries as a GTF file
    print_emphasized("Step 3: Save the augmented entries as a GTF file")
    write_gff_lines_as_gtf(augmented_entries, augmented_gtf_file_path)

    # --- Test ---
    print("> End of Step 3: Save the augmented entries as a GTF file\n")
    df = read_gtf_into_dataframe(augmented_gtf_file_path, attributes_as_dict=True)

    for feature in ['transcript', 'gene', 'CDS', 'exon']:
        # Count the number of unique reference IDs, transcript IDs, and ORF IDs, 
        # assuming that attribute is represented by dictionary
        print_unique_values(df, feature, keys=['reference_id', 'transcript_id', 'orf_id'])

    # ... number of unique transcript IDs: 1693
    df = read_gtf_into_dataframe(augmented_gtf_file_path, attributes_as_dict=False, feature_types=['transcript', ])
    A = df['transcript_id'].unique()
    print("[3+] Number of unique transcript IDs (Me): ", len(A))  # 1 extra? 
    if len(A) > len(tx_set):
        print("... len(A)={} > len(tx_set)={}".format(len(A), len(tx_set)))

        # Find the difference
        diff = set(A).difference(set(tx_set))
        print("... Difference: {}".format(diff))

    # Check for duplicate rows
    duplicate_rows = df.duplicated()
    num_duplicate_rows = duplicate_rows.sum()
    print("[3+] Number of duplicate rows in df (prior to update with df_ref): ", num_duplicate_rows)
    if num_duplicate_rows > 0:
        # Group by all columns, count the size of each group, and sort in descending order
        duplicate_sizes = df.groupby(df.columns.tolist()).size().sort_values(ascending=False)
        print(duplicate_sizes)

        # Drop duplicate rows
        df = df.drop_duplicates()

    assert df.duplicated().sum() == 0
    # -----------------------------------

    reference_gtf_path = os.path.join("uORF_explorer/pub_results", "gtex_uORFconnected_txs.gtf")   # 'path/to/gtex_uORFconnected_txs.gtf'
    df_uorf = read_gtf_into_dataframe(reference_gtf_path)
    print("> Reference GTF file: {}\n".format(reference_gtf_path))
    
    for feature in ['transcript', 'gene', 'CDS', 'exon']:
        print_unique_values(df_uorf, feature, keys=['reference_id', 'transcript_id', 'orf_id'])
    
    df_uorf = read_gtf_into_dataframe(reference_gtf_path, attributes_as_dict=False, feature_types=['transcript', ])
    num_duplicate_rows_ref = df_uorf.duplicated().sum()
    print("[3+] Number of duplicate rows in df_uorf: ", num_duplicate_rows_ref)
    if num_duplicate_rows_ref > 0:
        n = 10  # Replace with the number of rows you want to display
        duplicated_rows = df_uorf[df_uorf.duplicated(keep=False)]
        print(duplicated_rows.head(n))

    B = df_uorf['reference_id'].unique()

    print_emphasized(f"=> Number of common transcript IDs: {len(set(A).intersection(set(B)))}\n")
    # ... number of unique transcript IDs: 2977
    # ... number of unique reference IDs: 1714

    # Given the input GTF file, update the attributes to match the reference GTF file

    ########################################################################
    print_emphasized("Step 3.5: Update GTF file so that it can be correctly parsed by uORFinder")
    # update_gtf_attributes(augmented_gtf_file_path, reference_gtf_path, output_gtf_file_path)
    # update_gtf_attributes_via_dictionary(
    #     augmented_gtf_file_path, augmented_gtf_file_path, 
    #     {'gene_biotype': 'protein_coding'})
    ########################################################################

    # --- Test --- 
    print()
    print("> Reading the updated GTF file from (final query_uorfs.gtf):\n{}\n".format(output_gtf_file_path))

    df_query = \
        read_gtf_into_dataframe(output_gtf_file_path, 
                attributes_as_dict=True, feature_types=None)
    shape1 = df_query.shape[0]

    print("... shape(df_query): {}\n".format(shape1))
    print("... cols(df_uorf): {}\n".format(list(df_uorf.columns)))
    for feature in ['transcript', 'gene', 'CDS', 'exon']:
        print_unique_values(df_query, feature, keys=['reference_id', 'transcript_id', 'orf_id'])

    # test_features_presence(df_query)
    # display_example_rows(df_query, columns_to_display=['seqname', 'start', 'end', 'feature'])

    # Step 4: Create a gffutils database from the augmented GTF file
    print_emphasized("Step 4: Create a gffutils database from the augmented GTF file")
    # create_gffutils_db(augmented_gtf_file_path, gffutils_db_file_path)
    # verify_gtf_file(augmented_gtf_file_path)

    # Step 5. Reorder rows to make it easier to read 
    print_emphasized("Step 5: Reorder rows to make it easier to read")
    # - A common way to order gene annotation files is by 'seqname', 'start', 'end', and 'feature' 
    #   in that order. Within 'feature', the rows are often ordered as 'gene', 'transcript', 'CDS', and 'exon'.

    # Define the order of the features
    feature_order = ['gene', 'transcript', 'CDS', 'exon']

    # Create a categorical column with the defined order
    df_query['feature'] = pd.Categorical(df_query['feature'], categories=feature_order, ordered=True)

    # Sort the DataFrame by 'seqname', 'start', 'end', and 'feature'
    df_query = df_query.sort_values(by=['seqname', 'start', 'end', 'feature'])

    # Reset the index of the DataFrame
    df_query = df_query.reset_index(drop=True)

    print(f"> Writing the updated GTF file to: {output_gtf_file_path}")
    write_dataframe_into_gtf(df_query, output_gtf_file_path)


    return

def demo_create_query_uorfs_gtf(**kargs): 
    from meta_spliceai.analyze_pub_results import make_query_uorfs_fasta

    data_prefix = os.path.join(config.get_proj_dir(), 'data')
    # output_dir = os.path.join(os.getenv('HOME'), "work/meta-spliceai/output")
    output_dir = os.path.join(config.get_proj_dir(), 'output')

    print("> Create query_uorfs.gtf file ...")
    # create_query_uorfs_gtf_from_pub_results(output_dir=output_dir)

    input_dir =  os.path.join(data_prefix, 'ORF') # "/path/to/meta-spliceai/data/ORF"
    path_to_experimental_source = os.path.join(input_dir, "NIHMS1854551-supplement-supplementary_tables.xlsx")
    assert os.path.exists(path_to_experimental_source), f"Invalid input path: {input_path}"

    path_to_query_uorfs_gtf = os.path.join(output_dir, 'query_uorfs.gtf')
    create_query_uorfs_gtf(input_path=path_to_experimental_source, 
                           output_path=path_to_query_uorfs_gtf, verbose=1)

    print("> Create query_uorfs.fa file ...")
    make_query_uorfs_fasta(output_dir=output_dir)

    return

def demo_write_dataframe_into_gtf(**kargs):

    # Example usage
    df = pd.DataFrame({
        'seqname': ['chr1', 'chr1'],
        'source': ['ensembl', 'ensembl'],
        'feature': ['gene', 'transcript'],
        'start': [1000, 1050],
        'end': [5000, 4000],
        'score': ['.', '.'],
        'strand': ['+', '-'],
        'frame': ['.', '.'],
        'gene_id': ['ENSG000001', 'ENSG000002'],
        'transcript_id': ['ENST000001', 'ENST000002'],
        'gene_name': ['Gene1', 'Gene2']
    })

    write_dataframe_into_gtf(df, 'test.gtf', attribute_columns=['gene_id', 'transcript_id', 'gene_name'])

    return

def test(**kargs): 

    # demo_write_dataframe_into_gtf()

    demo_create_query_uorfs_gtf(**kargs)



    return

if __name__ == "__main__":
    test()