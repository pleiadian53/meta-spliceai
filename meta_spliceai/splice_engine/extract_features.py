import os

import pandas as pd
import polars as pl
import numpy as np

from pybedtools import BedTool
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm

from .utils_fs import (
    build_gtf_database, 
    extract_all_gtf_annotations, 
    read_splice_sites
)
# NOTE: 
#    ├── splice_engine/
#    │   ├── __init__.py
#    │   ├── extract_features.py
#    │   └── utils_fs.py
# 
#    python -m splice_engine.extract_features
#

def get_gene_sequences(fasta_file, gene_names):
    """
    Retrieve DNA sequences for a set of genes from a FASTA file and return as a DataFrame.

    Parameters:
    fasta_file (str): Path to the FASTA file containing DNA sequences.
    gene_names (set or list): A set or list of gene names to retrieve sequences for.

    Returns:
    pd.DataFrame: A DataFrame with columns 'Gene' and 'Sequence'.

    Example:

    # Example usage:
    fasta_file = "your_genome_sequences.fasta"  # Replace with your FASTA file path
    gene_names = {"GENE1", "GENE2", "GENE3"}  # Replace with your set of gene names
    gene_sequences_df = get_gene_sequences_df(fasta_file, gene_names)
    print(gene_sequences_df)
    """
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        gene_id = record.id
        if gene_id in gene_names:
            data.append({'Gene': gene_id, 'Sequence': str(record.seq)})
    
    return pd.DataFrame(data)


def extract_splice_sites(gtf_file):
    """
    Extract splice site coordinates from a GTF file.
    
    This function extracts donor and acceptor sites from a GTF/GFF file and saves 
    them in a BED format

    Example usage:
    splice_sites = extract_splice_sites("your_annotations.gtf")
    splice_sites.to_csv("splice_sites.bed", sep="\t", index=False, header=False)
    """
    gtf = BedTool(gtf_file)
    exons = gtf.filter(lambda x: x[2] == 'exon').saveas()
    
    splice_sites = []
    
    for exon in exons:
        chrom = exon.chrom
        strand = exon.strand
        exon_start = exon.start
        exon_end = exon.end
        
        # Donor site: 3' end of the exon (start for positive strand)
        if strand == '+':
            splice_sites.append((chrom, exon_end, exon_end + 1, "donor"))
        else:
            splice_sites.append((chrom, exon_start - 1, exon_start, "donor"))
        
        # Acceptor site: 5' end of the exon (end for positive strand)
        if strand == '+':
            splice_sites.append((chrom, exon_start - 1, exon_start, "acceptor"))
        else:
            splice_sites.append((chrom, exon_end, exon_end + 1, "acceptor"))
    
    # Convert to DataFrame for further processing
    splice_sites_df = pd.DataFrame(splice_sites, columns=["chrom", "start", "end", "type"])
    return splice_sites_df


def extract_sequences(fasta_file, annotations):
    """Extract DNA sequences corresponding to splice sites."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id = record.id
        if seq_id in annotations['seqname'].values:
            sequences[seq_id] = str(record.seq)
    return sequences

def label_nucleotides(annotations, sequences):
    """Label nucleotides as donor, acceptor, or neither."""
    labeled_data = []
    for _, row in annotations.iterrows():
        if row['feature'] == 'exon':
            seq_id = row['seqname']
            sequence = sequences[seq_id]
            start, end = row['start'], row['end']
            for i in range(start, end + 1):
                if i == start:
                    labeled_data.append((sequence[i-1], 'donor'))
                elif i == end:
                    labeled_data.append((sequence[i-1], 'acceptor'))
                else:
                    labeled_data.append((sequence[i-1], 'neither'))
    return labeled_data

def prepare_dataset(labeled_data):
    """Prepare the dataset for model training."""
    sequences, labels = zip(*labeled_data)
    return train_test_split(sequences, labels, test_size=0.2, random_state=42)

def prepare_data_for_spliceator(gtf_file, sequence_fasta):
    """
    Extract and format data for Spliceator.
    
    Example usage:
        spliceator_input_data = prepare_data_for_spliceator("your_annotations.gtf", "genome_sequence.fasta")
    """
    splice_sites_df = extract_splice_sites(gtf_file)
    
    # Load the corresponding DNA sequences from FASTA
    from Bio import SeqIO
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(sequence_fasta, "fasta")}
    
    spliceator_data = []
    
    for _, row in splice_sites_df.iterrows():
        chrom, start, end, splice_type = row
        seq = sequences[chrom][start:end]
        
        # Example: store as tuple (sequence, label)
        spliceator_data.append((seq, splice_type))
    
    return spliceator_data


def extract_splice_sites_for_all_genes_v0(db):
    """
    Extract all donor and acceptor sites for all genes using gffutils, accounting for strand information.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.

    Returns:
    - all_splice_sites (dict): A dictionary where the key is the transcript_id, and the value is a dictionary
                               containing 'donor' and 'acceptor' sites with their genomic coordinates.
    """
    # from collections import defaultdict
    all_splice_sites = defaultdict(lambda: {'donor': [], 'acceptor': []})

    for transcript in db.features_of_type('transcript'):
        transcript_id = transcript['transcript_id'][0]
        strand = transcript.strand

        # Loop through all exons for this transcript
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        for exon in exons:
            exon_start = exon.start
            exon_end = exon.end

            if strand == '+':
                # Donor sites on + strand: end of the exon (exon-intron boundary)
                # Acceptor sites on + strand: start of the exon (intron-exon boundary)
                donor_start, donor_end = exon_end - 2, exon_end + 3  # Including consensus sequence length
                acceptor_start, acceptor_end = exon_start - 3, exon_start + 2  # Including consensus sequence length

            else:
                # Donor sites on - strand: start of the exon (exon-intron boundary)
                # Acceptor sites on - strand: end of the exon (intron-exon boundary)
                donor_start, donor_end = exon_start - 3, exon_start + 2  # Including consensus sequence length
                acceptor_start, acceptor_end = exon_end - 2, exon_end + 3  # Including consensus sequence length

            # Append donor and acceptor sites to the respective lists
            all_splice_sites[transcript_id]['donor'].append((exon.chrom, donor_start, donor_end, strand))
            all_splice_sites[transcript_id]['acceptor'].append((exon.chrom, acceptor_start, acceptor_end, strand))

    return all_splice_sites


def extract_splice_sites_for_all_genes_v0(db, consensus_window=2, save=False, return_df=True, **kargs):
    """
    Extract all donor and acceptor sites for all genes using gffutils, accounting for strand information.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - consensus_window (int): Number of nucleotides to include before and after the splice site.

    Returns:
    - all_splice_sites (list of dicts): A list where each element is a dictionary containing
      'chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id'.

    """
    all_splice_sites = []
    transcript_counter = 0

    for transcript in tqdm(db.features_of_type('transcript'), desc="Processing transcripts"):
    # for transcript in db.features_of_type('transcript'):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # For negative strand, reverse the exons to process in transcriptional order
        if strand == '-':
            exons = exons[::-1]

        # Iterate over exon boundaries to find splice sites
        for i in range(len(exons)):
            exon = exons[i]
            exon_start = exon.start
            exon_end = exon.end

            # Acceptor site: start of the current exon (except for the first exon)
            if i > 0:
                if strand == '+':
                    position = exon_start
                else:
                    position = exon_end
                # Extract consensus sequence region
                acceptor_start = position - consensus_window
                acceptor_end = position + consensus_window   

                # Test 
                # if transcript_counter < 10:
                #     print(f"> Acceptor site - Position: {position}, Start: {acceptor_start}, End: {acceptor_end}")

                all_splice_sites.append({
                    'chrom': chrom,
                    'start': acceptor_start,
                    'end': acceptor_end,
                    'strand': strand,
                    'site_type': 'acceptor',
                    'gene_id': gene_id,
                    'transcript_id': transcript_id
                })

            # Donor site: end of the current exon (except for the last exon)
            if i < len(exons) - 1:
                if strand == '+':
                    position = exon_end
                else:
                    position = exon_start
                # Extract consensus sequence region
                donor_start = position - consensus_window
                donor_end = position + consensus_window     # - 1  # Inclusive of consensus_window nucleotides

                # Test
                # if transcript_counter < 10:
                #     print(f"Donor site - Position: {position}, Start: {donor_start}, End: {donor_end}")

                all_splice_sites.append({
                    'chrom': chrom,
                    'start': donor_start,
                    'end': donor_end,
                    'strand': strand,
                    'site_type': 'donor',
                    'gene_id': gene_id,
                    'transcript_id': transcript_id
                })
        transcript_counter += 1

    # End of foreach transcript

    if save:
        sep = kargs.get('sep', '\t')
        format = 'csv' if sep == ',' else 'tsv'
        output_file = kargs.get('output_file', f'splice_sites.{format}')

        # Convert the list of dictionaries to a DataFrame
        splice_sites_df = pd.DataFrame(all_splice_sites)

        # Save the DataFrame to a CSV file
        splice_sites_df.to_csv(output_file, index=False, sep=sep)
        print(f"[i/o] Splice sites saved to {output_file}")

    # Return DataFrame if return_df is True, otherwise return list of dictionaries
    if return_df:
        return pd.DataFrame(all_splice_sites)
    else:
        return all_splice_sites


import pandas as pd
import gffutils
from tqdm import tqdm

def extract_junctions_for_gene(db, gene_identifier, consensus_window=0):
    """
    Extract all annotated transcripts and their associated junctions for a given gene,
    considering strand orientation and exon indexing.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - gene_identifier (str): Either gene name or gene ID (e.g., Ensembl ID).
    - consensus_window (int): Number of nucleotides to include before and after the splice site.

    Returns:
    - gene_junctions (pd.DataFrame): DataFrame containing junctions for all annotated transcripts.

    Example usage:
    db = gffutils.FeatureDB('path_to_gffutils_database.db')
    gene_junctions_df = extract_junctions_for_gene(db, 'ENSG00000123456')
    print(gene_junctions_df)
    """
    gene_junctions = []

    # Find the gene using the provided gene_identifier
    try:
        gene = db[gene_identifier]
    except gffutils.FeatureNotFoundError:
        print(f"Gene '{gene_identifier}' not found in the database.")
        return pd.DataFrame()

    for transcript in db.children(gene, featuretype='transcript', order_by='start'):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # Adjust exon order for negative strand
        if strand == '-':
            exons = exons[::-1]  # transcript is in reverse order for negative strand

        # ----xxxxxx----xxxxx---------xxxxxx----------> 
        #           d   a    d       a

        # <---xxxxxx----xxxxx---------xxxxxx-----------
        #           a   d    a       d

        # Iterate over exon boundaries to find junctions
        for i in range(len(exons) - 1):
            donor_end = exons[i].end if strand == '+' else exons[i + 1].start
            acceptor_start = exons[i + 1].start if strand == '+' else exons[i].end

            # Store junction information
            junction_name = f"{transcript_id}_JUNC{i:08d}"
            gene_junctions.append({
                'chrom': chrom,
                'start': donor_end,  # Start position in BED is 0-based
                'end': acceptor_start,  # End position is 1-based
                'name': junction_name,
                'score': 0,
                'strand': strand,
            })

    # Convert to DataFrame
    return pd.DataFrame(gene_junctions, columns=['chrom', 'start', 'end', 'name', 'score', 'strand'])


def extract_splice_sites_for_all_genes_v0(db, consensus_window=1, save=False, return_df=True, **kargs):
    """
    Extract all donor and acceptor sites for all genes using gffutils, accounting for strand information.
    Unlike extract_splice_sites_for_all_genes_v0(), we are assuming that the DNA sequences for 
    the negative strand have been reverse-complemented. 

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - consensus_window (int): Number of nucleotides to include before and after the splice site.

    Returns:
    - all_splice_sites (list of dicts): A list where each element is a dictionary containing
      'chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id'.

    """
    all_splice_sites = []
    transcript_counter = 0

    for transcript in tqdm(db.features_of_type('transcript'), desc="Processing transcripts"):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # We no longer need to reverse the exons for negative strand, 
        # because we assume sequences are reverse-complemented.
        
        # The exon boundaries remain the same for both strands because we assume that the DNA sequences 
        # for the negative strand have been reverse-complemented. 
        # This means the actual genomic coordinates remain constant, but the sequence itself 
        # has been flipped to follow the same processing logic as the positive strand.

        # Iterate over exon boundaries to find splice sites
        for i in range(len(exons)):
            exon = exons[i]
            exon_start = exon.start
            exon_end = exon.end

            # Acceptor site: start of the current exon (except for the first exon)
            if i > 0:
                # Regardless of strand, the acceptor site is before the exon start
                position = exon_start - 1  # The last nucleotide of the intron

                # Extract consensus sequence region
                acceptor_start = position - consensus_window
                acceptor_end = position + consensus_window   

                all_splice_sites.append({
                    'chrom': chrom,
                    'start': acceptor_start,
                    'end': acceptor_end,
                    'strand': strand,
                    'site_type': 'acceptor',
                    'gene_id': gene_id,
                    'transcript_id': transcript_id
                })

            # Donor site: end of the current exon (except for the last exon)
            if i < len(exons) - 1:
                # Regardless of strand, the donor site is after the exon end
                position = exon_end + 1  # The first nucleotide of the intron

                # Extract consensus sequence region
                donor_start = position - consensus_window
                donor_end = position + consensus_window     

                all_splice_sites.append({
                    'chrom': chrom,
                    'start': donor_start,
                    'end': donor_end,
                    'strand': strand,
                    'site_type': 'donor',
                    'gene_id': gene_id,
                    'transcript_id': transcript_id
                })
        transcript_counter += 1

    # Save or return the splice sites
    if save:
        sep = kargs.get('sep', '\t')
        format = 'csv' if sep == ',' else 'tsv'
        output_file = kargs.get('output_file', f'splice_sites.{format}')

        splice_sites_df = pd.DataFrame(all_splice_sites)
        splice_sites_df.to_csv(output_file, index=False, sep=sep)
        print(f"[i/o] Splice sites saved to {output_file}")

    return pd.DataFrame(all_splice_sites) if return_df else all_splice_sites


def extract_splice_sites_for_all_genes(db, consensus_window=1, save=False, return_df=True, **kargs):
    """
    Extract all donor and acceptor sites for all genes using gffutils, accounting for strand information.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - consensus_window (int): Number of nucleotides to include before and after the splice site.
    - save (bool): Whether to save the results to a file.
    - return_df (bool): Whether to return the results as a DataFrame.
    - **kargs: Additional keyword arguments for saving the file.

    Returns:
    - all_splice_sites (list of dicts): A list where each element is a dictionary containing
      'chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id'.

    Memo: 
    - Strand-Specific Splice Site Logic:
        * For the positive strand (+):
            - Acceptor Site: Located at the start of the current exon, except for the first exon.
            - Donor Site: Located at the end of the current exon, except for the last exon.
        * For the negative strand (-):
            - Acceptor Site: 
                Located at the end of the current exon, except for the last exon (as seen in genomic coordinates).
            - Donor Site: 
                Located at the start of the current exon, except for the first exon (as seen in genomic coordinates).

        ----xxxxxx----xxxxx---------xxxxxx---------->
                d   a    d       a

        <---xxxxxx----xxxxx---------xxxxxx-----------
                a   d    a       d

    """
    all_splice_sites = []
    transcript_counter = 0

    for transcript in tqdm(db.features_of_type('transcript'), desc="Processing transcripts"):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # Iterate over exon boundaries to find splice sites
        for i in range(len(exons)):
            exon = exons[i]

            if strand == '+':
                # Positive strand: Donor at exon end, acceptor at exon start
                exon_start = exon.start
                exon_end = exon.end

                # Acceptor site: start of the current exon (except for the first exon)
                if i > 0:
                    position = exon_start - 1  # The last nucleotide of the intron before the exon

                    # Extract consensus sequence region
                    acceptor_start = position - consensus_window
                    acceptor_end = position + consensus_window   

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': acceptor_start,
                        'end': acceptor_end,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: end of the current exon (except for the last exon)
                if i < len(exons) - 1:
                    position = exon_end + 1  # The first nucleotide of the intron after the exon

                    # Extract consensus sequence region
                    donor_start = position - consensus_window
                    donor_end = position + consensus_window     

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': donor_start,
                        'end': donor_end,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

            elif strand == '-':
                # Negative strand: Donor at exon start, acceptor at exon end (reverse logic)
                exon_start = exon.start
                exon_end = exon.end

                # Acceptor site: end of the current exon (except for the last exon in reverse order)
                if i < len(exons) - 1:
                    position = exon_end + 1  # The first nucleotide of the intron after the exon

                    # Extract consensus sequence region
                    acceptor_start = position - consensus_window
                    acceptor_end = position + consensus_window   

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': acceptor_start,
                        'end': acceptor_end,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: start of the current exon (except for the first exon in reverse order)
                if i > 0:
                    position = exon_start - 1  # The last nucleotide of the intron before the exon

                    # Extract consensus sequence region
                    donor_start = position - consensus_window
                    donor_end = position + consensus_window     

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': donor_start,
                        'end': donor_end,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

        transcript_counter += 1

    # Save or return the splice sites
    if save:
        sep = kargs.get('sep', '\t')
        format = 'csv' if sep == ',' else 'tsv'
        output_file = kargs.get('output_file', f'splice_sites.{format}')

        splice_sites_df = pd.DataFrame(all_splice_sites)
        splice_sites_df.to_csv(output_file, index=False, sep=sep)
        print(f"[i/o] Splice sites saved to {output_file}")

    return pd.DataFrame(all_splice_sites) if return_df else all_splice_sites


def extract_splice_sites_for_all_genes_v1(db, consensus_window=1, save=False, return_df=True, **kargs):
    """
    Extract all donor and acceptor sites for all genes using gffutils, accounting for strand information.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - consensus_window (int): Number of nucleotides to include before and after the splice site.
    - save (bool): Whether to save the results to a file.
    - return_df (bool): Whether to return the results as a DataFrame.
    - **kargs: Additional keyword arguments for saving the file.

    Returns:
    - pd.DataFrame or list: DataFrame or list of dictionaries containing splice site information.
    """
    all_splice_sites = []

    for transcript in tqdm(db.features_of_type('transcript'), desc="Processing transcripts"):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # Iterate over exon boundaries to find splice sites
        for i in range(len(exons)):
            exon = exons[i]
            exon_start = exon.start
            exon_end = exon.end

            if strand == '+':
                # Positive strand logic

                # Acceptor site: start of the current exon (except for the first exon)
                if i > 0:
                    position = exon_start - 1  # The last nucleotide of the intron

                    # Extract consensus sequence region
                    acceptor_start = position - consensus_window
                    acceptor_end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': acceptor_start,
                        'end': acceptor_end,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: end of the current exon (except for the last exon)
                if i < len(exons) - 1:
                    position = exon_end + 1  # The first nucleotide of the intron

                    # Extract consensus sequence region
                    donor_start = position - consensus_window
                    donor_end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': donor_start,
                        'end': donor_end,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

            elif strand == '-':
                # Negative strand logic

                # Acceptor site: end of the current exon (except for the last exon)
                if i < len(exons) - 1:
                    position = exon_end + 1  # The first nucleotide of the intron

                    # Extract consensus sequence region
                    acceptor_start = position - consensus_window
                    acceptor_end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': acceptor_start,
                        'end': acceptor_end,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: start of the current exon (except for the first exon)
                if i > 0:
                    position = exon_start - 1  # The last nucleotide of the intron

                    # Extract consensus sequence region
                    donor_start = position - consensus_window
                    donor_end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': donor_start,
                        'end': donor_end,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

    # Save or return the splice sites
    if save:
        sep = kargs.get('sep', '\t')
        file_format = 'csv' if sep == ',' else 'tsv'
        output_file = kargs.get('output_file', f'splice_sites.{file_format}')

        splice_sites_df = pd.DataFrame(all_splice_sites)
        splice_sites_df.to_csv(output_file, index=False, sep=sep)
        print(f"[i/o] Splice sites saved to {output_file}")

    return pd.DataFrame(all_splice_sites) if return_df else all_splice_sites


def create_asymmetric_labels(window_size, site_type):
    """
    Create asymmetric probability labels for splice sites based on type.
    
    - window_size: Size of the consensus window around the splice site.
    - site_type: Either 'donor' or 'acceptor'.
    
    Returns:
    - A list of probability labels decreasing away from the splice site.
    """
    labels = [1]  # Exact site has label 1

    # Probability drops based on consensus window, decreasing away from the site
    for i in range(1, window_size + 1):
        if site_type == 'donor':
            labels.insert(0, 1 - i * 0.1)  # Adjust these values based on expected importance
        elif site_type == 'acceptor':
            labels.append(1 - i * 0.1)  # Adjust based on expected importance
    
    return labels


def extend_annotated_sites_with_consensus(annotated_splice_df, consensus_sequences, tolerance):
    """
    Extend annotated splice sites to account for consensus sequences around donor and acceptor sites, considering strand.

    Parameters:
    - annotated_splice_df (pd.DataFrame): Annotated splice sites DataFrame.
    - consensus_sequences (dict): A dictionary with 'donor' and 'acceptor' keys and their consensus sequences as values.
    - tolerance (int): Number of nucleotides around the splice site boundary to be considered a match.

    Returns:
    - extended_regions (dict): A dictionary of annotated regions extended by consensus sequences.
    """
    extended_regions = defaultdict(lambda: {'donor': [], 'acceptor': []})

    donor_consensus_length = len(consensus_sequences['donor'])
    acceptor_consensus_length = len(consensus_sequences['acceptor'])

    for idx, row in annotated_splice_df.iterrows():
        transcript_id = row['transcript_id']
        chrom = row['chrom']
        strand = row['strand']
        splice_type = row['splice_type']

        if splice_type == 'donor':
            if strand == '+':
                start = row['end'] - donor_consensus_length - tolerance
                end = row['end'] + tolerance
            else:
                start = row['start'] - tolerance
                end = row['start'] + donor_consensus_length + tolerance

            extended_regions[transcript_id]['donor'].append((chrom, start, end, strand))

        elif splice_type == 'acceptor':
            if strand == '+':
                start = row['start'] - tolerance
                end = row['start'] + acceptor_consensus_length + tolerance
            else:
                start = row['end'] - acceptor_consensus_length - tolerance
                end = row['end'] + tolerance

            extended_regions[transcript_id]['acceptor'].append((chrom, start, end, strand))

    return extended_regions


def match_predicted_to_annotated_sites(spliceai_predictions, annotated_sites, tolerance=2):
    """
    Match predicted splice sites to annotated splice sites within a positional tolerance.

    Parameters:
    - spliceai_predictions (pd.DataFrame): DataFrame containing SpliceAI predictions.
    - annotated_sites (pd.DataFrame): DataFrame of annotated splice sites from extract_splice_sites_for_all_genes().
    - tolerance (int): Number of nucleotides to consider around the annotated site for a match.

    Returns:
    - results_df (pd.DataFrame): DataFrame containing matched predictions with labels (TP, FP, FN).
    """
    import pandas as pd

    # Ensure 'position' is included in annotated_sites
    annotated_sites['position'] = (annotated_sites['start'] + annotated_sites['end']) // 2

    # Create sets for quick lookup
    annotated_positions = set(zip(
        annotated_sites['chrom'],
        annotated_sites['position'],
        annotated_sites['strand'],
        annotated_sites['site_type']
    ))

    # Initialize lists to store results
    true_positives = []
    false_positives = []
    false_negatives = []

    # Iterate over predicted positions
    for idx, row in spliceai_predictions.iterrows():
        chrom = row['chrom']
        position = row['position']
        strand = row['strand']

        # Check for donor prediction
        if row['donor_prob'] >= 0.5:  # Threshold can be adjusted
            matched = False
            for delta in range(-tolerance, tolerance + 1):
                key = (chrom, position + delta, strand, 'donor')
                if key in annotated_positions:
                    true_positives.append(row)
                    matched = True
                    break
            if not matched:
                false_positives.append(row)
        # Check for acceptor prediction
        if row['acceptor_prob'] >= 0.5:
            matched = False
            for delta in range(-tolerance, tolerance + 1):
                key = (chrom, position + delta, strand, 'acceptor')
                if key in annotated_positions:
                    true_positives.append(row)
                    matched = True
                    break
            if not matched:
                false_positives.append(row)

    # Find false negatives
    predicted_positions = set(zip(
        spliceai_predictions['chrom'],
        spliceai_predictions['position'],
        spliceai_predictions['strand']
    ))
    for idx, row in annotated_sites.iterrows():
        chrom = row['chrom']
        position = row['position']
        strand = row['strand']
        site_type = row['site_type']

        matched = False
        for delta in range(-tolerance, tolerance + 1):
            key = (chrom, position + delta, strand)
            if key in predicted_positions:
                matched = True
                break
        if not matched:
            false_negatives.append(row)

    # Compile results
    tp_df = pd.DataFrame(true_positives)
    fp_df = pd.DataFrame(false_positives)
    fn_df = pd.DataFrame(false_negatives)

    return tp_df, fp_df, fn_df


def extract_annotations(gtf_file, db_file='annotations.db', output_file='annotations_all_transcripts.tsv', **kargs):
    # Step 1: Build the database
    db = build_gtf_database(gtf_file, db_file=db_file, overwrite=False)

    # Step 2: Extract exon, CDS, 5'UTR, and 3'UTR annotations for all transcripts from the DB
    annotations_df = extract_all_gtf_annotations(db)

    # Step 3: Save annotations to a file
    annotations_df.to_csv(output_file, index=False, **kargs)
    print(f"[info] Annotations extracted and saved to {output_file}")


def transcript_sequence_retrieval_workflow(gtf_file, genome_fasta, gene_tx_map=None, output_file="tx_sequence.parquet"):
    from .utils_bio import (
        extract_transcripts_from_gtf,
        extract_transcript_sequences,
        save_sequences, 
        save_sequences_by_chromosome, 
        load_sequences, 
        load_sequences_by_chromosome
    )

    # data_prefix = "/path/to/meta-spliceai/data/ensembl"
    # local_dir = "/path/to/meta-spliceai/data/ensembl"
    # genome_annot = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf") 
    # genome_fasta = os.path.join(data_prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") # "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
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
    transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids=tx_ids, ignore_version=True)
    print(transcripts_df.head())  # Display the extracted transcript information

    additional_columns = ['seqname', 'start', 'end', 'strand', ]
    seq_df = extract_transcript_sequences(transcripts_df, genome_fasta, output_format='dataframe', include_columns=additional_columns)
    
    format = 'parquet'
    output_path = output_file  # os.path.join(local_dir, f"tx_sequence.{format}")
    assert os.path.exists(os.path.dirname(output_path))

    print(f"[i/o] Saving transcript sequences to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    
    
    # NOTE: Columns: ['transcript_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']
    # save_sequences(seq_df, output_path, format=format)

    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    save_sequences_by_chromosome(seq_df, output_path, format=format, chromosomes_to_save=chromosomes)

    return 


def gene_specific_sequence_retrieval_workflow(gtf_file, genome_fasta, gene_tx_map=None, output_file="gene_sequence.parquet", mode='minmax'): 
    from .utils_bio import (
        extract_genes_from_gtf,
        extract_gene_sequences,
        save_sequences_by_chromosome
    )
   
    if mode == 'minmax': 
        print("[info] Extract gene-specific sequences with min start and max end across all transcripts ...")
        return gene_sequence_retrieval_minmax_workflow(gtf_file, genome_fasta, gene_tx_map=gene_tx_map, output_file=output_file)
    else: 
        print("[info] Extract the entire DNA sequence for each gene ...")

    genes_df = extract_genes_from_gtf(gtf_file)
    print("[info] Gene dataframe:")
    print(genes_df.head())  # Display the extracted gene information

    seq_df = extract_gene_sequences(genes_df, genome_fasta, output_format='dataframe', include_columns=None)

    format = 'parquet'
    output_path = output_file  

    # Check if the directory part of the output_path is empty
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        # If output_path is just a file name, assume the current directory as the parent directory
        output_path = os.path.join(os.getcwd(), output_path)
        output_dir = os.getcwd()
    assert os.path.exists(output_dir), "Invalid output path: {}".format(output_path)

    print(f"[i/o] Saving the full DNA sequence for each gene to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    
    # NOTE: Columns: ['transcript_id', 'gene_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']
    # save_sequences(seq_df, output_path, format=format)

    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    save_sequences_by_chromosome(seq_df, output_path, format=format, chromosomes_to_save=chromosomes)  


def gene_sequence_retrieval_minmax_workflow(gtf_file, genome_fasta, gene_tx_map=None, output_file="gene_sequence.parquet"): 
    
    from .utils_bio import (
        extract_transcripts_from_gtf,
        extract_gene_sequences_minmax,
        save_sequences_by_chromosome, 
        load_sequences_by_chromosome
    )

    # src_dir = "/path/to/meta-spliceai/data/ensembl"
    # local_dir = "/path/to/meta-spliceai/data/ensembl"
    # genome_annot = os.path.join(src_dir, "Homo_sapiens.GRCh38.112.gtf") 
    # genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")  # "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

    # print("[info] Extract gene-specific sequences with min start and max end across all transcripts ...")

    # Extract transcripts and their features
    transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids=gene_tx_map, ignore_version=True)
    print("> Transcript dataframe:")
    print(transcripts_df.head())  # Display the extracted transcript information

    # additional_columns = ['seqname', 'start', 'end', 'strand', ]
    seq_df = extract_gene_sequences_minmax(transcripts_df, genome_fasta, output_format='dataframe', include_columns=None)
    print("> DNA sequence dataframe:")
    print("... cols(seq_df): ", list(seq_df.columns))
    # A DataFrame with the following minimum columns:
    #     - gene_id: The unique identifier for each gene.
    #     - sequence: The extracted DNA sequence for the gene.
    #     - strand: The strand on which the gene is located ('+' or '-').
    #     - start: The 1-based start coordinate of the gene sequence.
    #     - end: The end coordinate of the gene sequence.

    # Assuming gene_sequences_df is the dataframe with the columns mentioned
    min_start_position = seq_df['start'].min()
    print(f"[info] Minimum start position: {min_start_position}")

    # Test: Determine if the annotation is 1-based or 0-based
    if min_start_position == 0:
        print("The annotation is 0-based.")
    else:
        print("The annotation is 1-based.")

    # Save the gene-specific DNA sequences to a file
    format = 'parquet'
    output_path = output_file  
    assert os.path.exists(os.path.dirname(output_path)), "Invalid output path: {}".format(output_path)

    print(f"[i/o] Saving gene-specific DNA sequences to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    
    # NOTE: Columns: ['transcript_id', 'gene_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']
    # save_sequences(seq_df, output_path, format=format)

    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    save_sequences_by_chromosome(seq_df, output_path, format=format, chromosomes_to_save=chromosomes)    


def extract_splice_sites_workflow(data_prefix, gtf_file, consensus_window=2, **kargs):
    """
    Extract splice sites workflow.

    Parameters:
    - data_prefix (str): The prefix directory for data files.
    - gtf_file (str): The path to the gene annotation GTF file.
    - consensus_window (int): The consensus window size for extracting splice sites (default is 2).
    """
    import gffutils

    assert os.path.exists(gtf_file), f"GTF file not found: {gtf_file}"
    db_file = os.path.join(data_prefix, "annotations.db")
    # NOTE: extract_annotations() will create the database file if it does not exist

    print("Step 1: Build gene annotation database and extract all annotations ...")
    format = 'tsv'
    output_file = os.path.join(data_prefix, f"annotations_all_transcripts.{format}")
    
    # Check if the output file exists and its size
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        print(f"[info] Annotation file not found at:\n{output_file}\n")

        # Extract exon, CDS, 5'UTR, and 3'UTR annotations for all transcripts from the DB
        extract_annotations(gtf_file, db_file=db_file, output_file=output_file, sep='\t')
    else:
        print(f"[info] Found the extracted annotation file at:\n{output_file}\n")

    # Load the annotations from the saved file
    annotations_df = pd.read_csv(output_file, sep='\t')
    # NOTE: By default, pd.read_csv() reads the first row of the CSV file as the header; i.e. header=0 by default

    print("(extract_splice_sites_workflow) annotations_df:")
    print(annotations_df.head())

    print("Step 2: Extract splice sites ...")
    db = gffutils.FeatureDB(db_file)
    print(f"[info] Database loaded from {db_file}")

    format = 'tsv'
    output_file = os.path.join(data_prefix, f"splice_sites.{format}")
    extract_splice_sites_for_all_genes(db, consensus_window=consensus_window, save=True, output_file=output_file)


def test(): 

    demo_workflow()


if __name__ == "__main__": 
    test()

