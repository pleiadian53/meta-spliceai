import os, sys
import pandas as pd
import csv
# from commons import write_gtf_line

import requests
import urllib.request
import subprocess

import numpy as np
import pprint

from meta_spliceai.system.config import Config
from Bio import SeqIO

import meta_spliceai.system.sys_config as config

from meta_spliceai.utils.bio_utils import (
    count_sequences, 
    create_gtf, 
    parse_gtf,
    parse_gff,
    find_orfs
)


def write_gtf_line(gtf_fh, ctg, feature, start, end, strand, gid, oid, tid, ref_id, splice=False, sites=None):
    """
    Writes a single line to a GTF file with optional splicing information.
    """
    attributes = f'gene_id "{gid}"; orf_id "{oid}"; transcript_id "{tid}"; reference_id "{ref_id}";'
    if splice:
        attributes += f" splice; sites ({', '.join(map(str, sites))});"
    gtf_fh.write(f"{ctg}\tuORFexplorer\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attributes}\n")


#############################################
def gunzip(file):
    if os.path.exists(file):
        try:
            subprocess.check_call(['gunzip', file])
        except subprocess.CalledProcessError as e:
            print(f"Gunzip failed with error: {e}")
    else:
        print(f"File {file} does not exist.")


def wget(url, filename):
    """
    The wget function downloads a file from a given URL and saves it to a specified location. 
    It uses the requests library for HTTP/HTTPS URLs and urllib for FTP URLs.

    Memo
    ----
    1. urllib is a package that collects several modules for working with URLs, and 
       urllib.request is a module for opening and reading URLs. 
       
       The urlretrieve function in the urllib.request module is used to download a URL to a local file. 
       This function is suitable for downloading files from both HTTP/HTTPS and FTP URLs.


    """
    # import urllib.request
    if url.startswith('ftp://'):
        print(f"Downloading file from {url} to {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download completed.")
    else:
        response = requests.get(url, stream=True)
        # NOTE: It uses the requests library to make a GET request to the URL, and writes the response content to a file. 
        #       If the status code of the response is not 200, it prints an error message. 

        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"Failed to download file, status code: {response.status_code}")

def download_dff3_from_gencode(url=None, filepath=None, **kargs): 

    # Download Gencode human release 43 annotation files
    if url is None:
        url = 'ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gff3.gz'
     
    if filepath is None:
        data_dir = kargs.get("data_dir", "./data/gencode/")

        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        fname = 'gencode.v43.annotation.gff3.gz'
        filepath = os.path.join(data_dir, fname) 

    print("> Download gencode human release 43 annotation files to: {} ...".format(filepath))
    # Run commands: 
    # - wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gff3.gz
    # - gunzip gencode.v43.annotation.gff3.gz
    run_wget_and_gunzip(url, filepath)

    return

def download_fasta_from_gencode(url=None, filepath=None, **kargs):
    url = "ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/GRCh38.primary_assembly.genome.fa.gz"
    data_dir = kargs.get("data_dir", "./data/gencode/")

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, "GRCh38.primary_assembly.genome.fa.gz")

    print("> Downloading the FASTA file from GENCODE to {} ...".format(output_path))
    run_wget_and_gunzip(url, output_path)

    return

def run_wget_and_gunzip(url, filepath):
    """
    This function downloads a file from a given URL and saves it to a specified location. 
    It then unzips the file using the gunzip command.
    """
    wget(url, filepath)
    gunzip(filepath)
    return

def download_gencode_files(url=None, filepath=None, **kargs): 

    data_dir = kargs.get("data_dir", "./data/gencode/")

    # Download annotations for the target genome (GFF3)
    # url = 'ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gff3.gz'
    # fname = 'gencode.v43.annotation.gff3.gz'
    # output_path = os.path.join(data_dir, fname)
    download_dff3_from_gencode(data_dir=data_dir)

    # Download the FASTA file from GENCODE
    # wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/GRCh38.primary_assembly.genome.fa.gz
    # gunzip GRCh38.primary_assembly.genome.fa.gz
    # url = "ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/GRCh38.primary_assembly.genome.fa.gz"
    # fname = "GRCh38.primary_assembly.genome.fa.gz"
    # output_path = os.path.join(data_dir, fname)
    # run_wget_and_gunzip(url, output_path)
    download_fasta_from_gencode(data_dir=data_dir)
    
    return

#############################################


def run_uorfinder_data_preparation_workflow(data_dir=None):
    from meta_spliceai.utils.download_genomes import (
        download_pr1_genome, 
        download_ash1_genome, 
        download_chm13_genome, 
        download_han1_genome,
        # download_genome_and_annotation_given_accession_number
        is_fasta, 
        is_gff
    )
    from meta_spliceai.utils.utils_excel import load_excel_sheets_into_dataframes
    # from analyze_uorf_data import prepare_input_data_for_uorfinder

    if data_dir is None: 
        data_dir = os.path.join(config.get_proj_dir(), "data")

    # dfs = prepare_input_data_for_uorfinder(input_dir=data_dir)
    # df = dfs['S2. uORF-connected transcripts']
    # print(df.head())
    # print("uORF-connected transcripts: Columns={}".format(list(df.columns)))
    
    # reference_ids = df['reference_id'].unique()
    # print("> Number of reference IDs: {}".format(len(reference_ids)))

    # Download Gencode annotation v43 and fasta
    data_dir = os.path.join(config.get_proj_dir(), "data/gencode")
    # download_gencode_files(data_dir=data_dir)

    # Verify the downloaded GFF file
    filepath = "/path/to/meta-spliceai/data/gencode/gencode.v43.annotation.gff3"
    is_gff(filepath, num_lines=50)

    # Verify the downloaded FASTA file
    filepath = "/path/to/meta-spliceai/data/gencode/GRCh38.primary_assembly.genome.fa"
    is_fasta(filepath, num_lines=50)

    # PR1 Genome (Puerto Rican individual, v3.0)
    output_dir = os.path.join(config.get_proj_dir(), "data/PR1")
    genome_path, annotation_path, transcript_path = \
        download_pr1_genome(data_dir=output_dir, extract_transcript=True)  
    print("[test] Genome path: {}".format(genome_path))
    print("[test] Annotation path: {}".format(annotation_path))

    # output_dir = os.path.join(data_dir, "PR1")
    # download_genome_and_annotation_given_accession("GCA_018873775.2", output_dir=output_dir)
    # Todo: Given accession number, download genome and annotation files

    return

def verify_genome_data():     
    # PR1 Genome (Puerto Rican individual, v3.0)
    output_dir = os.path.join(config.get_proj_dir(), "data/PR1")

    is_fasta(genome_path, num_lines=50)
    is_gff(annotation_path, num_lines=50)


def demo_create_gtf():

    output_dir = os.path.join(os.getcwd(), "data")

    # Create a list of feature dictionaries
    print("(demo_create_gtf) Creating a list of feature dictionaries ...")
    features = [
        {"ctg": "chr1", "feature": "transcript", "start": 100352597, "end": 100499379, "strand": "+", 
        "gid": "ENSG00000079335.20", "oid": "c1riboseqorf129", "tid": "uorft_1", "ref_id": "ENST00000644676.1", 
        "splice": True, "sites": [100352936, 100353761]},
        {"ctg": "chr1", "feature": "exon", "start": 100352597, "end": 100352936, "strand": "+", 
        "gid": "ENSG00000079335.20", "oid": "c1riboseqorf129", "tid": "uorft_1", "ref_id": "ENST00000644676.1", 
        "splice": True, "sites": [100352936, 100353761]},
        # Add more features as needed
    ]

    print("(create_gtf_v0) Writing GTF lines to a file ...")
    create_gtf_v0(features, os.path.join(output_dir, 'example_v0.gtf'))

    ##################################

    # Example usage
    features = [
        {
            'seqname': 'chr1',   # transcripts.chromosome
            'source': 'uORFexplorer',  # TXdb
            'feature': 'transcript', # 
            'start': 100352597,  # transcripts.start
            'end': 100499379,  # transcripts.end
            'score': '.',   # '.'
            'strand': '+',  # transcripts.strand
            'frame': '.',   # '.'
            'attributes': {
                'gene_id': 'ENSG00000079335.20', # txdb.genes
                
                'transcript_id': 'uorft_1', # txdb.transcripts
                'associated_gene': 'MIST1',  # txdb.genes
                'transcript_biotype': '', # tx_type_ref, tx_type_pred

                'orf_id': 'c1riboseqorf129',
                'reference_id': 'ENST00000644676.1',
                'splice': '',
                'sites': '(100352936, 100353761)'
            }
        },
        {
            'seqname': 'chr1',
            'source': 'uORFexplorer',
            'feature': 'exon',
            'start': 100352597,
            'end': 100352936,
            'score': '.',
            'strand': '+',
            'frame': '.',
            'attributes': {
                'gene_id': 'ENSG00000079335.20',
                'orf_id': 'c1riboseqorf129',
                'transcript_id': 'uorft_1',
                'reference_id': 'ENST00000644676.1',
                'splice': '',
                'sites': '(100352936, 100353761)'
            }
        },
        {
            'seqname': 'chr1',
            'source': 'uORFexplorer',
            'feature': 'CDS',
            'start': 100352597,
            'end': 100352936,
            'score': '.',
            'strand': '+',
            'frame': '.',
            'attributes': {
                'gene_id': 'ENSG00000079335.20',
                'orf_id': 'c1riboseqorf129',
                'transcript_id': 'uorft_1',
                'reference_id': 'ENST00000644676.1',
                'splice': '',
                'sites': '(100352936, 100353761)'
            }
        },
        # Add more entries as needed...
    ]

    print("(create_gtf) Writing GTF lines to a file ...")
    output_path = os.path.join(output_dir, 'example_v1.gtf')
    create_gtf(features, output_path)

    features_prime = parse_gtf(output_path)
    assert features == features_prime

    print("(parse_gtf) Example row from the parsed GTF file:")
    # print(np.random.choice(features_prime, 1)[0])
    pprint.pprint(features_prime)


    return


def test(): 

    # demo_create_gtf()

    # Prepare and analyze experimental data for uORFinder
    run_uorfinder_data_preparation_workflow()

    return



if __name__ == "__main__": 
    test()