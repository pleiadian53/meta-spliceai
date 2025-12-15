import os 
import requests

from urllib.parse import urlparse
import subprocess

from .utils_sys import (
    wget,
    gunzip,
    run_wget_and_gunzip
)


def download_dff3_from_gencode(url=None, filepath=None, **kargs): 
    release = kargs.get("version", 43)

    # Download Gencode human release 43 annotation files
    if url is None:
        # url = 'ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gff3.gz'
        url = f'ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/gencode.v{release}.annotation.gff3.gz'

    if filepath is None:
        data_dir = kargs.get("data_dir", "./data/gencode/")

        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # fname = 'gencode.v43.annotation.gff3.gz'
        fname = f'gencode.v{release}.annotation.gff3.gz'
        filepath = os.path.join(data_dir, fname) 

    print(f"> Download gencode human release {release} annotation files to: {filepath} ...")
    # Run commands: 
    # - wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/gencode.v{release}.annotation.gff3.gz
    # - gunzip gencode.v{release}.annotation.gff3.gz
    run_wget_and_gunzip(url, filepath)

    return


def download_genome_from_gencode(url=None, filepath=None, **kargs):
    release = kargs.get("version", 43)

    url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/GRCh38.primary_assembly.genome.fa.gz"
    data_dir = kargs.get("data_dir", "./data/gencode/")

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, "GRCh38.primary_assembly.genome.fa.gz")

    print("> Downloading the FASTA file from GENCODE to {} ...".format(output_path))
    run_wget_and_gunzip(url, output_path)

    return

def download_transcriptome_from_gencode(url=None, filepath=None, **kargs):
    release = kargs.get("version", 43)

    url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/gencode.v{release}.transcripts.fa.gz"
    data_dir = kargs.get("data_dir", "./data/gencode/")

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, f"gencode.v{release}.transcripts.fa.gz")

    print("> Downloading the transcriptome FASTA file from GENCODE to {} ...".format(output_path))
    run_wget_and_gunzip(url, output_path)

    return

######################

def download_data(url, data_dir, filename):
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, filename)
    
    print(f"> Downloading from {url} to {output_path} ...")
    run_wget_and_gunzip(url, output_path)
    return output_path

# def download_genome_from_gencode(version=43, data_dir="./data/gencode/"):
#     url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/GRCh38.primary_assembly.genome.fa.gz"
#     filename = "GRCh38.primary_assembly.genome.fa.gz"
#     return download_data(url, data_dir, filename)

# def download_transcriptome_from_gencode(version=43, data_dir="./data/gencode/"):
#     url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/gencode.v{version}.transcripts.fa.gz"
#     filename = f"gencode.v{version}.transcripts.fa.gz"
#     return download_data(url, data_dir, filename)

def is_gff(filename, num_lines=10):
    # Check if file exists
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return False

    # Check file size
    if os.path.getsize(filename) == 0:
        print(f"File {filename} is empty.")
        return False

    with open(filename, "r") as file:
        for i, line in enumerate(file):
            if i >= num_lines:  # Only check first num_lines lines
                break
            if not line.strip():  # Ignore empty lines
                continue
            # Check if line starts with '#' (comment) or contains 9 tab-separated fields (data)
            if not (line.startswith("#") or len(line.split("\t")) == 9):
                print(f"Invalid line in file {filename}: {line}")
                return False

    print(f"File {filename} is verified as GFF.")
    return True

def is_fasta(filename, num_lines=10):
    # Check if file exists
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return False

    # Check file size
    if os.path.getsize(filename) == 0:
        print(f"File {filename} is empty.")
        return False

    with open(filename, "r") as file:
        first_line = file.readline()
        if not first_line.startswith(">"):  # Check if description line starts with '>'
            print(f"File {filename} does not start with '>': {first_line}")
            return False

        for i, line in enumerate(file):
            if not line.strip():  # Ignore empty lines
                continue
            # Check if line is shorter than 80 characters and contains only valid characters
            if len(line) > 80 or not set(line.strip()).issubset(set("ACGTNacgtn")):
                print(f"Invalid line in file {filename}: {line}")
                return False
            if i >= num_lines:  # Only check first num_lines lines
                break

    print(f"File {filename} is verified as FASTA.")
    return True


def download_genome_and_annotation(genome_url, annotation_url, genome_name, data_dir=None, extract_transcript=True):
    # from urllib.parse import urlparse
    # import subprocess
    
    # Define the data directory if not provided
    if data_dir is None:
        data_dir = f"./data/genomes/{genome_name}/"

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    print("(download_genome_and_annotation) data_dir: ", data_dir)

    # Extract filenames from the URLs
    genome_filename = os.path.basename(urlparse(genome_url).path)
    annotation_filename = os.path.basename(urlparse(annotation_url).path)

    # Define the paths where the files will be saved
    genome_path = os.path.join(data_dir, genome_filename)
    annotation_path = os.path.join(data_dir, annotation_filename)

    # Verify the uncompressed file paths
    genome_decompressed = os.path.splitext(genome_path)[0]
    annotation_decompressed = os.path.splitext(annotation_path)[0]
    
    if os.path.exists(genome_decompressed): 
        is_fasta(genome_decompressed, num_lines=50)
        print(f"> Genome {genome_name} already exists at {genome_decompressed} ...")
        genome_path = genome_decompressed
    else: 
        # Download the genome and annotation
        print(f"> Downloading genome {genome_name} to {genome_path} ...")
        run_wget_and_gunzip(genome_url, genome_path)

    if os.path.exists(annotation_decompressed):
        is_gff(annotation_decompressed, num_lines=50)
        print(f"> Annotation for {genome_name} already exists at {annotation_decompressed} ...")
        annotation_path = annotation_decompressed
    else: 
        print(f"> Downloading annotation for {genome_name} to {annotation_path} ...")
        run_wget_and_gunzip(annotation_url, annotation_path)

    # Extract transcript sequences from genome using gffread
    transcript_path = None
    if extract_transcript: 
        transcript_path = os.path.join(data_dir, f"{genome_name}_transcripts.fa")
        print("> Extracting transcript sequences from genome using gffread and save to: ", transcript_path)
        gffread_cmd = f"gffread -w {transcript_path} -g {genome_path} {annotation_path}"
        subprocess.run(gffread_cmd, shell=True)
    
    return genome_path, annotation_path, transcript_path


def download_pr1_genome(data_dir="./data/genomes/PR1/", extract_transcript=True):
    genome_url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/018/873/775/GCA_018873775.2_hg01243.v3.0/GCA_018873775.2_hg01243.v3.0_genomic.fna.gz"
    annotation_url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/018/873/775/GCA_018873775.2_hg01243.v3.0/GCA_018873775.2_hg01243.v3.0_genomic.gff.gz"
    
    print("[test] URL: ", genome_url)
    
    # genome_filename = "PR1.fa.gz"
    # annotation_filename = "pr1v3.0_final.gff3.gz"
    
    # genome_path = download_data(genome_url, data_dir, genome_filename)
    # annotation_path = download_data(annotation_url, data_dir, annotation_filename)

    # # Extract transcript sequences from genome using gffread
    # transcript_path = None
    # if extract_transcript: 
    #     transcript_path = os.path.join(data_dir, "PR1_transcripts.fa")
    #     gffread_cmd = f"gffread -w {transcript_path} -g {genome_path} {annotation_path}"
    #     subprocess.run(gffread_cmd, shell=True)

    genome_path, annotation_path, transcript_path = \
        download_genome_and_annotation(
            genome_url=genome_url, 
            annotation_url=annotation_url, 
            genome_name="PR1", 
            data_dir=data_dir)
    
    return genome_path, annotation_path, transcript_path


def download_ash1_genome(): 
    """
    wget ftp://ftp.ccb.jhu.edu/pub/data/Homo_sapiens/Ash1/v2.2/Assembly/Ash1_v2.2.fa.gz
    wget ftp://ftp.ccb.jhu.edu/pub/data/Homo_sapiens/Ash1/v2.2/Annotation/Ash1_v2.2.gff3.gz
    """
    genome_url = "ftp://ftp.ccb.jhu.edu/pub/data/Homo_sapiens/Ash1/v2.2/Assembly/Ash1_v2.2.fa.gz"
    annotation_url = "ftp://ftp.ccb.jhu.edu/pub/data/Homo_sapiens/Ash1/v2.2/Annotation/Ash1_v2.2.gff3.gz"

    genome_path, annotation_path, transcript_path = \
        download_genome_and_annotation(
            genome_url=genome_url, 
            annotation_url=annotation_url, 
            genome_name="Ash1")
    
    return genome_path, annotation_path, transcript_path


def download_han1_genome(data_dir="./data/genomes/HAN1/", extract_transcript=True):
    genome_url = "ftp://ftp.ccb.jhu.edu/pub/data/T2T-CHM13/Han1.fasta"
    # ftp://ftp.ccb.jhu.edu/pub/data/Homo_sapiens/Han1/v1.0/Assembly/Han1_v1.2.fasta
    
    annotation_url = "ftp://ftp.ccb.jhu.edu/pub/data/T2T-CHM13/Han1.gff3"
    # ftp://ftp.ccb.jhu.edu/pub/data/Homo_sapiens/Han1/v1.0/Annotation/Han1_v1.4.gff3
    
    # genome_filename = os.path.basename(urlparse(genome_url).path)
    # annotation_filename = os.path.basename(urlparse(annotation_url).path)
    
    # genome_path = download_data(genome_url, data_dir, genome_filename)
    # annotation_path = download_data(annotation_url, data_dir, annotation_filename)

    # # Extract transcript sequences from genome using gffread
    # transcript_path = None
    # if extract_transcript: 
    #     transcript_path = os.path.join(data_dir, "Han1_transcripts.fa")
    #     gffread_cmd = f"gffread -w {transcript_path} -g {genome_path} {annotation_path}"
    #     subprocess.run(gffread_cmd, shell=True)

    genome_path, annotation_path, transcript_path = \
        download_genome_and_annotation(
            genome_url=genome_url, 
            annotation_url=annotation_url, 
            genome_name="HAN1")
    
    return genome_path, annotation_path, transcript_path


def download_chm13_genome(data_dir="./data/genomes/CHM13/", extract_transcript=True):
    """
    Using NCBI links
    wget "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_009914755.1/download?include_annotation_type=GENOME_FASTA&include_annotation_type=GENOME_GFF&include_annotation_type=RNA_FASTA&include_annotation_type=CDS_FASTA&include_annotation_type=PROT_FASTA&include_annotation_type=SEQUENCE_REPORT&hydrated=FULLY_HYDRATED" -O chm13v2.0_ncbi.zip

    Other links: 
    wget https://github.com/marbl/CHM13/releases/download/v2.0/CHM13v2.0.fa.gz
    wget https://github.com/marbl/CHM13/releases/download/v2.0/CHM13v2.0.gff3.gz
    gunzip CHM13v2.0.fa.gz
    gunzip CHM13v2.0.gff3.gz
    """
    genome_url = "https://github.com/marbl/CHM13/releases/download/v2.0/CHM13v2.0.fa.gz"
    annotation_url = "https://github.com/marbl/CHM13/releases/download/v2.0/CHM13v2.0.gff3.gz"
    
    # genome_filename = os.path.basename(urlparse(genome_url).path)
    # annotation_filename = os.path.basename(urlparse(annotation_url).path)
    
    # genome_path = download_data(genome_url, data_dir, genome_filename)
    # annotation_path = download_data(annotation_url, data_dir, annotation_filename)

    # # Extract transcript sequences from genome using gffread
    # transcript_path = None
    # if extract_transcript: 
    #     transcript_path = os.path.join(data_dir, "CHM13v2.0_transcripts.fa")
    #     gffread_cmd = f"gffread -w {transcript_path} -g {genome_path} {annotation_path}"
    #     subprocess.run(gffread_cmd, shell=True)

    genome_path, annotation_path, transcript_path = \
        download_genome_and_annotation(
            genome_url=genome_url, 
            annotation_url=annotation_url, 
            genome_name="CHM13")
    
    return genome_path, annotation_path, transcript_path


def download_and_decompress_genome(url, genome, filename=None, **kargs): 
    # from urllib.parse import urlparse

    data_dir = kargs.get("data_dir", f"./data/{genome}/")

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # If filename is not provided, extract it from the URL
    if filename is None:
        filename = os.path.basename(urlparse(url).path)
 
    output_path = os.path.join(data_dir, filename)

    print(f"> Downloading genome {genome} to {output_path} ...")
    run_wget_and_gunzip(url, output_path)

    return

######################

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def download_genome_and_annotation_given_accession_number(accession_number, genome_name, data_dir=None, extract_transcript=True):
    # Construct the URLs for the genome and annotation
    base_url = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
    genome_url = f"{base_url}/{accession_number}/{accession_number}_genomic.fna.gz"
    annotation_url = f"{base_url}/{accession_number}/{accession_number}_genomic.gff.gz"

    # Call the download_genome_and_annotation function with the constructed URLs
    genome_path, annotation_path, transcript_path = \
        download_genome_and_annotation(
            genome_url=genome_url, 
            annotation_url=annotation_url, 
            genome_name=genome_name, 
            data_dir=data_dir,
            extract_transcript=extract_transcript)
    
    return genome_path, annotation_path, transcript_path

def download_genome_and_annotation_given_accession_number_v0(accession, output_dir="./data/PR1/"):
    os.makedirs(output_dir, exist_ok=True)

    # Construct directory structure
    directory = f"{accession[:3]}/{accession[4:7]}/{accession[7:10]}/{accession}/{accession}_hg01243.v3.0/"
    # e.g. https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/018/873/775/GCA_018873775.2_hg01243.v3.0/

    # Correct file names for the genome and annotation
    genome_filename = f"{accession}_hg01243.v3.0_genomic.fna.gz"
    annotation_filename = f"{accession}_hg01243.v3.0_genomic.gff.gz"

    # Construct URLs
    base_url = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{directory}"
    genome_url = f"{base_url}{genome_filename}"
    annotation_url = f"{base_url}{annotation_filename}"

    # Local paths
    genome_path = os.path.join(output_dir, genome_filename)
    annotation_path = os.path.join(output_dir, annotation_filename)

    # Download files
    print(f"Downloading genome to {genome_path} ...")
    download_file(genome_url, genome_path)
    
    print(f"Downloading annotation to {annotation_path} ...")
    download_file(annotation_url, annotation_path)

    print("Download complete.")
    
    return genome_path, annotation_path


######################

def download_and_decompress_genome_v0(url, genome):
    # Define file paths
    gz_path = f'{genome}_genome.fasta.gz'
    fasta_path = f'{genome}_genome.fasta'

    # Check if the FASTA file already exists
    if os.path.exists(fasta_path):
        print(f'{genome} genome already downloaded and decompressed.')
        return

    # Download genome
    print(f'Downloading {genome} genome...')
    response = requests.get(url, stream=True)

    # Check if the download was successful
    if response.status_code != 200:
        print(f'Error downloading {genome} genome: {response.status_code}')
        return

    with open(gz_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f'{genome} genome downloaded to {gz_path}')

    # Decompress the downloaded file
    print(f'Decompressing {genome} genome...')
    subprocess.run(['gunzip', gz_path])
    print(f'{genome} genome decompressed.')

