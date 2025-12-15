import os, sys
import requests
# from subprocess import call, check_output
import subprocess
import pandas as pd
from collections import defaultdict
from meta_spliceai.utils.utils_doc import print_emphasized

from meta_spliceai.splice_engine.extract_genomic_features import (
    extract_junctions_for_gene,
)
from tqdm import tqdm


def get_transcript_info(gene_name, species='human'):
    """
    Get transcript information for a given gene using Ensembl REST API.
    
    Parameters:
    - gene_name (str): The name of the gene (e.g., STMN2, UNC13A).
    - species (str): The species of the gene (default is human).
    
    Returns:
    - dict: A dictionary of transcript information including exons.

        - Example dictionary structure:
            {
                "id": "ENSG00000112345",  # Ensembl Gene ID
                "seq_region_name": "19",  # Chromosome or contig name
                "start": 12345678,
                "end": 12356789,
                "strand": 1,  # 1 for positive strand, -1 for negative strand
                "Transcript": [
                    {
                        "id": "ENST00000312345",  # Ensembl Transcript ID
                        "seq_region_name": "19",
                        "start": 12345678,
                        "end": 12356789,
                        "strand": 1,
                        "Exon": [
                            {
                                "id": "ENSE00001234567",  # Ensembl Exon ID
                                "seq_region_name": "19",
                                "start": 12345678,
                                "end": 12345789,
                                "strand": 1
                            },
                            {
                                "id": "ENSE00001234568",
                                "seq_region_name": "19",
                                "start": 12346000,
                                "end": 12346100,
                                "strand": 1
                            }
                            # More exons...
                        ]
                        # More transcript-related data...
                    }
                    # More transcripts...
                ]
                # More gene-related data...
            }


    """
    server = "https://rest.ensembl.org"
    ext = f"/lookup/symbol/{species}/{gene_name}?expand=1"
    
    headers = {"Content-Type": "application/json"}
    
    try:
        r = requests.get(server + ext, headers=headers, timeout=10)
        r.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
        return None
    
    try:
        data = r.json()
    except ValueError:
        print("Error decoding JSON response")
        return None
    
    return data


# See meta_spliceai/splice_engine/extract_features.py
# def extract_junctions_for_gene(db, gene_identifier, consensus_window=0):
#     """
#     Extract all annotated transcripts and their associated junctions for a given gene,
#     considering strand orientation and exon indexing.

#     Parameters:
#     - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
#     - gene_identifier (str): Either gene name or gene ID (e.g., Ensembl ID).
#     - consensus_window (int): Number of nucleotides to include before and after the splice site.

#     Returns:
#     - gene_junctions (pd.DataFrame): DataFrame containing junctions for all annotated transcripts.

#     Example usage:
#     db = gffutils.FeatureDB('path_to_gffutils_database.db')
#     gene_junctions_df = extract_junctions_for_gene(db, 'ENSG00000123456')
#     print(gene_junctions_df)
#     """
#     pass


def extract_junctions_for_all_genes(db, save=False, return_df=True, **kargs):
    """
    Extract all junction coordinates for all genes, given a gffutils database (db) created from a GTF or GFF file. 


    Memo
    ----
    * Summary of the Function's Logic:
        - Positive Strand (strand == '+'):
            Loop through exons from first to last (in increasing genomic coordinates).
            For each exon (except the last), define:
                Donor Site: End of the current exon (exons[i].end).
                Acceptor Site: Start of the next exon (exons[i + 1].start).
            Store the junction coordinates as (donor_site, acceptor_site).
    
        - Negative Strand (strand == '-'):
            Loop through exons from last to first (in decreasing genomic coordinates).
            For each exon (starting from the last and moving towards the first), define:
                Donor Site: Start of the current exon (exons[i].start).
                Acceptor Site: End of the previous exon (exons[i - 1].end).
            Store the junction coordinates as (acceptor_site, donor_site) after ensuring start < end.

    """

    all_junctions = []

    for gene in tqdm(db.features_of_type('gene'), desc="Processing genes"):
        gene_id = gene.attributes.get('gene_id', [gene.id])[0]

        for transcript in db.children(gene, featuretype='transcript', order_by='start'):
            transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
            strand = transcript.strand
            chrom = transcript.chrom

            # Extract exons, sorted by genomic coordinates
            exons = list(db.children(transcript, featuretype='exon', order_by='start'))

            # Skip transcripts with less than 2 exons (no splicing)
            if len(exons) < 2:
                continue

            if strand == '+':
                # Positive strand logic
                for i in range(len(exons) - 1):
                    donor_site = exons[i].end      # Donor site at end of current exon
                    acceptor_site = exons[i + 1].start  # Acceptor site at start of next exon

                    # Store junction information
                    junction_name = f"{transcript_id}_JUNC{i:08d}"
                    all_junctions.append({
                        'chrom': chrom,
                        'start': donor_site,
                        'end': acceptor_site,
                        'name': junction_name,
                        'score': 0,
                        'strand': strand,
                    })

            elif strand == '-':
                # Negative strand logic
                for i in range(len(exons) - 1, 0, -1):  #  0 is the stopping index (exclusive), meaning the loop will stop before reaching index 0.
                    donor_site = exons[i].start       # Donor site at start of current exon
                    acceptor_site = exons[i - 1].end  # Acceptor site at end of previous exon

                    # Ensure start < end
                    start = min(donor_site, acceptor_site)
                    end = max(donor_site, acceptor_site)

                    # Store junction information
                    junction_name = f"{transcript_id}_JUNC{len(exons) - i:08d}"
                    all_junctions.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'name': junction_name,
                        'score': 0,
                        'strand': strand,
                    })

    # Convert to DataFrame
    junctions_df = pd.DataFrame(all_junctions, columns=['chrom', 'start', 'end', 'name', 'score', 'strand'])

    # Save or return the junctions
    if save:
        sep = kargs.get('sep', '\t')
        format = 'csv' if sep == ',' else 'tsv'
        output_file = kargs.get('output_file', f'junctions.{format}')
        junctions_df.to_csv(output_file, index=False, sep=sep)
        print(f"[i/o] Junctions saved to {output_file}")

    return junctions_df if return_df else all_junctions


def extract_junctions_from_transcript(transcript, normalize_strand=False, start_index=1):
    """
    Extract junction coordinates from a given transcript's exon data.
    
    Parameters:
    - transcript (dict): A transcript object from Ensembl API.
    - normalize_strand (bool): If True, normalize strand to '+' and '-'. Default is False.
    
    Returns:
    - list of tuples: List of junctions with (chrom, start, end, strand).

    Memo: 
    - the convention for naming exons and introns (junctions) typically follows the order of 
       their occurrence along the transcript. On the positive strand, this means going from 5' to 3', 
       where exon numbers increase as you move downstream. On the negative strand, the natural biological convention 
       is for exon numbers to decrease as you move downstream (from the 3' end to the 5' end).
    """
    exons = sorted(transcript['Exon'], key=lambda x: x['start'])
    # The exons of the transcript are sorted by their start coordinates to ensure they are in the correct order.

    junctions = []
    transcript_id = transcript.get('id', 'unknown')  # Use transcript ID or a placeholder
    strand = transcript.get('strand')
    
    # Name of the junctions
    # - Adjust the junction naming order based on strand
    if strand in (-1, '-'):
        junction_index = start_index + len(exons) - 2  # Start naming from the higher index for negative strand
    else:
        junction_index = start_index  # Start naming from the lower index for positive strand

    if normalize_strand: 
        strand = '+' if strand == 1 else '-'  # Normalize strand to '+' and '-'

    print("[info] transcript_id: {}, strand: {}".format(transcript_id, strand))  
    
    # Calculate Junction Coordinates
    for i in range(len(exons) - 1):
        chrom = exons[i]['seq_region_name']
        start = exons[i]['end'] + 1
        end = exons[i + 1]['start'] - 1
        exon_strand = exons[i]['strand']

        # Normalize strand if the option is chosen
        if normalize_strand:
            exon_strand = '+' if exon_strand == 1 else '-'

        # Check if the exon strand matches the transcript strand
        if exon_strand != strand:
            print(f"Warning: Strand mismatch in transcript {transcript_id}: "
                  f"Exon {i + 1} has strand {exon_strand}, but transcript has strand {strand}.")
            # continue  # Skip this exon or handle it as needed

        # Generate junction name including transcript ID
        name = f"{transcript_id}_JUNC{junction_index:08d}"
        score = 0  # Default score

        junctions.append((chrom, start, end, name, score, strand))

        # print(f"> tx_id: {transcript_id}, exon={i+1} ({start}), junction_index={junction_index}, strand=({exon_strand}=?={strand}), name={name}")
        
        # Update junction index based on strand
        if strand == '-':
            junction_index -= 1  # Decrease for negative strand
        else:
            junction_index += 1  # Increase for positive strand

    return junctions


def extract_junctions_and_exons_from_transcript(transcript, start_index=0):
    """
    Extract junctions and exons from a transcript dictionary in a BED-compatible format.

    Parameters:
    - transcript (dict): Transcript dictionary obtained from Ensembl API.
    - start_index (int): Starting index for naming junctions.

    Returns:
    - tuple: (junctions, exons)
        - junctions: List of tuples (chrom, start, end, name, score, strand)
        - exons: List of tuples (chrom, start, end, name, score, strand)

    Example usage:
        transcript = {
            'id': 'ENST00000335137',
            'strand': 1,
            'Exon': [
                {'seq_region_name': '1', 'start': 11868, 'end': 12227, 'strand': 1},
                {'seq_region_name': '1', 'start': 12612, 'end': 12721, 'strand': 1},
                # Add more exons as needed
            ]
        }
        junctions, exons = extract_junctions_and_exons_from_transcript(transcript)
    """
    exons = sorted(transcript['Exon'], key=lambda x: x['start'])
    junctions = []
    exons_list = []
    transcript_id = transcript.get('id', 'unknown')  # Use transcript ID or a placeholder
    strand = normalize_strand(transcript.get('strand'))

    # Adjust the junction naming order based on strand
    if strand == '-':
        junction_index = start_index + len(exons) - 2  # Start naming from the higher index for negative strand
    else:
        junction_index = start_index  # Start naming from the lower index for positive strand

    print(f"[info] transcript_id: {transcript_id}, strand: {strand}")  

    # Calculate Junction Coordinates
    for i in range(len(exons)):
        chrom = exons[i]['seq_region_name']
        start = exons[i]['start'] - 1  # Adjust for 0-based BED format
        end = exons[i]['end']  # No adjustment needed since BED end is exclusive
        exon_strand = normalize_strand(exons[i]['strand']) 

        # Check if the exon strand matches the transcript strand
        if exon_strand != strand:
            print(f"Warning: Strand mismatch in transcript {transcript_id}: "
                  f"Exon {i + 1} has strand {exon_strand}, but transcript has strand {strand}.")
            # continue  # Skip this exon or handle it as needed

        # Generate exon name including transcript ID
        exon_name = f"{transcript_id}_EXON{i+1:08d}"
        exon_score = 0  # Default score

        # Append exon to exons_list
        # print("[debug] adding exon (prior to index adjustment): ", (chrom, exons[i]['start'], exons[i]['end'], strand))
        exons_list.append((chrom, start, end, exon_name, exon_score, strand))

        # Calculate junctions if not the first or last exon
        if i < len(exons) - 1:
            junction_start = exons[i]['end']  # 0-based start of junction (exons[i]['end']+1-1 = exons[i]['end'])
            junction_end = exons[i + 1]['start'] - 1  # 0-based exclusive end
            junction_name = f"{transcript_id}_JUNC{junction_index:08d}"
            junction_score = 0  # Default score

            # [debug]
            # if junction_name == "ENST00000519716_JUNC00000019":
            #     print("[debug] junction_start: {}, junction_end: {}".format(junction_start, junction_end))
            #     print(f"... last exon end: {exons[i]['end']}, next exon start: {exons[i+1]['start']}")

            # Append junction to junctions list
            junctions.append((chrom, junction_start, junction_end, junction_name, junction_score, strand))

            # Update junction index based on strand
            if strand == '-':
                junction_index -= 1  # Decrease for negative strand
            else:
                junction_index += 1  # Increase for positive strand

    return junctions, exons_list


def extract_junctions_for_trio(transcript, exon_trio, junction_index=0):
    """
    Extract junctions for a given set of exons (upstream, cryptic, downstream) and handle strand-specific coordinates.
    
    Parameters:
    - transcript (dict): The transcript object containing exon data.
    - exon_trio (dict): Dictionary with keys 'upstream', 'cryptic', 'downstream'.
    - junction_index (int): Starting index for naming junctions.

    Returns:
    - list of tuples: Junctions and exons in BED format.
    """
    junctions = []
    exons_list = []
    transcript_id = transcript.get('id', 'unknown')
    strand = normalize_strand(transcript.get('strand'))  # Normalize strand to '+' or '-'
    chrom = transcript['Exon'][0]['seq_region_name']  # Assuming all exons in the transcript are on the same chromosome
    
    upstream_exon = exon_trio.get('upstream')
    cryptic_exon = exon_trio.get('cryptic')
    downstream_exon = exon_trio.get('downstream')

    # Handle upstream to cryptic junction
    if upstream_exon and cryptic_exon:  # if both upstream and cryptic exons are present (not None)

        if strand == '-':
            # On negative strand, the end is higher than the start
            junction_start = cryptic_exon[2]  # cryptic_exon[2]+1 -> (cryptic_exon[2]+1) - 1
            junction_end = upstream_exon[1] - 1  # (upstream_exon[1]-1) -> upstream_exon[1]-1 (no adjustment needed)
            junction_type = "3p"  # upstream junction is toward 3' end  on negative strand

            # NOTE: 
            # junction: (cryptic_exon[2]+1, upstream_exon[1]-1)
            # 1-based to 0-based (end exclusive): (cryptic_exon[2]+1)-1, upstream_exon[1]-1

        else:
            # On positive strand, the start is lower than the end
            junction_start = upstream_exon[2]
            junction_end = cryptic_exon[1] - 1
            junction_type = "5p"  # upstream junction is toward 5' end on positive strand

        # NOTE: 0-based start and end positions are used for BED format
        #       The end position is exclusive, so we subtract 1 from the end position.
        #       Suppose strand = '+', then
        #          upstream_exon[2]+1 if 1-based position is needed
        #          => upstream_exon[2]+1-1 = upstream_exon[2] for 0-based position in BED
        #          => cryptic_exon[1]-1 for end position in 1-based inclusive, which is the same as 0-based exclusive
        assert junction_start < junction_end, f"Invalid junction coordinates: {junction_start} >= {junction_end}"
        
        # junction_name = f"{transcript_id}_CRPT{junction_index:08d}_up"
        junction_name = f"{transcript_id}_CRPT{junction_index:08d}_{junction_type}"
        junctions.append((chrom, junction_start, junction_end, junction_name, 0, strand))
        junction_index += 1
    
    # Handle cryptic to downstream junction
    if cryptic_exon and downstream_exon:
        if strand == '-':
            # On negative strand, the end is higher than the start
            junction_start = downstream_exon[2]
            junction_end = cryptic_exon[1] - 1
            junction_type = "5p"  # downstream junction is on the 5' end on negative strand
        else:
            # On positive strand, the start is lower than the end
            junction_start = cryptic_exon[2]
            junction_end = downstream_exon[1] - 1
            junction_type = "3p"  # downstream junction is toward 3' end on positive strand

        assert junction_start < junction_end, f"Invalid junction coordinates: {junction_start} >= {junction_end}"

        # junction_name = f"{transcript_id}_CRPT{junction_index:08d}_down"
        junction_name = f"{transcript_id}_CRPT{junction_index:08d}_{junction_type}"
        junctions.append((chrom, junction_start, junction_end, junction_name, 0, strand))
    
    # Add the cryptic exon itself
    exon_name = f"{transcript_id}_CRPT_EXON{junction_index:08d}"
    exons_list.append((chrom, cryptic_exon[1] - 1, cryptic_exon[2], exon_name, 0, strand))

    return junctions, exons_list


def extract_junctions_and_exons_v0(transcript, exon_trio, start_index=0, verbose=0, **kargs):
    """
    Extract all junctions including canonical and those flanking the cryptic exon, along with exons.
    
    Parameters:
    - transcript: Transcript dictionary from Ensembl API.
    - exon_trio: Dictionary with keys 'upstream', 'cryptic', 'downstream'.
    - start_index: The starting index for naming junctions.

    Returns:
    - List of tuples: Combined list of junctions including canonical and cryptic.
    - List of tuples: Combined list of exons including canonical and cryptic.
    """
    # Extract canonical junctions and exons
    canonical_junctions, canonical_exons = extract_junctions_and_exons_from_transcript(transcript, start_index)
    print("[info] Canonical Junctions:")
    print("... Number of canonical junctions:", len(canonical_junctions))
    print("... Number of canonical exons:", len(canonical_exons))

    # Extract cryptic junctions and cryptic exons
    cryptic_junctions, cryptic_exons = extract_junctions_for_trio(transcript, exon_trio, start_index)
    print("[info] Cryptic Junctions:")
    print("... Number of cryptic junctions:", len(cryptic_junctions))
    print("... Number of cryptic exons:", len(cryptic_exons))

    # Combine junctions
    all_junctions = canonical_junctions + cryptic_junctions

    # Combine exons
    all_exons = canonical_exons + cryptic_exons

    # Optional: Sort combined junctions by coordinates (chrom, start, end, strand)
    all_junctions = sorted(all_junctions, key=lambda j: (j[0], j[1], j[2], j[5]))

    # Optional: Sort combined exons by coordinates (chrom, start, end, strand)
    all_exons = sorted(all_exons, key=lambda e: (e[0], e[1], e[2], e[5]))

    return all_junctions, all_exons


def extract_junctions_and_exons(transcript, exon_trio, start_index=0, verbose=0, **kargs):
    """
    Extract all junctions including canonical and those flanking the cryptic exon, along with exons.

    Parameters:
    - transcript: Transcript dictionary from Ensembl API.
    - exon_trio: Dictionary with keys 'upstream', 'cryptic', 'downstream'.
    - start_index: The starting index for naming junctions.

    Returns:
    - dict: Dictionary of junctions with keys 'canonical_junctions' and 'cryptic_junctions'.
    - dict: Dictionary of exons with keys 'canonical_exons' and 'cryptic_exons'.
    """
    # Step 1: Extract canonical junctions and exons
    canonical_junctions, canonical_exons = extract_junctions_and_exons_from_transcript(transcript, start_index)
    # print(f"(extract_junctions_and_exons) canonical exons")
    # display_exons(canonical_exons)   # check: ok 

    # Step 2: Extract cryptic junctions and cryptic exons
    cryptic_junctions, cryptic_exons = extract_junctions_for_trio(transcript, exon_trio, start_index)
    print("[info] n={} cryptic jucntions identified from the exon_trio: {}".format(len(cryptic_junctions), cryptic_junctions))
    # display_exons(cryptic_exons)  # check: ok

    # Step 3: Create combined sets with canonical + cryptic
    combined_junctions = set(canonical_junctions + cryptic_junctions)
    combined_exons = set(canonical_exons + cryptic_exons)
    # display_exons(sorted(combined_exons, key=lambda j: (j[0], j[1], j[2], j[5])))

    # Normalize strand for consistency
    strand = normalize_strand(transcript.get('strand'))

    # Retrieve upstream and downstream exons from the exon_trio
    upstream_exon = exon_trio.get('upstream')  # 1-based start and end positions from standard annotation
    downstream_exon = exon_trio.get('downstream')  # 1-based start and end positions from standard annotation

    # Remove any canonical junctions that are replaced by cryptic exon-induced junctions
    n_removed = 0
    for junc in canonical_junctions:
        j_start, j_end = junc[1], junc[2]
        j_name = junc[3]

        if upstream_exon and downstream_exon:
            if strand == '+':
                start = upstream_exon[2]  # End position of the upstream exon (1-based to 0-based) + 1
                # NOTE: 1-based to 0-based => upstream_exon[2]-1
                #       junction starts after the end of the upstream exon => (upstream_exon[2]-1)+1
                #       => upstream_exon[2] (as if no adjustment is needed)
                end = downstream_exon[1]-1  # Start position of the downstream exon (1-based)
            elif strand == '-':
                start = downstream_exon[2] # End position of the downstream exon (1-based to 0-based) + 1 
                end = upstream_exon[1] -1   # Start position of the upstream exon (1-based)

            # Debug 
            # if j_name == "ENST00000519716_JUNC00000019": 
            #     last_exon_end = min(upstream_exon[2], downstream_exon[2])
            #     next_exon_start = max(upstream_exon[1], downstream_exon[1])
            #     print(f"[debug] canonical junction: {junc}")
            #     print("... junction_start: {}, junction_end: {}".format(start, end))
            #     print(f"... last exon end: {last_exon_end}, next exon start: {next_exon_start}")

            # Check if the canonical junction matches the one induced by the cryptic exon
            # if j_start == start and j_end == end:
            if abs(j_start - start) < 2 and abs(j_end-end) < 2:
                print("[action] Removing canonical junction:", junc)
                combined_junctions.remove(junc)
                n_removed += 1

    print("[info] Removed {} canonical junctions replaced by cryptic exon-induced junctions.".format(n_removed))

    # Sort combined junctions and exons by coordinates (chrom, start, end, strand)
    combined_junctions = sorted(combined_junctions, key=lambda j: (j[0], j[1], j[2], j[5]))
    combined_exons = sorted(combined_exons, key=lambda e: (e[0], e[1], e[2], e[5]))

    # Return as dictionaries
    junctions = {
        'canonical': canonical_junctions,
        'cryptic': combined_junctions  # canonical + cryptic junctions
    }

    exons = {
        'canonical': canonical_exons,
        'cryptic': combined_exons  # canonical + cryptic exons
    }

    if verbose:
        print('-' * 80)
        tx_id = transcript.get('id', 'unknown')
        print("[info] Transcript ID:", tx_id)
        print("[info] Canonical junctions (n={}):".format(len(canonical_junctions)))
        print("[info] Cryptic junctions (n={}):".format(len(combined_junctions)))
        print("[info] Canonical exons (n={}):".format(len(canonical_exons)))
        print("[info] Cryptic exons (n={}):".format(len(combined_exons)))
        print('-' * 80)

    return junctions, exons



####################################################################################################

def display_exons(exons_of_interest, separator_char='-', margin=2):
    """
    Display each exon in the exons_of_interest list with better readability.
    
    Args:
        exons_of_interest (list of tuples): List of exons to display.
        separator_char (str): Character to use for the line separator.
        margin (int): Number of blank lines to add as margin.
    """
    if exons_of_interest is None: 
        return 

    # Check if the input is a single tuple
    if isinstance(exons_of_interest, tuple):
        exons_of_interest = [exons_of_interest, ]

    # Define the line separator
    separator_line = separator_char * 50

    # Print top margin
    for _ in range(margin):
        print()

    # Print the top separator line
    print(separator_line)

    for i, exon in enumerate(exons_of_interest):
        if exon is None:
            print(f"Exon {i + 1}: None")
            continue

        if len(exon) == 4: 
            chrom, start, end, strand = exon
        elif len(exon) >= 6: 
            # Assuming to be in BED format
            chrom, start, end, name, score, strand = exon
        else: 
            msg = f"Unexpected format for exon: {exon}" 
            raise ValueError(msg)

        print(f"Exon {i + 1}: Chromosome = {chrom}, Start = {start}, End = {end}, Strand = {strand}")

    # Print the bottom separator line
    print(separator_line)

    # Print bottom margin
    for _ in range(margin):
        print()
    

def normalize_chrom(chrom):
    """Normalize chromosome names to ensure consistency."""
    return chrom.lower().replace('chr', '')

def normalize_strand(strand):
    """Normalize strand to '+' or '-'."""
    if strand in [1, '+']:
        return '+'
    elif strand in [-1, '-']:
        return '-'
    else:
        raise ValueError(f"Unexpected strand value: {strand}")

def find_matching_transcript(gene_name, exon_trio, **kargs):
    """
    Find a transcript that contains the exons of interest or calculate the distance to the closest exons.
    
    Parameters:
    - gene_name (str): The name of the gene (e.g., STMN2, UNC13A).
    - exon_trio (dict): Dictionary with keys 'upstream', 'cryptic', 'downstream'.
       e.g.  {'upstream': ('19', 17642844, 17642960, '-'), 
              'cryptic': ('19', 17642413, 17642541, '-'),
              'downstream': ('19', 17641392, 17641556, '-')}
    
    Returns:
    - dict: The transcript that contains the exons of interest or the distances to the closest exons.
    """
    
    species = kargs.get('species', 'human')
    gene_info = get_transcript_info(gene_name, species)
    
    if not gene_info:
        print(f"Could not retrieve information for gene: {gene_name}")
        return None


    print(f"[info] Gene: {gene_name}" )
    # exons_of_interest = [exon_trio[key] for key in exon_trio if exon_trio[key] is not None]
    # display_exons(exons_of_interest)
    
    upstream_exon = exon_trio.get('upstream')
    downstream_exon = exon_trio.get('downstream')
    display_exons([upstream_exon, downstream_exon])

    best_match_transcript = None
    best_match_score = float('inf')  # Initialize with a high score

    for transcript in gene_info['Transcript']:
        exons = sorted(transcript['Exon'], key=lambda x: x['start'])
        
        total_distance = 0
        matched_exons = 0

        for e in [upstream_exon, downstream_exon]:
            if e is None:
                continue

            chrom_interest = normalize_chrom(e[0])
            strand_interest = normalize_strand(e[3])

            closest_exon = None
            min_distance = float('inf')

            for exon in exons:
                chrom_exon = normalize_chrom(exon['seq_region_name'])
                strand_exon = normalize_strand(exon['strand'])

                if chrom_exon == chrom_interest and strand_exon == strand_interest:
                    # Calculate the min distance
                    distance = min(abs(exon['start'] - e[1]), abs(exon['end'] - e[2]))

                    if distance < min_distance:
                        min_distance = distance
                        closest_exon = exon
            
            if closest_exon:
                total_distance += min_distance
                matched_exons += 1

        if matched_exons > 0 and total_distance < best_match_score:
            best_match_score = total_distance
            best_match_transcript = transcript

            print("... best_match_transcript={}, total_distance={}".format(
                best_match_transcript.get('id', 'unknown'), total_distance))

        if best_match_score == 0:  # Exact match found
            print(f"Exact matching transcript found: {best_match_transcript['id']}")
            return best_match_transcript

    if best_match_transcript:
        print(f"Best matching transcript found: {best_match_transcript['id']} with score {best_match_score}")
    else:
        print("No suitable matching transcript found.")
    
    return best_match_transcript


def find_best_matching_transcript(gene_name, exon_trio, normalize_strand=True, **kargs):
    """
    Find the canonical transcript that best matches the upstream and downstream exons of the exon_trio.
    
    Parameters:
    - gene_name (str): The name of the gene (e.g., STMN2, UNC13A).
    - exon_trio (dict): Dictionary with keys 'upstream', 'cryptic', 'downstream'.
       e.g.  {'upstream': ('19', 17642844, 17642960, '-'), 
              'cryptic': ('19', 17642413, 17642541, '-'),
              'downstream': ('19', 17641392, 17641556, '-')}
    
    Returns:
    - dict: The best matching transcript based on the upstream and downstream exons.
    """

    species = kargs.get('species', 'human')
    gene_info = get_transcript_info(gene_name, species)
    
    if not gene_info:
        print(f"Could not retrieve information for gene: {gene_name}")
        return None

    upstream_exon = exon_trio.get('upstream')
    downstream_exon = exon_trio.get('downstream')

    best_match_transcript = None
    best_match_score = float('inf')  # Initialize with a high score

    for transcript in gene_info['Transcript']:
        exons = sorted(transcript['Exon'], key=lambda x: x['start'])
        
        total_distance = 0
        matched_exons = 0

        for e in [upstream_exon, downstream_exon]:
            if e is None:
                continue

            chrom_interest = normalize_chrom(e[0])
            strand_interest = normalize_strand(e[3])

            closest_exon = None
            min_distance = float('inf')

            for exon in exons:
                chrom_exon = normalize_chrom(exon['seq_region_name'])
                strand_exon = normalize_strand(exon['strand'])

                if chrom_exon == chrom_interest and strand_exon == strand_interest:
                    # Calculate the Euclidean distance
                    distance = math.sqrt((exon['start'] - e[1]) ** 2 + (exon['end'] - e[2]) ** 2)

                    if distance < min_distance:
                        min_distance = distance
                        closest_exon = exon
            
            if closest_exon:
                total_distance += min_distance
                matched_exons += 1

        if matched_exons > 0 and total_distance < best_match_score:
            best_match_score = total_distance
            best_match_transcript = transcript

    if best_match_transcript:
        print(f"Best matching transcript found: {best_match_transcript['id']} with score {best_match_score}")
    else:
        print("No suitable matching transcript found.")
    
    return best_match_transcript

####################################################################################################

def infer_new_junctions_v0(canonical_junctions, cryptic_exon, include_length=False):
    """
    Infer the new junctions that will be created when a cryptic exon is expressed.
    
    Parameters:
    - canonical_junctions (list of tuples): Junctions from the canonical isoform.
    - cryptic_exon (tuple): A tuple (chrom, start, end, strand) representing the cryptic exon.
    
    Returns:
    - list of tuples: List of junctions with (chrom, start, end, strand) including cryptic exon.
    """
    chrom, cryptic_start, cryptic_end, strand, *rest = cryptic_exon
    new_junctions = []
    
    if include_length: 
        for i, (c_chrom, c_start, c_end, c_strand) in enumerate(canonical_junctions):
            # Infer junctions based on the position of the cryptic exon
            if c_chrom == chrom and c_start < cryptic_start < c_end:
                # Add junction from canonical start to just before cryptic exon start
                new_junctions.append((chrom, c_start, cryptic_start - 1, strand, cryptic_start - c_start))
                # Add junction from just after cryptic exon end to canonical end
                new_junctions.append((chrom, cryptic_end + 1, c_end, strand, c_end - cryptic_end))
            else:
                new_junctions.append((c_chrom, c_start, c_end, c_strand, c_end - c_start)) 
    else:
        for i, (c_chrom, c_start, c_end, c_strand) in enumerate(canonical_junctions):
            if c_chrom == chrom and c_start < cryptic_start < c_end:
                new_junctions.append((chrom, c_start, cryptic_start - 1, strand))  # Add junction from canonical start to just before cryptic exon start
                new_junctions.append((chrom, cryptic_end + 1, c_end, strand))  # Add junction from just after cryptic exon end to canonical end
            else:
                new_junctions.append((c_chrom, c_start, c_end, c_strand))
    
    return new_junctions


def infer_new_junctions(
        canonical_junctions, 
        cryptic_exon=None, 
        transcript="tx", 
        cryptic_exon_name="CRPT", raise_on_invalid=False):
    """
    Infer the new junctions that will be created when a cryptic exon is expressed.
    
    Parameters:
    - canonical_junctions (list of tuples): Junctions from the canonical isoform. The tuples can contain additional elements beyond the standard ones.
    - cryptic_exon (tuple, dict, None): 
        A tuple (chrom, start, end, strand) representing the cryptic exon, 
        a dictionary following BED format, or 
        None.
    - transcript (str, dict): The ID of the transcript from which the junctions are derived.
    - cryptic_exon_name (str): Base name for generating cryptic exon junction IDs.
    - raise_on_invalid (bool): If True, raises an exception when an invalid cryptic exon is encountered.
    
    Returns:
    - list of tuples: 
       List of junctions with (chrom, start, end, name, score, strand) including those induced by
       the given cryptic exon.
    """
    if cryptic_exon is None or len(cryptic_exon) == 0:
        print("[info] No cryptic exon provided. Returning canonical junctions.")
        return canonical_junctions

    if isinstance(cryptic_exon, dict):
        chrom = cryptic_exon.get('chrom')
        cryptic_start = cryptic_exon.get('start')
        cryptic_end = cryptic_exon.get('end')
        strand = cryptic_exon.get('strand')
    elif isinstance(cryptic_exon, tuple):
        chrom, cryptic_start, cryptic_end, strand, *_ = cryptic_exon
    else:
        raise ValueError("cryptic_exon must be a tuple, dict, or None")
    
    new_junctions = []
    cryptic_junction_index = 1

    transcript_id = "unknown"
    if isinstance(transcript, str): 
        transcript_id = transcript
    elif isinstance(transcript, dict):
        transcript_id = transcript.get('id', 'unknown')
    
    for junction in canonical_junctions:
        chrom, start, end, name, score, strand, *rest = junction  # Handle additional elements in junction tuple

        if chrom == chrom and start < cryptic_start < end:
            if cryptic_end + 1 >= end:
                # Raise an exception if the cryptic exon is invalid and the flag is set
                if raise_on_invalid:
                    raise ValueError(f"Invalid cryptic exon {cryptic_exon} for junction {junction}: exon overlaps or exceeds junction boundaries.")
                # Skip creating new junctions if the cryptic exon is invalid
                continue
            
            # Modify junctions to accommodate cryptic exon
            # cryptic_name_1 = f"{transcript_id}_{cryptic_exon_name}{cryptic_junction_index:08d}_up"
            cryptic_name_1 = f"{transcript_id}_{cryptic_exon_name}{cryptic_junction_index:08d}_5p"
            # cryptic_name_2 = f"{transcript_id}_{cryptic_exon_name}{cryptic_junction_index:08d}_down"
            cryptic_name_2 = f"{transcript_id}_{cryptic_exon_name}{cryptic_junction_index:08d}_3p"
            # NOTE: Standard junction names: 
            #       f"{transcript_id}_JUNC{junction_index:08d}"


            cryptic_junction_index += 1

            new_junctions.append((chrom, start, cryptic_start - 1, cryptic_name_1, score, strand, *rest))  # Junction to cryptic start
            new_junctions.append((chrom, cryptic_end + 1, end, cryptic_name_2, score, strand, *rest))  # Junction from cryptic end to canonical end
        else:
            new_junctions.append(junction)
    
    return new_junctions


def write_junctions_to_bed(junctions, bed_file_path):
    """
    Write a list of junctions to a BED file.

    Parameters:
    - junctions (list of tuples): List of junctions in the format 
      (chrom, start, end, name, score, strand).
    - bed_file_path (str): Path to the output BED file.
    """
    with open(bed_file_path, 'w') as bed_file:
        for j in junctions:
            chrom = j[0]    # Chromosome
            start = j[1]    # Start position (0-based, already adjusted in infer_new_junctions)
            end = j[2]      # End position
            name = j[3]     # Junction name
            score = j[4]    # Score (can be 0 if not applicable)
            strand = j[5]   # Strand (+ or -)

            # Construct the BED line
            bed_line = f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n"
            
            # Write to BED file
            bed_file.write(bed_line)

    print(f"[i/o] Junction BED file written to: {bed_file_path}")

    return


def write_exons_to_bed(exons, bed_file_path):
    """
    Write a list of exons to a BED file.

    Parameters:
    - exons (list of tuples): List of exons in the format 
      (chrom, start, end, name, score, strand).
    - bed_file_path (str): Path to the output BED file.
    """
    with open(bed_file_path, 'w') as bed_file:
        for e in exons:
            chrom = e[0]    # Chromosome
            start = e[1]    # Start position (0-based, already adjusted)
            end = e[2]      # End position
            name = e[3]     # Exon name
            score = e[4]    # Score (can be 0 if not applicable)
            strand = e[5]   # Strand (+ or -)

            # Construct the BED line
            bed_line = f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n"
            
            # Write to BED file
            bed_file.write(bed_line)

    print(f"[i/o] Exon BED file written to: {bed_file_path}")

    return


def inspect_junctions(transcript_info, junctions):
    """
    Print a readable summary of junctions for a given transcript.

    Parameters:
    - transcript_info (dict): A dictionary containing transcript information returned by get_transcript_info().
    - junctions (list of tuples): List of junctions with (chrom, start, end, name, score, strand, ...).
    """
    transcript_id = transcript_info.get('id')
    chrom = transcript_info.get('seq_region_name')
    strand = '+' if transcript_info.get('strand') == 1 else '-'

    print(f"Transcript ID: {transcript_id}")
    print(f"Chromosome: {chrom}")
    print(f"Strand: {strand}")
    print("Junctions:")
    print("-" * 60)
    
    for junction in junctions:
        chrom, start, end, name, score, strand, *rest = junction
        print(f"Junction Name: {name}")
        print(f"Coordinates: {chrom}:{start}-{end}")
        print(f"Strand: {strand}")
        print(f"Score: {score}")
        print("-" * 60)


def compare_junctions(canonical_junctions, new_junctions, transcript):
    """
    Compare canonical junctions with new junctions induced by a cryptic exon.
    
    Parameters:
    - canonical_junctions (list of tuples): List of canonical junctions 
      (chrom, start, end, name, score, strand).
    - new_junctions (list of tuples): List of new junctions including the cryptic exon.
    - transcript (dict): The transcript object containing relevant information.

    Returns:
    - dict: A dictionary with 'common_junctions' and 'unique_junctions' (from the new set).
    """
    # Sorting junctions by (chrom, start, end, strand) to ensure order
    canonical_junctions = sorted(canonical_junctions, key=lambda j: (j[0], j[1], j[2], j[5]))
    new_junctions = sorted(new_junctions, key=lambda j: (j[0], j[1], j[2], j[5]))

    # Convert the junction lists to sets for easier comparison (ignoring 'name' and 'score' for now)
    canonical_set = set((j[0], j[1], j[2], j[5]) for j in canonical_junctions)
    new_set = set((j[0], j[1], j[2], j[5]) for j in new_junctions)
    
    # Find common and unique junctions
    common_junctions = canonical_set.intersection(new_set)
    unique_junctions = new_set.difference(canonical_set)

    # Generate comparison results
    result = {
        "common_junctions": list(common_junctions),
        "unique_junctions": list(unique_junctions)
    }
    
    # Inspect and print the comparison results
    print(f"Transcript ID: {transcript['id']}")
    print(f"Chromosome: {transcript['seq_region_name']}, Strand: {transcript['strand']}\n")

    # Function to find the index of a junction in the canonical list
    def find_index(junction, junctions_list):
        for idx, j in enumerate(junctions_list):
            if junction[0] == j[0] and junction[1] == j[1] and junction[2] == j[2] and junction[3] == j[5]:
                return idx + 1  # Indexing starts at 1 for readability
        return None

    # Print Common Junctions with Index
    print("Common Junctions:")
    for j in result["common_junctions"]:
        index = find_index(j, canonical_junctions)
        print(f"{j[0]}:{j[1]}-{j[2]} (Index: {index}, Strand: {j[3]})")

    # Print Unique Junctions with Index (from the new set)
    print("\nUnique Junctions (New Set):")
    for j in result["unique_junctions"]:
        index = find_index(j, new_junctions)
        print(f"{j[0]}:{j[1]}-{j[2]} (Index: {index}, Strand: {j[3]})")
    
    return result




def get_junctions_with_cryptic_exon(gene_name, cryptic_exon, species='human', **kargs):
    """
    Get junctions for the canonical isoform and infer junctions with a cryptic exon.
    
    Parameters:
    - gene_name (str): The name of the gene (e.g., STMN2, UNC13A).
    - cryptic_exon (tuple): A tuple (chrom, start, end, strand) representing the cryptic exon.
    - species (str): The species of the gene (default is human).
    
    Returns:
    - dict: A dictionary with canonical and inferred junctions with cryptic exon.
    """
    gene_info = get_transcript_info(gene_name, species)
    
    if not gene_info:
        return None
    
    canonical_transcript = None
    
    # Find the canonical transcript
    for transcript in gene_info['Transcript']:
        if transcript.get('is_canonical'):
            canonical_transcript = transcript
            break

    canonical_junctions = []
    cryptic_junctions = []

    if canonical_transcript:  
        inferred_strand = infer_strand_from_transcript(canonical_transcript)
        tx_id = canonical_transcript.get('id', 'unknown')

        print_emphasized(f"[info] Gene name: {gene_name}")
        print(f"... inferred strand for transcript={tx_id} < gene={gene_name}): {inferred_strand}")

        # Assuming canonical_transcript is found
        
        print(f"... canonical transcript (dtype={type(canonical_transcript)}): {tx_id}")
        
        canonical_junctions = extract_junctions_from_transcript(canonical_transcript, normalize_strand=True)
        
        print(f"> Canonical junctions (n={len(canonical_junctions)}):")
        inspect_junctions(canonical_transcript, canonical_junctions)
        # print(f"... canonical junctions: \n{canonical_junctions}\n")
        print(f"... number of exons: {count_exons_from_junctions(canonical_junctions)}")
        # print(f"... junction lengths: \n{calculate_junction_lengths(canonical_junctions)}\n")

        cryptic_junctions = infer_new_junctions(canonical_junctions, cryptic_exon, transcript=canonical_transcript)
        
        print(f"> Cryptic junctions (n={len(cryptic_junctions)}):")
        inspect_junctions(canonical_transcript, cryptic_junctions)
        # print(f"... cryptic junctions: \n{cryptic_junctions}\n")
        print(f"... number of exons with cryptic exon: {count_exons_from_junctions(cryptic_junctions)}")
        # print(f"... junction lengths with cryptic exon: \n{calculate_junction_lengths(cryptic_junctions)}\n")

        result = compare_junctions(canonical_junctions, cryptic_junctions, canonical_transcript)

    else: 
        print("Canonical transcript not found.")

    junctions = {
            'canonical_junctions': canonical_junctions,  # canonical junctions only
            'cryptic_junctions': cryptic_junctions  # canonical + cryptic junctions
            }
    
    if kargs.get("return_transcript", False):
        return junctions, canonical_transcript

    return junctions


def count_exons_from_junctions(junctions):
    """
    Infer the number of exons from the number of junctions.
    
    Parameters:
    - junctions (list of tuples): List of junctions with (chrom, start, end, strand).
    
    Returns:
    - int: Number of exons.
    """
    # Number of exons is the number of junctions plus one
    return len(junctions) + 1


def calculate_junction_lengths(junctions):
    """
    Calculate the lengths of junctions (introns) from a list of junctions.
    
    Parameters:
    - junctions (list of tuples): List of junctions with (chrom, start, end, strand).
    
    Returns:
    - list of tuples: List of junctions with (chrom, start, end, strand, length).
    """
    junctions_with_lengths = []

    for junction in junctions:
        chrom, start, end, name, score, strand, *rest = junction  # Handle additional elements in junction tuple

        length = end - start + 1
        junctions_with_lengths.append((chrom, start, end, strand, length))
    
    return junctions_with_lengths


def infer_strand_from_transcript_v0(transcript):
    """
    Infer the strand of a given transcript.
    
    Parameters:
    - transcript (dict): A transcript object from Ensembl API.
    
    Returns:
    - str: Strand value ('+' or '-') of the transcript.
    """
    # Exons are sorted by start position, so we check the order to determine the strand
    exons = sorted(transcript['Exon'], key=lambda x: x['start'])
    
    # Compare the first and last exons to infer strand
    if exons[0]['start'] < exons[-1]['start']:
        return '+'
    else:
        return '-'


def infer_strand_from_transcript(transcript):
    """
    Directly retrieve the strand information from a given transcript.

    Parameters:
    - transcript (dict): A transcript object from Ensembl API.

    Returns:
    - str: Strand value ('+' or '-') of the transcript.
    """
    strand = transcript.get('strand')
    if strand == 1:
        return '+'
    elif strand == -1:
        return '-'
    else:
        raise ValueError("Strand information is missing or not recognized.")


def make_junction_bed_file(gene_exon_pairs, junction_type='cryptic_junctions', **kargs):
    """
    Extract junctions for a set of genes and their corresponding cryptic exons.
    
    Parameters:
    - gene_exon_pairs (list of tuples): List of tuples where each tuple contains a gene name and its cryptic exon.
    - junction_type (str): Type of junctions to collect ('cryptic_junctions' or 'canonical_junctions').

    Example usage: 
        gene_exon_pairs = [
            ('STMN2', ('8', 79616821, 79617048, '+')),
            ('UNC13A', ('19', 17642413, 17642541, '-'))
        ]

        demo_extract_junctions(gene_exon_pairs, junction_type='cryptic_junctions')
    """
    data_path = "/path/to/meta-spliceai/data/ensembl/ALS"
    # Ensure the directory exists
    os.makedirs(data_path, exist_ok=True)

    all_junctions = []

    for gene_name, cryptic_exon in gene_exon_pairs:
        print("[info] Processing gene:", gene_name)
        junctions, transcript = get_junctions_with_cryptic_exon(gene_name, cryptic_exon, return_transcript=True)

        # Sort junctions by coordinates
        junctions[junction_type] = sorted(junctions[junction_type], key=lambda j: (j[0], j[1], j[2], j[5]))  
        # Sort by (chrom, start, end, strand)
        # NOTE: chrom, start, end, name, score, strand, donor_score, acceptor_score 

        if junction_type in junctions:
            all_junctions.extend(junctions[junction_type])
            print(f"[i/o] Extracted {junction_type} for gene: {gene_name}")
            print(f"... n={len(junctions[junction_type])} junctions found.")
        else:
            print(f"[error] Junction type '{junction_type}' not found for gene: {gene_name}")

    prefix = kargs.get("output_prefix", "")
    if prefix: 
        bed_file_path = os.path.join(data_path, f"{prefix}_{junction_type}.bed")
    else: 
        bed_file_path = os.path.join(data_path, f"{junction_type}.bed")
    print(f"[i/o] Writing n={len(all_junctions)} junctions to BED file:\n{bed_file_path}\n ...")
    write_junctions_to_bed(all_junctions, bed_file_path)

    return bed_file_path


def make_junction_bed_file_v2(gene_exon_trio, junction_types=None, **kargs):
    """
    Extract junctions for a set of genes and their corresponding cryptic exons.
    
    Parameters:
    - gene_exon_trio (dict): Dictionary where each key is a gene name, and its value is a dictionary with upstream, cryptic, and downstream exons.
    - junction_types (str): Type of junctions to collect ('all_junctions').

    Returns:
    - str: Path to the generated BED file.
    """
    def get_bed_file_path(prefix, data_path, junction_type, gene_name=None, dtype='junctions'):
        """
        Construct the BED file path based on the prefix, junction type, and gene name.
        """
        if prefix:
            if gene_name:
                return os.path.join(data_path, f"{prefix}_{gene_name}_{junction_type}_{dtype}.bed")
            else:
                return os.path.join(data_path, f"{prefix}_{junction_type}_{dtype}.bed")
        else:
            if gene_name:
                return os.path.join(data_path, f"{gene_name}_{junction_type}_{dtype}.bed")
            else:
                return os.path.join(data_path, f"{junction_type}_{dtype}.bed")

    prefix = kargs.get("output_prefix", "")
    data_path = "/path/to/meta-spliceai/data/ensembl/ALS"
    os.makedirs(data_path, exist_ok=True)

    if junction_types is None: 
        junction_types = ['canonical',  'cryptic', ]
    
    junction_paths = {}
    exon_paths = {}

    base_transcript = {}

    for junction_type in junction_types:  # Process each junction type

        all_junctions = []
        all_exons = []

        for gene_name, exon_trio in gene_exon_trio.items():  # Process each gene
            print_emphasized(f"[info] Processing gene: {gene_name}, junction type: {junction_type}")

            if not gene_name in base_transcript: 
                base_transcript[gene_name] = find_matching_transcript(gene_name, exon_trio)
            
            transcript = base_transcript[gene_name]
            if not transcript:
                print(f"[error] No matching transcript found for gene: {gene_name}")
                continue

            junctions, exons = extract_junctions_and_exons(transcript, exon_trio, start_index=0, verbose=1)  

            all_junctions.extend(junctions[junction_type])
            all_exons.extend(exons[junction_type])
                    
            print(f"[i/o] Extracted {junction_type} for gene: {gene_name}")
            print(f"... n={len(junctions[junction_type])} junctions found.")
            print(f"... n={len(exons[junction_type])} exons found.")

            # Save gene-specific BED files
            gene_junction_path = os.path.join(data_path, junction_type, gene_name)
            os.makedirs(gene_junction_path, exist_ok=True)
            gene_junction_bed_file = get_bed_file_path(prefix, gene_junction_path, junction_type, gene_name=gene_name, dtype='junctions')
            write_junctions_to_bed(junctions[junction_type], gene_junction_bed_file)

            gene_exon_path = os.path.join(data_path, junction_type, gene_name)
            os.makedirs(gene_exon_path, exist_ok=True)
            gene_exon_bed_file = get_bed_file_path(prefix, gene_exon_path, junction_type, gene_name=gene_name, dtype='exons')
            write_exons_to_bed(exons[junction_type], gene_exon_bed_file)

            print(f"... Wrote n={len(junctions[junction_type])} junctions to gene-specific BED file:\n{gene_junction_bed_file}\n ...")
            print(f"... Wrote n={len(exons[junction_type])} exons to gene-specific BED file:\n{gene_exon_bed_file}\n ...")
        # End of gene loop
     
        # Prepare output including all junctions and exons for the given junction type
        junction_path = os.path.join(data_path, junction_type) 
        os.makedirs(junction_path, exist_ok=True)
        junction_bed_file = \
            get_bed_file_path(
                prefix, 
                junction_path, 
                junction_type, dtype='junctions')
        write_junctions_to_bed(all_junctions, junction_bed_file) # Write junctions to BED file

        # Write exons to BED file
        exon_path = os.path.join(data_path, junction_type) 
        os.makedirs(exon_path, exist_ok=True)
        exon_bed_file = \
            get_bed_file_path(
                prefix, 
                exon_path, 
                junction_type, dtype='exons')
        write_exons_to_bed(all_exons, exon_bed_file)

        junction_paths[junction_type] = junction_bed_file
        exon_paths[junction_type] = exon_bed_file
        print(f"[i/o] Wrote n={len(all_junctions)} junctions to BED file:\n{junction_bed_file}\n ...")
        print(f"[i/o] Wrote n={len(all_exons)} exons to BED file:\n{exon_bed_file}\n ...")

    return junction_paths, exon_paths


def make_junction_bed_file_from_exon_trio(gene_exon_trio, junction_type='cryptic_junctions', **kargs):
    """
    Extract junctions for a set of genes and their corresponding exon trios.

    Parameters:
    - gene_exon_trio (dict): Dictionary where keys are gene names and values are dictionaries with keys 'upstream', 
      'cryptic', 'downstream', and their corresponding exon coordinates.
    - junction_type (str): Type of junctions to collect ('cryptic_junctions' or 'canonical_junctions').

    Example usage: 
        gene_exon_trio = {
            'STMN2': {
                'upstream': ('8', 79611117, 79611214, '+'),
                'cryptic': ('8', 79616821, 79617048, '+'),
                'downstream': None
            },
            'UNC13A': {
                'upstream': ('19', 17642844, 17642960, '-'),
                'cryptic': ('19', 17642413, 17642541, '-'),
                'downstream': ('19', 17641392, 17641556, '-')
            }
        }
        make_junction_bed_file_from_exon_trio(gene_exon_trio, junction_type='cryptic_junctions')
    """
    data_dir = kargs.get('data_dir', "/path/to/meta-spliceai/data/ensembl/ALS") 
    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    all_junctions = []

    for gene_name, exon_trio in gene_exon_trio.items():
        print("[info] Processing gene:", gene_name)

        # Find the best matching transcript based on upstream and downstream exons
        best_transcript = find_matching_transcript(gene_name, exon_trio)
        if best_transcript:
            print(f"Found matching transcript for {gene_name}: {best_transcript.get('id')}")
            # Proceed with further processing using `best_transcript`
        else:
            print(f"No matching transcript found for {gene_name}.")

        if best_transcript:
            # Extract junctions using the best matching transcript and the cryptic exon
            junctions, _ = extract_junctions_for_trio(best_transcript, exon_trio, **kargs)
            
            # Sort junctions by coordinates
            sorted_junctions = sorted(junctions[junction_type], key=lambda j: (j[0], j[1], j[2], j[5]))
            # Sort by (chrom, start, end, strand)

            all_junctions.extend(sorted_junctions)
            print(f"[i/o] Extracted {junction_type} for gene: {gene_name}")
            print(f"... n={len(sorted_junctions)} junctions found.")
        else:
            print(f"[error] No suitable transcript found for gene: {gene_name}")

    # Prepare output path
    prefix = kargs.get("output_prefix", "")
    bed_file_path = os.path.join(data_dir, f"{prefix}_{junction_type}.bed" if prefix else f"{junction_type}.bed")

    # Write to BED file
    print(f"[i/o] Writing n={len(all_junctions)} junctions to BED file:\n{bed_file_path}\n ...")
    write_junctions_to_bed(all_junctions, bed_file_path)

    return bed_file_path


def display_junction_scores(junction_scores):
    """
    Display the junction scores dictionary in a readable format.
    
    Parameters:
    - junction_scores (dict): A dictionary mapping junction names to their average scores.

    Example usage:
        junction_scores = parse_junction_scores('path_to_junction_score_file.bed')
        display_junction_scores(junction_scores)
    """
    print(f"{'Junction Name':<20} {'Average Score':<15}")
    print("-" * 35)
    
    for name, score in sorted(junction_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{name:<20} {score:<15.2f}")


def parse_junction_scores(junction_score_file):
    """
    Parse the junction score file and return a dictionary mapping junction names to their average scores.
    
    Parameters:
    - junction_score_file (str): Path to the junction score file.

    Returns:
    - dict: A dictionary mapping junction names to their average scores.
    """
    junction_scores = {}
    
    with open(junction_score_file, 'r') as f:
        for line in f:
            chrom, start, end, name, score, strand, donor_score, acceptor_score = line.strip().split()
            average_score = (float(donor_score) + float(acceptor_score)) / 2
            junction_scores[name] = average_score
    
    return junction_scores


def compute_exon_scores_v0(junctions, junction_scores):
    """
    Compute exon scores based on the average of flanking junction scores.
    
    Parameters:
    - junctions (list of tuples): List of junction tuples (chrom, start, end, name, score, strand).
    - junction_scores (dict): Dictionary mapping junction names to their average scores.
    
    Returns:
    - list of tuples: List of exons with their calculated scores.
    """
    exon_scores = []
    
    if isinstance(junctions, dict): 
        junctions_by_transcript = junctions
    else: 
        # Group junctions by transcript ID
        junctions_by_transcript = defaultdict(list)
        
        for j in junctions:
            transcript_id = j[3].split('_')[0]  # Extract transcript ID from junction name
            junctions_by_transcript[transcript_id].append(j)
    
    # Process each transcript's junctions separately
    for transcript_id, transcript_junctions in junctions_by_transcript.items():
        transcript_junctions = sorted(transcript_junctions, key=lambda x: x[1])  # Sort by start position
        
        for i in range(len(transcript_junctions) - 1):
            chrom, start1, end1, name1, score1, strand1, *rest = transcript_junctions[i]
            chrom, start2, end2, name2, score2, strand2, *rest2 = transcript_junctions[i + 1]

            assert strand1 == strand2, f"Strand mismatch between junctions: {name1} ({strand1}) and {name2} ({strand2})"
            
            # Determine if the junctions are canonical or cryptic exon-related
            if "JUNC" in name1 and "JUNC" in name2:
                exon_name = name1.replace("JUNC", "EXON")
            elif "CRPT" in name1 or "CRPT" in name2:
                exon_name = f"{name1.split('_')[0]}_CRPT_EXON"
            else:
                exon_name = f"{name1}_EXON"  # Fallback in case patterns do not match

            # Average of the scores of the two flanking junctions
            if name1 in junction_scores and name2 in junction_scores:
                average_score = (junction_scores[name1] + junction_scores[name2]) / 2
                exon_scores.append((chrom, end1 + 1, start2 - 1, exon_name, average_score, strand1))
    
    return exon_scores


def compute_exon_scores(cryptic_junctions, cryptic_scores, canonical_junctions, canonical_scores):
    """
    Compute exon scores based on the average of flanking junction scores, separately for cryptic and canonical exons,
    and ensure that junctions are grouped by transcript ID.
    
    Parameters:
    - cryptic_junctions (list of tuples): List of junctions induced by cryptic exons.
    - cryptic_scores (dict): Dictionary mapping cryptic junction names to their average scores.
    - canonical_junctions (list of tuples): List of junctions from the canonical transcript.
    - canonical_scores (dict): Dictionary mapping canonical junction names to their average scores.
    
    Returns:
    - list of tuples: List of exons with their calculated scores.
    """
    exon_scores = []

    # Group cryptic junctions by transcript ID
    cryptic_junctions_by_transcript = defaultdict(list)
    for j in cryptic_junctions:
        transcript_id = j[3].split('_')[0]  # Extract transcript ID from junction name
        cryptic_junctions_by_transcript[transcript_id].append(j)

    # Group canonical junctions by transcript ID
    canonical_junctions_by_transcript = defaultdict(list)
    for j in canonical_junctions:
        transcript_id = j[3].split('_')[0]  # Extract transcript ID from junction name
        canonical_junctions_by_transcript[transcript_id].append(j)

    # Process cryptic junctions first
    for transcript_id, junctions in cryptic_junctions_by_transcript.items():
        junctions = sorted(junctions, key=lambda x: x[1])  # Sort junctions by start position
        # This orders the junctions from 5' to 3' in the transcript sequence

        for i in range(len(junctions) - 1):
            chrom, start1, end1, name1, score1, strand1, *rest = junctions[i]
            chrom, start2, end2, name2, score2, strand2, *rest2 = junctions[i + 1]

            assert strand1 == strand2, "Strand mismatch between junctions"

            # Check if the junctions are part of the cryptic exon
            print("[debug] any cryptic in names: ", "CRPT" in name1, "CRPT" in name2)
            print(f"... name2: {name2}")
            print(f"... name1: {name1}")
            if "CRPT" in name1 and "CRPT" in name2:
                # print(f"[debug] name1: {name1}, name2: {name2}")
                if "_5p" in name1 and "_3p" in name2:
                    # Extract just the numeric index from the CRPT part
                    exon_index = name1.split('_')[1].replace("CRPT", "")
                    exon_name = f"{transcript_id}_CRPT_EXON{exon_index}"
                    if name1 in cryptic_scores and name2 in cryptic_scores:
                        average_score = (cryptic_scores[name1] + cryptic_scores[name2]) / 2
                        exon_scores.append((chrom, end1 + 1, start2 - 1, exon_name, average_score, strand1))
                elif "_5p" in name1 or "_3p" in name2:
                    
                    continue

    n_cryptic_exons = len(exon_scores)
    print("[info] Collected n={} cryptic exon scores".format(len(exon_scores)))

    # Process canonical junctions
    for transcript_id, junctions in canonical_junctions_by_transcript.items():
        junctions = sorted(junctions, key=lambda x: x[1])  # Sort junctions by start position
        for i in range(len(junctions) - 1):
            chrom, start1, end1, name1, score1, strand1, *rest = junctions[i]
            chrom, start2, end2, name2, score2, strand2, *rest2 = junctions[i + 1]

            assert strand1 == strand2, "Strand mismatch between junctions"

            # Ensure that we are dealing with canonical junctions
            if "JUNC" in name1 and "JUNC" in name2:
                exon_index = name2.split('JUNC')[-1]  # Use the larger index for the exon
                exon_name = f"{transcript_id}_EXON{exon_index}"
                if name1 in canonical_scores and name2 in canonical_scores:
                    average_score = (canonical_scores[name1] + canonical_scores[name2]) / 2
                    exon_scores.append((chrom, end1 + 1, start2 - 1, exon_name, average_score, strand1))

    print("[info] Collected n={} canonical exon scores".format(len(exon_scores)-n_cryptic_exons))

    return exon_scores


def compute_exon_scores_v1(canonical_transcript, junction_scores):
    """
    Compute exon scores based on the scores of flanking junctions.

    Parameters:
    - canonical_transcript (dict): A transcript object from Ensembl API.
    - junction_scores (list of tuples): List of tuples with junction information (chrom, start, end, name, score, strand).

    Returns:
    - list of tuples: List of exons with their coordinates and computed scores.
    """
    exons = sorted(canonical_transcript['Exon'], key=lambda x: x['start'])
    transcript_id = canonical_transcript.get('id', 'unknown')

    exon_scores = []

    for i, exon in enumerate(exons):
        chrom = exon['seq_region_name']
        start = exon['start']
        end = exon['end']
        strand = exon['strand']

        # Filter junctions belonging to the current transcript
        relevant_junctions = [j for j in junction_scores if j[3].startswith(transcript_id)]

        # Find scores for flanking junctions
        prev_junction_score = None
        next_junction_score = None

        for j in relevant_junctions:
            j_start, j_end = j[1], j[2]
            if j_end == start - 1:  # Previous junction
                prev_junction_score = (j[5] + j[6]) / 2  # Average of donor and acceptor scores
            elif j_start == end + 1:  # Next junction
                next_junction_score = (j[5] + j[6]) / 2  # Average of donor and acceptor scores

        # Compute the exon score as the average of the two junction scores
        scores = []
        if prev_junction_score is not None:
            scores.append(prev_junction_score)
        if next_junction_score is not None:
            scores.append(next_junction_score)

        exon_score = sum(scores) / len(scores) if scores else 0

        exon_id = exon.get('id', f"{transcript_id}_Exon_{i + 1}")

        exon_scores.append((chrom, start, end, exon_id, exon_score, strand))

    return exon_scores


def parse_junction_bed_file(junction_bed_file):
    """
    Parse a junction BED file into a list of tuples.
    
    Parameters:
    - junction_bed_file (str): Path to the junction BED file.
    
    Returns:
    - list of tuples: Each tuple contains (chrom, start, end, name, score, strand).
    """
    print(f"[i/o] Parsing junction BED file: \n{junction_bed_file}\n")
    junctions = []
    with open(junction_bed_file, 'r') as f:
        for line in f:
            chrom, start, end, name, score, strand = line.strip().split()[:6]
            junctions.append((chrom, int(start), int(end), name, int(score), strand))
    print(f"... found n={len(junctions)} junctions")
    return junctions


def sort_and_write_bed_v0(exon_scores, bed_file_path):
    """
    Sort exon scores by chromosome, then by ascending start positions, and write them to a BED file.
    
    Parameters:
    - exon_scores (list of tuples): List of exons with their calculated scores.
    - bed_file_path (str): Path to the output BED file.
    """
    # Convert exon_scores to a DataFrame for easier sorting
    df = pd.DataFrame(exon_scores, columns=["chrom", "start", "end", "name", "score", "strand"])
    
    # Sort by chromosome (numerically and lexically), then by start, and finally by name to group transcripts
    df['chrom'] = pd.Categorical(df['chrom'], categories=sorted(df['chrom'].unique()), ordered=True)
    df = df.sort_values(by=['chrom', 'start', 'name'])
    
    # Write sorted DataFrame to a BED file
    df.to_csv(bed_file_path, sep='\t', header=False, index=False)
    
    print(f"Sorted BED file written to: {bed_file_path}")


def sort_and_write_bed(exon_scores, bed_file_path):
    """
    Sort exon scores by chromosome (numerically and lexically), then by ascending start positions, and write them to a BED file.
    
    Parameters:
    - exon_scores (list of tuples): List of exons with their calculated scores.
    - bed_file_path (str): Path to the output BED file.
    """
    # Convert exon_scores to a DataFrame for easier sorting
    df = pd.DataFrame(exon_scores, columns=["chrom", "start", "end", "name", "score", "strand"])
    
    # Define a custom sorting key for chromosomes
    def chrom_sort_key(chrom):
        try:
            # Try to convert to an integer (for numbered chromosomes)
            return (True, int(chrom))
        except ValueError:
            # For non-numbered chromosomes, sort alphabetically
            return (False, chrom)
    
    # Sort the DataFrame by chromosome using the custom key, then by start and name
    df['chrom'] = df['chrom'].apply(lambda x: chrom_sort_key(x))
    df = df.sort_values(by=['chrom', 'start', 'name'])
    
    # Drop the auxiliary sorting column (chrom) after sorting
    df['chrom'] = df['chrom'].apply(lambda x: x[1])
    
    # Write sorted DataFrame to a BED file
    df.to_csv(bed_file_path, sep='\t', header=False, index=False)
    
    print(f"Sorted BED file written to: {bed_file_path}")


def generate_exon_bed_file(exon_scores, exon_bed_file_path):
    """
    Generate a BED file containing exons and their calculated scores.
    
    Parameters:
    - exon_scores (list of tuples): List of exons with their calculated scores.
    - exon_bed_file_path (str): Path to the output exon BED file.
    """
    with open(exon_bed_file_path, 'w') as f:
        for exon in exon_scores:
            chrom, start, end, name, score, strand = exon
            f.write(f"{chrom}\t{start}\t{end}\t{name}\t{score:.6f}\t{strand}\n")


def plot_transcript(bed_file_path, output_file=None):
    """

    Example usage:
    plot_transcript("path_to_your_file/exon_scores.bed", "output_plot.png")
    plot_transcript("path_to_your_file/exon_scores.bed")
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the BED file
    df = pd.read_csv(bed_file_path, sep='\t', header=None, names=["chrom", "start", "end", "name", "score", "strand"])

    # Split the transcript ID and exon name from the 'name' column
    df['transcript_id'] = df['name'].apply(lambda x: x.split('_')[0])
    df['exon_name'] = df['name'].apply(lambda x: '_'.join(x.split('_')[1:]))

    # Sort by transcript_id, chrom, and start to ensure correct exon order
    df = df.sort_values(by=['transcript_id', 'chrom', 'start'])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    for transcript_id, group in df.groupby('transcript_id'):
        ax.plot(group['start'], group['score'], marker='o', label=transcript_id)

    ax.set_xlabel('Genomic Coordinate (Start)')
    ax.set_ylabel('Exon Score')
    ax.set_title('Exon Scores per Transcript')
    ax.legend(title='Transcript ID', bbox_to_anchor=(1.05, 1), loc='upper left')

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def extract_splice_junctions_workflow(data_prefix, gtf_file, consensus_window=2, **kargs):
    import gffutils

    verify = kargs.get("verify", False)
    assert os.path.exists(gtf_file), f"GTF file not found: {gtf_file}"
    
    db_file = os.path.join(data_prefix, "annotations.db")
    db = build_gtf_database(gtf_file, db_file=db_file, overwrite=False)

    format = 'tsv'
    output_file = os.path.join(data_prefix, f"junctions_all_transcripts.{format}")
    extract_junctions_for_all_genes(db, save=True, output_file=output_file)

    if verify: 
        # Load the junctions from the saved file
        junctions_df = pd.read_csv(output_file, sep='\t')
        # NOTE: By default, pd.read_csv() reads the first row of the CSV file as the header; i.e. header=0 by default

        print("(extract_splice_sites_workflow) Junction dataframe:")
        print(junctions_df.head())


def demo_extract_junctions(): 

    data_path = "/path/to/meta-spliceai/data/ensembl/ALS"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Example usage for STMN2
    stmn2_cryptic_exon = ('8', 79616821, 79617048, '+')
    junctions = get_junctions_with_cryptic_exon('STMN2', stmn2_cryptic_exon)
    print(junctions)

    # Example usage for UNC13A
    gene_name = "UNC13A"
    unc13a_cryptic_exon = ('19', 17642413, 17642541, '-')
    junctions = get_junctions_with_cryptic_exon(gene_name, unc13a_cryptic_exon)
    # print(junctions)

    bed_file_path = os.path.join(data_path, "junctions_with_cryptic_exon.bed")
    print(f"[i/o] Writing n={len(junctions['cryptic_junctions'])} junctions to BED file:\n{bed_file_path}\n ...")
    write_junctions_to_bed(junctions['cryptic_junctions'], bed_file_path)



def demo_make_junction_bed_file():

    data_dir = "/path/to/meta-spliceai/data/ensembl/ALS"
    cryptic_junc_fn = os.path.join(data_dir, "cryptic_junctions.bed")  # canonical + cryptic junctions
    canonical_junc_fn = os.path.join(data_dir, "canonical_junctions.bed")

    # Method 1 
    gene_exon_pairs = [
        ('STMN2', ('8', 79616821, 79617048, '+')),
        ('UNC13A', ('19', 17642413, 17642541, '-'))
    ]

    print("[info] Generating junction BED files by types ...")
    # make_junction_bed_file(gene_exon_pairs, junction_type='cryptic_junctions')   # cryptic + canonical junctions
    # make_junction_bed_file(gene_exon_pairs, junction_type='canonical_junctions')  # canonical junctions only

    ############################################################

    # Method 2
    gene_exon_trio = \
    {
        'STMN2': {
            'upstream': ('8', 79611117, 79611214, '+'),
            'cryptic': ('8', 79616821, 79617048, '+'),
            # Downstream is omitted or set to None because it doesn't exist in the alternative isoform.
            'downstream': None
        },
        'UNC13A': {
            # 'upstream': ('19', 17642844, 17642960, '-'),
            # 'cryptic': ('19', 17642413, 17642541, '-'),
            # 'downstream': ('19', 17641392, 17641556, '-')

            'upstream': ('19', 17642845, 17642960, '-'),
            'cryptic': ('19', 17642414, 17642541, '-'),
            'downstream': ('19', 17641393, 17641556, '-')
        }
    }

    make_junction_bed_file_v2(gene_exon_trio)  # cryptic + canonical junctions

    return


def demo_workflow(): 

    print_emphasized("[action] Creating Junction BED files ...")
    # demo_extract_junctions()
    demo_make_junction_bed_file()
    # Output: Junction scores in BED format: 
    #         /path/to/meta-spliceai/data/ensembl/ALS/junction_score.bed

    data_dir = "/path/to/meta-spliceai/data/ensembl/ALS"
    cryptic_junc_fn = os.path.join(data_dir, "cryptic/cryptic_junctions.bed")
    canonical_junc_fn = os.path.join(data_dir, "canonical/canonical_junctions.bed")

    # Run Splam 
    # 1. Score cryptic junctions

    command = """
splam score -V \
-G /path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
-m ./splam/model/splam_script.pt \
-o /path/to/meta-spliceai/data/ensembl/ALS/cryptic/ \
/path/to/meta-spliceai/data/ensembl/ALS/cryptic/cryptic_junctions.bed
"""
    # subprocess.call(command, shell=True)
    # Use subprocess to run the command
    # result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

    # 2. Score canonical junctions
    command = """
splam score -V \
-G /path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
-m ./splam/model/splam_script.pt \
-o /path/to/meta-spliceai/data/ensembl/ALS/canonical \
/path/to/meta-spliceai/data/ensembl/ALS/canonical/canonical_junctions.bed
"""
    # subprocess.call(command, shell=True)

    # canonical_dir = os.path.join(data_dir, 'canonical')
    # cryptic_dir = os.path.join(data_dir, 'cryptic')

    # Analyze junction scores and use them to infer exon inclusion levels
    cryptic_score_file = os.path.join(data_dir, "cryptic/junction_score.bed")
    cryptic_junction_scores = parse_junction_scores(cryptic_score_file)
    print(f"> Cryptic Junction Scores (n={len(cryptic_junction_scores)}): {cryptic_junction_scores}")

    canonical_score_file = os.path.join(data_dir, "canonical/junction_score.bed")
    canonical_junction_scores = parse_junction_scores(canonical_score_file)
    print(f"> Canonical Junction Scores (n={len(canonical_junction_scores)}): {canonical_junction_scores}")

    # Display the junction scores
    # display_junction_scores(junction_scores)

    # Parse the junction BED file
    print(f"[i/o] Parsing cryptic junction BED file: \n{cryptic_junc_fn}\n")
    cryptic_junctions = parse_junction_bed_file(cryptic_junc_fn)
    print(f"[i/o] Parsing canonical junction BED file: \n{canonical_junc_fn}\n")
    canonical_junctions = parse_junction_bed_file(canonical_junc_fn)

    # Compute exon scores based on junction scores
    exon_scores = \
        compute_exon_scores(
            cryptic_junctions, 
            cryptic_junction_scores, 
            canonical_junctions, 
            canonical_junction_scores)
    # NOTE: junctions as a dictionary? mapping transcript ID to their junctions? 
    #       format: (chrom, end1 + 1, start2 - 1, exon_name, average_score, strand1)

    # Generate the exon BED file with calculated scores
    exon_bed_file_path = os.path.join(data_dir, "exon_scores.bed")
    # generate_exon_bed_file(exon_scores, exon_bed_file_path)
    sort_and_write_bed(exon_scores, exon_bed_file_path)
    print(f"[Output] Exon BED file written to: {exon_bed_file_path}")

    plot_path = os.path.join(data_dir, "exon_scores.pdf")
    plot_transcript(exon_bed_file_path, output_file=plot_path)

    return


def demo_verify_cryptic_exon_with_matched_canonical_transcript(): 

    # demo_make_junction_bed_file()
    # Output: Junction scores in BED format: 
    #         /path/to/meta-spliceai/data/ensembl/ALS/junction_score.bed

    data_dir = "/path/to/meta-spliceai/data/ensembl/ALS"
    cryptic_junc_fn = os.path.join(data_dir, "cryptic_junctions.bed")  # canonical + cryptic junctions
    canonical_junc_fn = os.path.join(data_dir, "canonical_junctions.bed")

    gene_exon_trio = \
    {
        'STMN2': {
            'upstream': ('8', 79611117, 79611214, '+'),
            'cryptic': ('8', 79616821, 79617048, '+'),
            # Downstream is omitted or set to None because it doesn't exist in the alternative isoform.
            'downstream': None
        },
        'UNC13A': {
            # 'upstream': ('19', 17642844, 17642960, '-'),
            # 'cryptic': ('19', 17642413, 17642541, '-'),
            # 'downstream': ('19', 17641392, 17641556, '-')

            'upstream': ('19', 17642845, 17642960, '-'),
            'cryptic': ('19', 17642414, 17642541, '-'),
            'downstream': ('19', 17641393, 17641556, '-')
        }
    }

    for gene, exon_trio in gene_exon_trio.items():
        matching_transcript = find_matching_transcript(gene, exon_trio)
        if matching_transcript:
            print(f"Found matching transcript for {gene}: {matching_transcript['id']}")
            # Proceed with further processing using `matching_transcript`
        else:
            print(f"No matching transcript found for {gene}.")

    all_junctions, all_exons = extract_junctions_and_exons(transcript, exon_trio, start_index=0)
    print("[info] Extracted junctions and exons:")
    print(f"... n={len(all_junctions)} junctions")
    print(f"... n={len(all_exons)} exons")



    return


def demo(): 

    # demo_verify_cryptic_exon_with_matched_canonical_transcript()

    demo_make_junction_bed_file()
    
    # From matching canonical transcript, extract and score junctions and exons, to plotting, etc. 
    # demo_workflow()

    


if __name__ == "__main__": 
    demo()
