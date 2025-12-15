
import os
import pandas as pd

import pybedtools
from pybedtools import BedTool


def extract_gtf_annotations_by_biopython(gtf_file, transcript_id):
    """
    Extract exon, CDS, 5'UTR, and 3'UTR annotations for a specific transcript from a GTF file using Biopython.

    Parameters:
    - gtf_file (str): Path to the GTF file containing transcript annotations.
    - transcript_id (str): The transcript ID to extract annotations for.

    Returns:
    - annotations (dict): Dictionary of features (exons, CDS, UTRs) with their coordinates.
    """
    from BCBio import GFF  # pip install bcbio-gff
    from Bio import SeqIO

    annotations = {'exons': [], 'CDS': [], '5UTR': [], '3UTR': []}
    
    # Read GTF file using GFF parsing in Biopython
    with open(gtf_file) as gtf_handle:
        for record in GFF.parse(gtf_handle):
            for feature in record.features:
                if 'transcript_id' in feature.qualifiers and transcript_id in feature.qualifiers['transcript_id']:
                    # Extract exons
                    if feature.type == "exon":
                        annotations['exons'].append((record.id, feature.location.start, feature.location.end, feature.strand))

                    # Extract CDS regions
                    elif feature.type == "CDS":
                        annotations['CDS'].append((record.id, feature.location.start, feature.location.end, feature.strand))

                    # Extract 5'UTR and 3'UTR regions
                    elif feature.type == "five_prime_UTR":
                        annotations['5UTR'].append((record.id, feature.location.start, feature.location.end, feature.strand))
                    elif feature.type == "three_prime_UTR":
                        annotations['3UTR'].append((record.id, feature.location.start, feature.location.end, feature.strand))

    return annotations


def extract_gtf_annotations_v0(gtf_file, transcript_id):
    """
    Extract exon, CDS, 5'UTR, and 3'UTR annotations for a specific transcript from a GTF file using pybedtools.

    Parameters:
    - gtf_file (str): Path to the GTF file containing transcript annotations.
    - transcript_id (str): The transcript ID to extract annotations for.

    Returns:
    - annotations (dict): Dictionary of features (exons, CDS, UTRs) with their coordinates.
    """
    # from pybedtools import BedTool
    gtf = BedTool(gtf_file)
    
    # Filter the GTF for exons, CDS, 5'UTR, and 3'UTR annotations
    transcript_exons = gtf.filter(lambda x: x.fields[2] == "exon" and 'transcript_id "{}"'.format(transcript_id) in x.fields[8]).saveas()
    transcript_cds = gtf.filter(lambda x: x.fields[2] == "CDS" and 'transcript_id "{}"'.format(transcript_id) in x.fields[8]).saveas()
    transcript_5utr = gtf.filter(lambda x: x.fields[2] == "five_prime_UTR" and 'transcript_id "{}"'.format(transcript_id) in x.fields[8]).saveas()
    transcript_3utr = gtf.filter(lambda x: x.fields[2] == "three_prime_UTR" and 'transcript_id "{}"'.format(transcript_id) in x.fields[8]).saveas()
    # NOTE: When using pybedtools.BedTool and performing operations like filtering with a lambda function, 
    #       the results are typically in-memory objects. To convert these results to a DataFrame, 
    #       you need to save them to a temporary file first.

    # Convert the filtered BedTool objects to DataFrames
    def to_dataframe(bedtool_obj):
        if len(bedtool_obj) > 0:
            return bedtool_obj.to_dataframe(names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no features were found

    exon_df = to_dataframe(transcript_exons)
    cds_df = to_dataframe(transcript_cds)
    utr5_df = to_dataframe(transcript_5utr)
    utr3_df = to_dataframe(transcript_3utr)
    
    # Combine the annotations into a dictionary, checking for empty DataFrames
    annotations = {
        'exons': exon_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not exon_df.empty else [],
        'CDS': cds_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not cds_df.empty else [],
        '5UTR': utr5_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not utr5_df.empty else [],
        '3UTR': utr3_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not utr3_df.empty else []
    }
    
    return annotations


def extract_gtf_annotations(gtf_file, transcript_id):
    """
    Extract exon, CDS, 5'UTR, and 3'UTR annotations for a specific transcript from a GTF file using pybedtools.

    Parameters:
    - gtf_file (str): Path to the GTF file containing transcript annotations.
    - transcript_id (str): The transcript ID to extract annotations for.

    Returns:
    - annotations (dict): Dictionary of features (exons, CDS, UTRs) with their coordinates.
    """
    # from pybedtools import BedTool
    gtf = BedTool(gtf_file)
    
    # Filter the GTF for exons and CDS annotations
    transcript_exons = gtf.filter(lambda x: x.fields[2] == "exon" and f'transcript_id "{transcript_id}"' in x.fields[8]).saveas()
    transcript_cds = gtf.filter(lambda x: x.fields[2] == "CDS" and f'transcript_id "{transcript_id}"' in x.fields[8]).saveas()

    # Convert the filtered BedTool objects to DataFrames
    def to_dataframe(bedtool_obj):
        if len(bedtool_obj) > 0:
            return bedtool_obj.to_dataframe(names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no features were found

    exon_df = to_dataframe(transcript_exons)
    cds_df = to_dataframe(transcript_cds)

    # Sort the DataFrames by start coordinate
    exon_df = exon_df.sort_values(by='start')
    cds_df = cds_df.sort_values(by='start')

    # Get transcript coordinates
    if not exon_df.empty:
        chrom = exon_df['chrom'].iloc[0]
        strand = exon_df['strand'].iloc[0]
        transcript_start = exon_df['start'].min()
        transcript_end = exon_df['end'].max()
    else:
        return {}  # If no exons are found, return an empty dictionary

    # Infer 5'UTR and 3'UTR based on strand and CDS regions
    if not cds_df.empty:
        if strand == '+':
            # Ensure that UTR coordinates do not go out of the transcript bounds
            utr5_start, utr5_end = transcript_start, max(transcript_start, cds_df['start'].min() - 1)
            utr3_start, utr3_end = min(transcript_end, cds_df['end'].max() + 1), transcript_end
        else:
            utr5_start, utr5_end = min(transcript_end, cds_df['end'].max() + 1), transcript_end
            utr3_start, utr3_end = transcript_start, max(transcript_start, cds_df['start'].min() - 1)

        # Ensure the coordinates are valid, i.e., start < end
        utr5_df = pd.DataFrame([[chrom, utr5_start, utr5_end, strand]], columns=['chrom', 'start', 'end', 'strand']) \
            if utr5_start < utr5_end else pd.DataFrame()

        utr3_df = pd.DataFrame([[chrom, utr3_start, utr3_end, strand]], columns=['chrom', 'start', 'end', 'strand']) \
            if utr3_start < utr3_end else pd.DataFrame()
    else:
        # If no CDS is found, the entire transcript could be non-coding or incomplete
        utr5_df = pd.DataFrame()
        utr3_df = pd.DataFrame()

    # Combine the annotations into a dictionary, checking for empty DataFrames
    annotations = {
        'exons': exon_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not exon_df.empty else [],
        'CDS': cds_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not cds_df.empty else [],
        '5UTR': utr5_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not utr5_df.empty else [],
        '3UTR': utr3_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not utr3_df.empty else []
    }
    
    return annotations



def infer_exons_with_gtf(junction_bed, gtf_file, transcript_id):
    """
    Infer exon coordinates from a junction BED file using GTF annotations.

    Parameters:
    - junction_bed (str): Path to the junction BED file with donor and acceptor sites.
    - gtf_file (str): Path to the GTF file containing transcript annotations.
    - transcript_id (str): The transcript ID for which to infer exon coordinates.

    Returns:
    - exons (list): List of inferred exon coordinates in the format [chrom, start, end, strand].
    """
    # Extract GTF annotations for the given transcript ID
    annotations = extract_gtf_annotations(gtf_file, transcript_id)
    exons_gtf = annotations['exons']
    transcript_cds = annotations['CDS']
    transcript_5utr = annotations['5UTR']
    transcript_3utr = annotations['3UTR']
    
    if len(exons_gtf) == 0:
        raise ValueError(f"No exons found for transcript {transcript_id} in the GTF file.")

    # Load the junction BED file
    junction_df = pd.read_csv(junction_bed, sep='\t', header=None, names=[
        'chrom', 'start', 'end', 'name', 'score', 'strand', 'donor_prob', 'acceptor_prob'
    ])

    # Initialize the list of inferred exons
    inferred_exons = []

    # Get transcript information from GTF annotations
    chrom = exons_gtf[0][0]  # Chromosome name from the first exon entry
    strand = exons_gtf[0][3]  # Strand information from the first exon entry

    # Loop through the junctions to infer exon coordinates
    if strand == '+':
        # Positive strand: exons are between donor end and acceptor start
        for i in range(len(junction_df) - 1):
            donor_end = junction_df.iloc[i]['end']
            acceptor_start = junction_df.iloc[i + 1]['start']
            exon_start = donor_end
            exon_end = acceptor_start
            inferred_exons.append([chrom, exon_start, exon_end, strand])

        # Add the first and last exons using the transcript start and end
        transcript_start = min(exons_gtf, key=lambda x: x[1])[1]  # Start position of the first exon
        first_exon_end = junction_df.iloc[0]['start']
        inferred_exons.insert(0, [chrom, transcript_start, first_exon_end, strand])

        transcript_end = max(exons_gtf, key=lambda x: x[2])[2]  # End position of the last exon
        last_exon_start = junction_df.iloc[-1]['end']
        inferred_exons.append([chrom, last_exon_start, transcript_end, strand])

    elif strand == '-':
        # Negative strand: exons are between acceptor end and donor start
        for i in range(len(junction_df) - 1):
            acceptor_end = junction_df.iloc[i]['end']
            donor_start = junction_df.iloc[i + 1]['start']
            exon_start = donor_start
            exon_end = acceptor_end
            inferred_exons.append([chrom, exon_start, exon_end, strand])

        # Add the first and last exons using the transcript start and end
        transcript_start = min(exons_gtf, key=lambda x: x[1])[1]  # Start position of the first exon
        first_exon_start = junction_df.iloc[0]['end']
        inferred_exons.insert(0, [chrom, first_exon_start, transcript_start, strand])

        transcript_end = max(exons_gtf, key=lambda x: x[2])[2]  # End position of the last exon
        last_exon_end = junction_df.iloc[-1]['start']
        inferred_exons.append([chrom, transcript_end, last_exon_end, strand])

    # Output inferred exons along with any UTR information
    exons_with_annotations = {
        'exons': inferred_exons,
        'CDS': transcript_cds,
        '5UTR': transcript_5utr,
        '3UTR': transcript_3utr
    }

    return exons_with_annotations


def merge_exons(exons):
    """
    Merge overlapping or adjacent exons and refine boundaries.
    """
    merged_exons = []
    current_exon = exons[0]

    for next_exon in exons[1:]:
        # Merge overlapping or adjacent exons
        if next_exon[1] <= current_exon[2]:  
            current_exon = (current_exon[0], current_exon[1], max(current_exon[2], next_exon[2]), current_exon[3], current_exon[4])
        else:
            merged_exons.append(current_exon)
            current_exon = next_exon

    merged_exons.append(current_exon)
    return merged_exons


def demo_infer_exons_with_gtf():

    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    local_dir = "/path/to/meta-spliceai/data/ensembl/ALS"
    genome_annot = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf") 

    genome_fasta = os.path.join(
        data_prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")

    assert os.path.exists(genome_annot)
    assert os.path.exists(genome_fasta)
    
    # Example usage
    junction_data = {
        'chrom': ['19', '19', '19', '19'],
        'start': [17606354, 17610099, 17611855, 17617849],
        'end': [17609940, 17611763, 17617702, 17618421],
        'name': ['ENST00000519716_JUNC_42', 'ENST00000519716_JUNC_41', 'ENST00000519716_JUNC_40', 'ENST00000519716_JUNC_39'],
        'score': [996.21, 995.28, 947.42, 997.84],
        'strand': ['-', '-', '-', '-'],
        'donor_prob': [0.9952, 0.9903, 0.9151, 0.9962],
        'acceptor_prob': [0.9950, 0.9980, 0.9776, 0.9972]
    }
    junction_bed_df = pd.DataFrame(junction_data)
    gtf_file_path = genome_annot
    transcript_id = 'ENST00000519716'

    # inferred_exons = infer_exons_with_gtf(junction_bed_df, gtf_file_path, transcript_id)
    # print("Inferred Exons with UTRs and CDS:", inferred_exons)

    method = 'bedtools' # 'biopython', 'bedtools'

    if method == 'biopython':
        annotations = extract_gtf_annotations_by_biopython(gtf_file_path, transcript_id)
    else:
        annotations = extract_gtf_annotations(gtf_file_path, transcript_id)
    
    print("Exons:", annotations['exons'])
    print("CDS:", annotations['CDS'])
    print("5' UTR:", annotations['5UTR'])
    print("3' UTR:", annotations['3UTR'])

    return



def test(): 

    demo_infer_exons_with_gtf()


    return


if __name__ == "__main__":
    test() 