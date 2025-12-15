import pandas as pd
from pybedtools import BedTool
from Bio import SeqIO

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

