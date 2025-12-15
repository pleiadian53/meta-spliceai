from Bio import SeqIO
import pandas as pd

def extract_utrs(gff_file, genome_file):
    # Use GffRead or similar tool to extract 5' UTRs
    pass

def find_orfs(sequence, min_length=30):
    orfs = []
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}

    for frame in range(3):  # Check all three reading frames
        for i in range(frame, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if codon == start_codon:
                for j in range(i, len(sequence) - 2, 3):
                    stop_codon = sequence[j:j+3]
                    if stop_codon in stop_codons:
                        orf_length = j - i + 3
                        if orf_length >= min_length:
                            orfs.append((i, j + 3, frame))
                        break
    return orfs

def create_gtf(features, output_gtf_path):
    data = []
    for feature in features:
        attributes_str = ' '.join(f'{key} "{value}";' for key, value in feature['attributes'].items())
        data.append([
            feature['seqname'],
            feature['source'],
            feature['feature'],
            feature['start'],
            feature['end'],
            feature['score'],
            feature['strand'],
            feature['frame'],
            attributes_str
        ])
    
    df = pd.DataFrame(data, columns=[
        'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'
    ])
    
    df.to_csv(output_gtf_path, sep='\t', header=False, index=False, quoting=pd.io.common.csv.QUOTE_NONE, quotechar='')


def run_example_workflow(): 
    """
    This function demonstrates how to identify uORFs in any reading frame within the 5' UTR sequences 
    and annotate them appropriately in a GTF file. 
    The `find_orfs` function considers all three reading frames, ensuring comprehensive uORF detection.
    """

    # Example usage
    utrs = extract_utrs("annotations.gff", "genome.fa")
    features = []

    for record in utrs:
        sequence = str(record.seq)
        orfs = find_orfs(sequence)
        for orf in orfs:
            start, end, frame = orf
            feature = {
                'seqname': record.id,
                'source': 'uORFexplorer',
                'feature': 'uORF',
                'start': start + 1,  # GTF is 1-based
                'end': end,
                'score': '.',
                'strand': record.annotations['strand'],
                'frame': frame,
                'attributes': {
                    'gene_id': record.id,
                    'orf_id': f'uorf_{start}_{end}',
                    'transcript_id': record.id
                }
            }
            features.append(feature)

    create_gtf(features, 'output.gtf')


def demo(): 
 
    run_example_workflow()


if __name__ == '__main__':
    demo()  
