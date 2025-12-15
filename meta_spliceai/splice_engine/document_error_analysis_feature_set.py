import os
import pandas as pd

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
)

from .model_evaluator import ModelEvaluationFileHandler
from .extract_genomic_features import FeatureAnalyzer, SpliceAnalyzer

# Define the feature names and their descriptions
feature_data = [
    ('gene_id', 'Unique identifier for a gene.'),
    ('transcript_id', 'Unique identifier for a transcript.'),
    ('position', 'Relative position of the splice site (strand-dependent: relative to gene start on positive strand, and gene end on negative strand).'),
    ('score', 'SpliceAI-predicted probability of a splice site.'),
    ('splice_type', 'Type of splice site: donor or acceptor.'),
    ('chrom', 'Chromosome where the gene is located.'),
    ('strand', 'Strand direction of the gene: "+" or "-".'),
    ('gc_content', 'Proportion of GC content in the sequence context window.'),
    ('sequence_complexity', 'Measure of sequence complexity (e.g., Shannon entropy) in the context window.'),
    ('sequence_length', 'Length of the sequence window.'),
    ('transcript_length', 'Total length of the transcript.'),
    ('tx_start', 'Start position of the transcript in the genome.'),
    ('tx_end', 'End position of the transcript in the genome.'),
    ('gene_start', 'Start position of the gene in the genome.'),
    ('gene_end', 'End position of the gene in the genome.'),
    ('gene_type', 'Biotype of the gene (e.g., protein-coding, lncRNA).'),
    ('gene_length', 'Length of the gene in base pairs.'),
    ('num_exons', 'Total number of exons in the gene.'),
    ('avg_exon_length', 'Average length of exons in the gene.'),
    ('median_exon_length', 'Median exon length in the gene.'),
    ('total_exon_length', 'Sum of all exon lengths in the gene.'),
    ('total_intron_length', 'Sum of all intron lengths in the gene.'),
    ('n_splice_sites', 'Total number of splice sites for the gene.'),
    ('num_overlaps', 'Number of overlapping genes with the current gene.'),
    ('absolute_position', 'Absolute genomic position of the splice site.'),
    ('distance_to_start', 'Distance from the predicted splice site to the gene start.'),
    ('distance_to_end', 'Distance from the predicted splice site to the gene end.'),
    ('has_consensus', 'Binary flag indicating whether the splice site matches canonical consensus sequences.'),
    # General description for k-mer features
    ('^\d+mer_.*', 'Counts of specific k-mers in the sequence context window (e.g., 3mer_ACC, 2mer_GT, with their naming following the regex r"^\d+mer_.*").')
]

# Additional documentation for exon_length related features
exon_length_calculation_doc = """
Exon Length Related Features Calculation:
- num_exons: Calculated as the count of exon lengths per transcript.
- avg_exon_length: Calculated as the mean of exon lengths per transcript.
- median_exon_length: Calculated as the median of exon lengths per transcript.
- total_exon_length: Calculated as the sum of exon lengths per transcript.

These features are derived from the exon_df DataFrame using group_by and aggregation operations.
"""

# Create a DataFrame for the feature codebook
feature_codebook = pd.DataFrame(feature_data, columns=["Feature Name", "Description"])

# Save the codebook to an Excel file
file_path = os.path.join(FeatureAnalyzer.analysis_dir, "error_analysis_feature_codebook.xlsx")

# Create a DataFrame for the documentation
exon_doc_df = pd.DataFrame({"Documentation": [exon_length_calculation_doc]})

# feature_codebook.to_excel(file_path, index=False)
# Use ExcelWriter to save multiple sheets
with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
    feature_codebook.to_excel(writer, sheet_name='Feature Codebook', index=False)
    exon_doc_df.to_excel(writer, sheet_name='Exon Length Documentation', index=False, header=False)

print_emphasized(f"Feature codebook saved to {file_path}")