import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
import os
import sys

# Add the project root to the path to import the SpliceAnalyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meta_spliceai.splice_engine.meta_models.core.analyzers import SpliceAnalyzer

def analyze_overlapping_genes(file_path):
    """
    Comprehensive analysis of gene overlap patterns from TSV file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the overlapping gene counts TSV file
        
    Returns
    -------
    dict
        Dictionary containing summary statistics and analysis results
    """
    # Step 1: Load and validate data
    print("Step 1: Loading and validating data...")
    df = pd.read_csv(file_path, sep='\t')
    print(f"Total rows in dataset: {len(df)}")
    
    # Check if gene_name columns exist, create them from gene_id if not
    if 'gene_name_1' not in df.columns:
        df['gene_name_1'] = df['gene_id_1'].str.split('.').str[0]
    if 'gene_name_2' not in df.columns:
        df['gene_name_2'] = df['gene_id_2'].str.split('.').str[0]

    # Step 2: Analyze overlap distribution
    print("\nStep 2: Analyzing overlap distribution...")
    overlap_counts = df['num_overlaps'].value_counts().sort_index()
    print("\nDistribution of overlaps:")
    print(overlap_counts)

    # Step 3: Check pair reciprocity
    print("\nStep 3: Analyzing pair reciprocity...")
    df['canonical_pair'] = df.apply(lambda x: '-'.join(sorted([x['gene_id_1'], x['gene_id_2']])), axis=1)
    unique_pairs = df.groupby('canonical_pair').agg({
        'num_overlaps': ['count', 'nunique', 'first'],
        'gene_name_1': 'first',
        'gene_name_2': 'first'
    })
    print(f"\nTotal unique gene pairs: {len(unique_pairs)}")
    print(f"Pairs with non-matching overlap counts: {len(unique_pairs[unique_pairs['num_overlaps']['nunique'] > 1])}")

    # Step 4: Analyze pair orientations
    pairs_with_both = unique_pairs[unique_pairs['num_overlaps']['count'] == 2]
    pairs_with_one = unique_pairs[unique_pairs['num_overlaps']['count'] == 1]
    print(f"\nPairs with both orientations: {len(pairs_with_both)}")
    print(f"Pairs with only one orientation: {len(pairs_with_one)}")

    # Step 5: Analyze high-overlap pairs
    print("\nStep 5: Analyzing high-overlap pairs...")
    max_overlaps = df['num_overlaps'].max()
    high_overlap_df = df[df['num_overlaps'] == max_overlaps]
    print(f"\nHighest number of overlaps: {max_overlaps}")
    print(f"Number of pairs with {max_overlaps} overlaps: {len(high_overlap_df)}")
    print("\nExample high-overlap pairs:")
    print(high_overlap_df[['gene_id_1', 'gene_id_2', 'gene_name_1', 'gene_name_2', 'num_overlaps']].head())

    # Step 6: Analyze gene families
    print("\nStep 6: Analyzing gene families in high-overlap pairs...")
    gene_families = pd.DataFrame({
        'Family': high_overlap_df['gene_name_1'].str.extract(r'([A-Za-z]+)')[0].value_counts()
    })
    print("\nGene families in high-overlap pairs:")
    print(gene_families)

    # NEW Step 7: Analyze chromosome distribution
    print("\nStep 7: Analyzing chromosomal distribution of overlapping genes...")
    chrom_distribution = df['chrom'].value_counts().sort_index()
    print("\nChromosome distribution:")
    print(chrom_distribution)
    
    # NEW Step 8: Analyze strand orientation patterns
    print("\nStep 8: Analyzing strand orientation patterns...")
    df['strand_pattern'] = df['strand_1'] + df['strand_2']
    strand_patterns = df['strand_pattern'].value_counts()
    print("\nStrand orientation patterns:")
    print(strand_patterns)
    
    # NEW Step 9: Analyze overlap extent
    print("\nStep 9: Analyzing extent of overlaps...")
    df['overlap_length'] = np.minimum(df['end_1'], df['end_2']) - np.maximum(df['start_1'], df['start_2'])
    df['overlap_pct_1'] = (df['overlap_length'] / (df['end_1'] - df['start_1'])) * 100
    df['overlap_pct_2'] = (df['overlap_length'] / (df['end_2'] - df['start_2'])) * 100
    
    print("\nOverlap statistics:")
    print(f"Average overlap length: {df['overlap_length'].mean():.1f} bp")
    print(f"Median overlap length: {df['overlap_length'].median():.1f} bp")
    print(f"Average overlap percentage (gene 1): {df['overlap_pct_1'].mean():.1f}%")
    print(f"Average overlap percentage (gene 2): {df['overlap_pct_2'].mean():.1f}%")

    # Generate visualizations
    output_dir = os.path.dirname(file_path)
    visualization_path = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualization_path, exist_ok=True)
    
    # Visualization 1: Overlap distribution
    plt.figure(figsize=(12, 6))
    overlap_counts.plot(kind='bar')
    plt.title('Distribution of Gene Overlap Counts')
    plt.xlabel('Number of Overlaps')
    plt.ylabel('Number of Gene Pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_path, 'overlap_distribution.png'))
    plt.close()
    
    # Visualization 2: Chromosome distribution
    plt.figure(figsize=(14, 7))
    chrom_distribution.plot(kind='bar', color='skyblue')
    plt.title('Chromosomal Distribution of Overlapping Genes')
    plt.xlabel('Chromosome')
    plt.ylabel('Number of Overlapping Gene Pairs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_path, 'chrom_distribution.png'))
    plt.close()
    
    # Visualization 3: Strand patterns
    plt.figure(figsize=(10, 6))
    strand_patterns.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("Set3"))
    plt.title('Strand Orientation Patterns of Overlapping Genes')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_path, 'strand_patterns.png'))
    plt.close()
    
    # Visualization 4: Overlap length vs. number of overlaps
    plt.figure(figsize=(12, 6))
    plt.scatter(df['overlap_length'], df['num_overlaps'], alpha=0.5)
    plt.title('Relationship Between Overlap Length and Number of Overlaps')
    plt.xlabel('Overlap Length (bp)')
    plt.ylabel('Number of Overlaps')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_path, 'overlap_length_vs_count.png'))
    plt.close()

    # Return summary statistics
    return {
        'total_pairs': len(df),
        'unique_pairs': len(unique_pairs),
        'max_overlaps': max_overlaps,
        'high_overlap_pairs': len(high_overlap_df),
        'overlap_distribution': overlap_counts.to_dict(),
        'chrom_distribution': chrom_distribution.to_dict(),
        'strand_patterns': strand_patterns.to_dict(),
        'avg_overlap_length': df['overlap_length'].mean(),
        'median_overlap_length': df['overlap_length'].median()
    }

if __name__ == "__main__":
    # Use SpliceAnalyzer to get the path to the overlapping gene metadata file
    analyzer = SpliceAnalyzer()
    # NOTE: Example path
    # data_dir = "/path/to/meta-spliceai/data/ensembl"
    # file_path = Path(f"{data_dir}/test_overlapping_genes/overlapping_gene_counts.tsv")
    
    # Example path: "/path/to/meta-spliceai/data/ensembl/overlapping_gene_counts.tsv"
    file_path = analyzer.path_to_overlapping_gene_metadata
    print(f"[info] Path to overlapping gene metadata: {file_path}")
    
    # Check if custom path is provided as an argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find file at {file_path}")
        exit(1)

    try:
        results = analyze_overlapping_genes(file_path)
        print("\nAnalysis Summary:")
        print(f"Total pairs analyzed: {results['total_pairs']}")
        print(f"Unique pairs: {results['unique_pairs']}")
        print(f"Maximum overlaps found: {results['max_overlaps']}")
        print(f"Pairs with maximum overlaps: {results['high_overlap_pairs']}")
        print(f"Average overlap length: {results['avg_overlap_length']:.1f} bp")
        
        print(f"\nVisualizations saved to: {os.path.dirname(file_path)}/visualizations/")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
