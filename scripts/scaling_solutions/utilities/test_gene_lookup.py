#!/usr/bin/env python3
"""
Test Gene Lookup Functionality
==============================
Demonstrates how to use the gene-position index for efficient gene-to-feature-vector lookups.
"""

import sys
from pathlib import Path
import polars as pl
import json

# Add the utilities directory to the path
sys.path.append(str(Path(__file__).parent))
from create_gene_index import load_gene_positions


def test_gene_lookups():
    """Test gene lookup functionality with example genes."""
    
    # Paths
    index_path = "train_pc_1000/master/gene_position_index.csv"
    dataset_dir = "train_pc_1000/master/"
    
    # Example genes from the manifest
    test_genes = [
        "ENSG00000137710",  # RDX
        "ENSG00000137513",  # NARS2  
        "ENSG00000076554",  # TPD52
    ]
    
    print("=== Gene Lookup Test ===\n")
    
    # Load the index to see statistics
    index_df = pl.read_csv(index_path)
    print(f"Index contains {index_df.height:,} genes")
    print(f"Total positions across all genes: {index_df.select('position_count').sum().item():,}")
    
    # Show some statistics
    avg_positions = index_df.select(pl.col("position_count").mean()).item()
    max_positions = index_df.select(pl.col("position_count").max()).item()
    min_positions = index_df.select(pl.col("position_count").min()).item()
    
    print(f"Average positions per gene: {avg_positions:.1f}")
    print(f"Min positions per gene: {min_positions}")
    print(f"Max positions per gene: {max_positions}")
    
    # Test lookups for each gene
    for i, gene_id in enumerate(test_genes, 1):
        print(f"\n--- Gene {i}: {gene_id} ---")
        
        try:
            # Load gene data
            gene_data = load_gene_positions(index_path, gene_id, dataset_dir)
            
            print(f"✅ Successfully loaded {len(gene_data):,} positions")
            print(f"Available columns: {gene_data.columns[:10]}...")  # Show first 10 columns
            
            # Show basic statistics
            chrom_val = gene_data['chrom'][0] if 'chrom' in gene_data.columns else 'N/A'
            print(f"Chromosome: {chrom_val}")
            
            # Analyze splice sites
            splice_sites = gene_data.filter(pl.col("splice_type") != "neither")
            print(f"Splice sites: {len(splice_sites):,} (donor + acceptor)")
            
            if len(splice_sites) > 0:
                donor_sites = splice_sites.filter(pl.col("splice_type") == "donor")
                acceptor_sites = splice_sites.filter(pl.col("splice_type") == "acceptor")
                print(f"  - Donor sites: {len(donor_sites):,}")
                print(f"  - Acceptor sites: {len(acceptor_sites):,}")
            
            # Show feature statistics (if available)
            if "donor_score" in gene_data.columns:
                donor_stats = gene_data.select([
                    pl.col("donor_score").mean().alias("avg_donor_score"),
                    pl.col("donor_score").max().alias("max_donor_score"),
                    pl.col("acceptor_score").mean().alias("avg_acceptor_score"),
                    pl.col("acceptor_score").max().alias("max_acceptor_score")
                ])
                print(f"Score statistics:")
                print(f"  - Avg donor score: {donor_stats['avg_donor_score'][0]:.3f}")
                print(f"  - Max donor score: {donor_stats['max_donor_score'][0]:.3f}")
                print(f"  - Avg acceptor score: {donor_stats['avg_acceptor_score'][0]:.3f}")
                print(f"  - Max acceptor score: {donor_stats['max_acceptor_score'][0]:.3f}")
            
            # Show position range
            positions = gene_data.select("position").to_series().to_list()
            print(f"Position range: {min(positions):,} to {max(positions):,}")
            
        except Exception as e:
            print(f"❌ Error loading gene {gene_id}: {e}")
    
    print("\n=== Use Cases Demonstrated ===")
    print("1. ✅ Gene-specific feature vector lookup")
    print("2. ✅ Splice site analysis per gene")
    print("3. ✅ Score statistics for meta-model training")
    print("4. ✅ Position range analysis")
    print("5. ✅ Efficient loading (only loads specific file)")


def demonstrate_use_cases():
    """Demonstrate additional use cases."""
    
    print("\n=== Additional Use Cases ===")
    
    use_cases = [
        "1. **Meta-model Training**: Load gene-specific data for training meta-models",
        "2. **Splice Site Prediction**: Get all positions for a gene to predict donor/acceptor scores",
        "3. **Unseen Position Analysis**: Identify positions not in training data for validation",
        "4. **Gene Comparison**: Compare splice patterns between different genes",
        "5. **Quality Control**: Verify gene coverage and data completeness",
        "6. **Feature Engineering**: Extract gene-level features from position-level data",
        "7. **Model Interpretation**: Analyze which positions contribute most to predictions",
        "8. **Data Validation**: Check for missing or inconsistent data per gene",
        "9. **Performance Analysis**: Measure prediction accuracy gene-by-gene",
        "10. **Research Applications**: Study specific genes of interest"
    ]
    
    for use_case in use_cases:
        print(use_case)


if __name__ == "__main__":
    test_gene_lookups()
    demonstrate_use_cases() 