#!/usr/bin/env python
"""
Test script to validate the gtf_utils module, particularly the get_gene_chromosomes function.
This script tests with ALS-related genes (STMN2 and UNC13A) to verify chromosome mapping.
"""

import os
import sys
import time
import polars as pl

# Add project root to path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the utility function
from meta_spliceai.utils.gtf_utils import get_gene_chromosomes, get_chromosome_genes
from meta_spliceai.splice_engine.analyzer import Analyzer


def test_get_gene_chromosomes():
    """Test get_gene_chromosomes function with ALS-related genes."""
    # Get GTF file path from Analyzer class
    gtf_file = Analyzer.gtf_file
    
    # Define ALS-related genes
    als_genes = [
        'STMN2',     # Stathmin-2 - affected by TDP-43 loss in ALS
        'UNC13A',    # UNC13A - contains ALS-risk SNPs that affect splicing
        'ENSG00000104435'  # Ensembl ID for STMN2
    ]
    
    # Print test information
    print(f"Testing get_gene_chromosomes with ALS-related genes")
    print(f"GTF file: {gtf_file}")
    print(f"Target genes: {', '.join(als_genes)}")
    
    # Measure execution time
    start_time = time.time()
    
    # Get chromosomes for ALS genes
    gene_to_chrom = get_gene_chromosomes(gtf_file, als_genes)
    
    # Print results
    execution_time = time.time() - start_time
    print(f"\nResults (execution time: {execution_time:.2f} seconds):")
    
    if gene_to_chrom:
        print("\nGene to Chromosome mapping:")
        for gene, chrom in gene_to_chrom.items():
            print(f"  {gene}: chromosome {chrom}")
    else:
        print("No matching genes found in the GTF file.")
    
    # Verify the expected chromosomes
    expected = {
        'STMN2': '8',          # STMN2 should be on chromosome 8
        'UNC13A': '19',        # UNC13A should be on chromosome 19
        'ENSG00000104435': '8' # Ensembl ID for STMN2 should also map to chromosome 8
    }
    
    print("\nValidation:")
    for gene, expected_chrom in expected.items():
        if gene in gene_to_chrom:
            actual_chrom = gene_to_chrom[gene]
            if actual_chrom == expected_chrom:
                print(f"✓ {gene}: Found on expected chromosome {expected_chrom}")
            else:
                print(f"✗ {gene}: Found on chromosome {actual_chrom}, but expected {expected_chrom}")
        else:
            print(f"✗ {gene}: Not found in GTF file")


def test_get_chromosome_genes():
    """Test get_chromosome_genes function to verify the reverse mapping."""
    # Get GTF file path from Analyzer class
    gtf_file = Analyzer.gtf_file
    
    # Define chromosomes of interest
    test_chromosomes = ["8", "19"]  # Should contain STMN2 and UNC13A
    
    print(f"\nTesting get_chromosome_genes with chromosomes: {', '.join(test_chromosomes)}")
    
    # Measure execution time
    start_time = time.time()
    
    # Get genes on specified chromosomes
    chrom_to_genes = get_chromosome_genes(gtf_file, test_chromosomes)
    
    # Print results
    execution_time = time.time() - start_time
    print(f"Results (execution time: {execution_time:.2f} seconds):")
    
    if chrom_to_genes:
        for chrom, genes in chrom_to_genes.items():
            gene_count = len(genes)
            print(f"  Chromosome {chrom}: {gene_count} genes")
            
            # Look for our ALS genes in the results
            if chrom == "8":
                stmn2_ids = [g for g in genes if "STMN2" in g or g == "ENSG00000104435"]
                if stmn2_ids:
                    print(f"    ✓ STMN2 found: {', '.join(stmn2_ids)}")
                else:
                    print(f"    ✗ STMN2 not found")
            
            if chrom == "19":
                unc13a_ids = [g for g in genes if "UNC13A" in g]
                if unc13a_ids:
                    print(f"    ✓ UNC13A found: {', '.join(unc13a_ids)}")
                else:
                    print(f"    ✗ UNC13A not found")
    else:
        print("No chromosomes or genes found in the GTF file.")


if __name__ == "__main__":
    # Run tests
    test_get_gene_chromosomes()
    test_get_chromosome_genes()
    
    print("\nTest completed!")
