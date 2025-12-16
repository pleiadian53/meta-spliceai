#!/usr/bin/env python3
"""
Example demonstrating how to use the enhance_splice_sites_with_features utility
to add gene and transcript information to splice site annotations.
Also includes genomic data science analysis for splicing patterns across gene types.
"""

import os
import polars as pl
from pathlib import Path

from meta_spliceai.splice_engine.meta_models.utils import enhance_splice_sites_with_features
from meta_spliceai.splice_engine.meta_models.utils import analyze_splicing_patterns

def main():
    # Get project directory by finding the root of the splice-surveyor project
    script_path = Path(__file__)
    # Start from script location and move up until we find the project root
    # (where we either find a .git folder or the meta_spliceai package)
    current_dir = script_path.parent
    while current_dir.name:
        if (current_dir / '.git').exists() or (current_dir / 'setup.py').exists() or current_dir.name == 'splice-surveyor':
            project_dir = str(current_dir)
            break
        current_dir = current_dir.parent
    else:
        # Fallback if we can't detect automatically
        project_dir = '/home/bchiu/work/splice-surveyor'
    
    print(f"Project directory: {project_dir}")
    
    # Default file paths based on project structure
    splice_sites_path = os.path.join(project_dir, "data", "ensembl", "splice_sites.tsv")
    gene_features_path = os.path.join(project_dir, "data", "ensembl", "spliceai_analysis", "gene_features.tsv")
    transcript_features_path = os.path.join(project_dir, "data", "ensembl", "spliceai_analysis", "transcript_features.tsv")
    
    # Load enhanced splice sites with both gene and transcript features
    print("\n=== Loading Enhanced Splice Sites ===")
    try:
        enhanced_df = enhance_splice_sites_with_features(
            splice_sites_path=splice_sites_path,
            gene_features_path=gene_features_path,
            transcript_features_path=transcript_features_path,
            verbose=1
        )
        
        print(f"\nEnhanced dataset columns: {enhanced_df.columns}")
        print(f"Total rows: {enhanced_df.shape[0]}")
        
        # Save the complete enhanced dataset
        output_path = os.path.join(project_dir, "data", "ensembl", "enhanced_splice_sites.tsv")
        enhanced_df.write_csv(output_path, separator='\t')
        print(f"\nSaved enhanced splice sites to: {output_path}")
        
        # 1. Analysis for protein-coding genes only
        protein_coding_df = analyze_splicing_patterns(
            enhanced_df, 
            gene_types=["protein_coding"],
            title="Protein-Coding Gene Splicing Analysis"
        )
        
        # 2. Analysis including lncRNA genes
        coding_lncrna_df = analyze_splicing_patterns(
            enhanced_df,
            gene_types=["protein_coding", "lncRNA"],
            title="Protein-Coding and lncRNA Gene Splicing Analysis"
        )
        
        # 3. Analysis including snRNA and snoRNA genes
        rna_types_df = analyze_splicing_patterns(
            enhanced_df,
            gene_types=["protein_coding", "lncRNA", "snRNA", "snoRNA"],
            title="RNA Gene Splicing Analysis (Protein-Coding, lncRNA, snRNA, snoRNA)"
        )
        
        # Get and save protein-coding splice sites for SpliceAI analysis
        protein_coding_only = enhanced_df.filter(pl.col('gene_type') == 'protein_coding')
        protein_coding_path = os.path.join(project_dir, "data", "ensembl", "protein_coding_splice_sites.tsv")
        protein_coding_only.write_csv(protein_coding_path, separator='\t')
        print(f"\nSaved protein-coding splice sites to: {protein_coding_path}")
        
    except Exception as e:
        print(f"Error analyzing splice sites: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
