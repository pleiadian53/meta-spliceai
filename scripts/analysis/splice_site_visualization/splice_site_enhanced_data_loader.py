#!/usr/bin/env python3
"""Enhanced data loader for splice site visualization.

This module provides efficient data loading for specific genes without requiring
the full dataset to be loaded into memory. It supports:
1. Gene ID and gene name lookup
2. Cross-referencing with gene/transcript/splice site metadata
3. Memory-efficient loading for visualization purposes
4. Flexible filtering and sampling strategies

Usage:
    from splice_site_enhanced_data_loader import SpliceSiteDataLoader
    
    loader = SpliceSiteDataLoader()
    data = loader.load_genes(['ENSG00000205592', 'MUC19'], dataset_path='train_pc_1000/master')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpliceSiteDataLoader:
    """Enhanced data loader for splice site visualization with gene-specific loading."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gene_features = None
        self.transcript_features = None
        self.splice_sites = None
        self._gene_name_to_id_map = {}
        self._gene_id_to_name_map = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with appropriate level."""
        if self.verbose:
            if level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
    
    def load_metadata(self, gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv",
                     transcript_features_path: str = "data/ensembl/spliceai_analysis/transcript_features.tsv",
                     splice_sites_path: str = "data/ensembl/splice_sites.tsv") -> None:
        """Load gene, transcript, and splice site metadata for cross-referencing."""
        
        # Load gene features
        if Path(gene_features_path).exists():
            self.log(f"Loading gene features from: {gene_features_path}")
            self.gene_features = pd.read_csv(gene_features_path, sep='\t')
            self.log(f"Loaded {len(self.gene_features)} gene features")
            
            # Create gene name/ID mapping
            if 'gene_name' in self.gene_features.columns:
                # Gene name to ID mapping
                name_to_id = self.gene_features.dropna(subset=['gene_name']).set_index('gene_name')['gene_id'].to_dict()
                self._gene_name_to_id_map.update(name_to_id)
                
                # Gene ID to name mapping
                id_to_name = self.gene_features.dropna(subset=['gene_name']).set_index('gene_id')['gene_name'].to_dict()
                self._gene_id_to_name_map.update(id_to_name)
                
                self.log(f"Created mappings for {len(name_to_id)} gene names")
        else:
            self.log(f"Gene features file not found: {gene_features_path}", "WARNING")
            
        # Load transcript features
        if Path(transcript_features_path).exists():
            self.log(f"Loading transcript features from: {transcript_features_path}")
            self.transcript_features = pd.read_csv(transcript_features_path, sep='\t')
            self.log(f"Loaded {len(self.transcript_features)} transcript features")
        else:
            self.log(f"Transcript features file not found: {transcript_features_path}", "WARNING")
            
        # Load splice sites
        if Path(splice_sites_path).exists():
            self.log(f"Loading splice sites from: {splice_sites_path}")
            self.splice_sites = pd.read_csv(splice_sites_path, sep='\t')
            self.log(f"Loaded {len(self.splice_sites)} splice sites")
        else:
            self.log(f"Splice sites file not found: {splice_sites_path}", "WARNING")
    
    def resolve_gene_identifiers(self, gene_list: List[str]) -> List[str]:
        """Resolve mixed gene names/IDs to gene IDs."""
        resolved_ids = []
        
        for gene in gene_list:
            if gene.startswith('ENSG'):
                # Already a gene ID
                resolved_ids.append(gene)
            elif gene in self._gene_name_to_id_map:
                # Gene name that can be mapped to ID
                gene_id = self._gene_name_to_id_map[gene]
                resolved_ids.append(gene_id)
                self.log(f"Resolved gene name '{gene}' to ID '{gene_id}'")
            else:
                # Try to find partial matches or similar names
                possible_matches = [name for name in self._gene_name_to_id_map.keys() 
                                  if gene.upper() in name.upper() or name.upper() in gene.upper()]
                if possible_matches:
                    self.log(f"Gene '{gene}' not found exactly, possible matches: {possible_matches[:5]}", "WARNING")
                else:
                    self.log(f"Gene '{gene}' not found in gene features", "WARNING")
                # Add as-is in case it's a valid ID not in our metadata
                resolved_ids.append(gene)
        
        return resolved_ids
    
    def get_gene_display_info(self, gene_id: str) -> Dict[str, str]:
        """Get display information for a gene."""
        info = {'gene_id': gene_id, 'gene_name': gene_id}  # Default
        
        if gene_id in self._gene_id_to_name_map:
            info['gene_name'] = self._gene_id_to_name_map[gene_id]
        
        # Add additional metadata if available
        if self.gene_features is not None:
            gene_info = self.gene_features[self.gene_features['gene_id'] == gene_id]
            if len(gene_info) > 0:
                info.update({
                    'chromosome': gene_info.get('chromosome', 'unknown').iloc[0] if 'chromosome' in gene_info.columns else 'unknown',
                    'start': gene_info.get('start', 0).iloc[0] if 'start' in gene_info.columns else 0,
                    'end': gene_info.get('end', 0).iloc[0] if 'end' in gene_info.columns else 0,
                    'strand': gene_info.get('strand', '.').iloc[0] if 'strand' in gene_info.columns else '.',
                    'biotype': gene_info.get('biotype', 'unknown').iloc[0] if 'biotype' in gene_info.columns else 'unknown'
                })
        
        return info
    
    def load_genes_targeted(self, gene_ids: List[str], dataset_path: str, 
                           max_genes_to_sample: int = 1000) -> pd.DataFrame:
        """Load data for specific genes using targeted sampling."""
        self.log(f"Loading data for {len(gene_ids)} specific genes from {dataset_path}")
        
        try:
            # Use hierarchical sampling with enough genes to capture our targets
            from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
            
            # Sample enough genes to likely capture our targets, but not too many for memory
            sample_size = max(max_genes_to_sample, len(gene_ids) * 20)
            sample_size = min(sample_size, 2000)  # Cap at reasonable size
            
            self.log(f"Sampling {sample_size} genes to capture target genes")
            df = load_dataset_sample(dataset_path, sample_genes=sample_size, random_seed=42)
            
            # Convert to pandas if needed
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            
            # Filter to target genes
            target_data = df[df['gene_id'].isin(gene_ids)]
            
            # Check which genes were found
            found_genes = set(target_data['gene_id'].unique())
            missing_genes = set(gene_ids) - found_genes
            
            if missing_genes:
                self.log(f"Warning: {len(missing_genes)} target genes not found in sample: {list(missing_genes)[:5]}", "WARNING")
                if len(missing_genes) > len(found_genes):
                    self.log("Many target genes missing - consider increasing sample size or checking gene IDs", "WARNING")
            
            self.log(f"Successfully loaded data for {len(found_genes)} genes with {len(target_data)} positions")
            
            return target_data
            
        except Exception as e:
            self.log(f"Error in targeted gene loading: {e}", "ERROR")
            # Fallback to different approach
            return self._load_genes_fallback(gene_ids, dataset_path)
    
    def _load_genes_fallback(self, gene_ids: List[str], dataset_path: str) -> pd.DataFrame:
        """Fallback method for loading specific genes."""
        self.log("Using fallback loading method...")
        
        try:
            # Try loading with higher row limit
            original_row_cap = os.environ.get('SS_MAX_ROWS', None)
            os.environ['SS_MAX_ROWS'] = '500000'  # Increase row limit temporarily
            
            from meta_spliceai.splice_engine.meta_models.training import datasets
            df = datasets.load_dataset(dataset_path)
            
            # Restore original setting
            if original_row_cap is not None:
                os.environ['SS_MAX_ROWS'] = original_row_cap
            else:
                if 'SS_MAX_ROWS' in os.environ:
                    del os.environ['SS_MAX_ROWS']
            
            # Convert to pandas if needed
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            
            # Filter to target genes
            target_data = df[df['gene_id'].isin(gene_ids)]
            
            self.log(f"Fallback loading successful: {len(target_data)} positions for target genes")
            return target_data
            
        except Exception as e:
            self.log(f"Fallback loading also failed: {e}", "ERROR")
            raise
    
    def load_genes(self, gene_identifiers: List[str], dataset_path: str,
                   filter_kmers: bool = True, max_sample_genes: int = 1000) -> pd.DataFrame:
        """Main method to load data for specified genes.
        
        Args:
            gene_identifiers: List of gene IDs or gene names
            dataset_path: Path to the dataset
            filter_kmers: Whether to remove k-mer features for memory efficiency
            max_sample_genes: Maximum number of genes to sample when searching
            
        Returns:
            DataFrame with data for the specified genes
        """
        # Resolve gene names to IDs
        gene_ids = self.resolve_gene_identifiers(gene_identifiers)
        
        # Load targeted data
        df = self.load_genes_targeted(gene_ids, dataset_path, max_sample_genes)
        
        # Filter k-mer features if requested
        if filter_kmers and len(df) > 0:
            original_cols = len(df.columns)
            kmer_patterns = ['6mer_', 'kmer_', '_mer_', 'mer_']
            kmer_cols = [col for col in df.columns 
                        if any(pattern in col.lower() for pattern in kmer_patterns)]
            
            if kmer_cols:
                self.log(f"Filtering out {len(kmer_cols)} k-mer features for memory efficiency")
                df = df.drop(columns=kmer_cols)
                self.log(f"Reduced from {original_cols} to {len(df.columns)} columns")
        
        # Add gene display information
        if len(df) > 0:
            self._add_gene_display_info(df)
        
        return df
    
    def _add_gene_display_info(self, df: pd.DataFrame) -> None:
        """Add gene display information to the dataframe."""
        if 'gene_name' not in df.columns and self.gene_features is not None:
            # Add gene names
            gene_name_map = self._gene_id_to_name_map
            df['gene_name'] = df['gene_id'].map(gene_name_map).fillna(df['gene_id'])
        
        # Add gene boundaries if available
        if self.gene_features is not None and 'gene_start' not in df.columns:
            gene_boundaries = self.gene_features[['gene_id', 'start', 'end', 'strand']].rename(
                columns={'start': 'gene_start', 'end': 'gene_end'})
            df = df.merge(gene_boundaries, on='gene_id', how='left')
    
    def get_gene_transcript_info(self, gene_id: str) -> Dict[str, any]:
        """Get transcript information for a gene."""
        info = {'gene_id': gene_id, 'transcript_count': 0, 'transcripts': []}
        
        if self.transcript_features is not None:
            gene_transcripts = self.transcript_features[self.transcript_features['gene_id'] == gene_id]
            info['transcript_count'] = len(gene_transcripts)
            
            if len(gene_transcripts) > 0:
                info['transcripts'] = gene_transcripts[['transcript_id', 'transcript_name']].to_dict('records')
        
        return info
    
    def get_gene_splice_sites(self, gene_id: str) -> Dict[str, any]:
        """Get annotated splice sites for a gene."""
        info = {'gene_id': gene_id, 'annotated_donors': [], 'annotated_acceptors': []}
        
        if self.splice_sites is not None:
            gene_splice_sites = self.splice_sites[self.splice_sites['gene_id'] == gene_id]
            
            if len(gene_splice_sites) > 0:
                donors = gene_splice_sites[gene_splice_sites['splice_type'] == 'donor']
                acceptors = gene_splice_sites[gene_splice_sites['splice_type'] == 'acceptor']
                
                info['annotated_donors'] = donors['position'].tolist() if len(donors) > 0 else []
                info['annotated_acceptors'] = acceptors['position'].tolist() if len(acceptors) > 0 else []
        
        return info
    
    def validate_genes_availability(self, gene_identifiers: List[str], dataset_path: str) -> Dict[str, bool]:
        """Check which genes are available in the dataset without loading full data."""
        gene_ids = self.resolve_gene_identifiers(gene_identifiers)
        
        # Quick check using small sample
        try:
            from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
            sample_df = load_dataset_sample(dataset_path, sample_genes=100, random_seed=42)
            
            if hasattr(sample_df, 'to_pandas'):
                sample_df = sample_df.to_pandas()
            
            available_genes = set(sample_df['gene_id'].unique())
            
            return {gene_id: gene_id in available_genes for gene_id in gene_ids}
            
        except Exception as e:
            self.log(f"Error checking gene availability: {e}", "WARNING")
            return {gene_id: True for gene_id in gene_ids}  # Assume available
    
    def suggest_similar_genes(self, gene_query: str, max_suggestions: int = 5) -> List[Dict[str, str]]:
        """Suggest similar gene names for a query."""
        suggestions = []
        
        if self.gene_features is not None and 'gene_name' in self.gene_features.columns:
            gene_names = self.gene_features['gene_name'].dropna().unique()
            
            # Simple similarity: contains query or query contains name
            query_upper = gene_query.upper()
            for name in gene_names:
                name_upper = name.upper()
                if query_upper in name_upper or name_upper in query_upper:
                    gene_id = self._gene_name_to_id_map.get(name, 'unknown')
                    suggestions.append({'gene_name': name, 'gene_id': gene_id})
                    
                    if len(suggestions) >= max_suggestions:
                        break
        
        return suggestions


# Convenience functions for common use cases
def load_genes_for_visualization(gene_identifiers: List[str], dataset_path: str,
                                gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv",
                                verbose: bool = True) -> pd.DataFrame:
    """Convenience function to load genes for visualization."""
    loader = SpliceSiteDataLoader(verbose=verbose)
    loader.load_metadata(gene_features_path=gene_features_path)
    return loader.load_genes(gene_identifiers, dataset_path)


def discover_genes_by_name(name_patterns: List[str], 
                          gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv") -> List[Dict[str, str]]:
    """Discover genes by name patterns."""
    loader = SpliceSiteDataLoader()
    loader.load_metadata(gene_features_path=gene_features_path)
    
    all_suggestions = []
    for pattern in name_patterns:
        suggestions = loader.suggest_similar_genes(pattern, max_suggestions=10)
        all_suggestions.extend(suggestions)
    
    return all_suggestions 