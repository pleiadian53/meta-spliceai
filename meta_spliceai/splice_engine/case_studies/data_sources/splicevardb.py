"""
SpliceVarDB ingester.

Ingests splice variant data from SpliceVarDB (Sullivan et al. 2024),
a comprehensive database of experimentally validated splice variants.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import gzip
from urllib.parse import urljoin

from .base import BaseIngester, SpliceMutation, IngestionResult, SpliceEventType, ClinicalSignificance
from ..formats.hgvs_parser import HGVSParser


class SpliceVarDBIngester(BaseIngester):
    """Ingester for SpliceVarDB splice variant database."""
    
    def __init__(self, output_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize SpliceVarDB ingester.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save processed data
        cache_dir : Path, optional
            Directory for caching downloaded data
        """
        super().__init__(output_dir, cache_dir)
        
        # SpliceVarDB endpoints (adjust based on actual API)
        self.base_url = "https://splicevardb.org/"  # Placeholder URL
        self.api_endpoints = {
            "variants": "api/variants",
            "genes": "api/genes", 
            "evidence": "api/evidence"
        }
        
        # Alternative: Direct file download URLs if API not available
        self.download_urls = {
            "all_variants": "downloads/splicevardb_all_variants.tsv.gz",
            "validated_variants": "downloads/splicevardb_validated.tsv.gz",
            "pathogenic_variants": "downloads/splicevardb_pathogenic.tsv.gz"
        }
        
        self.hgvs_parser = HGVSParser()
    
    def download_data(self, force_refresh: bool = False, variant_set: str = "validated") -> Path:
        """
        Download SpliceVarDB data.
        
        Parameters
        ----------
        force_refresh : bool
            Whether to redownload even if cached
        variant_set : str
            Which variant set to download ("all", "validated", "pathogenic")
            
        Returns
        -------
        Path
            Path to downloaded data file
        """
        # Determine download target
        if variant_set not in self.download_urls:
            raise ValueError(f"Unknown variant set: {variant_set}. Choose from {list(self.download_urls.keys())}")
        
        cache_file = self.cache_dir / f"splicevardb_{variant_set}.tsv.gz"
        
        # Check if cached file exists and is recent
        if cache_file.exists() and not force_refresh:
            print(f"Using cached SpliceVarDB data: {cache_file}")
            return cache_file
        
        # Download the data
        download_url = urljoin(self.base_url, self.download_urls[variant_set])
        print(f"Downloading SpliceVarDB {variant_set} variants from {download_url}")
        
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded SpliceVarDB data to {cache_file}")
            return cache_file
            
        except requests.RequestException as e:
            # Fallback: try to load demo/test data
            print(f"Download failed: {e}")
            print("Creating demo SpliceVarDB data for development...")
            return self._create_demo_data()
    
    def _create_demo_data(self) -> Path:
        """Create demo data for development when real data is unavailable."""
        demo_file = self.cache_dir / "splicevardb_demo.tsv"
        
        # Create sample data based on common SpliceVarDB format
        demo_data = [
            {
                "variant_id": "SPLVAR_000001",
                "gene_symbol": "CFTR", 
                "gene_id": "ENSG00000001626",
                "transcript_id": "ENST00000003084",
                "hgvs_c": "c.3718-2477C>T",
                "hgvs_g": "g.117199644C>T",
                "chromosome": "7",
                "position": 117199644,
                "ref_allele": "C",
                "alt_allele": "T",
                "splice_event": "cryptic_exon_activation",
                "clinical_significance": "pathogenic",
                "disease": "cystic_fibrosis",
                "experimental_validation": "RNA-seq",
                "validation_pmid": "12345678",
                "splice_site_type": "cryptic_donor",
                "distance_to_exon": 2477
            },
            {
                "variant_id": "SPLVAR_000002", 
                "gene_symbol": "MET",
                "gene_id": "ENSG00000105976",
                "transcript_id": "ENST00000397752",
                "hgvs_c": "c.3028+1G>A", 
                "hgvs_g": "g.116412043G>A",
                "chromosome": "7",
                "position": 116412043,
                "ref_allele": "G",
                "alt_allele": "A",
                "splice_event": "exon_skipping",
                "clinical_significance": "pathogenic",
                "disease": "lung_cancer",
                "experimental_validation": "RT-PCR",
                "validation_pmid": "23456789",
                "splice_site_type": "canonical_donor",
                "distance_to_exon": 1
            },
            {
                "variant_id": "SPLVAR_000003",
                "gene_symbol": "BRCA1",
                "gene_id": "ENSG00000012048", 
                "transcript_id": "ENST00000357654",
                "hgvs_c": "c.4357+1G>A",
                "hgvs_g": "g.41244000G>A",
                "chromosome": "17",
                "position": 41244000,
                "ref_allele": "G", 
                "alt_allele": "A",
                "splice_event": "intron_retention",
                "clinical_significance": "pathogenic",
                "disease": "breast_cancer",
                "experimental_validation": "minigene_assay",
                "validation_pmid": "34567890",
                "splice_site_type": "canonical_donor",
                "distance_to_exon": 1
            }
        ]
        
        df = pd.DataFrame(demo_data)
        df.to_csv(demo_file, sep='\t', index=False)
        print(f"Created demo SpliceVarDB data: {demo_file}")
        return demo_file
    
    def parse_raw_data(self, data_path: Path) -> List[SpliceMutation]:
        """
        Parse SpliceVarDB TSV data to SpliceMutation objects.
        
        Parameters
        ---------- 
        data_path : Path
            Path to SpliceVarDB data file
            
        Returns
        -------
        List[SpliceMutation]
            List of parsed splice mutations
        """
        mutations = []
        
        # Load data
        if data_path.suffix == '.gz':
            df = pd.read_csv(data_path, sep='\t', compression='gzip')
        else:
            df = pd.read_csv(data_path, sep='\t')
        
        print(f"Parsing {len(df)} SpliceVarDB records...")
        
        for idx, row in df.iterrows():
            try:
                # Parse HGVS notation if available
                hgvs_c = row.get('hgvs_c', '')
                hgvs_parsed = None
                if hgvs_c:
                    hgvs_parsed = self.hgvs_parser.parse(hgvs_c)
                
                # Map splice event types
                splice_event_str = row.get('splice_event', '').lower()
                splice_event_map = {
                    'cryptic_exon_activation': SpliceEventType.PSEUDOEXON_ACTIVATION,
                    'cryptic_donor': SpliceEventType.CRYPTIC_DONOR,
                    'cryptic_acceptor': SpliceEventType.CRYPTIC_ACCEPTOR,
                    'exon_skipping': SpliceEventType.EXON_SKIPPING,
                    'intron_retention': SpliceEventType.INTRON_RETENTION,
                    'partial_exon_deletion': SpliceEventType.PARTIAL_EXON_DELETION,
                    'canonical_site_loss': SpliceEventType.CANONICAL_SITE_LOSS
                }
                splice_event_type = splice_event_map.get(splice_event_str, SpliceEventType.CRYPTIC_DONOR)
                
                # Map clinical significance
                clin_sig_str = row.get('clinical_significance', '').lower()
                clin_sig_map = {
                    'pathogenic': ClinicalSignificance.PATHOGENIC,
                    'likely_pathogenic': ClinicalSignificance.LIKELY_PATHOGENIC,
                    'benign': ClinicalSignificance.BENIGN,
                    'likely_benign': ClinicalSignificance.LIKELY_BENIGN,
                    'uncertain': ClinicalSignificance.UNCERTAIN
                }
                clinical_significance = clin_sig_map.get(clin_sig_str, ClinicalSignificance.UNCERTAIN)
                
                # Determine splice site position
                splice_site_position = None
                if hgvs_parsed and hgvs_parsed.is_valid:
                    if hgvs_parsed.intronic_offset is not None:
                        # Calculate splice site position from coding position + offset
                        splice_site_position = hgvs_parsed.start_position + hgvs_parsed.intronic_offset
                    else:
                        splice_site_position = hgvs_parsed.start_position
                
                # Create mutation object
                mutation = SpliceMutation(
                    chrom=str(row.get('chromosome', '')),
                    position=int(row.get('position', 0)),
                    ref_allele=str(row.get('ref_allele', '')),
                    alt_allele=str(row.get('alt_allele', '')),
                    gene_id=str(row.get('gene_id', '')),
                    gene_symbol=str(row.get('gene_symbol', '')),
                    transcript_id=str(row.get('transcript_id', '')),
                    splice_event_type=splice_event_type,
                    affected_site_type=str(row.get('splice_site_type', '')),
                    splice_site_position=splice_site_position,
                    clinical_significance=clinical_significance,
                    disease_context=str(row.get('disease', '')),
                    experimentally_validated=bool(row.get('experimental_validation', False)),
                    validation_method=str(row.get('experimental_validation', '')),
                    source_database="SpliceVarDB",
                    source_id=str(row.get('variant_id', f"SPLVAR_{idx}")),
                    hgvs_notation=hgvs_c,
                    metadata={
                        'validation_pmid': str(row.get('validation_pmid', '')),
                        'distance_to_exon': row.get('distance_to_exon', None),
                        'hgvs_g': str(row.get('hgvs_g', '')),
                        'original_row_index': idx
                    }
                )
                
                mutations.append(mutation)
                
            except Exception as e:
                self.processing_stats["parsing_errors"] += 1
                print(f"Error parsing row {idx}: {e}")
                continue
        
        print(f"Successfully parsed {len(mutations)} SpliceVarDB mutations")
        return mutations
    
    def validate_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """
        Validate and filter SpliceVarDB mutations.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            Raw parsed mutations
            
        Returns
        -------
        List[SpliceMutation]
            Validated and filtered mutations
        """
        validated = []
        
        for mutation in mutations:
            # Basic validation checks
            validation_errors = []
            
            # Check required fields
            if not mutation.chrom:
                validation_errors.append("Missing chromosome")
            if not mutation.position or mutation.position <= 0:
                validation_errors.append("Invalid position")
            if not mutation.gene_symbol:
                validation_errors.append("Missing gene symbol")
            if not mutation.ref_allele or not mutation.alt_allele:
                validation_errors.append("Missing alleles")
            
            # Check chromosome format
            if mutation.chrom and not (mutation.chrom.isdigit() or mutation.chrom in ['X', 'Y', 'MT']):
                if mutation.chrom.startswith('chr'):
                    mutation.chrom = mutation.chrom[3:]  # Remove chr prefix
                else:
                    validation_errors.append(f"Invalid chromosome format: {mutation.chrom}")
            
            # Filter by experimental validation if desired
            if not mutation.experimentally_validated:
                # Still include but mark as lower confidence
                if mutation.metadata is None:
                    mutation.metadata = {}
                mutation.metadata['confidence'] = 'low'
            
            # Only include if basic validation passes
            if not validation_errors:
                validated.append(mutation)
            else:
                self.processing_stats["validation_errors"] += 1
                print(f"Validation failed for {mutation.source_id}: {'; '.join(validation_errors)}")
        
        print(f"Validated {len(validated)} out of {len(mutations)} SpliceVarDB mutations")
        return validated
    
    def get_disease_specific_mutations(self, mutations: List[SpliceMutation], 
                                     diseases: List[str]) -> List[SpliceMutation]:
        """
        Filter mutations for specific diseases.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            All mutations
        diseases : List[str]
            Disease names to filter for
            
        Returns
        -------
        List[SpliceMutation]
            Disease-specific mutations
        """
        disease_lower = [d.lower() for d in diseases]
        filtered = []
        
        for mutation in mutations:
            if mutation.disease_context:
                if any(disease in mutation.disease_context.lower() for disease in disease_lower):
                    filtered.append(mutation)
        
        return filtered
    
    def get_validated_mutations_only(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get only experimentally validated mutations."""
        return [m for m in mutations if m.experimentally_validated]
    
    def get_pathogenic_mutations_only(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get only pathogenic or likely pathogenic mutations."""
        pathogenic_categories = {ClinicalSignificance.PATHOGENIC, ClinicalSignificance.LIKELY_PATHOGENIC}
        return [m for m in mutations if m.clinical_significance in pathogenic_categories] 