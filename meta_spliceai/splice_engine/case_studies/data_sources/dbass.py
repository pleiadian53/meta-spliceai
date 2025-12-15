"""
DBASS ingester.

Ingests cryptic splice site data from DBASS5 and DBASS3 databases,
which catalog cryptic splice-site activations induced by mutations.
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


class DBASSIngester(BaseIngester):
    """Ingester for DBASS cryptic splice site database."""
    
    def __init__(self, output_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize DBASS ingester.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save processed data
        cache_dir : Path, optional
            Directory for caching downloaded data
        """
        super().__init__(output_dir, cache_dir)
        
        # DBASS endpoints
        self.base_urls = {
            "dbass5": "http://www.dbass.org.uk/",
            "dbass3": "http://www.dbass.org.uk/"
        }
        
        # Download URLs for data files
        self.download_urls = {
            "dbass5_data": "downloads/DBASS5_data.tsv",
            "dbass3_data": "downloads/DBASS3_data.tsv",
            "combined_data": "downloads/DBASS_combined.tsv"
        }
        
        self.hgvs_parser = HGVSParser()
    
    def download_data(self, force_refresh: bool = False, dataset: str = "combined_data") -> Path:
        """
        Download DBASS data.
        
        Parameters
        ----------
        force_refresh : bool
            Whether to redownload even if cached
        dataset : str
            Which dataset to download ("dbass5_data", "dbass3_data", "combined_data")
            
        Returns
        -------
        Path
            Path to downloaded data file
        """
        if dataset not in self.download_urls:
            raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(self.download_urls.keys())}")
        
        cache_file = self.cache_dir / f"dbass_{dataset}.tsv"
        
        # Check if cached file exists
        if cache_file.exists() and not force_refresh:
            print(f"Using cached DBASS data: {cache_file}")
            return cache_file
        
        # For DBASS, the real data might require manual download or API access
        # For now, create demo data based on typical DBASS format
        print(f"Creating demo DBASS {dataset} data for development...")
        return self._create_demo_data(dataset)
    
    def _create_demo_data(self, dataset: str) -> Path:
        """Create demo DBASS data for development."""
        demo_file = self.cache_dir / f"dbass_{dataset}_demo.tsv"
        
        # Create sample data based on DBASS format
        demo_data = [
            {
                "entry_id": "DBASS5_001",
                "gene_name": "CFTR",
                "gene_symbol": "CFTR",
                "chromosome": "7",
                "genomic_position": 117199644,
                "mutation": "c.3718-2477C>T",
                "mutation_type": "deep_intronic",
                "ref_allele": "C",
                "alt_allele": "T",
                "cryptic_site_type": "donor",
                "cryptic_site_position": 117197167,  # Position of activated cryptic site
                "cryptic_site_sequence": "GTAAGT",
                "strength_score": 8.2,
                "wild_type_score": 2.1,
                "mutant_score": 8.2,
                "fold_change": 3.9,
                "disease": "Cystic Fibrosis",
                "phenotype": "severe",
                "pmid": "12345678",
                "validation_method": "RT-PCR",
                "patient_origin": "European",
                "transcript_id": "NM_000492.3",
                "exon_affected": "cryptic_exon",
                "splice_effect": "pseudoexon_inclusion"
            },
            {
                "entry_id": "DBASS3_002",
                "gene_name": "BRCA1",
                "gene_symbol": "BRCA1", 
                "chromosome": "17",
                "genomic_position": 41244435,
                "mutation": "c.4358-43A>G",
                "mutation_type": "intronic",
                "ref_allele": "A",
                "alt_allele": "G",
                "cryptic_site_type": "acceptor",
                "cryptic_site_position": 41244392,
                "cryptic_site_sequence": "TTTAG",
                "strength_score": 7.8,
                "wild_type_score": 1.5,
                "mutant_score": 7.8,
                "fold_change": 5.2,
                "disease": "Breast Cancer",
                "phenotype": "familial",
                "pmid": "23456789",
                "validation_method": "minigene_assay",
                "patient_origin": "Ashkenazi",
                "transcript_id": "NM_007294.3",
                "exon_affected": "11",
                "splice_effect": "cryptic_acceptor_use"
            },
            {
                "entry_id": "DBASS5_003",
                "gene_name": "DMD",
                "gene_symbol": "DMD",
                "chromosome": "X",
                "genomic_position": 32346896,
                "mutation": "c.6614+861C>T",
                "mutation_type": "deep_intronic",
                "ref_allele": "C",
                "alt_allele": "T",
                "cryptic_site_type": "donor",
                "cryptic_site_position": 32347757,
                "cryptic_site_sequence": "GTGAGT",
                "strength_score": 9.1,
                "wild_type_score": 2.8,
                "mutant_score": 9.1,
                "fold_change": 3.3,
                "disease": "Duchenne Muscular Dystrophy",
                "phenotype": "severe",
                "pmid": "34567890",
                "validation_method": "RNA_seq",
                "patient_origin": "Mixed",
                "transcript_id": "NM_004006.2",
                "exon_affected": "45",
                "splice_effect": "pseudoexon_activation"
            },
            {
                "entry_id": "DBASS3_004",
                "gene_name": "NF1",
                "gene_symbol": "NF1",
                "chromosome": "17",
                "genomic_position": 29421945,
                "mutation": "c.1466-89T>C",
                "mutation_type": "intronic",
                "ref_allele": "T",
                "alt_allele": "C",
                "cryptic_site_type": "acceptor",
                "cryptic_site_position": 29421856,
                "cryptic_site_sequence": "TTCAG",
                "strength_score": 6.9,
                "wild_type_score": 1.2,
                "mutant_score": 6.9,
                "fold_change": 5.8,
                "disease": "Neurofibromatosis Type 1",
                "phenotype": "typical",
                "pmid": "45678901",
                "validation_method": "hybrid_minigene",
                "patient_origin": "Caucasian",
                "transcript_id": "NM_000267.3",
                "exon_affected": "10b",
                "splice_effect": "cryptic_exon_insertion"
            }
        ]
        
        df = pd.DataFrame(demo_data)
        df.to_csv(demo_file, sep='\t', index=False)
        print(f"Created demo DBASS data: {demo_file}")
        return demo_file
    
    def parse_raw_data(self, data_path: Path) -> List[SpliceMutation]:
        """
        Parse DBASS TSV data to SpliceMutation objects.
        
        Parameters
        ----------
        data_path : Path
            Path to DBASS data file
            
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
        
        print(f"Parsing {len(df)} DBASS records...")
        
        for idx, row in df.iterrows():
            try:
                # Parse mutation HGVS notation if available
                hgvs_notation = row.get('mutation', '')
                hgvs_parsed = None
                if hgvs_notation:
                    hgvs_parsed = self.hgvs_parser.parse(hgvs_notation)
                
                # Map splice effects to event types
                splice_effect_str = row.get('splice_effect', '').lower()
                splice_effect_map = {
                    'pseudoexon_inclusion': SpliceEventType.PSEUDOEXON_ACTIVATION,
                    'pseudoexon_activation': SpliceEventType.PSEUDOEXON_ACTIVATION,
                    'cryptic_donor_use': SpliceEventType.CRYPTIC_DONOR,
                    'cryptic_acceptor_use': SpliceEventType.CRYPTIC_ACCEPTOR,
                    'cryptic_exon_insertion': SpliceEventType.PSEUDOEXON_ACTIVATION,
                    'partial_intron_retention': SpliceEventType.INTRON_RETENTION
                }
                splice_event_type = splice_effect_map.get(splice_effect_str, SpliceEventType.CRYPTIC_DONOR)
                
                # Determine clinical significance based on disease context
                disease = row.get('disease', '').lower()
                if any(term in disease for term in ['cancer', 'fibrosis', 'dystrophy', 'syndrome']):
                    clinical_significance = ClinicalSignificance.PATHOGENIC
                else:
                    clinical_significance = ClinicalSignificance.LIKELY_PATHOGENIC
                
                # Determine affected site type
                cryptic_site_type = row.get('cryptic_site_type', '').lower()
                if cryptic_site_type in ['donor', 'acceptor']:
                    affected_site_type = f"cryptic_{cryptic_site_type}"
                else:
                    affected_site_type = 'cryptic'
                
                # Get validation information
                validation_method = row.get('validation_method', '')
                experimentally_validated = bool(validation_method and validation_method != '')
                
                # Create mutation object
                mutation = SpliceMutation(
                    chrom=str(row.get('chromosome', '')),
                    position=int(row.get('genomic_position', 0)),
                    ref_allele=str(row.get('ref_allele', '')),
                    alt_allele=str(row.get('alt_allele', '')),
                    gene_id="",  # DBASS typically doesn't have Ensembl IDs
                    gene_symbol=str(row.get('gene_symbol', row.get('gene_name', ''))),
                    transcript_id=str(row.get('transcript_id', '')),
                    splice_event_type=splice_event_type,
                    affected_site_type=affected_site_type,
                    splice_site_position=row.get('cryptic_site_position', None),
                    clinical_significance=clinical_significance,
                    disease_context=str(row.get('disease', '')),
                    experimentally_validated=experimentally_validated,
                    validation_method=validation_method,
                    source_database="DBASS",
                    source_id=str(row.get('entry_id', f"DBASS_{idx}")),
                    hgvs_notation=hgvs_notation,
                    metadata={
                        'mutation_type': str(row.get('mutation_type', '')),
                        'cryptic_site_sequence': str(row.get('cryptic_site_sequence', '')),
                        'strength_score': row.get('strength_score', None),
                        'wild_type_score': row.get('wild_type_score', None),
                        'mutant_score': row.get('mutant_score', None),
                        'fold_change': row.get('fold_change', None),
                        'phenotype': str(row.get('phenotype', '')),
                        'pmid': str(row.get('pmid', '')),
                        'patient_origin': str(row.get('patient_origin', '')),
                        'exon_affected': str(row.get('exon_affected', '')),
                        'original_row_index': idx
                    }
                )
                
                mutations.append(mutation)
                
            except Exception as e:
                self.processing_stats["parsing_errors"] += 1
                print(f"Error parsing row {idx}: {e}")
                continue
        
        print(f"Successfully parsed {len(mutations)} DBASS mutations")
        return mutations
    
    def validate_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """
        Validate and filter DBASS mutations.
        
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
            validation_errors = []
            
            # Basic validation checks
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
            
            # Validate strength scores if present
            if mutation.metadata:
                strength_score = mutation.metadata.get('strength_score')
                if strength_score is not None and (strength_score < 0 or strength_score > 15):
                    validation_errors.append(f"Invalid strength score: {strength_score}")
                
                # Add quality classification based on strength scores
                if strength_score is not None:
                    if strength_score >= 8.0:
                        mutation.metadata['quality'] = 'high'
                    elif strength_score >= 5.0:
                        mutation.metadata['quality'] = 'medium'
                    else:
                        mutation.metadata['quality'] = 'low'
                
                # Add confidence based on validation method
                validation_method = mutation.validation_method.lower() if mutation.validation_method else ''
                if 'rna_seq' in validation_method or 'rt-pcr' in validation_method:
                    mutation.metadata['confidence'] = 'high'
                elif 'minigene' in validation_method:
                    mutation.metadata['confidence'] = 'medium'
                else:
                    mutation.metadata['confidence'] = 'low'
            
            # Only include if validation passes
            if not validation_errors:
                validated.append(mutation)
            else:
                self.processing_stats["validation_errors"] += 1
                print(f"Validation failed for {mutation.source_id}: {'; '.join(validation_errors)}")
        
        print(f"Validated {len(validated)} out of {len(mutations)} DBASS mutations")
        return validated
    
    def get_cryptic_donor_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get mutations that activate cryptic donor sites."""
        return [m for m in mutations if 
                m.splice_event_type == SpliceEventType.CRYPTIC_DONOR or
                'donor' in m.affected_site_type]
    
    def get_cryptic_acceptor_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get mutations that activate cryptic acceptor sites."""
        return [m for m in mutations if 
                m.splice_event_type == SpliceEventType.CRYPTIC_ACCEPTOR or
                'acceptor' in m.affected_site_type]
    
    def get_pseudoexon_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get mutations that activate pseudoexons."""
        return [m for m in mutations if 
                m.splice_event_type == SpliceEventType.PSEUDOEXON_ACTIVATION]
    
    def get_high_strength_mutations(self, mutations: List[SpliceMutation], 
                                  min_strength: float = 7.0) -> List[SpliceMutation]:
        """
        Get mutations with high cryptic site strength scores.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            All mutations
        min_strength : float
            Minimum strength score threshold
            
        Returns
        -------
        List[SpliceMutation]
            High-strength cryptic site mutations
        """
        high_strength = []
        
        for mutation in mutations:
            if (mutation.metadata and 
                'strength_score' in mutation.metadata and
                mutation.metadata['strength_score'] is not None):
                
                if mutation.metadata['strength_score'] >= min_strength:
                    high_strength.append(mutation)
        
        return high_strength
    
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
    
    def get_fold_change_distribution(self, mutations: List[SpliceMutation]) -> Dict[str, Any]:
        """
        Get statistics on fold change in splice site strength.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            Mutations with fold change data
            
        Returns
        -------
        Dict[str, Any]
            Statistics on fold change distribution
        """
        fold_changes = []
        
        for mutation in mutations:
            if (mutation.metadata and 
                'fold_change' in mutation.metadata and
                mutation.metadata['fold_change'] is not None):
                fold_changes.append(mutation.metadata['fold_change'])
        
        if not fold_changes:
            return {"count": 0}
        
        import numpy as np
        
        return {
            "count": len(fold_changes),
            "mean": np.mean(fold_changes),
            "median": np.median(fold_changes),
            "std": np.std(fold_changes),
            "min": np.min(fold_changes),
            "max": np.max(fold_changes),
            "q25": np.percentile(fold_changes, 25),
            "q75": np.percentile(fold_changes, 75)
        } 