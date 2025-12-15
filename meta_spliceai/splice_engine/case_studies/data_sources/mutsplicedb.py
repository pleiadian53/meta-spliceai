"""
MutSpliceDB ingester.

Ingests splice mutation data from MutSpliceDB (NCI), which contains
validated splice-site mutations from TCGA and CCLE datasets.
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


class MutSpliceDBIngester(BaseIngester):
    """Ingester for MutSpliceDB splice mutation database."""
    
    def __init__(self, output_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize MutSpliceDB ingester.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save processed data
        cache_dir : Path, optional
            Directory for caching downloaded data
        """
        super().__init__(output_dir, cache_dir)
        
        # MutSpliceDB endpoints (adjust based on actual API)
        self.base_url = "https://mutsplicedb.nci.nih.gov/"  # Placeholder URL
        self.api_endpoints = {
            "mutations": "api/mutations",
            "genes": "api/genes",
            "cancer_types": "api/cancer_types"
        }
        
        # Alternative: Direct file download URLs
        self.download_urls = {
            "tcga_mutations": "downloads/mutsplicedb_tcga.tsv.gz",
            "ccle_mutations": "downloads/mutsplicedb_ccle.tsv.gz", 
            "all_mutations": "downloads/mutsplicedb_all.tsv.gz",
            "validated_mutations": "downloads/mutsplicedb_validated.tsv.gz"
        }
        
        self.hgvs_parser = HGVSParser()
    
    def download_data(self, force_refresh: bool = False, dataset: str = "tcga_mutations") -> Path:
        """
        Download MutSpliceDB data.
        
        Parameters
        ----------
        force_refresh : bool
            Whether to redownload even if cached
        dataset : str
            Which dataset to download ("tcga_mutations", "ccle_mutations", "all_mutations", "validated_mutations")
            
        Returns
        -------
        Path
            Path to downloaded data file
        """
        if dataset not in self.download_urls:
            raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(self.download_urls.keys())}")
        
        cache_file = self.cache_dir / f"mutsplicedb_{dataset}.tsv.gz"
        
        # Check if cached file exists and is recent
        if cache_file.exists() and not force_refresh:
            print(f"Using cached MutSpliceDB data: {cache_file}")
            return cache_file
        
        # Download the data
        download_url = urljoin(self.base_url, self.download_urls[dataset])
        print(f"Downloading MutSpliceDB {dataset} from {download_url}")
        
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded MutSpliceDB data to {cache_file}")
            return cache_file
            
        except requests.RequestException as e:
            # Fallback: create demo data
            print(f"Download failed: {e}")
            print("Creating demo MutSpliceDB data for development...")
            return self._create_demo_data(dataset)
    
    def _create_demo_data(self, dataset: str) -> Path:
        """Create demo data for development when real data is unavailable."""
        demo_file = self.cache_dir / f"mutsplicedb_{dataset}_demo.tsv"
        
        # Create sample data based on TCGA/CCLE format
        demo_data = [
            {
                "mutation_id": "MUT_000001",
                "sample_id": "TCGA-AA-3502-01A",
                "cancer_type": "NSCLC",
                "gene_symbol": "MET",
                "gene_id": "ENSG00000105976",
                "transcript_id": "ENST00000397752",
                "hgvs_c": "c.3028+1G>A",
                "hgvs_g": "g.116412043G>A", 
                "chromosome": "7",
                "position": 116412043,
                "ref_allele": "G",
                "alt_allele": "A",
                "mutation_type": "splice_donor",
                "splice_effect": "exon_skipping",
                "rna_evidence": "confirmed",
                "rna_support_reads": 45,
                "normal_reads": 98,
                "tumor_reads": 87,
                "vaf": 0.52,
                "clinical_annotation": "pathogenic",
                "therapeutic_target": "capmatinib",
                "pmid": "31883929"
            },
            {
                "mutation_id": "MUT_000002",
                "sample_id": "TCGA-A2-A04P-01A",
                "cancer_type": "BRCA",
                "gene_symbol": "BRCA1",
                "gene_id": "ENSG00000012048",
                "transcript_id": "ENST00000357654",
                "hgvs_c": "c.4357+1G>A",
                "hgvs_g": "g.41244000G>A",
                "chromosome": "17",
                "position": 41244000,
                "ref_allele": "G",
                "alt_allele": "A",
                "mutation_type": "splice_donor",
                "splice_effect": "intron_retention",
                "rna_evidence": "confirmed",
                "rna_support_reads": 23,
                "normal_reads": 67,
                "tumor_reads": 78,
                "vaf": 0.48,
                "clinical_annotation": "pathogenic",
                "therapeutic_target": "PARP_inhibitor",
                "pmid": "24240700"
            },
            {
                "mutation_id": "MUT_000003",
                "sample_id": "CCLE_A549",
                "cancer_type": "LUAD",
                "gene_symbol": "TP53",
                "gene_id": "ENSG00000141510",
                "transcript_id": "ENST00000269305",
                "hgvs_c": "c.375+2T>A",
                "hgvs_g": "g.7573927T>A",
                "chromosome": "17",
                "position": 7573927,
                "ref_allele": "T",
                "alt_allele": "A",
                "mutation_type": "splice_donor",
                "splice_effect": "cryptic_donor_activation",
                "rna_evidence": "confirmed",
                "rna_support_reads": 34,
                "normal_reads": 0,  # Cell line
                "tumor_reads": 89,
                "vaf": 0.38,
                "clinical_annotation": "pathogenic",
                "therapeutic_target": "MDM2_inhibitor",
                "pmid": "19934046"
            },
            {
                "mutation_id": "MUT_000004",
                "sample_id": "TCGA-BR-8371-01A",
                "cancer_type": "COAD",
                "gene_symbol": "APC",
                "gene_id": "ENSG00000134982",
                "transcript_id": "ENST00000257430",
                "hgvs_c": "c.1744-2A>G",
                "hgvs_g": "g.112128215A>G",
                "chromosome": "5",
                "position": 112128215,
                "ref_allele": "A",
                "alt_allele": "G",
                "mutation_type": "splice_acceptor",
                "splice_effect": "cryptic_acceptor_activation",
                "rna_evidence": "probable",
                "rna_support_reads": 12,
                "normal_reads": 56,
                "tumor_reads": 43,
                "vaf": 0.28,
                "clinical_annotation": "likely_pathogenic",
                "therapeutic_target": "WNT_inhibitor",
                "pmid": "21478906"
            }
        ]
        
        df = pd.DataFrame(demo_data)
        df.to_csv(demo_file, sep='\t', index=False)
        print(f"Created demo MutSpliceDB data: {demo_file}")
        return demo_file
    
    def parse_raw_data(self, data_path: Path) -> List[SpliceMutation]:
        """
        Parse MutSpliceDB TSV data to SpliceMutation objects.
        
        Parameters
        ----------
        data_path : Path
            Path to MutSpliceDB data file
            
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
        
        print(f"Parsing {len(df)} MutSpliceDB records...")
        
        for idx, row in df.iterrows():
            try:
                # Parse HGVS notation
                hgvs_c = row.get('hgvs_c', '')
                hgvs_parsed = None
                if hgvs_c:
                    hgvs_parsed = self.hgvs_parser.parse(hgvs_c)
                
                # Map splice effects to event types
                splice_effect_str = row.get('splice_effect', '').lower()
                splice_effect_map = {
                    'exon_skipping': SpliceEventType.EXON_SKIPPING,
                    'intron_retention': SpliceEventType.INTRON_RETENTION,
                    'cryptic_donor_activation': SpliceEventType.CRYPTIC_DONOR,
                    'cryptic_acceptor_activation': SpliceEventType.CRYPTIC_ACCEPTOR,
                    'canonical_site_loss': SpliceEventType.CANONICAL_SITE_LOSS,
                    'partial_exon_deletion': SpliceEventType.PARTIAL_EXON_DELETION,
                    'pseudoexon_activation': SpliceEventType.PSEUDOEXON_ACTIVATION
                }
                splice_event_type = splice_effect_map.get(splice_effect_str, SpliceEventType.CANONICAL_SITE_LOSS)
                
                # Map clinical annotations 
                clin_annotation_str = row.get('clinical_annotation', '').lower()
                clin_annotation_map = {
                    'pathogenic': ClinicalSignificance.PATHOGENIC,
                    'likely_pathogenic': ClinicalSignificance.LIKELY_PATHOGENIC,
                    'benign': ClinicalSignificance.BENIGN,
                    'likely_benign': ClinicalSignificance.LIKELY_BENIGN,
                    'uncertain': ClinicalSignificance.UNCERTAIN,
                    'conflicting': ClinicalSignificance.CONFLICTING
                }
                clinical_significance = clin_annotation_map.get(clin_annotation_str, ClinicalSignificance.UNCERTAIN)
                
                # Determine affected site type from mutation type
                mutation_type = row.get('mutation_type', '').lower()
                if 'donor' in mutation_type:
                    affected_site_type = 'donor'
                elif 'acceptor' in mutation_type:
                    affected_site_type = 'acceptor'
                else:
                    affected_site_type = 'unknown'
                
                # Determine splice site position from HGVS
                splice_site_position = None
                if hgvs_parsed and hgvs_parsed.is_valid:
                    if hgvs_parsed.intronic_offset is not None:
                        splice_site_position = hgvs_parsed.start_position + hgvs_parsed.intronic_offset
                    else:
                        splice_site_position = hgvs_parsed.start_position
                
                # Check for RNA evidence
                rna_evidence = row.get('rna_evidence', '').lower()
                experimentally_validated = rna_evidence in ['confirmed', 'probable']
                
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
                    affected_site_type=affected_site_type,
                    splice_site_position=splice_site_position,
                    clinical_significance=clinical_significance,
                    disease_context=str(row.get('cancer_type', '')),
                    experimentally_validated=experimentally_validated,
                    validation_method=rna_evidence,
                    source_database="MutSpliceDB",
                    source_id=str(row.get('mutation_id', f"MUT_{idx}")),
                    hgvs_notation=hgvs_c,
                    metadata={
                        'sample_id': str(row.get('sample_id', '')),
                        'cancer_type': str(row.get('cancer_type', '')),
                        'hgvs_g': str(row.get('hgvs_g', '')),
                        'rna_support_reads': row.get('rna_support_reads', None),
                        'normal_reads': row.get('normal_reads', None),
                        'tumor_reads': row.get('tumor_reads', None),
                        'vaf': row.get('vaf', None),  # Variant allele frequency
                        'therapeutic_target': str(row.get('therapeutic_target', '')),
                        'pmid': str(row.get('pmid', '')),
                        'original_row_index': idx
                    }
                )
                
                mutations.append(mutation)
                
            except Exception as e:
                self.processing_stats["parsing_errors"] += 1
                print(f"Error parsing row {idx}: {e}")
                continue
        
        print(f"Successfully parsed {len(mutations)} MutSpliceDB mutations")
        return mutations
    
    def validate_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """
        Validate and filter MutSpliceDB mutations.
        
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
            
            # Validate VAF if present
            if mutation.metadata and 'vaf' in mutation.metadata:
                vaf = mutation.metadata['vaf']
                if vaf is not None and (vaf < 0 or vaf > 1):
                    validation_errors.append(f"Invalid VAF: {vaf}")
            
            # Add quality scores based on RNA evidence
            if mutation.metadata:
                rna_reads = mutation.metadata.get('rna_support_reads', 0)
                if rna_reads and rna_reads >= 10:
                    mutation.metadata['quality'] = 'high'
                elif rna_reads and rna_reads >= 5:
                    mutation.metadata['quality'] = 'medium'
                else:
                    mutation.metadata['quality'] = 'low'
            
            # Only include if validation passes
            if not validation_errors:
                validated.append(mutation)
            else:
                self.processing_stats["validation_errors"] += 1
                print(f"Validation failed for {mutation.source_id}: {'; '.join(validation_errors)}")
        
        print(f"Validated {len(validated)} out of {len(mutations)} MutSpliceDB mutations")
        return validated
    
    def get_cancer_specific_mutations(self, mutations: List[SpliceMutation], 
                                    cancer_types: List[str]) -> List[SpliceMutation]:
        """
        Filter mutations for specific cancer types.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            All mutations
        cancer_types : List[str]
            Cancer type codes (e.g., ['NSCLC', 'BRCA', 'COAD'])
            
        Returns
        -------
        List[SpliceMutation]
            Cancer-specific mutations
        """
        cancer_lower = [c.lower() for c in cancer_types]
        filtered = []
        
        for mutation in mutations:
            if mutation.disease_context:
                if mutation.disease_context.lower() in cancer_lower:
                    filtered.append(mutation)
            # Also check metadata
            if mutation.metadata and 'cancer_type' in mutation.metadata:
                if mutation.metadata['cancer_type'].lower() in cancer_lower:
                    filtered.append(mutation)
        
        return filtered
    
    def get_tcga_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get mutations from TCGA samples only."""
        return [m for m in mutations if m.metadata and 
                'sample_id' in m.metadata and 
                m.metadata['sample_id'].startswith('TCGA-')]
    
    def get_ccle_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get mutations from CCLE cell lines only."""
        return [m for m in mutations if m.metadata and 
                'sample_id' in m.metadata and 
                ('CCLE' in m.metadata['sample_id'] or not m.metadata['sample_id'].startswith('TCGA-'))]
    
    def get_high_quality_mutations(self, mutations: List[SpliceMutation], 
                                 min_rna_reads: int = 10) -> List[SpliceMutation]:
        """
        Get high-quality mutations with strong RNA evidence.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            All mutations
        min_rna_reads : int
            Minimum RNA supporting reads required
            
        Returns
        -------
        List[SpliceMutation]
            High-quality mutations
        """
        high_quality = []
        
        for mutation in mutations:
            if (mutation.experimentally_validated and 
                mutation.metadata and 
                'rna_support_reads' in mutation.metadata):
                
                rna_reads = mutation.metadata.get('rna_support_reads', 0)
                if rna_reads >= min_rna_reads:
                    high_quality.append(mutation)
        
        return high_quality
    
    def get_therapeutic_targets(self, mutations: List[SpliceMutation]) -> Dict[str, List[SpliceMutation]]:
        """
        Group mutations by therapeutic targets.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            All mutations
            
        Returns
        -------
        Dict[str, List[SpliceMutation]]
            Mutations grouped by therapeutic target
        """
        targets = {}
        
        for mutation in mutations:
            if mutation.metadata and 'therapeutic_target' in mutation.metadata:
                target = mutation.metadata['therapeutic_target']
                if target and target != '':
                    if target not in targets:
                        targets[target] = []
                    targets[target].append(mutation)
        
        return targets 