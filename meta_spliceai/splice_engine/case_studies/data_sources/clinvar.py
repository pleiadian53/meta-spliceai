"""
ClinVar ingester.

Ingests splice-affecting variants from ClinVar database with clinical
significance annotations and pathogenicity classifications.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import gzip
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import logging
import re
from datetime import datetime
import time

from .base import BaseIngester, SpliceMutation, IngestionResult, SpliceEventType, ClinicalSignificance
from ..formats.hgvs_parser import HGVSParser


class ClinVarIngester(BaseIngester):
    """Ingester for ClinVar splice-affecting variants."""
    
    def __init__(self, output_dir: Path, cache_dir: Optional[Path] = None, genome_build: str = "GRCh38"):
        """
        Initialize ClinVar ingester.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save processed data
        cache_dir : Path, optional
            Directory for caching downloaded data
        genome_build : str, optional
            Genome build to use (GRCh37, GRCh38). Default: GRCh38
        """
        super().__init__(output_dir, cache_dir)
        
        self.genome_build = genome_build
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.hgvs_parser = HGVSParser()
        
        # ClinVar FTP configuration
        self.base_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/"
        self.vcf_base_path = f"vcf_{genome_build}/"
        
        # File patterns for auto-detection
        self.vcf_pattern = re.compile(r'clinvar_(\d{8})\.vcf\.gz$')
        self.index_pattern = re.compile(r'clinvar_(\d{8})\.vcf\.gz\.tbi$')
    
    def list_available_files(self) -> List[Dict[str, Any]]:
        """
        List available ClinVar files on the FTP server.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of available files with metadata
        """
        try:
            # Get directory listing
            listing_url = urljoin(self.base_url, self.vcf_base_path)
            response = requests.get(listing_url)
            response.raise_for_status()
            
            files = []
            for line in response.text.split('\n'):
                # Parse FTP directory listing
                vcf_match = self.vcf_pattern.search(line)
                if vcf_match:
                    date_str = vcf_match.group(1)
                    filename = vcf_match.group(0)
                    
                    # Extract file size if available
                    size_match = re.search(r'\s+(\d+)\s+', line)
                    file_size = int(size_match.group(1)) if size_match else None
                    
                    files.append({
                        'filename': filename,
                        'date': date_str,
                        'size': file_size,
                        'url': urljoin(listing_url, filename)
                    })
            
            # Sort by date (newest first)
            files.sort(key=lambda x: x['date'], reverse=True)
            return files
            
        except Exception as e:
            self.logger.warning(f"Could not list available files: {e}")
            return []
    
    def get_latest_file_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the latest available ClinVar file."""
        files = self.list_available_files()
        return files[0] if files else None
    
    def download_data(self, force_refresh: bool = False, dataset_type: str = "vcf", 
                     auto_detect_latest: bool = True) -> Path:
        """
        Download ClinVar data with comprehensive functionality.
        
        Parameters
        ----------
        force_refresh : bool
            Whether to redownload even if cached
        dataset_type : str
            Type of dataset to download ("vcf", "variant_summary", "xml")
        auto_detect_latest : bool
            Whether to auto-detect and download the latest file
            
        Returns
        -------
        Path
            Path to downloaded data file
        """
        if dataset_type == "vcf" and auto_detect_latest:
            return self._download_latest_vcf(force_refresh)
        else:
            return self._download_standard_dataset(force_refresh, dataset_type)
    
    def _download_latest_vcf(self, force_refresh: bool) -> Path:
        """Download the latest ClinVar VCF file with auto-detection."""
        # Check for existing files first
        if not force_refresh:
            existing_files = list(self.cache_dir.glob("clinvar_*.vcf.gz"))
            if existing_files:
                # Return the most recent local file
                latest_local = sorted(existing_files)[-1]
                self.logger.info(f"Using cached ClinVar VCF: {latest_local}")
                return latest_local
        
        # Get latest file info from server
        latest_info = self.get_latest_file_info()
        if not latest_info:
            self.logger.warning("Could not detect latest file, using fallback")
            return self._download_standard_dataset(force_refresh, "vcf")
        
        filename = latest_info['filename']
        download_url = latest_info['url']
        local_path = self.cache_dir / filename
        
        # Check if we already have this specific file
        if local_path.exists() and not force_refresh:
            self.logger.info(f"Latest file already cached: {local_path}")
            return local_path
        
        # Download the file
        self.logger.info(f"Downloading latest ClinVar VCF: {filename}")
        self.logger.info(f"File date: {latest_info['date']}")
        if latest_info['size']:
            self.logger.info(f"File size: {latest_info['size']:,} bytes")
        
        try:
            self._download_file_with_progress(download_url, local_path)
            
            # Also download the index file if available
            index_filename = filename + '.tbi'
            index_url = urljoin(self.base_url, self.vcf_base_path + index_filename)
            index_path = self.cache_dir / index_filename
            
            try:
                self._download_file_with_progress(index_url, index_path, show_progress=False)
                self.logger.info(f"Downloaded index file: {index_filename}")
            except Exception as e:
                self.logger.warning(f"Could not download index file: {e}")
            
            # Save metadata
            self._save_download_metadata(local_path, latest_info)
            
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to download latest VCF: {e}")
            return self._create_demo_data("vcf")
    
    def _download_standard_dataset(self, force_refresh: bool, dataset_type: str) -> Path:
        """Download standard ClinVar datasets (non-VCF or fallback)."""
        # Map dataset types to URLs
        dataset_urls = {
            "vcf": f"{self.vcf_base_path}clinvar.vcf.gz",
            "variant_summary": "tab_delimited/variant_summary.txt.gz",
            "xml": "xml/clinvar_public.xml.gz"
        }
        
        if dataset_type not in dataset_urls:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from {list(dataset_urls.keys())}")
        
        cache_file = self.cache_dir / f"clinvar_{dataset_type}.gz"
        
        # Check if cached file exists and is recent
        if cache_file.exists() and not force_refresh:
            self.logger.info(f"Using cached ClinVar data: {cache_file}")
            return cache_file
        
        # Download the data
        download_url = urljoin(self.base_url, dataset_urls[dataset_type])
        self.logger.info(f"Downloading ClinVar {dataset_type} from {download_url}")
        
        try:
            self._download_file_with_progress(download_url, cache_file)
            return cache_file
            
        except Exception as e:
            self.logger.warning(f"Download failed: {e}")
            self.logger.info("Creating demo ClinVar data for development...")
            return self._create_demo_data(dataset_type)
    
    def _download_file_with_progress(self, url: str, local_path: Path, show_progress: bool = True):
        """Download a file with progress indication."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if show_progress and total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            self.logger.info(f"Download progress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)")
        
        self.logger.info(f"Download completed: {local_path}")
    
    def _save_download_metadata(self, file_path: Path, file_info: Dict[str, Any]):
        """Save metadata about the downloaded file."""
        metadata = {
            'filename': file_info['filename'],
            'date': file_info['date'],
            'size': file_info['size'],
            'download_timestamp': datetime.now().isoformat(),
            'genome_build': self.genome_build,
            'local_path': str(file_path)
        }
        
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved download metadata: {metadata_path}")
    
    def _create_demo_data(self, dataset: str) -> Path:
        """Create demo ClinVar data for development."""
        demo_file = self.cache_dir / f"clinvar_{dataset}_demo.txt"
        
        # Create sample data based on ClinVar variant_summary format
        demo_data = [
            {
                "VariationID": "17853",
                "Name": "NM_000492.3(CFTR):c.3718-2477C>T",
                "GeneSymbol": "CFTR",
                "HGVS(c.)": "NM_000492.3:c.3718-2477C>T",
                "HGVS(g.)": "NC_000007.14:g.117199644C>T",
                "Assembly": "GRCh38",
                "Chromosome": "7",
                "Start": 117199644,
                "Stop": 117199644,
                "ReferenceAllele": "C",
                "AlternateAllele": "T",
                "ClinicalSignificance": "Pathogenic",
                "PhenotypeList": "Cystic fibrosis",
                "ReviewStatus": "criteria provided, multiple submitters, no conflicts",
                "SubmitterCategories": "3",
                "DateLastEvaluated": "2023-05-15",
                "OriginSimple": "germline",
                "Type": "single nucleotide variant",
                "MolecularConsequence": "splice_site_variant",
                "AlleleID": "35033",
                "GeneID": "1080"
            },
            {
                "VariationID": "42368",
                "Name": "NM_007294.3(BRCA1):c.4357+1G>A",
                "GeneSymbol": "BRCA1",
                "HGVS(c.)": "NM_007294.3:c.4357+1G>A",
                "HGVS(g.)": "NC_000017.11:g.43124017G>A",
                "Assembly": "GRCh38",
                "Chromosome": "17",
                "Start": 43124017,
                "Stop": 43124017,
                "ReferenceAllele": "G",
                "AlternateAllele": "A",
                "ClinicalSignificance": "Pathogenic",
                "PhenotypeList": "Hereditary breast and ovarian cancer syndrome",
                "ReviewStatus": "criteria provided, multiple submitters, no conflicts",
                "SubmitterCategories": "3",
                "DateLastEvaluated": "2023-03-20",
                "OriginSimple": "germline",
                "Type": "single nucleotide variant",
                "MolecularConsequence": "splice_donor_variant",
                "AlleleID": "57003",
                "GeneID": "672"
            },
            {
                "VariationID": "318534",
                "Name": "NM_004006.2(DMD):c.6614+861C>T",
                "GeneSymbol": "DMD",
                "HGVS(c.)": "NM_004006.2:c.6614+861C>T",
                "HGVS(g.)": "NC_000023.11:g.32346896C>T",
                "Assembly": "GRCh38",
                "Chromosome": "X",
                "Start": 32346896,
                "Stop": 32346896,
                "ReferenceAllele": "C",
                "AlternateAllele": "T",
                "ClinicalSignificance": "Pathogenic",
                "PhenotypeList": "Duchenne muscular dystrophy",
                "ReviewStatus": "criteria provided, single submitter",
                "SubmitterCategories": "1",
                "DateLastEvaluated": "2022-11-08",
                "OriginSimple": "germline",
                "Type": "single nucleotide variant",
                "MolecularConsequence": "intron_variant",
                "AlleleID": "333459",
                "GeneID": "1756"
            },
            {
                "VariationID": "158442",
                "Name": "NM_000267.3(NF1):c.1466-2A>G",
                "GeneSymbol": "NF1",
                "HGVS(c.)": "NM_000267.3:c.1466-2A>G",
                "HGVS(g.)": "NC_000017.11:g.31234567A>G",
                "Assembly": "GRCh38",
                "Chromosome": "17",
                "Start": 31234567,
                "Stop": 31234567,
                "ReferenceAllele": "A",
                "AlternateAllele": "G",
                "ClinicalSignificance": "Pathogenic",
                "PhenotypeList": "Neurofibromatosis, type 1",
                "ReviewStatus": "criteria provided, multiple submitters, no conflicts",
                "SubmitterCategories": "3",
                "DateLastEvaluated": "2023-01-12",
                "OriginSimple": "germline",
                "Type": "single nucleotide variant",
                "MolecularConsequence": "splice_acceptor_variant",
                "AlleleID": "173035",
                "GeneID": "4763"
            },
            {
                "VariationID": "95784",
                "Name": "NM_000038.5(APC):c.1744-1G>A",
                "GeneSymbol": "APC",
                "HGVS(c.)": "NM_000038.5:c.1744-1G>A",
                "HGVS(g.)": "NC_000005.10:g.112840253G>A",
                "Assembly": "GRCh38",
                "Chromosome": "5",
                "Start": 112840253,
                "Stop": 112840253,
                "ReferenceAllele": "G",
                "AlternateAllele": "A",
                "ClinicalSignificance": "Pathogenic",
                "PhenotypeList": "Familial adenomatous polyposis",
                "ReviewStatus": "criteria provided, multiple submitters, no conflicts",
                "SubmitterCategories": "3",
                "DateLastEvaluated": "2023-04-05",
                "OriginSimple": "germline",
                "Type": "single nucleotide variant",
                "MolecularConsequence": "splice_acceptor_variant",
                "AlleleID": "110360",
                "GeneID": "324"
            }
        ]
        
        df = pd.DataFrame(demo_data)
        df.to_csv(demo_file, sep='\t', index=False)
        print(f"Created demo ClinVar data: {demo_file}")
        return demo_file
    
    def parse_raw_data(self, data_path: Path) -> List[SpliceMutation]:
        """
        Parse ClinVar TSV data to SpliceMutation objects.
        
        Parameters
        ----------
        data_path : Path
            Path to ClinVar data file
            
        Returns
        -------
        List[SpliceMutation]
            List of parsed splice mutations
        """
        mutations = []
        
        # Load data
        if data_path.suffix == '.gz':
            df = pd.read_csv(data_path, sep='\t', compression='gzip', low_memory=False)
        else:
            df = pd.read_csv(data_path, sep='\t', low_memory=False)
        
        print(f"Parsing {len(df)} ClinVar records...")
        
        # Filter for splice-affecting variants
        splice_terms = [
            'splice', 'donor', 'acceptor', 'intronic', 'intron_variant',
            'splice_site_variant', 'splice_donor_variant', 'splice_acceptor_variant'
        ]
        
        # Filter based on molecular consequence
        if 'MolecularConsequence' in df.columns:
            splice_mask = df['MolecularConsequence'].str.lower().str.contains('|'.join(splice_terms), na=False)
            df_splice = df[splice_mask].copy()
        else:
            # Fallback: filter based on HGVS notation patterns
            if 'HGVS(c.)' in df.columns:
                splice_mask = df['HGVS(c.)'].str.contains(r'[\+\-]\d+', na=False)  # Intronic positions
                df_splice = df[splice_mask].copy()
            else:
                df_splice = df.copy()
        
        print(f"Found {len(df_splice)} potential splice-affecting variants")
        
        for idx, row in df_splice.iterrows():
            try:
                # Parse HGVS notation
                hgvs_c = row.get('HGVS(c.)', '')
                hgvs_g = row.get('HGVS(g.)', '')
                hgvs_parsed = None
                if hgvs_c:
                    hgvs_parsed = self.hgvs_parser.parse(hgvs_c)
                
                # Map molecular consequence to splice event type
                mol_consequence = row.get('MolecularConsequence', '').lower()
                mol_consequence_map = {
                    'splice_donor_variant': SpliceEventType.CANONICAL_SITE_LOSS,
                    'splice_acceptor_variant': SpliceEventType.CANONICAL_SITE_LOSS,
                    'splice_site_variant': SpliceEventType.CANONICAL_SITE_LOSS,
                    'intron_variant': SpliceEventType.CRYPTIC_DONOR,  # Assume deep intronic = cryptic
                }
                
                splice_event_type = mol_consequence_map.get(mol_consequence, SpliceEventType.CANONICAL_SITE_LOSS)
                
                # If HGVS parsing shows deep intronic, likely cryptic site activation
                if hgvs_parsed and hgvs_parsed.intronic_offset is not None:
                    if abs(hgvs_parsed.intronic_offset) > 20:  # Deep intronic
                        if hgvs_parsed.intronic_offset > 0:
                            splice_event_type = SpliceEventType.CRYPTIC_DONOR
                        else:
                            splice_event_type = SpliceEventType.CRYPTIC_ACCEPTOR
                
                # Map clinical significance
                clin_sig_str = row.get('ClinicalSignificance', '').lower()
                clin_sig_map = {
                    'pathogenic': ClinicalSignificance.PATHOGENIC,
                    'likely pathogenic': ClinicalSignificance.LIKELY_PATHOGENIC,
                    'benign': ClinicalSignificance.BENIGN,
                    'likely benign': ClinicalSignificance.LIKELY_BENIGN,
                    'uncertain significance': ClinicalSignificance.UNCERTAIN,
                    'conflicting interpretations of pathogenicity': ClinicalSignificance.CONFLICTING
                }
                
                clinical_significance = None
                for key, value in clin_sig_map.items():
                    if key in clin_sig_str:
                        clinical_significance = value
                        break
                if clinical_significance is None:
                    clinical_significance = ClinicalSignificance.UNCERTAIN
                
                # Determine affected site type
                if 'donor' in mol_consequence:
                    affected_site_type = 'donor'
                elif 'acceptor' in mol_consequence:
                    affected_site_type = 'acceptor'
                elif hgvs_parsed and hgvs_parsed.intronic_offset is not None:
                    if hgvs_parsed.intronic_offset > 0:
                        affected_site_type = 'donor'
                    else:
                        affected_site_type = 'acceptor'
                else:
                    affected_site_type = 'splice_site'
                
                # Determine experimental validation
                review_status = row.get('ReviewStatus', '').lower()
                experimentally_validated = any(term in review_status for term in [
                    'criteria provided', 'reviewed by expert panel', 'practice guideline'
                ])
                
                # Extract position information
                chrom = str(row.get('Chromosome', ''))
                position = int(row.get('Start', 0)) if row.get('Start', 0) else 0
                
                # Create mutation object
                mutation = SpliceMutation(
                    chrom=chrom,
                    position=position,
                    ref_allele=str(row.get('ReferenceAllele', '')),
                    alt_allele=str(row.get('AlternateAllele', '')),
                    gene_id=str(row.get('GeneID', '')),
                    gene_symbol=str(row.get('GeneSymbol', '')),
                    transcript_id="",  # ClinVar doesn't always have transcript IDs in summary
                    splice_event_type=splice_event_type,
                    affected_site_type=affected_site_type,
                    splice_site_position=position,  # Use variant position as splice site position
                    clinical_significance=clinical_significance,
                    disease_context=str(row.get('PhenotypeList', '')),
                    experimentally_validated=experimentally_validated,
                    validation_method=review_status,
                    source_database="ClinVar",
                    source_id=str(row.get('VariationID', f"CV_{idx}")),
                    hgvs_notation=hgvs_c,
                    metadata={
                        'variation_id': str(row.get('VariationID', '')),
                        'allele_id': str(row.get('AlleleID', '')),
                        'name': str(row.get('Name', '')),
                        'hgvs_g': hgvs_g,
                        'assembly': str(row.get('Assembly', '')),
                        'molecular_consequence': mol_consequence,
                        'review_status': review_status,
                        'submitter_categories': str(row.get('SubmitterCategories', '')),
                        'date_last_evaluated': str(row.get('DateLastEvaluated', '')),
                        'origin': str(row.get('OriginSimple', '')),
                        'variant_type': str(row.get('Type', '')),
                        'stop_position': int(row.get('Stop', 0)) if row.get('Stop', 0) else None,
                        'original_row_index': idx
                    }
                )
                
                mutations.append(mutation)
                
            except Exception as e:
                self.processing_stats["parsing_errors"] += 1
                print(f"Error parsing row {idx}: {e}")
                continue
        
        print(f"Successfully parsed {len(mutations)} ClinVar splice mutations")
        return mutations
    
    def validate_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """
        Validate and filter ClinVar mutations.
        
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
            
            # Add quality classification based on review status
            if mutation.metadata:
                review_status = mutation.metadata.get('review_status', '').lower()
                submitter_categories = mutation.metadata.get('submitter_categories', '')
                
                # Classify based on review status and number of submitters
                if 'expert panel' in review_status or 'practice guideline' in review_status:
                    mutation.metadata['quality'] = 'high'
                    mutation.metadata['confidence'] = 'high'
                elif 'multiple submitters' in review_status and 'no conflicts' in review_status:
                    mutation.metadata['quality'] = 'high'
                    mutation.metadata['confidence'] = 'high'
                elif 'criteria provided' in review_status:
                    try:
                        num_submitters = int(submitter_categories) if submitter_categories.isdigit() else 1
                        if num_submitters >= 2:
                            mutation.metadata['quality'] = 'medium'
                            mutation.metadata['confidence'] = 'medium'
                        else:
                            mutation.metadata['quality'] = 'low'
                            mutation.metadata['confidence'] = 'low'
                    except:
                        mutation.metadata['quality'] = 'medium'
                        mutation.metadata['confidence'] = 'medium'
                else:
                    mutation.metadata['quality'] = 'low'
                    mutation.metadata['confidence'] = 'low'
            
            # Only include if validation passes
            if not validation_errors:
                validated.append(mutation)
            else:
                self.processing_stats["validation_errors"] += 1
                print(f"Validation failed for {mutation.source_id}: {'; '.join(validation_errors)}")
        
        print(f"Validated {len(validated)} out of {len(mutations)} ClinVar mutations")
        return validated
    
    def get_pathogenic_variants(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get pathogenic and likely pathogenic variants."""
        pathogenic_categories = {ClinicalSignificance.PATHOGENIC, ClinicalSignificance.LIKELY_PATHOGENIC}
        return [m for m in mutations if m.clinical_significance in pathogenic_categories]
    
    def get_benign_variants(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get benign and likely benign variants."""
        benign_categories = {ClinicalSignificance.BENIGN, ClinicalSignificance.LIKELY_BENIGN}
        return [m for m in mutations if m.clinical_significance in benign_categories]
    
    def get_high_confidence_variants(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get variants with high confidence annotations."""
        return [m for m in mutations if 
                m.metadata and 
                m.metadata.get('confidence') == 'high']
    
    def get_canonical_splice_variants(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Get variants affecting canonical splice sites (+/-1,2)."""
        canonical = []
        for mutation in mutations:
            if mutation.hgvs_notation:
                hgvs_parsed = self.hgvs_parser.parse(mutation.hgvs_notation)
                if (hgvs_parsed and hgvs_parsed.intronic_offset is not None and
                    abs(hgvs_parsed.intronic_offset) <= 2):
                    canonical.append(mutation)
            elif 'donor' in mutation.affected_site_type or 'acceptor' in mutation.affected_site_type:
                # If molecular consequence suggests canonical site
                if (mutation.metadata and 
                    'molecular_consequence' in mutation.metadata and
                    any(term in mutation.metadata['molecular_consequence'] for term in 
                        ['splice_donor_variant', 'splice_acceptor_variant'])):
                    canonical.append(mutation)
        return canonical
    
    def get_deep_intronic_variants(self, mutations: List[SpliceMutation], min_distance: int = 20) -> List[SpliceMutation]:
        """Get deep intronic variants that may activate cryptic sites."""
        deep_intronic = []
        for mutation in mutations:
            if mutation.hgvs_notation:
                hgvs_parsed = self.hgvs_parser.parse(mutation.hgvs_notation)
                if (hgvs_parsed and hgvs_parsed.intronic_offset is not None and
                    abs(hgvs_parsed.intronic_offset) >= min_distance):
                    deep_intronic.append(mutation)
        return deep_intronic
    
    def get_phenotype_specific_variants(self, mutations: List[SpliceMutation], 
                                      phenotypes: List[str]) -> List[SpliceMutation]:
        """
        Filter variants for specific phenotypes/diseases.
        
        Parameters
        ----------
        mutations : List[SpliceMutation]
            All mutations
        phenotypes : List[str]
            Phenotype terms to filter for
            
        Returns
        -------
        List[SpliceMutation]
            Phenotype-specific mutations
        """
        phenotype_lower = [p.lower() for p in phenotypes]
        filtered = []
        
        for mutation in mutations:
            if mutation.disease_context:
                if any(phenotype in mutation.disease_context.lower() for phenotype in phenotype_lower):
                    filtered.append(mutation)
        
        return filtered
    
    def get_review_status_summary(self, mutations: List[SpliceMutation]) -> Dict[str, int]:
        """Get summary of review status categories."""
        review_counts = {}
        
        for mutation in mutations:
            if mutation.metadata and 'review_status' in mutation.metadata:
                status = mutation.metadata['review_status']
                review_counts[status] = review_counts.get(status, 0) + 1
        
        return review_counts 