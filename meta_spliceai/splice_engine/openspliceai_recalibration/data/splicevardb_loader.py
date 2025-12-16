"""
Complete SpliceVarDB data loader with API integration.

Ingests and processes SpliceVarDB's >50K experimentally validated splice variants
for training OpenSpliceAI recalibration models.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpliceVarDBRecord:
    """Structured representation of a SpliceVarDB variant record."""
    
    # Variant identification
    chrom: str
    pos: int  # 1-based genomic position
    ref: str
    alt: str
    gene: str
    build: str = "GRCh38"
    
    # SpliceVarDB classification
    classification: str = ""  # splice-altering, not splice-altering, low-frequency
    splicing_outcome: str = ""  # exon_skip, intron_retention, cryptic_site, etc.
    
    # Experimental evidence
    assay_types: List[str] = field(default_factory=list)
    evidence_strength: str = ""  # strong, moderate, weak
    pmids: List[str] = field(default_factory=list)
    
    # Metadata
    source_id: str = ""
    raw_record: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "chrom": self.chrom,
            "pos": self.pos,
            "ref": self.ref,
            "alt": self.alt,
            "gene": self.gene,
            "build": self.build,
            "classification": self.classification,
            "splicing_outcome": self.splicing_outcome,
            "assay_types": ",".join(self.assay_types) if self.assay_types else "",
            "evidence_strength": self.evidence_strength,
            "pmids": ",".join(self.pmids) if self.pmids else "",
            "source_id": self.source_id,
        }
    
    @property
    def is_splice_altering(self) -> bool:
        """Check if variant is classified as splice-altering."""
        return "splice-altering" in self.classification.lower()
    
    @property
    def has_strong_evidence(self) -> bool:
        """Check if variant has strong experimental evidence."""
        return self.evidence_strength.lower() in ["strong", "high"]


class SpliceVarDBLoader:
    """
    Loader for SpliceVarDB data with comprehensive API integration.
    
    Features:
    - Paginated API downloads with rate limiting
    - Local caching to avoid redundant downloads
    - Data validation and quality control
    - Export to multiple formats (TSV, VCF, Parquet)
    - Demo data fallback for development
    """
    
    # API Configuration (based on SpliceVarDB documentation)
    DEFAULT_BASE_URL = "https://compbio.ccia.org.au/splicevardb-api"
    VARIANTS_ENDPOINT = "/variants"
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_downloads: bool = True,
        verbose: int = 1,
        use_systematic_paths: bool = True
    ):
        """
        Initialize SpliceVarDB loader.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory for saving processed data. If None and use_systematic_paths=True,
            uses systematic path from genomic resource manager:
            data/ensembl/case_studies/splicevardb/
        api_token : str, optional
            SpliceVarDB API token (can also use SPLICEVARDB_TOKEN env var)
        base_url : str, optional
            API base URL (defaults to official API)
        cache_downloads : bool
            Whether to cache downloaded data
        verbose : int
            Verbosity level (0=silent, 1=progress, 2=debug)
        use_systematic_paths : bool
            Whether to use systematic path management from genomic_resources
        """
        # Determine output directory
        if output_dir is None and use_systematic_paths:
            # Use systematic path from resource manager
            try:
                from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
                    CaseStudyResourceManager
                )
                manager = CaseStudyResourceManager()
                self.output_dir = manager.case_study_paths.splicevardb
                logger.info(f"Using systematic path: {self.output_dir}")
            except ImportError:
                # Fallback to default if resource manager not available
                self.output_dir = Path("./data/splicevardb")
                logger.warning("Resource manager not available, using fallback path")
        else:
            self.output_dir = Path(output_dir or "./data/splicevardb")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_token = api_token or os.getenv("SPLICEVARDB_TOKEN")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.cache_downloads = cache_downloads
        self.verbose = verbose
        
        # Setup logging
        if self.verbose >= 2:
            logging.basicConfig(level=logging.DEBUG)
        elif self.verbose >= 1:
            logging.basicConfig(level=logging.INFO)
        
        # Create systematic subdirectories: raw/, processed/, cache/
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.cache_dir = self.output_dir / "cache"
        
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized SpliceVarDBLoader")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Raw data: {self.raw_dir}")
        logger.info(f"  Processed data: {self.processed_dir}")
        logger.info(f"  Cache: {self.cache_dir}")
    
    def load_validated_variants(
        self,
        build: str = "GRCh38",
        force_refresh: bool = False,
        max_variants: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load validated splice-altering variants from SpliceVarDB.
        
        Parameters
        ----------
        build : str
            Genome build (GRCh38 or GRCh37/hg19)
        force_refresh : bool
            Force redownload even if cached
        max_variants : int, optional
            Maximum number of variants to load (for testing)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with validated variants
        """
        cache_file = self.cache_dir / f"splicevardb_validated_{build}.parquet"
        
        # Check cache
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        
        # Download from API
        logger.info(f"Downloading SpliceVarDB variants (build={build})...")
        records = self._download_variants(build=build, max_variants=max_variants)
        
        if not records:
            logger.warning("No records downloaded. Using demo data.")
            return self._create_demo_data()
        
        # Convert to DataFrame
        df = pd.DataFrame([rec.to_dict() for rec in records])
        
        # Cache if enabled
        if self.cache_downloads:
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached data to {cache_file}")
        
        logger.info(f"Loaded {len(df)} variants from SpliceVarDB")
        return df
    
    def _download_variants(
        self,
        build: str = "GRCh38",
        max_variants: Optional[int] = None,
        page_size: int = 1000
    ) -> List[SpliceVarDBRecord]:
        """
        Download variants from SpliceVarDB API with pagination.
        
        Parameters
        ----------
        build : str
            Genome build
        max_variants : int, optional
            Maximum variants to download
        page_size : int
            Variants per API request
            
        Returns
        -------
        List[SpliceVarDBRecord]
            List of variant records
        """
        session = self._create_session()
        records = []
        offset = 0
        
        while True:
            if max_variants and len(records) >= max_variants:
                break
            
            try:
                # Fetch page
                page_records = self._fetch_page(
                    session, build, offset, page_size
                )
                
                if not page_records:
                    break
                
                records.extend(page_records)
                offset += len(page_records)
                
                if self.verbose >= 1:
                    print(f"\rDownloaded {len(records)} variants...", end="", flush=True)
                
                # Break if incomplete page (last page)
                if len(page_records) < page_size:
                    break
                
                # Rate limiting
                time.sleep(0.2)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                if len(records) == 0:
                    logger.warning("No records downloaded. Falling back to demo data.")
                    return []
                else:
                    logger.warning(f"Partial download complete: {len(records)} records")
                    break
        
        if self.verbose >= 1:
            print()  # New line after progress
        
        return records
    
    def _fetch_page(
        self,
        session: requests.Session,
        build: str,
        offset: int,
        limit: int
    ) -> List[SpliceVarDBRecord]:
        """
        Fetch single page from API.
        
        Parameters
        ----------
        session : requests.Session
            HTTP session with auth headers
        build : str
            Genome build
        offset : int
            Starting position
        limit : int
            Number of records to fetch
            
        Returns
        -------
        List[SpliceVarDBRecord]
            Records from this page
        """
        url = f"{self.base_url}{self.VARIANTS_ENDPOINT}"
        params = {
            "build": build,
            "offset": offset,
            "limit": limit
        }
        
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        payload = response.json()
        
        # Handle different response formats
        if isinstance(payload, dict) and "results" in payload:
            items = payload["results"]
        elif isinstance(payload, list):
            items = payload
        else:
            logger.warning(f"Unexpected payload format at offset={offset}")
            return []
        
        # Parse records
        records = []
        for item in items:
            try:
                record = self._parse_record(item)
                records.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse record: {e}")
                continue
        
        return records
    
    def _parse_record(self, raw: Dict[str, Any]) -> SpliceVarDBRecord:
        """
        Parse raw API record into structured SpliceVarDBRecord.
        
        Parameters
        ----------
        raw : dict
            Raw record from API
            
        Returns
        -------
        SpliceVarDBRecord
            Structured record
        """
        # Extract core variant fields
        chrom = str(raw.get("chromosome") or raw.get("chrom", "")).replace("chr", "")
        pos = int(raw.get("position") or raw.get("pos", 0))
        ref = str(raw.get("ref", ""))
        alt = str(raw.get("alt", ""))
        gene = str(raw.get("gene_symbol") or raw.get("gene", ""))
        build = str(raw.get("genome_build") or raw.get("build", "GRCh38"))
        
        # Extract classification
        classification = str(raw.get("classification", ""))
        splicing_outcome = str(raw.get("splicing_outcome") or raw.get("outcome", ""))
        
        # Extract evidence
        assays = raw.get("assays") or raw.get("assay") or []
        if isinstance(assays, str):
            assays = [a.strip() for a in assays.split(",")]
        assays = [str(a) for a in assays]
        
        evidence_strength = str(raw.get("evidence_strength") or raw.get("evidence", ""))
        
        # Extract PMIDs
        pmids = raw.get("pmids") or raw.get("pmid") or []
        if isinstance(pmids, str):
            pmids = [p.strip() for p in pmids.split(",")]
        pmids = [str(p) for p in pmids]
        
        # Metadata
        source_id = str(raw.get("id") or raw.get("_id", ""))
        
        return SpliceVarDBRecord(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            gene=gene,
            build=build,
            classification=classification,
            splicing_outcome=splicing_outcome,
            assay_types=assays,
            evidence_strength=evidence_strength,
            pmids=pmids,
            source_id=source_id,
            raw_record=raw
        )
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with authentication."""
        session = requests.Session()
        if self.api_token:
            session.headers.update({
                "Authorization": f"Bearer {self.api_token}"
            })
        session.headers.update({
            "Accept": "application/json",
            "User-Agent": "MetaSpliceAI-OpenSpliceAI-Recalibration/0.1.0"
        })
        return session
    
    def _create_demo_data(self) -> pd.DataFrame:
        """
        Create demo/test data for development.
        
        Returns
        -------
        pd.DataFrame
            Small demo dataset
        """
        logger.info("Creating demo SpliceVarDB data for development")
        
        # Classic splice-altering variants
        demo_variants = [
            # CFTR cryptic exon (classic)
            {
                "chrom": "7",
                "pos": 117199644,
                "ref": "C",
                "alt": "T",
                "gene": "CFTR",
                "build": "GRCh38",
                "classification": "splice-altering",
                "splicing_outcome": "cryptic_pseudoexon",
                "assay_types": "minigene,RT-PCR",
                "evidence_strength": "strong",
                "pmids": "26942284",
                "source_id": "demo_001"
            },
            # BRCA1 splice variant
            {
                "chrom": "17",
                "pos": 43094464,
                "ref": "A",
                "alt": "G",
                "gene": "BRCA1",
                "build": "GRCh38",
                "classification": "splice-altering",
                "splicing_outcome": "exon_skip",
                "assay_types": "RNA-seq,minigene",
                "evidence_strength": "strong",
                "pmids": "12345678",
                "source_id": "demo_002"
            },
            # DMD non-splice-altering control
            {
                "chrom": "X",
                "pos": 32867916,
                "ref": "G",
                "alt": "A",
                "gene": "DMD",
                "build": "GRCh38",
                "classification": "not splice-altering",
                "splicing_outcome": "no_effect",
                "assay_types": "RNA-seq",
                "evidence_strength": "strong",
                "pmids": "87654321",
                "source_id": "demo_003"
            },
        ]
        
        return pd.DataFrame(demo_variants)
    
    def export_to_tsv(self, df: pd.DataFrame, output_file: Optional[str] = None):
        """Export to TSV format."""
        if output_file is None:
            output_file = self.processed_dir / "splicevardb.tsv"
        else:
            output_file = Path(output_file)
        
        df.to_csv(output_file, sep="\t", index=False)
        logger.info(f"Exported TSV to {output_file}")
    
    def export_to_vcf(self, df: pd.DataFrame, output_file: Optional[str] = None):
        """Export to VCF format."""
        if output_file is None:
            output_file = self.processed_dir / "splicevardb.vcf"
        else:
            output_file = Path(output_file)
        
        with open(output_file, "w") as f:
            # VCF header
            f.write("##fileformat=VCFv4.2\n")
            f.write(f"##fileDate={datetime.now().strftime('%Y%m%d')}\n")
            f.write("##source=SpliceVarDB\n")
            f.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene symbol\">\n")
            f.write("##INFO=<ID=CLASS,Number=1,Type=String,Description=\"Splice classification\">\n")
            f.write("##INFO=<ID=OUTCOME,Number=1,Type=String,Description=\"Splicing outcome\">\n")
            f.write("##INFO=<ID=ASSAY,Number=.,Type=String,Description=\"Validation assays\">\n")
            f.write("##INFO=<ID=PMID,Number=.,Type=String,Description=\"PubMed IDs\">\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            
            # Variants
            for _, row in df.iterrows():
                chrom = row["chrom"]
                pos = row["pos"]
                ref = row["ref"]
                alt = row["alt"]
                gene = row["gene"]
                classification = row["classification"]
                outcome = row["splicing_outcome"]
                assay = row["assay_types"]
                pmid = row["pmids"]
                
                info = f"GENE={gene};CLASS={classification};OUTCOME={outcome};ASSAY={assay};PMID={pmid}"
                f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t{info}\n")
        
        logger.info(f"Exported VCF to {output_file}")
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_variants": len(df),
            "unique_genes": df["gene"].nunique(),
            "splice_altering": (df["classification"].str.contains("splice-altering")).sum(),
            "not_splice_altering": (df["classification"] == "not splice-altering").sum(),
            "chromosomes": df["chrom"].nunique(),
            "builds": df["build"].unique().tolist(),
        }
        
        # Assay type distribution
        all_assays = []
        for assays_str in df["assay_types"].dropna():
            if assays_str:
                all_assays.extend(assays_str.split(","))
        stats["assay_distribution"] = pd.Series(all_assays).value_counts().to_dict()
        
        return stats


def main():
    """Command-line interface for SpliceVarDB loader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and process SpliceVarDB variants"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/splicevardb",
        help="Output directory"
    )
    parser.add_argument(
        "--build",
        default="GRCh38",
        choices=["GRCh38", "GRCh37", "hg38", "hg19"],
        help="Genome build"
    )
    parser.add_argument(
        "--token",
        default=os.getenv("SPLICEVARDB_TOKEN"),
        help="API token (or use SPLICEVARDB_TOKEN env var)"
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        help="Maximum variants to download (for testing)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force redownload even if cached"
    )
    parser.add_argument(
        "--export-tsv",
        action="store_true",
        help="Export to TSV format"
    )
    parser.add_argument(
        "--export-vcf",
        action="store_true",
        help="Export to VCF format"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity"
    )
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = SpliceVarDBLoader(
        output_dir=args.output_dir,
        api_token=args.token,
        verbose=args.verbose
    )
    
    # Load data
    df = loader.load_validated_variants(
        build=args.build,
        force_refresh=args.force_refresh,
        max_variants=args.max_variants
    )
    
    # Print statistics
    stats = loader.get_statistics(df)
    print("\n=== SpliceVarDB Statistics ===")
    print(f"Total variants: {stats['total_variants']}")
    print(f"Unique genes: {stats['unique_genes']}")
    print(f"Splice-altering: {stats['splice_altering']}")
    print(f"Not splice-altering: {stats['not_splice_altering']}")
    print(f"Chromosomes: {stats['chromosomes']}")
    print(f"Builds: {', '.join(stats['builds'])}")
    
    # Export if requested
    if args.export_tsv:
        loader.export_to_tsv(df)
    
    if args.export_vcf:
        loader.export_to_vcf(df)
    
    print(f"\n✅ Data saved to {args.output_dir}")


if __name__ == "__main__":
    main()

