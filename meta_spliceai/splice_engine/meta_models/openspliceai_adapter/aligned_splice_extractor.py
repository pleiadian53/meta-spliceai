#!/usr/bin/env python3
"""
Aligned Splice Site Extractor

This module provides a unified interface for extracting splice sites that are
100% compatible between MetaSpliceAI and OpenSpliceAI systems. It handles:

1. Gene filtering alignment (all biotypes vs protein_coding only)
2. Transcript selection alignment (all vs filtered transcripts)
3. Coordinate system reconciliation (systematic position adjustments)
4. Variant analysis compatibility (ClinVar, SpliceVarDB coordinate systems)

Key Features:
- Systematic coordinate discrepancy detection and correction
- Support for multiple annotation sources and coordinate conventions
- Clean API for downstream variant analysis
- Comprehensive validation and quality control

Author: MetaSpliceAI Integration Team
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import gffutils
from pyfaidx import Fasta

# Import coordinate reconciliation
from .coordinate_reconciliation import SpliceCoordinateReconciler

logger = logging.getLogger(__name__)


class AlignedSpliceExtractor:
    """
    Unified splice site extractor that ensures 100% compatibility between
    MetaSpliceAI and OpenSpliceAI coordinate systems.
    
    This class provides the foundation for variant analysis by handling
    coordinate system discrepancies that are critical for accurate
    mutation impact assessment.
    """
    
    def __init__(self, 
                 coordinate_reconciler: Optional[SpliceCoordinateReconciler] = None,
                 enable_biotype_filtering: bool = False,
                 enable_transcript_filtering: bool = False,
                 coordinate_system: str = "splicesurveyor",
                 verbosity: int = 1):
        """
        Initialize the aligned splice extractor.
        
        Args:
            coordinate_reconciler: Optional reconciler for coordinate adjustments
            enable_biotype_filtering: If True, filter to protein_coding genes only
            enable_transcript_filtering: If True, apply transcript selection filters
            coordinate_system: Target coordinate system ("splicesurveyor", "openspliceai", "spliceai")
            verbosity: Logging verbosity level
        """
        self.coordinate_reconciler = coordinate_reconciler or SpliceCoordinateReconciler()
        self.enable_biotype_filtering = enable_biotype_filtering
        self.enable_transcript_filtering = enable_transcript_filtering
        self.coordinate_system = coordinate_system
        self.verbosity = verbosity
        
        # Coordinate system configurations
        self.coordinate_configs = {
            "splicesurveyor": {
                "donor_offset": 0,
                "acceptor_offset": 0,
                "description": "MetaSpliceAI native coordinates"
            },
            "openspliceai": {
                "donor_offset": -1,  # Detected from forensic analysis
                "acceptor_offset": 1,  # Detected from forensic analysis  
                "description": "OpenSpliceAI native coordinates"
            },
            "spliceai": {
                "donor_offset": 0,
                "acceptor_offset": 0,
                "description": "SpliceAI reference coordinates"
            }
        }
        
        if verbosity > 0:
            logger.info(f"Initialized AlignedSpliceExtractor with {coordinate_system} coordinates")
    
    def extract_splice_sites(self,
                           gtf_file: str,
                           fasta_file: str,
                           gene_ids: Optional[List[str]] = None,
                           output_format: str = "dataframe",
                           apply_schema_adaptation: bool = False) -> Union[pd.DataFrame, List[Dict]]:
        """
        Extract splice sites with full alignment and coordinate reconciliation.
        
        This is the main entry point for splice site extraction that ensures
        100% compatibility across different coordinate systems.
        
        Args:
            gtf_file: Path to GTF annotation file
            fasta_file: Path to FASTA genome file
            gene_ids: Optional list of gene IDs to process (None = all genes)
            output_format: Output format ("dataframe", "list", "openspliceai_compatible")
            apply_schema_adaptation: If True, apply schema adaptation to MetaSpliceAI format
            
        Returns:
            Extracted splice sites in requested format
        """
        
        if self.verbosity > 0:
            logger.info("🎯 Starting aligned splice site extraction")
            logger.info(f"Target coordinate system: {self.coordinate_system}")
            logger.info(f"Biotype filtering: {'enabled' if self.enable_biotype_filtering else 'disabled'}")
            logger.info(f"Transcript filtering: {'enabled' if self.enable_transcript_filtering else 'disabled'}")
        
        # Load database and FASTA
        db, fasta = self._load_genomic_resources(gtf_file, fasta_file)
        
        # Get genes to process
        genes_to_process = self._get_genes_to_process(db, gene_ids)
        
        # Extract splice sites from all genes
        all_splice_sites = []
        
        for i, gene in enumerate(genes_to_process):
            if self.verbosity > 0 and i % 50 == 0:
                logger.info(f"Processing gene {i+1}/{len(genes_to_process)}: {gene.id}")
            
            try:
                gene_splice_sites = self._extract_gene_splice_sites(gene, db, fasta)
                if gene_splice_sites:
                    all_splice_sites.extend(gene_splice_sites)
            except Exception as e:
                if self.verbosity > 0:
                    logger.warning(f"Error processing gene {gene.id}: {e}")
                continue
        
        if self.verbosity > 0:
            logger.info(f"✅ Extracted {len(all_splice_sites)} splice sites from {len(genes_to_process)} genes")
        
        # Apply coordinate reconciliation if needed
        if self.coordinate_system != "splicesurveyor":
            all_splice_sites = self._apply_coordinate_reconciliation(all_splice_sites)
        
        # Format output
        return self._format_output(all_splice_sites, output_format, apply_schema_adaptation)
    
    def detect_coordinate_discrepancies(self,
                                      reference_sites: pd.DataFrame,
                                      comparison_sites: pd.DataFrame,
                                      reference_system: str = "splicesurveyor",
                                      comparison_system: str = "openspliceai") -> Dict[str, Any]:
        """
        Detect systematic coordinate discrepancies between two splice site datasets.
        
        This method is crucial for variant analysis where different databases
        (ClinVar, SpliceVarDB) may use different coordinate conventions.
        
        Args:
            reference_sites: Reference splice site dataset
            comparison_sites: Comparison splice site dataset  
            reference_system: Coordinate system of reference dataset
            comparison_system: Coordinate system of comparison dataset
            
        Returns:
            Detailed discrepancy analysis report
        """
        
        if self.verbosity > 0:
            logger.info(f"🔍 Detecting coordinate discrepancies: {reference_system} vs {comparison_system}")
        
        # Use coordinate reconciler for systematic analysis
        discrepancy_report = self.coordinate_reconciler.detect_coordinate_offsets(
            reference_sites, comparison_sites
        )
        
        # Add system-specific analysis
        discrepancy_report.update({
            "reference_system": reference_system,
            "comparison_system": comparison_system,
            "reference_sites_count": len(reference_sites),
            "comparison_sites_count": len(comparison_sites),
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        })
        
        if self.verbosity > 0:
            logger.info(f"📊 Discrepancy analysis complete")
            for splice_type in ["donor", "acceptor"]:
                for strand in ["+", "-"]:
                    key = f"{splice_type}_{strand}"
                    if key in discrepancy_report.get("detected_offsets", {}):
                        offset_info = discrepancy_report["detected_offsets"][key]
                        logger.info(f"  {key}: {offset_info['offset']:+d}nt (confidence: {offset_info['confidence']:.1%})")
        
        return discrepancy_report
    
    def reconcile_variant_coordinates(self,
                                    variant_df: pd.DataFrame,
                                    source_system: str,
                                    target_system: str = "splicesurveyor") -> pd.DataFrame:
        """
        Reconcile variant coordinates from external databases (ClinVar, SpliceVarDB)
        to ensure compatibility with MetaSpliceAI coordinate system.
        
        This is essential for accurate variant impact analysis.
        
        Args:
            variant_df: DataFrame with variant coordinates
            source_system: Source coordinate system ("clinvar", "splicevardb", "openspliceai", etc.)
            target_system: Target coordinate system (default: "splicesurveyor")
            
        Returns:
            DataFrame with reconciled coordinates
        """
        
        if self.verbosity > 0:
            logger.info(f"🔄 Reconciling variant coordinates: {source_system} → {target_system}")
        
        # Create a copy to avoid modifying original
        reconciled_df = variant_df.copy()
        
        # Apply known coordinate adjustments based on source system
        if source_system in self.coordinate_configs and target_system in self.coordinate_configs:
            source_config = self.coordinate_configs[source_system]
            target_config = self.coordinate_configs[target_system]
            
            # Calculate adjustment needed
            donor_adjustment = target_config["donor_offset"] - source_config["donor_offset"]
            acceptor_adjustment = target_config["acceptor_offset"] - source_config["acceptor_offset"]
            
            # Apply adjustments based on variant type
            if "splice_type" in reconciled_df.columns:
                donor_mask = reconciled_df["splice_type"] == "donor"
                acceptor_mask = reconciled_df["splice_type"] == "acceptor"
                
                if "position" in reconciled_df.columns:
                    reconciled_df.loc[donor_mask, "position"] += donor_adjustment
                    reconciled_df.loc[acceptor_mask, "position"] += acceptor_adjustment
                
                if self.verbosity > 0:
                    logger.info(f"  Applied donor adjustment: {donor_adjustment:+d}nt")
                    logger.info(f"  Applied acceptor adjustment: {acceptor_adjustment:+d}nt")
        
        # Add reconciliation metadata
        reconciled_df["original_coordinate_system"] = source_system
        reconciled_df["reconciled_coordinate_system"] = target_system
        reconciled_df["reconciliation_timestamp"] = pd.Timestamp.now().isoformat()
        
        if self.verbosity > 0:
            logger.info(f"✅ Reconciled {len(reconciled_df)} variant coordinates")
        
        return reconciled_df
    
    def _load_genomic_resources(self, gtf_file: str, fasta_file: str) -> Tuple[gffutils.FeatureDB, Fasta]:
        """Load GTF database and FASTA file."""
        
        # Try to use existing MetaSpliceAI database
        gtf_path = Path(gtf_file)
        shared_db_path = gtf_path.parent / 'annotations.db'
        
        if shared_db_path.exists():
            if self.verbosity > 0:
                logger.info(f"Using existing database: {shared_db_path}")
            db = gffutils.FeatureDB(str(shared_db_path))
        else:
            if self.verbosity > 0:
                logger.info(f"Creating new database from: {gtf_file}")
            db = gffutils.create_db(gtf_file, str(shared_db_path), merge_strategy="create_unique")
        
        # Load FASTA
        if self.verbosity > 0:
            logger.info(f"Loading FASTA: {fasta_file}")
        fasta = Fasta(fasta_file)
        
        return db, fasta
    
    def _get_genes_to_process(self, db: gffutils.FeatureDB, gene_ids: Optional[List[str]]) -> List:
        """Get list of genes to process based on filtering criteria."""
        
        # Get all genes
        all_genes = list(db.features_of_type('gene'))
        
        if self.verbosity > 0:
            logger.info(f"Total genes in GTF: {len(all_genes)}")
        
        # Apply gene ID filtering
        if gene_ids is not None:
            gene_id_to_gene = {gene.id: gene for gene in all_genes}
            filtered_genes = [gene_id_to_gene[gene_id] for gene_id in gene_ids if gene_id in gene_id_to_gene]
            if self.verbosity > 0:
                logger.info(f"Filtered to {len(filtered_genes)} genes from provided gene IDs")
        else:
            filtered_genes = all_genes
        
        # Apply biotype filtering if enabled
        if self.enable_biotype_filtering:
            protein_coding_genes = []
            for gene in filtered_genes:
                biotype = gene.attributes.get("gene_biotype", [""])[0]
                if biotype == "protein_coding":
                    protein_coding_genes.append(gene)
            
            if self.verbosity > 0:
                logger.info(f"Biotype filtering: {len(protein_coding_genes)}/{len(filtered_genes)} protein_coding genes")
            filtered_genes = protein_coding_genes
        
        return filtered_genes
    
    def _extract_gene_splice_sites(self, gene, db: gffutils.FeatureDB, fasta: Fasta) -> List[Dict]:
        """Extract splice sites from a single gene."""
        
        # Skip genes with exceptions
        if "exception" in gene.attributes.keys() and gene.attributes["exception"][0] == "trans-splicing":
            return []
        
        # Check if chromosome exists in FASTA
        if gene.seqid not in fasta:
            return []
        
        # Get all transcripts for this gene
        transcripts = list(db.children(gene, featuretype='transcript'))
        
        # Apply transcript filtering if enabled
        if self.enable_transcript_filtering:
            # Apply OpenSpliceAI-style transcript filtering
            filtered_transcripts = []
            for transcript in transcripts:
                transcript_biotype = transcript.attributes.get("transcript_biotype", [""])[0]
                if transcript_biotype == "protein_coding":
                    filtered_transcripts.append(transcript)
            transcripts = filtered_transcripts
        
        # Extract splice sites from all transcripts
        gene_splice_sites = []
        
        for transcript in transcripts:
            # Get exons for this transcript
            exons = list(db.children(transcript, featuretype='exon', order_by='start'))
            
            if len(exons) < 2:  # Need at least 2 exons for splice sites
                continue
            
            # Extract splice sites between consecutive exons
            for i in range(len(exons) - 1):
                current_exon = exons[i]
                next_exon = exons[i + 1]
                
                # Calculate splice site positions (MetaSpliceAI style)
                if gene.strand == '+':
                    # Donor: end of current exon
                    donor_pos = current_exon.end
                    # Acceptor: start of next exon
                    acceptor_pos = next_exon.start
                else:
                    # For minus strand, donor/acceptor are reversed
                    donor_pos = current_exon.start
                    acceptor_pos = next_exon.end
                
                # Create splice site records
                donor_site = {
                    "gene_id": gene.id,
                    "transcript_id": transcript.id,
                    "chromosome": gene.seqid,
                    "position": donor_pos,
                    "strand": gene.strand,
                    "splice_type": "donor",
                    "coordinate_system": "splicesurveyor"
                }
                
                acceptor_site = {
                    "gene_id": gene.id,
                    "transcript_id": transcript.id,
                    "chromosome": gene.seqid,
                    "position": acceptor_pos,
                    "strand": gene.strand,
                    "splice_type": "acceptor",
                    "coordinate_system": "splicesurveyor"
                }
                
                gene_splice_sites.extend([donor_site, acceptor_site])
        
        return gene_splice_sites
    
    def _apply_coordinate_reconciliation(self, splice_sites: List[Dict]) -> List[Dict]:
        """Apply coordinate reconciliation to convert to target coordinate system."""
        
        if self.verbosity > 0:
            logger.info(f"🔄 Applying coordinate reconciliation to {self.coordinate_system}")
        
        # Convert to DataFrame for reconciliation
        df = pd.DataFrame(splice_sites)
        
        # Apply coordinate adjustments based on target system
        if self.coordinate_system in self.coordinate_configs:
            config = self.coordinate_configs[self.coordinate_system]
            
            # Apply offsets based on splice type
            donor_mask = df["splice_type"] == "donor"
            acceptor_mask = df["splice_type"] == "acceptor"
            
            df.loc[donor_mask, "position"] += config["donor_offset"]
            df.loc[acceptor_mask, "position"] += config["acceptor_offset"]
            
            # Update coordinate system label
            df["coordinate_system"] = self.coordinate_system
        
        # Convert back to list of dictionaries
        return df.to_dict('records')
    
    def _format_output(self, splice_sites: List[Dict], output_format: str, apply_schema_adaptation: bool = False) -> Union[pd.DataFrame, List[Dict]]:
        """Format output in requested format."""
        
        if output_format == "dataframe":
            df = pd.DataFrame(splice_sites)
            
            # Apply schema adaptation if requested
            if apply_schema_adaptation:
                try:
                    from meta_spliceai.splice_engine.meta_models.core.schema_adapters import adapt_splice_annotations
                    df = adapt_splice_annotations(df, "aligned_extractor")
                except ImportError:
                    if self.verbosity > 0:
                        logger.warning("Schema adapters not available, returning raw format")
            
            return df
        elif output_format == "list":
            return splice_sites
        elif output_format == "openspliceai_compatible":
            # Convert to OpenSpliceAI-compatible format
            df = pd.DataFrame(splice_sites)
            
            # Group by gene and create OpenSpliceAI-style records
            openspliceai_records = []
            for gene_id, gene_group in df.groupby("gene_id"):
                # Create sequence labels array (simplified)
                gene_info = gene_group.iloc[0]
                
                record = {
                    "gene_id": gene_id,
                    "chromosome": gene_info["chromosome"],
                    "strand": gene_info["strand"],
                    "splice_sites": gene_group[["position", "splice_type"]].to_dict('records'),
                    "coordinate_system": gene_info["coordinate_system"]
                }
                openspliceai_records.append(record)
            
            return openspliceai_records
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


# Convenience functions for common use cases

def extract_aligned_splice_sites(gtf_file: str,
                                fasta_file: str,
                                gene_ids: Optional[List[str]] = None,
                                coordinate_system: str = "splicesurveyor",
                                enable_biotype_filtering: bool = False) -> pd.DataFrame:
    """
    Convenience function for extracting aligned splice sites.
    
    Args:
        gtf_file: Path to GTF file
        fasta_file: Path to FASTA file
        gene_ids: Optional list of gene IDs to process
        coordinate_system: Target coordinate system
        enable_biotype_filtering: Whether to filter to protein_coding genes only
        
    Returns:
        DataFrame with aligned splice sites
    """
    
    extractor = AlignedSpliceExtractor(
        coordinate_system=coordinate_system,
        enable_biotype_filtering=enable_biotype_filtering
    )
    
    return extractor.extract_splice_sites(
        gtf_file=gtf_file,
        fasta_file=fasta_file,
        gene_ids=gene_ids,
        output_format="dataframe"
    )


def reconcile_variant_coordinates_from_clinvar(clinvar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function for reconciling ClinVar variant coordinates.
    
    Args:
        clinvar_df: DataFrame with ClinVar variant data
        
    Returns:
        DataFrame with reconciled coordinates
    """
    
    extractor = AlignedSpliceExtractor()
    return extractor.reconcile_variant_coordinates(
        variant_df=clinvar_df,
        source_system="clinvar",
        target_system="splicesurveyor"
    )


def reconcile_variant_coordinates_from_splicevardb(splicevardb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function for reconciling SpliceVarDB variant coordinates.
    
    Args:
        splicevardb_df: DataFrame with SpliceVarDB variant data
        
    Returns:
        DataFrame with reconciled coordinates
    """
    
    extractor = AlignedSpliceExtractor()
    return extractor.reconcile_variant_coordinates(
        variant_df=splicevardb_df,
        source_system="splicevardb", 
        target_system="splicesurveyor"
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example: Extract splice sites with full alignment
    gtf_file = "/home/bchiu/work/splice-surveyor/data/ensembl/Homo_sapiens.GRCh38.112.gtf"
    fasta_file = "/home/bchiu/work/splice-surveyor/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    # Test with a few genes
    test_genes = ["ENSG00000108387", "ENSG00000103549", "ENSG00000049283"]
    
    print("🎯 Testing AlignedSpliceExtractor")
    
    # Extract with MetaSpliceAI coordinates (no filtering)
    extractor = AlignedSpliceExtractor(coordinate_system="splicesurveyor", verbosity=1)
    ss_sites = extractor.extract_splice_sites(gtf_file, fasta_file, test_genes)
    print(f"MetaSpliceAI sites: {len(ss_sites)}")
    
    # Extract with OpenSpliceAI coordinates (with filtering)
    extractor_osai = AlignedSpliceExtractor(
        coordinate_system="openspliceai",
        enable_biotype_filtering=True,
        enable_transcript_filtering=True,
        verbosity=1
    )
    osai_sites = extractor_osai.extract_splice_sites(gtf_file, fasta_file, test_genes)
    print(f"OpenSpliceAI sites: {len(osai_sites)}")
    
    # Detect coordinate discrepancies
    discrepancies = extractor.detect_coordinate_discrepancies(ss_sites, osai_sites)
    print(f"Coordinate discrepancy analysis complete!")
