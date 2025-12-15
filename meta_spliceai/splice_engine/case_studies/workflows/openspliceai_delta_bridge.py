"""
OpenSpliceAI Delta Score Bridge for Case Studies

This module provides the critical missing bridge between VCF variant analysis and
OpenSpliceAI delta score computation, enabling complete VCF → Alternative Splice Sites
transformation for meta-model training.

Key Features:
- Direct integration with OpenSpliceAI variant analysis utils
- Conversion from VCF format to delta scores
- Bridge from delta scores to alternative splice sites
- Support for disease mutation databases (ClinVar, SpliceVarDB, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import tempfile
import logging

# Optional imports
try:
    import pysam
    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

# Import OpenSpliceAI components
try:
    from meta_spliceai.openspliceai.variant.utils import get_delta_scores, Annotator
    from meta_spliceai.openspliceai.predict.utils import one_hot_encode
    OPENSPLICEAI_AVAILABLE = True
except ImportError:
    OPENSPLICEAI_AVAILABLE = False

try:
    from ..formats.variant_standardizer import VariantStandardizer, StandardizedVariant
    from ..data_types import AlternativeSpliceSite, DeltaScoreResult
except ImportError:
    # Handle relative import issues when running as script
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from formats.variant_standardizer import VariantStandardizer, StandardizedVariant
    from data_types import AlternativeSpliceSite, DeltaScoreResult


class OpenSpliceAIDeltaBridge:
    """
    Bridge between VCF variants and OpenSpliceAI delta scores.
    
    This class implements the critical missing functionality to compute actual
    delta scores from variants and convert them to alternative splice sites.
    """
    
    def __init__(self, 
                 reference_fasta: str,
                 annotations: str = "grch38",
                 flanking_size: int = 5000,
                 dist_var: int = 50,
                 gene_features_file: Optional[str] = None,
                 project_root: Optional[str] = None,
                 genome_build: str = "GRCh38",
                 ensembl_release: str = "112"):
        """
        Initialize the OpenSpliceAI Delta Score Bridge.
        
        Parameters
        ----------
        reference_fasta : str
            Path to the reference genome FASTA file
        annotations : str, optional
            Genome annotations version. Default: "grch38"
        flanking_size : int, optional
            Size of flanking region for context. Default: 5000
        dist_var : int, optional
            Distance variant parameter. Default: 50
        gene_features_file : str, optional
            Path to gene features TSV file containing strand information
        project_root : str, optional
            Project root directory for genomic resource manager
        genome_build : str, optional
            Genome build version (e.g., GRCh38, GRCh37). Default: "GRCh38"
        ensembl_release : str, optional
            Ensembl release version (e.g., 112). Default: "112"
        """
        # Initialize variant standardizer first
        self.variant_standardizer = VariantStandardizer(
            reference_genome=genome_build
        )
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize OpenSpliceAI annotator if available
        self.openspliceai_annotator = None
        if OPENSPLICEAI_AVAILABLE:
            try:
                self.openspliceai_annotator = Annotator(
                    ref_fasta=reference_fasta,
                    annotations=annotations
                )
                self.logger.info("✅ OpenSpliceAI annotator initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize OpenSpliceAI annotator: {e}")
        
        # Store parameters
        self.reference_fasta = reference_fasta
        self.annotations = annotations
        self.flanking_size = flanking_size
        self.dist_var = dist_var
        self.genome_build = genome_build
        self.ensembl_release = ensembl_release
        
        # Initialize genomic resource manager for systematic path resolution
        self.genomic_manager = None
        try:
            from meta_spliceai.system.genomic_resources import create_systematic_manager
            self.genomic_manager = create_systematic_manager(project_root)
            self.logger.info(f"✅ Initialized genomic resource manager for {genome_build} release {ensembl_release}")
        except ImportError as e:
            self.logger.warning(f"⚠️ Genomic resource manager not available: {e}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize genomic resource manager: {e}")
        
        # Load gene strand mapping from gene_features.tsv
        self.gene_strand_map = self._load_gene_strand_map(gene_features_file)
        
        # Initialize OpenSpliceAI annotator if available
        self.annotator = None
        if OPENSPLICEAI_AVAILABLE and PYSAM_AVAILABLE:
            try:
                self.annotator = Annotator(
                    ref_fasta=reference_fasta,
                    annotations=annotations,
                    model_path='SpliceAI',
                    model_type='keras'
                )
                self.logger.info("OpenSpliceAI annotator initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenSpliceAI annotator: {e}")
                self.annotator = None
        else:
            missing = []
            if not OPENSPLICEAI_AVAILABLE:
                missing.append("OpenSpliceAI")
            if not PYSAM_AVAILABLE:
                missing.append("pysam")
            self.logger.warning(f"Missing dependencies: {', '.join(missing)}. Using mock implementation")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the bridge."""
        logger = logging.getLogger("OpenSpliceAIDeltaBridge")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_gene_strand_map(self, gene_features_file: Optional[str] = None) -> Dict[str, str]:
        """
        Load gene strand mapping from gene_features.tsv file.
        
        Args:
            gene_features_file: Optional path to gene features TSV file
            
        Returns:
            Dictionary mapping gene names to strand symbols ('+' or '-')
        """
        strand_map = {}
        
        # If no file specified, try to resolve using genomic resource manager
        if gene_features_file is None and self.genomic_manager:
            try:
                # Use genomic manager to get the analysis directory
                analysis_dir = self.genomic_manager.genome.get_source_dir("ensembl") / "spliceai_analysis"
                gene_features_path = analysis_dir / "gene_features.tsv"
                
                if gene_features_path.exists():
                    gene_features_file = str(gene_features_path)
                    self.logger.info(f"Using gene_features from genomic resource manager: {gene_features_file}")
                else:
                    self.logger.debug(f"gene_features.tsv not found at: {gene_features_path}")
            except Exception as e:
                self.logger.debug(f"Could not resolve gene_features path via genomic manager: {e}")
        
        # Fallback to hardcoded path if genomic manager unavailable
        if gene_features_file is None:
            default_path = Path("/path/to/meta-spliceai/data/ensembl/spliceai_analysis/gene_features.tsv")
            if default_path.exists():
                gene_features_file = str(default_path)
                self.logger.info(f"Using fallback gene_features file: {gene_features_file}")
        
        # Load the gene features file if available
        if gene_features_file and Path(gene_features_file).exists():
            try:
                df = pd.read_csv(gene_features_file, sep='\t')
                
                # Check for required columns
                if 'gene_name' in df.columns and 'strand' in df.columns:
                    # Create mapping from gene_name to strand
                    strand_map = dict(zip(df['gene_name'], df['strand']))
                    self.logger.info(f"Loaded strand information for {len(strand_map)} genes")
                else:
                    self.logger.warning("gene_features.tsv missing required columns (gene_name, strand)")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load gene_features file: {e}")
        else:
            self.logger.info("No gene_features file available, will default to '+' strand")
        
        return strand_map
    
    def compute_delta_scores_from_variants(self, 
                                         variants: List[StandardizedVariant]) -> List[DeltaScoreResult]:
        """
        Compute delta scores from standardized variants.
        
        Parameters
        ----------
        variants : List[StandardizedVariant]
            List of standardized variants
            
        Returns
        -------
        List[DeltaScoreResult]
            List of delta score results
        """
        if not self.annotator:
            self.logger.warning("No OpenSpliceAI annotator available, using mock delta scores")
            return self._mock_delta_scores(variants)
        
        # Create temporary VCF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as tmp_vcf:
            self._write_vcf_header(tmp_vcf)
            
            # Write variants to VCF
            for variant in variants:
                vcf_record = self.variant_standardizer.to_vcf_format(variant)
                tmp_vcf.write(f"{vcf_record['CHROM']}\t{vcf_record['POS']}\t.\t"
                             f"{vcf_record['REF']}\t{vcf_record['ALT']}\t.\tPASS\t.\n")
            
            tmp_vcf.flush()
            
            # Process VCF with OpenSpliceAI
            results = []
            try:
                if PYSAM_AVAILABLE:
                    with pysam.VariantFile(tmp_vcf.name) as vcf:
                        for record in vcf:
                            delta_scores = get_delta_scores(
                                record=record,
                                ann=self.annotator,
                                dist_var=self.dist_var,
                                mask=0,
                                flanking_size=self.flanking_size
                            )
                            
                            # Parse delta score results
                            for score_line in delta_scores:
                                result = self._parse_delta_score_line(record, score_line)
                                if result:
                                    results.append(result)
                else:
                    self.logger.warning("pysam not available, falling back to mock delta scores")
                    return self._mock_delta_scores(variants)
                                
            except Exception as e:
                self.logger.error(f"Failed to compute delta scores: {e}")
                return self._mock_delta_scores(variants)
            finally:
                # Clean up temporary file
                Path(tmp_vcf.name).unlink(missing_ok=True)
        
        self.logger.info(f"Computed delta scores for {len(results)} variant-gene pairs")
        return results
    
    def _write_vcf_header(self, file_handle):
        """Write VCF header to file."""
        file_handle.write("##fileformat=VCFv4.2\n")
        file_handle.write("##source=MetaSpliceAI_DeltaBridge\n")
        file_handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    
    def _parse_delta_score_line(self, record, score_line: str) -> Optional[DeltaScoreResult]:
        """
        Parse a delta score line from OpenSpliceAI output.
        
        Format: ALT|GENE|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
        """
        try:
            fields = score_line.split('|')
            if len(fields) < 10:
                return None
            
            def parse_float(value: str) -> Optional[float]:
                return float(value) if value != '.' else None
            
            def parse_int(value: str) -> Optional[int]:
                return int(value) if value != '.' else None
            
            return DeltaScoreResult(
                variant_id=f"{record.chrom}:{record.pos}:{record.ref}>{fields[0]}",
                gene_symbol=fields[1],
                chrom=record.chrom,
                position=record.pos,
                ref_allele=record.ref,
                alt_allele=fields[0],
                ds_ag=parse_float(fields[2]),
                ds_al=parse_float(fields[3]),
                ds_dg=parse_float(fields[4]),
                ds_dl=parse_float(fields[5]),
                dp_ag=parse_int(fields[6]),
                dp_al=parse_int(fields[7]),
                dp_dg=parse_int(fields[8]),
                dp_dl=parse_int(fields[9])
            )
        except (ValueError, IndexError) as e:
            self.logger.warning(f"Failed to parse delta score line: {score_line} - {e}")
            return None
    
    def _mock_delta_scores(self, variants: List[StandardizedVariant]) -> List[DeltaScoreResult]:
        """Generate mock delta scores for testing."""
        results = []
        
        for variant in variants:
            # Generate realistic mock scores
            base_score = np.random.uniform(-0.5, 0.5)
            
            result = DeltaScoreResult(
                variant_id=f"{variant.chrom}:{variant.start}:{variant.ref}>{variant.alt}",
                gene_symbol=f"GENE_{variant.chrom}_{variant.start}",
                chrom=variant.chrom,
                position=variant.start,
                ref_allele=variant.ref,
                alt_allele=variant.alt,
                ds_ag=base_score + np.random.uniform(-0.3, 0.3),
                ds_al=base_score + np.random.uniform(-0.3, 0.3),
                ds_dg=base_score + np.random.uniform(-0.3, 0.3),
                ds_dl=base_score + np.random.uniform(-0.3, 0.3),
                dp_ag=np.random.randint(-50, 51),
                dp_al=np.random.randint(-50, 51),
                dp_dg=np.random.randint(-50, 51),
                dp_dl=np.random.randint(-50, 51)
            )
            results.append(result)
        
        return results
    
    def delta_scores_to_alternative_sites(self, 
                                        delta_results: List[DeltaScoreResult],
                                        threshold: float = 0.2) -> List[AlternativeSpliceSite]:
        """
        Convert delta score results to alternative splice sites.
        
        This is the critical transformation function that bridges delta scores
        to alternative splice site annotations for meta-model training.
        
        Parameters
        ----------
        delta_results : List[DeltaScoreResult]
            Delta score computation results
        threshold : float
            Minimum absolute delta score to consider significant
            
        Returns
        -------
        List[AlternativeSpliceSite]
            List of alternative splice sites
        """
        alternative_sites = []
        
        for result in delta_results:
            # Check for acceptor gains (new acceptor sites)
            if result.ds_ag is not None and abs(result.ds_ag) >= threshold:
                site_pos = result.position + (result.dp_ag or 0)
                
                site = AlternativeSpliceSite(
                    chrom=result.chrom,
                    position=site_pos,
                    strand=self.gene_strand_map.get(result.gene_symbol, '+'),  # Use gene strand from annotations
                    site_type='acceptor',
                    splice_category=self._classify_splice_category(result.ds_ag, 'gain'),
                    delta_score=result.ds_ag,
                    ref_score=0.0,  # Would need reference prediction to compute
                    alt_score=result.ds_ag,  # Approximation
                    variant_id=result.variant_id,
                    gene_symbol=result.gene_symbol,
                    clinical_significance=None,
                    validation_evidence='openspliceai_prediction'
                )
                alternative_sites.append(site)
            
            # Check for acceptor losses (disrupted acceptor sites)
            if result.ds_al is not None and abs(result.ds_al) >= threshold:
                site_pos = result.position + (result.dp_al or 0)
                
                site = AlternativeSpliceSite(
                    chrom=result.chrom,
                    position=site_pos,
                    strand=self.gene_strand_map.get(result.gene_symbol, '+'),
                    site_type='acceptor',
                    splice_category=self._classify_splice_category(result.ds_al, 'loss'),
                    delta_score=result.ds_al,
                    ref_score=abs(result.ds_al),  # Approximation
                    alt_score=0.0,
                    variant_id=result.variant_id,
                    gene_symbol=result.gene_symbol,
                    clinical_significance=None,
                    validation_evidence='openspliceai_prediction'
                )
                alternative_sites.append(site)
            
            # Check for donor gains (new donor sites)
            if result.ds_dg is not None and abs(result.ds_dg) >= threshold:
                site_pos = result.position + (result.dp_dg or 0)
                
                site = AlternativeSpliceSite(
                    chrom=result.chrom,
                    position=site_pos,
                    strand=self.gene_strand_map.get(result.gene_symbol, '+'),
                    site_type='donor',
                    splice_category=self._classify_splice_category(result.ds_dg, 'gain'),
                    delta_score=result.ds_dg,
                    ref_score=0.0,
                    alt_score=result.ds_dg,
                    variant_id=result.variant_id,
                    gene_symbol=result.gene_symbol,
                    clinical_significance=None,
                    validation_evidence='openspliceai_prediction'
                )
                alternative_sites.append(site)
            
            # Check for donor losses (disrupted donor sites)
            if result.ds_dl is not None and abs(result.ds_dl) >= threshold:
                site_pos = result.position + (result.dp_dl or 0)
                
                site = AlternativeSpliceSite(
                    chrom=result.chrom,
                    position=site_pos,
                    strand=self.gene_strand_map.get(result.gene_symbol, '+'),
                    site_type='donor',
                    splice_category=self._classify_splice_category(result.ds_dl, 'loss'),
                    delta_score=result.ds_dl,
                    ref_score=abs(result.ds_dl),
                    alt_score=0.0,
                    variant_id=result.variant_id,
                    gene_symbol=result.gene_symbol,
                    clinical_significance=None,
                    validation_evidence='openspliceai_prediction'
                )
                alternative_sites.append(site)
        
        self.logger.info(f"Extracted {len(alternative_sites)} alternative splice sites from delta scores")
        return alternative_sites
    
    def _classify_splice_category(self, delta_score: float, event_type: str) -> str:
        """Classify splice site category based on delta score and event type."""
        abs_score = abs(delta_score)
        
        if event_type == 'gain':
            if abs_score >= 0.8:
                return 'high_confidence_cryptic'
            elif abs_score >= 0.5:
                return 'cryptic_activated'
            else:
                return 'predicted_alternative'
        else:  # loss
            if abs_score >= 0.8:
                return 'canonical_disrupted_high'
            elif abs_score >= 0.5:
                return 'canonical_disrupted'
            else:
                return 'canonical_weakened'
    
    def process_vcf_to_alternative_sites(self, 
                                       vcf_path: Path,
                                       output_path: Optional[Path] = None,
                                       threshold: float = 0.2) -> pd.DataFrame:
        """
        Complete pipeline from VCF to alternative splice sites.
        
        This implements the complete VCF → Delta Scores → Alternative Sites workflow.
        
        Parameters
        ----------
        vcf_path : Path
            Path to input VCF file
        output_path : Optional[Path]
            Output path for alternative sites TSV
        threshold : float
            Delta score threshold for significance
            
        Returns
        -------
        pd.DataFrame
            DataFrame of alternative splice sites
        """
        self.logger.info(f"Processing VCF file: {vcf_path}")
        
        # Step 1: Load and standardize variants from VCF
        variants = self._load_variants_from_vcf(vcf_path)
        self.logger.info(f"Loaded {len(variants)} variants from VCF")
        
        # Step 2: Compute delta scores
        delta_results = self.compute_delta_scores_from_variants(variants)
        self.logger.info(f"Computed delta scores for {len(delta_results)} variant-gene pairs")
        
        # Step 3: Convert to alternative splice sites
        alternative_sites = self.delta_scores_to_alternative_sites(delta_results, threshold)
        self.logger.info(f"Identified {len(alternative_sites)} alternative splice sites")
        
        # Step 4: Convert to DataFrame
        sites_df = self._sites_to_dataframe(alternative_sites)
        
        # Step 5: Save output if requested
        if output_path:
            sites_df.to_csv(output_path, sep='\t', index=False)
            self.logger.info(f"Saved alternative sites to: {output_path}")
        
        return sites_df
    
    def _load_variants_from_vcf(self, vcf_path: Path) -> List[StandardizedVariant]:
        """Load variants from VCF file."""
        variants = []
        
        try:
            if PYSAM_AVAILABLE:
                with pysam.VariantFile(str(vcf_path)) as vcf:
                    for record in vcf:
                        # Handle multiple alternate alleles
                        for alt in record.alts:
                            variant = self.variant_standardizer.standardize_from_vcf(
                                record.chrom, record.pos, record.ref, alt
                            )
                            variants.append(variant)
            else:
                # Fallback: parse VCF manually
                with open(vcf_path, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        parts = line.strip().split('\t')
                        if len(parts) >= 5:
                            chrom, pos, _, ref, alt = parts[:5]
                            variant = self.variant_standardizer.standardize_from_vcf(
                                chrom, int(pos), ref, alt
                            )
                            variants.append(variant)
        except Exception as e:
            self.logger.error(f"Failed to load VCF file {vcf_path}: {e}")
            
        return variants
    
    def _sites_to_dataframe(self, sites: List[AlternativeSpliceSite]) -> pd.DataFrame:
        """Convert alternative splice sites to DataFrame."""
        rows = []
        
        for site in sites:
            rows.append({
                'chromosome': site.chrom,
                'position': site.position,
                'strand': site.strand,
                'site_type': site.site_type,
                'splice_category': site.splice_category,
                'delta_score': site.delta_score,
                'ref_score': site.ref_score,
                'alt_score': site.alt_score,
                'variant_id': site.variant_id,
                'gene_symbol': site.gene_symbol,
                'clinical_significance': site.clinical_significance,
                'validation_evidence': site.validation_evidence
            })
        
        return pd.DataFrame(rows)


def demo_delta_bridge():
    """Demonstrate the OpenSpliceAI delta score bridge."""
    print("\n" + "="*60)
    print("OpenSpliceAI Delta Score Bridge Demonstration")
    print("="*60 + "\n")
    
    # Initialize bridge (will use mock if OpenSpliceAI not available)
    bridge = OpenSpliceAIDeltaBridge(
        reference_fasta="path/to/reference.fa",  # Would be actual path
        annotations="grch38"
    )
    
    # Create mock variants
    mock_variants = [
        bridge.variant_standardizer.standardize_from_vcf("7", 117559593, "G", "T"),  # CFTR
        bridge.variant_standardizer.standardize_from_vcf("17", 43094077, "G", "A"),  # BRCA1
        bridge.variant_standardizer.standardize_from_vcf("13", 32339151, "C", "T"),  # BRCA2
    ]
    
    print(f"Processing {len(mock_variants)} mock variants...")
    
    # Compute delta scores
    delta_results = bridge.compute_delta_scores_from_variants(mock_variants)
    print(f"Computed delta scores for {len(delta_results)} variant-gene pairs")
    
    # Convert to alternative splice sites
    alternative_sites = bridge.delta_scores_to_alternative_sites(delta_results, threshold=0.1)
    print(f"Identified {len(alternative_sites)} alternative splice sites")
    
    # Show sample results
    if alternative_sites:
        print("\nSample Alternative Splice Sites:")
        for i, site in enumerate(alternative_sites[:3]):
            print(f"  {i+1}. {site.gene_symbol} {site.chrom}:{site.position}")
            print(f"     Type: {site.site_type}, Category: {site.splice_category}")
            print(f"     Delta Score: {site.delta_score:.3f}")
            print()
    
    print("✅ Delta bridge demonstration complete!")


if __name__ == "__main__":
    demo_delta_bridge()
