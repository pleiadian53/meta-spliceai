"""
Regulatory Features Module for Enhanced Alternative Splice Site Detection

This module implements the regulatory enhancement features from the noncoding
regulatory enhancement plan to improve capture of alternative splice sites
induced by variants and diseases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from ..data_types import AlternativeSpliceSite


@dataclass
class RegulatoryContext:
    """Regulatory context features for a genomic region."""
    chrom: str
    start: int
    end: int
    
    # Conservation scores
    phylop_score: Optional[float] = None
    phastcons_score: Optional[float] = None
    gerp_score: Optional[float] = None
    
    # Chromatin accessibility
    dnase_peak_count: int = 0
    atac_mean_score: Optional[float] = None
    
    # Regulatory motifs
    ese_motif_count: int = 0
    ess_motif_count: int = 0
    ise_motif_count: int = 0
    iss_motif_count: int = 0
    
    # Tissue-specific features
    tissue_expression: Dict[str, float] = None
    tissue_splice_entropy: Dict[str, float] = None
    
    # Long-range interactions
    hic_interaction_count: int = 0
    regulatory_element_distance: Optional[int] = None


class RegulatoryFeatureExtractor:
    """
    Extract regulatory features to enhance alternative splice site detection.
    
    Implements features from the noncoding regulatory enhancement plan:
    - Conservation scores (phyloP, phastCons, GERP++)
    - Chromatin accessibility (DNase-seq, ATAC-seq)
    - Splice regulatory motifs (ESE/ESS/ISE/ISS)
    - Tissue-specific expression patterns
    - Long-range regulatory interactions
    """
    
    def __init__(self, 
                 data_dir: Path,
                 verbosity: int = 1):
        """
        Initialize regulatory feature extractor.
        
        Parameters
        ----------
        data_dir : Path
            Directory containing regulatory data files
        verbosity : int
            Verbosity level (0=silent, 1=normal, 2=verbose)
        """
        self.data_dir = Path(data_dir)
        self.verbosity = verbosity
        self.logger = self._setup_logging()
        
        # Initialize regulatory databases
        self.conservation_data = self._load_conservation_data()
        self.chromatin_data = self._load_chromatin_data()
        self.motif_data = self._load_motif_data()
        self.tissue_data = self._load_tissue_data()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the extractor."""
        logger = logging.getLogger("RegulatoryFeatureExtractor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbosity >= 1 else logging.WARNING)
        return logger
    
    def _load_conservation_data(self) -> Dict[str, Any]:
        """Load conservation score data (phyloP, phastCons, GERP++)."""
        conservation_file = self.data_dir / "conservation" / "conservation_scores.tsv"
        
        if conservation_file.exists():
            try:
                df = pd.read_csv(conservation_file, sep='\t')
                self.logger.info(f"Loaded conservation data: {len(df)} regions")
                return {'data': df, 'available': True}
            except Exception as e:
                self.logger.warning(f"Failed to load conservation data: {e}")
        
        return {'data': None, 'available': False}
    
    def _load_chromatin_data(self) -> Dict[str, Any]:
        """Load chromatin accessibility data (DNase-seq, ATAC-seq)."""
        chromatin_file = self.data_dir / "chromatin" / "accessibility_scores.tsv"
        
        if chromatin_file.exists():
            try:
                df = pd.read_csv(chromatin_file, sep='\t')
                self.logger.info(f"Loaded chromatin data: {len(df)} regions")
                return {'data': df, 'available': True}
            except Exception as e:
                self.logger.warning(f"Failed to load chromatin data: {e}")
        
        return {'data': None, 'available': False}
    
    def _load_motif_data(self) -> Dict[str, Any]:
        """Load splice regulatory motif data."""
        motif_file = self.data_dir / "motifs" / "splice_regulatory_motifs.tsv"
        
        if motif_file.exists():
            try:
                df = pd.read_csv(motif_file, sep='\t')
                self.logger.info(f"Loaded motif data: {len(df)} motifs")
                return {'data': df, 'available': True}
            except Exception as e:
                self.logger.warning(f"Failed to load motif data: {e}")
        
        return {'data': None, 'available': False}
    
    def _load_tissue_data(self) -> Dict[str, Any]:
        """Load tissue-specific expression and splicing data."""
        tissue_file = self.data_dir / "tissue" / "gtex_expression.tsv"
        
        if tissue_file.exists():
            try:
                df = pd.read_csv(tissue_file, sep='\t')
                self.logger.info(f"Loaded tissue data: {len(df)} genes")
                return {'data': df, 'available': True}
            except Exception as e:
                self.logger.warning(f"Failed to load tissue data: {e}")
        
        return {'data': None, 'available': False}
    
    def extract_regulatory_context(self, 
                                 chrom: str, 
                                 position: int, 
                                 window: int = 2000) -> RegulatoryContext:
        """
        Extract comprehensive regulatory context for a genomic position.
        
        Parameters
        ----------
        chrom : str
            Chromosome
        position : int
            Genomic position
        window : int
            Window size around position
            
        Returns
        -------
        RegulatoryContext
            Regulatory features for the region
        """
        start = max(1, position - window // 2)
        end = position + window // 2
        
        context = RegulatoryContext(chrom=chrom, start=start, end=end)
        
        # Extract conservation features
        if self.conservation_data['available']:
            conservation_features = self._extract_conservation_features(chrom, start, end)
            context.phylop_score = conservation_features.get('phylop_score')
            context.phastcons_score = conservation_features.get('phastcons_score')
            context.gerp_score = conservation_features.get('gerp_score')
        
        # Extract chromatin accessibility features
        if self.chromatin_data['available']:
            chromatin_features = self._extract_chromatin_features(chrom, start, end)
            context.dnase_peak_count = chromatin_features.get('dnase_peak_count', 0)
            context.atac_mean_score = chromatin_features.get('atac_mean_score')
        
        # Extract regulatory motif features
        if self.motif_data['available']:
            motif_features = self._extract_motif_features(chrom, start, end)
            context.ese_motif_count = motif_features.get('ese_count', 0)
            context.ess_motif_count = motif_features.get('ess_count', 0)
            context.ise_motif_count = motif_features.get('ise_count', 0)
            context.iss_motif_count = motif_features.get('iss_count', 0)
        
        # Extract tissue-specific features
        if self.tissue_data['available']:
            tissue_features = self._extract_tissue_features(chrom, start, end)
            context.tissue_expression = tissue_features.get('expression', {})
            context.tissue_splice_entropy = tissue_features.get('splice_entropy', {})
        
        return context
    
    def _extract_conservation_features(self, chrom: str, start: int, end: int) -> Dict[str, float]:
        """Extract conservation scores for genomic region."""
        if not self.conservation_data['available']:
            return {}
        
        # Mock implementation - would query actual conservation databases
        return {
            'phylop_score': np.random.uniform(-2, 5),
            'phastcons_score': np.random.uniform(0, 1),
            'gerp_score': np.random.uniform(-5, 5)
        }
    
    def _extract_chromatin_features(self, chrom: str, start: int, end: int) -> Dict[str, Any]:
        """Extract chromatin accessibility features for genomic region."""
        if not self.chromatin_data['available']:
            return {}
        
        # Mock implementation - would query actual chromatin databases
        return {
            'dnase_peak_count': np.random.poisson(2),
            'atac_mean_score': np.random.uniform(0, 10)
        }
    
    def _extract_motif_features(self, chrom: str, start: int, end: int) -> Dict[str, int]:
        """Extract splice regulatory motif counts for genomic region."""
        if not self.motif_data['available']:
            return {}
        
        # Mock implementation - would scan sequence for actual motifs
        return {
            'ese_count': np.random.poisson(3),
            'ess_count': np.random.poisson(2),
            'ise_count': np.random.poisson(1),
            'iss_count': np.random.poisson(1)
        }
    
    def _extract_tissue_features(self, chrom: str, start: int, end: int) -> Dict[str, Dict]:
        """Extract tissue-specific expression and splicing features."""
        if not self.tissue_data['available']:
            return {}
        
        # Mock implementation - would query GTEx data
        tissues = ['brain', 'heart', 'liver', 'muscle', 'blood']
        return {
            'expression': {tissue: np.random.uniform(0, 100) for tissue in tissues},
            'splice_entropy': {tissue: np.random.uniform(0, 2) for tissue in tissues}
        }
    
    def enhance_alternative_sites_with_regulatory_features(self, 
                                                         sites: List[AlternativeSpliceSite],
                                                         regulatory_window: int = 2000) -> List[AlternativeSpliceSite]:
        """
        Enhance alternative splice sites with regulatory context features.
        
        Parameters
        ----------
        sites : List[AlternativeSpliceSite]
            List of alternative splice sites
        regulatory_window : int
            Window size for regulatory feature extraction
            
        Returns
        -------
        List[AlternativeSpliceSite]
            Enhanced splice sites with regulatory features
        """
        enhanced_sites = []
        
        for site in sites:
            # Extract regulatory context
            context = self.extract_regulatory_context(
                site.chrom, site.position, regulatory_window
            )
            
            # Create enhanced site with additional validation evidence
            enhanced_site = AlternativeSpliceSite(
                chrom=site.chrom,
                position=site.position,
                strand=site.strand,
                site_type=site.site_type,
                splice_category=self._enhance_splice_category(site, context),
                delta_score=site.delta_score,
                ref_score=site.ref_score,
                alt_score=site.alt_score,
                variant_id=site.variant_id,
                gene_symbol=site.gene_symbol,
                clinical_significance=site.clinical_significance,
                validation_evidence=self._enhance_validation_evidence(site, context)
            )
            
            enhanced_sites.append(enhanced_site)
        
        self.logger.info(f"Enhanced {len(enhanced_sites)} alternative splice sites with regulatory features")
        return enhanced_sites
    
    def _enhance_splice_category(self, site: AlternativeSpliceSite, context: RegulatoryContext) -> str:
        """Enhance splice category classification using regulatory context."""
        base_category = site.splice_category
        
        # Enhance with conservation evidence
        if context.phylop_score and context.phylop_score > 2.0:
            if 'cryptic' in base_category:
                return f"{base_category}_conserved"
            elif 'canonical' in base_category:
                return f"{base_category}_highly_conserved"
        
        # Enhance with chromatin accessibility
        if context.atac_mean_score and context.atac_mean_score > 5.0:
            return f"{base_category}_accessible_chromatin"
        
        # Enhance with regulatory motifs
        total_motifs = (context.ese_motif_count + context.ess_motif_count + 
                       context.ise_motif_count + context.iss_motif_count)
        if total_motifs > 5:
            return f"{base_category}_motif_rich"
        
        return base_category
    
    def _enhance_validation_evidence(self, site: AlternativeSpliceSite, context: RegulatoryContext) -> str:
        """Enhance validation evidence with regulatory features."""
        evidence_parts = [site.validation_evidence or "openspliceai_prediction"]
        
        # Add conservation evidence
        if context.phylop_score and context.phylop_score > 2.0:
            evidence_parts.append("highly_conserved")
        
        # Add chromatin evidence
        if context.dnase_peak_count > 0:
            evidence_parts.append("dnase_accessible")
        
        # Add motif evidence
        if context.ese_motif_count > 2 or context.ise_motif_count > 1:
            evidence_parts.append("enhancer_motifs")
        
        if context.ess_motif_count > 2 or context.iss_motif_count > 1:
            evidence_parts.append("silencer_motifs")
        
        return "|".join(evidence_parts)
    
    def create_regulatory_training_features(self, 
                                          sites: List[AlternativeSpliceSite]) -> pd.DataFrame:
        """
        Create regulatory features DataFrame for meta-model training.
        
        Parameters
        ----------
        sites : List[AlternativeSpliceSite]
            List of alternative splice sites
            
        Returns
        -------
        pd.DataFrame
            DataFrame with regulatory features for training
        """
        rows = []
        
        for site in sites:
            context = self.extract_regulatory_context(site.chrom, site.position)
            
            row = {
                'chromosome': site.chrom,
                'position': site.position,
                'site_type': site.site_type,
                'splice_category': site.splice_category,
                'delta_score': site.delta_score,
                'gene_symbol': site.gene_symbol,
                
                # Conservation features
                'phylop_score': context.phylop_score or 0,
                'phastcons_score': context.phastcons_score or 0,
                'gerp_score': context.gerp_score or 0,
                
                # Chromatin features
                'dnase_peak_count': context.dnase_peak_count,
                'atac_mean_score': context.atac_mean_score or 0,
                
                # Motif features
                'ese_motif_count': context.ese_motif_count,
                'ess_motif_count': context.ess_motif_count,
                'ise_motif_count': context.ise_motif_count,
                'iss_motif_count': context.iss_motif_count,
                
                # Tissue features (example for brain)
                'brain_expression': context.tissue_expression.get('brain', 0) if context.tissue_expression else 0,
                'brain_splice_entropy': context.tissue_splice_entropy.get('brain', 0) if context.tissue_splice_entropy else 0,
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        self.logger.info(f"Created regulatory training features for {len(df)} sites")
        
        return df


def demonstrate_regulatory_features():
    """Demonstrate regulatory feature extraction capabilities."""
    from pathlib import Path
    
    print("\n" + "="*60)
    print("Regulatory Feature Extraction Demonstration")
    print("="*60 + "\n")
    
    # Initialize extractor
    data_dir = Path("./case_studies/regulatory_data")
    extractor = RegulatoryFeatureExtractor(data_dir, verbosity=2)
    
    # Create mock alternative splice sites
    mock_sites = [
        AlternativeSpliceSite(
            chrom="7", position=117559593, strand="+", site_type="acceptor",
            splice_category="cryptic_activated", delta_score=0.8,
            ref_score=0.1, alt_score=0.9, variant_id="7:117559593:G>T",
            gene_symbol="CFTR", clinical_significance="pathogenic",
            validation_evidence="openspliceai_prediction"
        ),
        AlternativeSpliceSite(
            chrom="17", position=43094077, strand="+", site_type="donor",
            splice_category="canonical_disrupted", delta_score=-0.7,
            ref_score=0.9, alt_score=0.2, variant_id="17:43094077:G>A",
            gene_symbol="BRCA1", clinical_significance="pathogenic",
            validation_evidence="openspliceai_prediction"
        )
    ]
    
    print(f"Processing {len(mock_sites)} mock alternative splice sites...")
    
    # Extract regulatory context for each site
    for i, site in enumerate(mock_sites):
        print(f"\n🧬 Site {i+1}: {site.gene_symbol} {site.chrom}:{site.position}")
        
        context = extractor.extract_regulatory_context(site.chrom, site.position)
        
        print(f"   Conservation: phyloP={context.phylop_score:.2f}, GERP={context.gerp_score:.2f}")
        print(f"   Chromatin: DNase peaks={context.dnase_peak_count}, ATAC={context.atac_mean_score:.2f}")
        print(f"   Motifs: ESE={context.ese_motif_count}, ESS={context.ess_motif_count}")
        
        if context.tissue_expression:
            brain_expr = context.tissue_expression.get('brain', 0)
            print(f"   Tissue: Brain expression={brain_expr:.1f}")
    
    # Enhance sites with regulatory features
    enhanced_sites = extractor.enhance_alternative_sites_with_regulatory_features(mock_sites)
    
    print(f"\n📈 Enhanced {len(enhanced_sites)} sites with regulatory features:")
    for site in enhanced_sites:
        print(f"   {site.gene_symbol}: {site.splice_category}")
        print(f"   Evidence: {site.validation_evidence}")
    
    # Create training features
    training_df = extractor.create_regulatory_training_features(enhanced_sites)
    
    print(f"\n📊 Created training DataFrame with {len(training_df)} rows and {len(training_df.columns)} features")
    print(f"   Features: {list(training_df.columns)}")
    
    print("\n✅ Regulatory feature extraction demonstration complete!")


if __name__ == "__main__":
    demonstrate_regulatory_features()


