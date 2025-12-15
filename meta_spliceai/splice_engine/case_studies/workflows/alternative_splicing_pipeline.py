"""
Alternative Splicing Pipeline: VCF to Training Data Transformation

This module implements the critical missing functionality to convert
OpenSpliceAI delta scores from variant analysis into alternative splice
site annotations for meta-model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

from ..formats.variant_standardizer import VariantStandardizer
from ..data_sources.base import SpliceMutation
from ..data_types import AlternativeSpliceSite

# Import delta bridge without circular dependency
try:
    from .openspliceai_delta_bridge import OpenSpliceAIDeltaBridge
except ImportError:
    # Handle circular import - will be imported later if needed
    OpenSpliceAIDeltaBridge = None


class AlternativeSplicingPipeline:
    """
    Pipeline for converting VCF variant analysis to alternative splice site annotations.
    
    This implements the 5-stage workflow:
    1. VCF Standardization
    2. Delta Score Analysis (via OpenSpliceAI)
    3. Alternative Site Extraction
    4. Data Integration
    5. Training Data Preparation
    """
    
    def __init__(self, work_dir: Path, 
                 reference_fasta: Optional[str] = None,
                 annotations: str = "grch38"):
        """
        Initialize the alternative splicing pipeline.
        
        Parameters
        ----------
        work_dir : Path
            Working directory for outputs
        reference_fasta : Optional[str]
            Path to reference genome FASTA (required for real OpenSpliceAI analysis)
        annotations : str
            Gene annotations for OpenSpliceAI ("grch37", "grch38", or GTF path)
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.variant_standardizer = VariantStandardizer()
        
        # Initialize OpenSpliceAI delta bridge
        if reference_fasta and OpenSpliceAIDeltaBridge is not None:
            self.delta_bridge = OpenSpliceAIDeltaBridge(
                reference_fasta=reference_fasta,
                annotations=annotations
            )
        else:
            self.delta_bridge = None
        
    def extract_alternative_splice_sites_from_delta_scores(
        self,
        delta_scores_df: pd.DataFrame,
        threshold: float = 0.2,
        window_size: int = 50,
        include_regulatory_context: bool = True,
        regulatory_window: int = 2000
    ) -> List[AlternativeSpliceSite]:
        """
        Extract alternative splice sites from OpenSpliceAI delta scores.
        
        This is the CRITICAL transformation function that converts variant-level
        delta scores into splice site-level training annotations.
        
        Parameters
        ----------
        delta_scores_df : pd.DataFrame
            DataFrame with columns: chrom, position, ref_score, alt_score, delta_score
        threshold : float
            Minimum absolute delta score to consider site as alternative
        window_size : int
            Window around variant to search for splice sites
            
        Returns
        -------
        List[AlternativeSpliceSite]
            List of identified alternative splice sites
        """
        alternative_sites = []
        
        for _, row in delta_scores_df.iterrows():
            # Skip low-impact variants
            if abs(row['delta_score']) < threshold:
                continue
                
            # Identify splice site positions within window
            variant_pos = row['position']
            
            # Check for canonical splice site disruption
            if row.get('is_canonical_site', False):
                if row['delta_score'] < -threshold:
                    # Canonical site loss
                    site = AlternativeSpliceSite(
                        chrom=row['chrom'],
                        position=variant_pos,
                        strand=row.get('strand', '+'),
                        site_type=row.get('site_type', 'unknown'),
                        splice_category='canonical_disrupted',
                        delta_score=row['delta_score'],
                        ref_score=row['ref_score'],
                        alt_score=row['alt_score'],
                        variant_id=row.get('variant_id', f"{row['chrom']}:{variant_pos}"),
                        gene_symbol=row.get('gene_symbol', ''),
                        clinical_significance=row.get('clinical_significance'),
                        validation_evidence=row.get('validation_evidence')
                    )
                    alternative_sites.append(site)
            
            # Check for cryptic site activation
            if row['delta_score'] > threshold:
                # Scan for high-scoring positions that could be cryptic sites
                for offset in range(-window_size, window_size + 1):
                    if offset == 0:
                        continue
                    
                    scan_pos = variant_pos + offset
                    
                    # Check if position shows significant gain
                    if row.get(f'score_at_{offset}', 0) > threshold:
                        site_type = self._infer_site_type(row, offset)
                        
                        site = AlternativeSpliceSite(
                            chrom=row['chrom'],
                            position=scan_pos,
                            strand=row.get('strand', '+'),
                            site_type=site_type,
                            splice_category='cryptic_activated',
                            delta_score=row.get(f'delta_at_{offset}', row['delta_score']),
                            ref_score=row.get(f'ref_score_at_{offset}', 0),
                            alt_score=row.get(f'alt_score_at_{offset}', row['alt_score']),
                            variant_id=row.get('variant_id', f"{row['chrom']}:{variant_pos}"),
                            gene_symbol=row.get('gene_symbol', ''),
                            clinical_significance=row.get('clinical_significance'),
                            validation_evidence=row.get('validation_evidence')
                        )
                        alternative_sites.append(site)
        
        return alternative_sites
    
    def _infer_site_type(self, row: pd.Series, offset: int) -> str:
        """Infer whether site is donor or acceptor based on sequence context."""
        # Simplified logic - in practice would use sequence motifs
        if 'site_type' in row:
            return row['site_type']
        
        # Use offset as heuristic (donors typically upstream, acceptors downstream)
        if offset < 0:
            return 'donor'
        else:
            return 'acceptor'
    
    def process_vcf_to_alternative_sites(
        self,
        vcf_path: Path,
        gene_annotations: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Complete pipeline from VCF to alternative splice sites.
        
        Parameters
        ----------
        vcf_path : Path
            Path to input VCF file
        gene_annotations : pd.DataFrame
            Gene annotation data with canonical splice sites
        output_path : Optional[Path]
            Output path for alternative_splice_sites.tsv
            
        Returns
        -------
        pd.DataFrame
            DataFrame of alternative splice sites ready for meta-model training
        """
        print(f"[Pipeline] Processing VCF: {vcf_path}")
        
        # Stage 1: Load and standardize VCF
        variants = self._load_vcf(vcf_path)
        standardized_variants = self.variant_standardizer.batch_standardize(
            variants, input_format='vcf'
        )
        print(f"[Pipeline] Standardized {len(standardized_variants)} variants")
        
        # Stage 2: Compute delta scores using delta bridge
        if self.delta_bridge:
            # Use real OpenSpliceAI delta score computation
            alternative_sites = self.delta_bridge.delta_scores_to_alternative_sites(
                self.delta_bridge.compute_delta_scores_from_variants(standardized_variants)
            )
            print(f"[Pipeline] Computed {len(alternative_sites)} alternative sites via OpenSpliceAI")
        else:
            # Mock delta scores for demonstration
            delta_scores_df = self._mock_delta_scores(standardized_variants)
            alternative_sites = self.extract_alternative_splice_sites_from_delta_scores(
                delta_scores_df, threshold=0.2
            )
            print(f"[Pipeline] Generated {len(alternative_sites)} mock alternative sites")
        
        # Stage 3: Convert to DataFrame
        sites_df = self._sites_to_dataframe(alternative_sites)
        
        # Stage 4: Integrate with canonical sites
        integrated_df = self._integrate_with_canonical_sites(sites_df, gene_annotations)
        
        # Save output
        if output_path is None:
            output_path = self.work_dir / "alternative_splice_sites.tsv"
        
        integrated_df.to_csv(output_path, sep='\t', index=False)
        print(f"[Pipeline] Saved {len(integrated_df)} sites to {output_path}")
        
        return integrated_df
    
    def _load_vcf(self, vcf_path: Path) -> List[Dict]:
        """Load VCF file and parse variants."""
        variants = []
        
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    variants.append({
                        'CHROM': parts[0],
                        'POS': int(parts[1]),
                        'REF': parts[3],
                        'ALT': parts[4]
                    })
        
        return variants
    
    def _mock_delta_scores(self, variants: List) -> pd.DataFrame:
        """Generate mock delta scores for demonstration."""
        rows = []
        
        for var in variants:
            # Generate mock scores
            ref_score = np.random.uniform(0, 1)
            alt_score = ref_score + np.random.uniform(-0.5, 0.5)
            
            rows.append({
                'chrom': var.chrom.replace('chr', ''),
                'position': var.start,
                'ref_score': ref_score,
                'alt_score': alt_score,
                'delta_score': alt_score - ref_score,
                'variant_id': f"{var.chrom}:{var.start}:{var.ref}>{var.alt}",
                'gene_symbol': 'MOCK_GENE',
                'is_canonical_site': np.random.random() < 0.1,
                'site_type': np.random.choice(['donor', 'acceptor'])
            })
        
        return pd.DataFrame(rows)
    
    def _compute_delta_scores_openspliceai(
        self, 
        variants: List,
        gene_annotations: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute actual delta scores using OpenSpliceAI adapter."""
        # This would integrate with the actual OpenSpliceAI adapter
        # For now, return mock data
        return self._mock_delta_scores(variants)
    
    def _sites_to_dataframe(self, sites: List[AlternativeSpliceSite]) -> pd.DataFrame:
        """Convert list of AlternativeSpliceSite objects to DataFrame."""
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
    
    def _integrate_with_canonical_sites(
        self,
        alternative_sites_df: pd.DataFrame,
        gene_annotations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Integrate alternative sites with canonical sites for comprehensive training data.
        """
        # Add source column
        alternative_sites_df['source'] = 'alternative'
        
        # Extract canonical sites from gene annotations if available
        if 'splice_sites' in gene_annotations.columns:
            canonical_sites = []
            # Extract canonical sites logic here
            # For now, just return alternative sites
            return alternative_sites_df
        
        return alternative_sites_df
    
    def generate_training_manifest(
        self,
        alternative_sites_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate training manifest for meta-model.
        
        Parameters
        ----------
        alternative_sites_df : pd.DataFrame
            Alternative splice sites data
        output_path : Optional[Path]
            Output path for manifest
            
        Returns
        -------
        Dict[str, Any]
            Training manifest with statistics and file paths
        """
        manifest = {
            'total_sites': len(alternative_sites_df),
            'site_categories': alternative_sites_df['splice_category'].value_counts().to_dict(),
            'site_types': alternative_sites_df['site_type'].value_counts().to_dict(),
            'genes': alternative_sites_df['gene_symbol'].nunique(),
            'mean_delta_score': alternative_sites_df['delta_score'].mean(),
            'files': {
                'alternative_sites': str(self.work_dir / 'alternative_splice_sites.tsv'),
                'training_ready': str(self.work_dir / 'training_data.parquet')
            }
        }
        
        if output_path is None:
            output_path = self.work_dir / 'training_manifest.json'
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"[Pipeline] Generated training manifest: {output_path}")
        
        return manifest


def run_example_pipeline():
    """Example usage of the Alternative Splicing Pipeline."""
    from pathlib import Path
    
    # Initialize pipeline with mock reference (for demonstration)
    work_dir = Path("./case_studies/alternative_splicing")
    pipeline = AlternativeSplicingPipeline(
        work_dir=work_dir,
        reference_fasta=None,  # Will use mock delta scores
        annotations="grch38"
    )
    
    # Process VCF to alternative sites
    vcf_path = Path("./data/example_variants.vcf")
    gene_annotations = pd.DataFrame()  # Would load actual annotations
    
    if vcf_path.exists():
        sites_df = pipeline.process_vcf_to_alternative_sites(
            vcf_path, gene_annotations
        )
        
        # Generate training manifest
        manifest = pipeline.generate_training_manifest(sites_df)
        
        print(f"\n✅ Pipeline complete!")
        print(f"   Total alternative sites: {manifest['total_sites']}")
        print(f"   Site categories: {manifest['site_categories']}")
        print(f"   Mean delta score: {manifest['mean_delta_score']:.3f}")
    else:
        print(f"⚠️  Example VCF not found: {vcf_path}")
        print("   Run with mock data instead...")
        
        # Create mock VCF data for demonstration
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as tmp:
            tmp.write("##fileformat=VCFv4.2\n")
            tmp.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            tmp.write("7\t117559593\t.\tG\tT\t.\tPASS\tGENE=CFTR\n")
            tmp.write("17\t43094077\t.\tG\tA\t.\tPASS\tGENE=BRCA1\n")
            tmp.flush()
            
            # Process mock VCF
            sites_df = pipeline.process_vcf_to_alternative_sites(
                Path(tmp.name), gene_annotations
            )
            
            # Generate training manifest
            manifest = pipeline.generate_training_manifest(sites_df)
            
            print(f"\n✅ Mock pipeline complete!")
            print(f"   Total alternative sites: {manifest['total_sites']}")
            print(f"   Site categories: {manifest['site_categories']}")
            print(f"   Mean delta score: {manifest['mean_delta_score']:.3f}")
            
            # Clean up
            Path(tmp.name).unlink()


if __name__ == "__main__":
    run_example_pipeline()
