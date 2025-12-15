#!/usr/bin/env python3
"""
VCF to OpenSpliceAI Integration Example

This script demonstrates the complete pipeline from ClinVar VCF parsing
to OpenSpliceAI delta score computation and splice pattern analysis.

Requirements:
- Processed ClinVar data from vcf_clinvar_tutorial.py
- Reference genome FASTA file
- OpenSpliceAI model weights

Usage:
    python vcf_openspliceai_integration.py --input-dir data/ensembl/clinvar/splice_variants
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from case_studies.formats.variant_standardizer import VariantStandardizer
from case_studies.analysis.splicing_pattern_analyzer import SplicingPatternAnalyzer, SpliceSite, SplicingPattern
from case_studies.workflows.openspliceai_delta_bridge import OpenSpliceAIDeltaBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VCFOpenSpliceAIIntegrator:
    """Integrates VCF variant analysis with OpenSpliceAI predictions."""
    
    def __init__(
        self,
        fasta_path: Optional[str] = None,
        openspliceai_model_path: Optional[str] = None
    ):
        """
        Initialize the integrator.
        
        Parameters
        ----------
        fasta_path : Optional[str]
            Path to reference genome FASTA file
        openspliceai_model_path : Optional[str]
            Path to OpenSpliceAI model weights
        """
        self.fasta_path = fasta_path
        self.openspliceai_model_path = openspliceai_model_path
        
        # Initialize components
        self.standardizer = VariantStandardizer()
        self.pattern_analyzer = SplicingPatternAnalyzer()
        
        # Initialize OpenSpliceAI bridge if model available
        self.openspliceai_bridge = None
        if openspliceai_model_path and os.path.exists(openspliceai_model_path):
            try:
                self.openspliceai_bridge = OpenSpliceAIDeltaBridge(
                    model_path=openspliceai_model_path
                )
                logger.info("OpenSpliceAI bridge initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenSpliceAI bridge: {e}")
        else:
            logger.info("OpenSpliceAI bridge not available - using mock predictions")
    
    def load_processed_variants(self, input_path: str) -> pd.DataFrame:
        """
        Load processed variants from ClinVar tutorial output.
        
        Parameters
        ----------
        input_path : str
            Path to processed variants TSV file
        
        Returns
        -------
        pd.DataFrame
            Loaded variant data
        """
        try:
            df = pd.read_csv(input_path, sep='\t')
            logger.info(f"Loaded {len(df)} processed variants from {input_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading variants: {e}")
            return pd.DataFrame()
    
    def compute_openspliceai_scores(
        self,
        variants_df: pd.DataFrame,
        context_size: int = 10000
    ) -> pd.DataFrame:
        """
        Compute OpenSpliceAI delta scores for variants.
        
        Parameters
        ----------
        variants_df : pd.DataFrame
            DataFrame with variant information and sequences
        context_size : int
            Context size for OpenSpliceAI analysis
        
        Returns
        -------
        pd.DataFrame
            Variants with OpenSpliceAI scores
        """
        results = []
        
        for _, variant in variants_df.iterrows():
            try:
                if self.openspliceai_bridge:
                    # Use real OpenSpliceAI predictions
                    scores = self._compute_real_openspliceai_scores(variant, context_size)
                else:
                    # Use mock predictions for demonstration
                    scores = self._compute_mock_openspliceai_scores(variant)
                
                # Add scores to variant record
                variant_with_scores = variant.to_dict()
                variant_with_scores.update(scores)
                results.append(variant_with_scores)
                
            except Exception as e:
                logger.warning(f"Error computing scores for variant {variant.get('clinvar_id', 'unknown')}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        logger.info(f"Computed OpenSpliceAI scores for {len(results_df)} variants")
        return results_df
    
    def _compute_real_openspliceai_scores(
        self,
        variant: pd.Series,
        context_size: int
    ) -> Dict[str, float]:
        """Compute real OpenSpliceAI scores using the bridge."""
        # Extract sequences
        wt_seq = variant.get('wt_sequence', '')
        alt_seq = variant.get('alt_sequence', '')
        
        if not wt_seq or not alt_seq:
            raise ValueError("Missing WT or ALT sequence")
        
        # Compute delta scores using the bridge
        delta_scores = self.openspliceai_bridge.compute_delta_scores(
            wt_sequence=wt_seq,
            alt_sequence=alt_seq,
            variant_position=variant.get('var_position_in_seq', len(wt_seq) // 2)
        )
        
        return {
            'delta_score_acceptor': delta_scores.get('acceptor_gain', 0.0),
            'delta_score_donor': delta_scores.get('donor_gain', 0.0),
            'delta_score_acceptor_loss': delta_scores.get('acceptor_loss', 0.0),
            'delta_score_donor_loss': delta_scores.get('donor_loss', 0.0),
            'max_delta_score': max(abs(score) for score in delta_scores.values())
        }
    
    def _compute_mock_openspliceai_scores(self, variant: pd.Series) -> Dict[str, float]:
        """Generate mock OpenSpliceAI scores for demonstration."""
        # Generate realistic-looking mock scores based on variant type
        var_type = variant.get('var_type', 'SNV')
        
        # Base scores on variant type and position
        np.random.seed(hash(variant.get('clinvar_id', '')) % 2**32)
        
        if var_type == 'SNV':
            base_score = np.random.uniform(0.1, 0.8)
        elif var_type in ['Deletion', 'Insertion']:
            base_score = np.random.uniform(0.3, 0.9)
        else:
            base_score = np.random.uniform(0.2, 0.7)
        
        # Add some noise
        acceptor_gain = base_score + np.random.normal(0, 0.1)
        donor_gain = base_score * 0.8 + np.random.normal(0, 0.1)
        acceptor_loss = -base_score * 0.6 + np.random.normal(0, 0.05)
        donor_loss = -base_score * 0.7 + np.random.normal(0, 0.05)
        
        return {
            'delta_score_acceptor': max(0, acceptor_gain),
            'delta_score_donor': max(0, donor_gain),
            'delta_score_acceptor_loss': min(0, acceptor_loss),
            'delta_score_donor_loss': min(0, donor_loss),
            'max_delta_score': max(abs(acceptor_gain), abs(donor_gain), abs(acceptor_loss), abs(donor_loss))
        }
    
    def analyze_splicing_patterns(
        self,
        variants_with_scores: pd.DataFrame,
        score_threshold: float = 0.2
    ) -> pd.DataFrame:
        """
        Analyze splicing patterns from OpenSpliceAI scores.
        
        Parameters
        ----------
        variants_with_scores : pd.DataFrame
            Variants with computed delta scores
        score_threshold : float
            Minimum delta score threshold for pattern detection
        
        Returns
        -------
        pd.DataFrame
            Variants with splicing pattern analysis
        """
        results = []
        
        for _, variant in variants_with_scores.iterrows():
            try:
                # Create SpliceSite objects from delta scores
                splice_sites = self._create_splice_sites_from_scores(variant)
                
                # Analyze splicing patterns
                patterns = self.pattern_analyzer.analyze_variant_impact(
                    splice_sites=splice_sites,
                    variant_position=variant.get('var_position_in_seq', 0),
                    variant_type=variant.get('var_type', 'SNV')
                )
                
                # Filter patterns by score threshold
                significant_patterns = [
                    p for p in patterns 
                    if p.confidence_score >= score_threshold
                ]
                
                # Add pattern analysis to variant
                variant_dict = variant.to_dict()
                variant_dict.update({
                    'num_patterns_detected': len(significant_patterns),
                    'pattern_types': [p.pattern_type for p in significant_patterns],
                    'max_pattern_confidence': max([p.confidence_score for p in significant_patterns], default=0.0),
                    'predicted_impact': self._classify_impact(significant_patterns)
                })
                
                results.append(variant_dict)
                
            except Exception as e:
                logger.warning(f"Error analyzing patterns for variant {variant.get('clinvar_id', 'unknown')}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        logger.info(f"Analyzed splicing patterns for {len(results_df)} variants")
        return results_df
    
    def _create_splice_sites_from_scores(self, variant: pd.Series) -> List[SpliceSite]:
        """Create SpliceSite objects from delta scores."""
        splice_sites = []
        
        # Create acceptor site if significant gain
        if variant.get('delta_score_acceptor', 0) > 0.1:
            splice_sites.append(SpliceSite(
                position=variant.get('start', 0),
                site_type='acceptor',
                score=variant.get('delta_score_acceptor', 0),
                sequence_context='NNNAGNNN'  # Mock context
            ))
        
        # Create donor site if significant gain
        if variant.get('delta_score_donor', 0) > 0.1:
            splice_sites.append(SpliceSite(
                position=variant.get('start', 0) + 1,
                site_type='donor',
                score=variant.get('delta_score_donor', 0),
                sequence_context='NNNGTNNNN'  # Mock context
            ))
        
        return splice_sites
    
    def _classify_impact(self, patterns: List[SplicingPattern]) -> str:
        """Classify the predicted impact based on detected patterns."""
        if not patterns:
            return 'No significant impact'
        
        max_confidence = max(p.confidence_score for p in patterns)
        pattern_types = set(p.pattern_type for p in patterns)
        
        if max_confidence > 0.8:
            impact = 'High impact'
        elif max_confidence > 0.5:
            impact = 'Moderate impact'
        else:
            impact = 'Low impact'
        
        # Add pattern type information
        if 'exon_skipping' in pattern_types:
            impact += ' (exon skipping)'
        elif 'cryptic_activation' in pattern_types:
            impact += ' (cryptic site activation)'
        elif 'splice_site_disruption' in pattern_types:
            impact += ' (splice site disruption)'
        
        return impact
    
    def generate_report(
        self,
        final_results: pd.DataFrame,
        output_dir: str
    ) -> None:
        """
        Generate comprehensive analysis report.
        
        Parameters
        ----------
        final_results : pd.DataFrame
            Final analysis results
        output_dir : str
            Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_path / 'openspliceai_analysis_results.tsv'
        final_results.to_csv(results_file, sep='\t', index=False)
        
        # Generate summary report
        report_file = output_path / 'analysis_summary_report.txt'
        with open(report_file, 'w') as f:
            f.write("OpenSpliceAI Splice Variant Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total variants analyzed: {len(final_results)}\n")
            
            # Impact distribution
            if 'predicted_impact' in final_results.columns:
                f.write("\nPredicted Impact Distribution:\n")
                impact_counts = final_results['predicted_impact'].value_counts()
                for impact, count in impact_counts.items():
                    f.write(f"  {impact}: {count}\n")
            
            # Score distribution
            if 'max_delta_score' in final_results.columns:
                scores = final_results['max_delta_score']
                f.write(f"\nDelta Score Statistics:\n")
                f.write(f"  Mean: {scores.mean():.3f}\n")
                f.write(f"  Median: {scores.median():.3f}\n")
                f.write(f"  Max: {scores.max():.3f}\n")
                f.write(f"  Variants with score > 0.5: {(scores > 0.5).sum()}\n")
            
            # Top variants
            if 'max_delta_score' in final_results.columns:
                f.write("\nTop 10 Variants by Delta Score:\n")
                top_variants = final_results.nlargest(10, 'max_delta_score')
                for _, variant in top_variants.iterrows():
                    f.write(f"  {variant.get('clinvar_id', 'N/A')}: {variant.get('max_delta_score', 0):.3f} "
                           f"({variant.get('predicted_impact', 'N/A')})\n")
        
        logger.info(f"Analysis report saved to: {report_file}")
        logger.info(f"Detailed results saved to: {results_file}")


def main():
    """Main function for the integration example."""
    parser = argparse.ArgumentParser(
        description='Integrate VCF variants with OpenSpliceAI analysis'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/ensembl/clinvar/splice_variants',
        help='Directory containing processed variants from ClinVar tutorial'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/ensembl/clinvar/openspliceai_analysis',
        help='Output directory for analysis results'
    )
    
    parser.add_argument(
        '--fasta-path',
        type=str,
        help='Path to reference genome FASTA file'
    )
    
    parser.add_argument(
        '--openspliceai-model',
        type=str,
        help='Path to OpenSpliceAI model weights'
    )
    
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.2,
        help='Minimum delta score threshold for pattern detection'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VCF to OpenSpliceAI Integration Analysis")
    print("=" * 80)
    
    # Initialize integrator
    integrator = VCFOpenSpliceAIIntegrator(
        fasta_path=args.fasta_path,
        openspliceai_model_path=args.openspliceai_model
    )
    
    # Load processed variants
    input_file = Path(args.input_dir) / 'clinvar_splice_variants_processed.tsv'
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please run vcf_clinvar_tutorial.py first to generate processed variants.")
        sys.exit(1)
    
    print(f"\n1. Loading processed variants from: {input_file}")
    variants_df = integrator.load_processed_variants(str(input_file))
    
    if variants_df.empty:
        print("No variants found. Exiting.")
        sys.exit(1)
    
    # Compute OpenSpliceAI scores
    print("\n2. Computing OpenSpliceAI delta scores...")
    variants_with_scores = integrator.compute_openspliceai_scores(variants_df)
    
    # Analyze splicing patterns
    print("\n3. Analyzing splicing patterns...")
    final_results = integrator.analyze_splicing_patterns(
        variants_with_scores,
        score_threshold=args.score_threshold
    )
    
    # Generate report
    print("\n4. Generating analysis report...")
    integrator.generate_report(final_results, args.output_dir)
    
    # Display summary
    print("\n5. Analysis Summary:")
    print("-" * 40)
    print(f"Variants processed: {len(final_results)}")
    
    if 'max_delta_score' in final_results.columns:
        high_impact = (final_results['max_delta_score'] > 0.5).sum()
        print(f"High-impact variants (score > 0.5): {high_impact}")
    
    if 'predicted_impact' in final_results.columns:
        impact_counts = final_results['predicted_impact'].value_counts()
        print("\nImpact distribution:")
        for impact, count in impact_counts.head().items():
            print(f"  {impact}: {count}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
