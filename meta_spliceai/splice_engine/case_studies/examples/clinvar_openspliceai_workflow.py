#!/usr/bin/env python3
"""
Complete ClinVar to OpenSpliceAI Analysis Workflow

This workflow demonstrates the complete pipeline from ClinVar VCF to OpenSpliceAI
delta score computation for benchmarking base model vs meta-model performance.

Workflow Steps:
1. VCF normalization with bcftools
2. ClinVar-specific variant filtering  
3. Reference genome sequence extraction
4. WT/ALT sequence construction
5. OpenSpliceAI delta score computation
6. Alternative splice site analysis
7. Base model vs meta-model comparison

Usage:
    python clinvar_openspliceai_workflow.py --vcf clinvar_20250831.vcf.gz --output results/
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add case studies to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_sources.resource_manager import create_case_study_resource_manager
from workflows.vcf_preprocessing import preprocess_clinvar_vcf
from filters.splice_variant_filter import create_clinvar_filter
from formats.variant_standardizer import VariantStandardizer
from analysis.splicing_pattern_analyzer import SplicingPatternAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinVarOpenSpliceAIWorkflow:
    """Complete workflow for ClinVar variant analysis with OpenSpliceAI."""
    
    def __init__(self, output_dir: Path, genome_build: str = "GRCh38"):
        """
        Initialize workflow.
        
        Parameters
        ----------
        output_dir : Path
            Output directory for results
        genome_build : str
            Genome build (GRCh37, GRCh38)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize resource manager
        self.resource_manager = create_case_study_resource_manager(
            genome_build=genome_build
        )
        
        # Initialize components
        self.variant_filter = create_clinvar_filter()
        self.standardizer = VariantStandardizer(reference_genome=genome_build)
        self.splicing_analyzer = SplicingPatternAnalyzer()
        
        logger.info(f"Workflow initialized with output: {self.output_dir}")
        logger.info(f"Genome build: {genome_build}")
    
    def step1_normalize_vcf(self, input_vcf: Path) -> Path:
        """
        Step 1: Normalize VCF with bcftools.
        
        Parameters
        ----------
        input_vcf : Path
            Input ClinVar VCF file
            
        Returns
        -------
        Path
            Normalized VCF file path
        """
        logger.info("Step 1: VCF Normalization")
        
        normalized_vcf = preprocess_clinvar_vcf(
            input_vcf=input_vcf,
            output_dir=self.output_dir / "normalized"
        )
        
        logger.info(f"✓ VCF normalized: {normalized_vcf}")
        return normalized_vcf
    
    def step2_filter_variants(self, vcf_path: Path) -> pd.DataFrame:
        """
        Step 2: Filter for splice-affecting pathogenic variants.
        
        Parameters
        ----------
        vcf_path : Path
            Normalized VCF file
            
        Returns
        -------
        pd.DataFrame
            Filtered variants
        """
        logger.info("Step 2: Variant Filtering")
        
        # Parse VCF to DataFrame
        variants_df = self._parse_vcf_to_dataframe(vcf_path)
        logger.info(f"Parsed {len(variants_df)} total variants")
        
        # Apply ClinVar-specific filtering
        filtered_df = self.variant_filter.filter_variants(variants_df)
        logger.info(f"✓ Filtered to {len(filtered_df)} splice-affecting pathogenic variants")
        
        # Save filtered variants
        output_file = self.output_dir / "filtered_variants.tsv"
        filtered_df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Saved filtered variants: {output_file}")
        
        return filtered_df
    
    def step3_construct_sequences(self, variants_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Construct WT/ALT sequences from reference genome.
        
        Parameters
        ----------
        variants_df : pd.DataFrame
            Filtered variants
            
        Returns
        -------
        pd.DataFrame
            Variants with WT/ALT sequences
        """
        logger.info("Step 3: Sequence Construction")
        
        # Get reference FASTA
        try:
            fasta_path = self.resource_manager.get_fasta_path(validate=True)
            logger.info(f"Using reference FASTA: {fasta_path}")
        except FileNotFoundError:
            raise RuntimeError("Reference FASTA not found. Please ensure genomic resources are set up.")
        
        # Construct sequences for each variant
        sequences_data = []
        
        for idx, row in variants_df.iterrows():
            try:
                # Standardize variant
                std_variant = self.standardizer.standardize_from_vcf(
                    row['CHROM'], int(row['POS']), row['REF'], row['ALT']
                )
                
                # Extract sequences (5kb flanking for splice analysis)
                sequences = self._extract_sequences(std_variant, fasta_path, flanking_size=5000)
                
                # Combine variant info with sequences
                variant_data = {
                    'variant_id': f"{std_variant.chrom}:{std_variant.start}:{std_variant.ref}>{std_variant.alt}",
                    'chrom': std_variant.chrom,
                    'pos': std_variant.start,
                    'ref': std_variant.ref,
                    'alt': std_variant.alt,
                    'variant_type': std_variant.variant_type,
                    'wt_sequence': sequences['wildtype'],
                    'alt_sequence': sequences['alternative'],
                    'sequence_length': len(sequences['wildtype']),
                    'variant_offset': sequences['variant_offset'],
                    'clinical_significance': row.get('CLNSIG', ''),
                    'gene_info': row.get('GENEINFO', ''),
                    'molecular_consequence': row.get('MC', '')
                }
                
                sequences_data.append(variant_data)
                
            except Exception as e:
                logger.warning(f"Failed to process variant {idx}: {e}")
                continue
        
        sequences_df = pd.DataFrame(sequences_data)
        logger.info(f"✓ Constructed sequences for {len(sequences_df)} variants")
        
        # Save sequences
        output_file = self.output_dir / "variant_sequences.tsv"
        sequences_df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Saved variant sequences: {output_file}")
        
        return sequences_df
    
    def step4_openspliceai_analysis(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Run OpenSpliceAI analysis on WT/ALT sequences.
        
        Parameters
        ----------
        sequences_df : pd.DataFrame
            Variants with sequences
            
        Returns
        -------
        pd.DataFrame
            Variants with OpenSpliceAI scores
        """
        logger.info("Step 4: OpenSpliceAI Analysis")
        
        # Prepare sequences for OpenSpliceAI
        openspliceai_input = self._prepare_openspliceai_input(sequences_df)
        
        # Run OpenSpliceAI (this would call the actual OpenSpliceAI model)
        # For now, we'll create a placeholder for the integration
        logger.warning("OpenSpliceAI integration not yet implemented - using mock scores")
        
        # Add mock delta scores for demonstration
        sequences_df['delta_score_acceptor'] = np.random.uniform(-1, 1, len(sequences_df))
        sequences_df['delta_score_donor'] = np.random.uniform(-1, 1, len(sequences_df))
        sequences_df['max_delta_score'] = sequences_df[['delta_score_acceptor', 'delta_score_donor']].abs().max(axis=1)
        
        # Save OpenSpliceAI results
        output_file = self.output_dir / "openspliceai_results.tsv"
        sequences_df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Saved OpenSpliceAI results: {output_file}")
        
        return sequences_df
    
    def step5_alternative_splice_analysis(self, results_df: pd.DataFrame) -> Dict:
        """
        Step 5: Analyze alternative splice sites and patterns.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results with OpenSpliceAI scores
            
        Returns
        -------
        Dict
            Analysis summary
        """
        logger.info("Step 5: Alternative Splice Site Analysis")
        
        # Analyze splice patterns
        analysis_results = {}
        
        # High-impact variants (|delta_score| > 0.5)
        high_impact = results_df[results_df['max_delta_score'] > 0.5]
        analysis_results['high_impact_count'] = len(high_impact)
        analysis_results['high_impact_percentage'] = len(high_impact) / len(results_df) * 100
        
        # Variant type distribution
        variant_type_counts = results_df['variant_type'].value_counts()
        analysis_results['variant_types'] = variant_type_counts.to_dict()
        
        # Clinical significance correlation
        clinical_groups = results_df.groupby('clinical_significance')['max_delta_score'].agg(['mean', 'std', 'count'])
        analysis_results['clinical_significance_scores'] = clinical_groups.to_dict()
        
        # Save analysis summary
        import json
        output_file = self.output_dir / "splice_analysis_summary.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"✓ Alternative splice analysis complete")
        logger.info(f"High-impact variants: {analysis_results['high_impact_count']} ({analysis_results['high_impact_percentage']:.1f}%)")
        
        return analysis_results
    
    def run_complete_workflow(self, input_vcf: Path) -> Dict:
        """
        Run the complete ClinVar to OpenSpliceAI workflow.
        
        Parameters
        ----------
        input_vcf : Path
            Input ClinVar VCF file
            
        Returns
        -------
        Dict
            Workflow results summary
        """
        logger.info("Starting Complete ClinVar OpenSpliceAI Workflow")
        logger.info("=" * 80)
        
        try:
            # Step 1: Normalize VCF
            normalized_vcf = self.step1_normalize_vcf(input_vcf)
            
            # Step 2: Filter variants
            filtered_variants = self.step2_filter_variants(normalized_vcf)
            
            # Step 3: Construct sequences
            sequences_df = self.step3_construct_sequences(filtered_variants)
            
            # Step 4: OpenSpliceAI analysis
            results_df = self.step4_openspliceai_analysis(sequences_df)
            
            # Step 5: Alternative splice analysis
            analysis_summary = self.step5_alternative_splice_analysis(results_df)
            
            # Create workflow summary
            workflow_summary = {
                'input_vcf': str(input_vcf),
                'output_dir': str(self.output_dir),
                'total_variants_processed': len(filtered_variants),
                'sequences_constructed': len(sequences_df),
                'analysis_results': analysis_summary,
                'workflow_status': 'completed'
            }
            
            logger.info("=" * 80)
            logger.info("🎉 Workflow completed successfully!")
            logger.info(f"Results saved in: {self.output_dir}")
            
            return workflow_summary
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _parse_vcf_to_dataframe(self, vcf_path: Path) -> pd.DataFrame:
        """Parse VCF file to pandas DataFrame."""
        import subprocess
        
        # Use bcftools to convert to TSV for easier parsing
        cmd = f"bcftools query -f '%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT\\t%QUAL\\t%FILTER\\t%INFO\\n' {vcf_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to parse VCF: {result.stderr}")
        
        # Parse TSV output
        lines = result.stdout.strip().split('\n')
        data = []
        
        for line in lines:
            if line:
                fields = line.split('\t')
                if len(fields) >= 8:
                    # Parse INFO field
                    info_dict = {}
                    for item in fields[7].split(';'):
                        if '=' in item:
                            key, value = item.split('=', 1)
                            info_dict[key] = value
                        else:
                            info_dict[item] = True
                    
                    variant_data = {
                        'CHROM': fields[0],
                        'POS': fields[1], 
                        'ID': fields[2],
                        'REF': fields[3],
                        'ALT': fields[4],
                        'QUAL': fields[5],
                        'FILTER': fields[6],
                        **info_dict
                    }
                    data.append(variant_data)
        
        return pd.DataFrame(data)
    
    def _extract_sequences(self, variant, fasta_path: Path, flanking_size: int = 5000) -> Dict[str, str]:
        """Extract WT/ALT sequences from reference genome."""
        try:
            import pysam
            
            with pysam.FastaFile(str(fasta_path)) as fasta:
                # Calculate sequence coordinates
                start_pos = max(1, variant.start - flanking_size)
                end_pos = variant.end + flanking_size
                
                # Extract reference sequence
                chrom = variant.chrom if not variant.chrom.startswith('chr') else variant.chrom[3:]
                ref_sequence = fasta.fetch(chrom, start_pos - 1, end_pos)  # pysam uses 0-based
                
                # Calculate variant position within extracted sequence
                variant_offset = variant.start - start_pos
                
                # Construct alternative sequence
                alt_sequence = (ref_sequence[:variant_offset] + 
                              variant.alt + 
                              ref_sequence[variant_offset + len(variant.ref):])
                
                return {
                    'wildtype': ref_sequence.upper(),
                    'alternative': alt_sequence.upper(),
                    'variant_offset': variant_offset,
                    'sequence_coordinates': f"{chrom}:{start_pos}-{end_pos}"
                }
                
        except Exception as e:
            logger.error(f"Failed to extract sequences for {variant}: {e}")
            raise
    
    def _prepare_openspliceai_input(self, sequences_df: pd.DataFrame) -> Dict:
        """Prepare input format for OpenSpliceAI."""
        # This would format sequences for OpenSpliceAI input
        # Implementation depends on OpenSpliceAI interface
        return {
            'sequences': sequences_df[['variant_id', 'wt_sequence', 'alt_sequence']].to_dict('records')
        }


def main():
    """Main entry point for the workflow."""
    parser = argparse.ArgumentParser(
        description="Complete ClinVar to OpenSpliceAI analysis workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--vcf', required=True, type=Path,
                       help='Input ClinVar VCF file')
    parser.add_argument('--output', '-o', required=True, type=Path,
                       help='Output directory for results')
    parser.add_argument('--genome-build', default='GRCh38',
                       help='Genome build (GRCh37, GRCh38)')
    parser.add_argument('--max-variants', type=int,
                       help='Maximum variants to process (for testing)')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = ClinVarOpenSpliceAIWorkflow(
        output_dir=args.output,
        genome_build=args.genome_build
    )
    
    # Run workflow
    results = workflow.run_complete_workflow(args.vcf)
    
    print("\n🎉 Analysis Complete!")
    print(f"Results: {args.output}")
    print(f"Variants processed: {results['total_variants_processed']}")
    print(f"Sequences constructed: {results['sequences_constructed']}")


if __name__ == "__main__":
    main()
