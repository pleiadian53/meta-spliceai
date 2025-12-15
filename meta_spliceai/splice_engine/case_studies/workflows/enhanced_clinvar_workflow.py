#!/usr/bin/env python3
"""
Enhanced ClinVar Variant Analysis Workflow

Combines the robust ClinVar 5-step pipeline with the Universal VCF Parser
for improved parsing capabilities while maintaining the complete evaluation framework.

Enhanced Features:
- Step 2.5: Universal VCF parsing with comprehensive splice detection
- Maintains Steps 1, 3, 4, 5 from original workflow
- Improved splice variant detection
- Configurable annotation system support
"""

from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd
import logging

from .clinvar_variant_analysis import ClinVarVariantAnalysisWorkflow, ClinVarAnalysisConfig
from .universal_vcf_parser import create_clinvar_parser, UniversalVCFParser, VCFParsingConfig, AnnotationSystem, SpliceDetectionMode


class EnhancedClinVarWorkflow(ClinVarVariantAnalysisWorkflow):
    """
    Enhanced ClinVar workflow with Universal VCF Parser integration.
    
    Inherits the complete 5-step pipeline but replaces Step 2 with
    the Universal VCF Parser for improved parsing capabilities.
    """
    
    def __init__(self, config: ClinVarAnalysisConfig, use_universal_parser: bool = True):
        """
        Initialize enhanced workflow.
        
        Parameters
        ----------
        config : ClinVarAnalysisConfig
            Base workflow configuration
        use_universal_parser : bool
            Whether to use Universal VCF Parser for Step 2
        """
        super().__init__(config)
        self.use_universal_parser = use_universal_parser
        
        if use_universal_parser:
            self.logger.info("Enhanced workflow: Using Universal VCF Parser for Step 2")
    
    def step2_filter_and_parse(self, normalized_vcf: Path) -> pd.DataFrame:
        """
        Enhanced Step 2: Use Universal VCF Parser for comprehensive parsing.
        
        Parameters
        ----------
        normalized_vcf : Path
            Normalized VCF from Step 1
            
        Returns
        -------
        pd.DataFrame
            Enhanced parsed variants with comprehensive splice detection
        """
        if not self.use_universal_parser:
            # Fall back to original implementation
            return super().step2_filter_and_parse(normalized_vcf)
        
        self.logger.info("=== Step 2 (Enhanced): Universal VCF Parsing ===")
        
        # Create Universal VCF Parser with ClinVar-optimized configuration
        parser = create_clinvar_parser(
            splice_detection="comprehensive",  # Enhanced splice detection
            include_uncertain=not self.config.apply_splice_filter,  # Include uncertain if no filtering
            include_sequences=False  # Don't need sequences for this workflow
        )
        
        # Parse VCF with universal parser
        parsed_variants = parser.parse_vcf(normalized_vcf)
        
        # Convert column names to match expected format
        column_mapping = {
            'chrom': 'CHROM',
            'pos': 'POS', 
            'id': 'ID',
            'ref': 'REF',
            'alt': 'ALT',
            'qual': 'QUAL',
            'filter': 'FILTER',
            'clinical_significance': 'CLNSIG',
            'review_status': 'CLNREVSTAT',
            'molecular_consequence': 'MC',
            'disease': 'CLNDN',
            'variant_type': 'TYPE'
        }
        
        # Rename columns to match expected format
        for old_col, new_col in column_mapping.items():
            if old_col in parsed_variants.columns and new_col not in parsed_variants.columns:
                parsed_variants[new_col] = parsed_variants[old_col]
        
        # Apply additional filtering if requested
        if self.config.clinical_significance_filter or self.config.review_status_filter:
            parsed_variants = self._apply_variant_filters(parsed_variants)
        
        # Save enhanced parsing results
        output_tsv = self.config.output_dir / "step2_enhanced_filtered_variants.tsv"
        parsed_variants.to_csv(output_tsv, sep='\t', index=False)
        
        self.logger.info(f"Enhanced Step 2 completed: {len(parsed_variants)} variants parsed")
        
        # Log enhancement statistics
        if 'affects_splicing' in parsed_variants.columns:
            splice_count = parsed_variants['affects_splicing'].sum()
            self.logger.info(f"Universal parser detected {splice_count} splice-affecting variants ({splice_count/len(parsed_variants)*100:.1f}%)")
        
        if 'splice_confidence' in parsed_variants.columns:
            confidence_stats = parsed_variants['splice_confidence'].value_counts()
            self.logger.info(f"Splice confidence distribution: {confidence_stats.to_dict()}")
        
        return parsed_variants


def create_enhanced_clinvar_workflow(
    input_vcf: Union[str, Path],
    output_dir: Union[str, Path],
    use_universal_parser: bool = True,
    **kwargs
) -> EnhancedClinVarWorkflow:
    """
    Create enhanced ClinVar workflow with Universal VCF Parser integration.
    
    Parameters
    ----------
    input_vcf : Union[str, Path]
        Input ClinVar VCF file
    output_dir : Union[str, Path]
        Output directory for results
    use_universal_parser : bool
        Whether to use Universal VCF Parser for enhanced Step 2
    **kwargs
        Additional configuration options
        
    Returns
    -------
    EnhancedClinVarWorkflow
        Enhanced workflow instance
    """
    config = ClinVarAnalysisConfig(
        input_vcf=Path(input_vcf),
        output_dir=Path(output_dir),
        **kwargs
    )
    
    return EnhancedClinVarWorkflow(config, use_universal_parser=use_universal_parser)


# CLI interface
def main():
    """Command-line interface for Enhanced ClinVar workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced ClinVar Variant Analysis Workflow with Universal VCF Parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run enhanced workflow with Universal VCF Parser
  python enhanced_clinvar_workflow.py \\
      --input-vcf clinvar_20250831_main_chroms.vcf.gz \\
      --output-dir results/enhanced_clinvar \\
      --use-universal-parser

  # Run specific step with enhancement
  python enhanced_clinvar_workflow.py \\
      --input-vcf normalized.vcf.gz \\
      --output-dir results/ \\
      --step 2 \\
      --use-universal-parser
        """
    )
    
    parser.add_argument("--input-vcf", required=True, type=Path,
                       help="Input ClinVar VCF file")
    parser.add_argument("--output-dir", required=True, type=Path,
                       help="Output directory for results")
    parser.add_argument("--genome-build", default="GRCh38",
                       help="Genome build (GRCh37, GRCh38)")
    parser.add_argument("--use-universal-parser", action="store_true", default=True,
                       help="Use Universal VCF Parser for enhanced Step 2")
    parser.add_argument("--no-universal-parser", action="store_true",
                       help="Use original parsing (disable Universal VCF Parser)")
    parser.add_argument("--no-splice-filter", action="store_true",
                       help="Skip splice filtering to avoid evaluation bias")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of threads to use")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5],
                       help="Run only specific step (default: run all)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine parser usage
    use_universal_parser = args.use_universal_parser and not args.no_universal_parser
    
    # Create enhanced workflow
    workflow = create_enhanced_clinvar_workflow(
        input_vcf=args.input_vcf,
        output_dir=args.output_dir,
        use_universal_parser=use_universal_parser,
        genome_build=args.genome_build,
        apply_splice_filter=not args.no_splice_filter,
        threads=args.threads
    )
    
    # Run workflow
    if args.step:
        # Run specific step
        if args.step == 1:
            workflow.step1_normalize_vcf()
        elif args.step == 2:
            normalized_vcf = Path(args.input_vcf)
            workflow.step2_filter_and_parse(normalized_vcf)
        elif args.step == 3:
            normalized_vcf = Path(args.input_vcf)
            workflow.step3_openspliceai_scoring(normalized_vcf)
        # Add other steps as needed
    else:
        # Run complete workflow
        results = workflow.run_complete_workflow()
        print(f"Enhanced workflow completed successfully. Results saved to {workflow.config.output_dir}")


if __name__ == "__main__":
    main()
