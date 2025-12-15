#!/usr/bin/env python3
"""
VCF Column Documentation Tool Entry Point

This script provides an easy way to run the VCF Column Documenter tool
for analyzing and documenting VCF column values and meanings.

Usage Examples:
    # Basic usage
    python run_vcf_column_documenter.py \\
        --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \\
        --output-dir data/ensembl/clinvar/vcf/docs/

    # With sample size limit
    python run_vcf_column_documenter.py \\
        --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \\
        --output-dir data/ensembl/clinvar/vcf/docs/ \\
        --max-variants 50000

    # JSON only output
    python run_vcf_column_documenter.py \\
        --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \\
        --output-dir data/ensembl/clinvar/vcf/docs/ \\
        --formats json

Author: MetaSpliceAI Team
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the path using systematic detection
from project_root_utils import setup_entry_point_imports
setup_entry_point_imports(__file__)

from meta_spliceai.splice_engine.case_studies.tools.vcf_column_documenter import (
    VCFColumnDocumenter,
    VCFDocumentationConfig
)


def main():
    """Simple command-line interface for VCF Column Documenter."""
    parser = argparse.ArgumentParser(
        description="VCF Column Documentation Tool: Analyze and document VCF column values",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--vcf', '-v', required=True,
                       help='Input VCF file (supports structured paths like data/ensembl/clinvar/vcf/clinvar.vcf.gz)')
    parser.add_argument('--output-dir', '-o', required=True,
                       help='Output directory for documentation')
    
    # Optional arguments
    parser.add_argument('--max-variants', type=int,
                       help='Maximum variants to analyze (default: all)')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Sample size for value enumeration (default: 10000)')
    parser.add_argument('--formats', nargs='+', 
                       choices=['json', 'markdown', 'csv'],
                       default=['json', 'markdown'],
                       help='Output formats (default: json markdown)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input
    vcf_path = Path(args.vcf)
    if not vcf_path.exists():
        print(f"❌ Error: VCF file not found: {vcf_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = VCFDocumentationConfig(
        input_vcf=vcf_path,
        output_dir=output_dir,
        max_variants=args.max_variants,
        sample_size=args.sample_size,
        output_formats=args.formats,
        verbose=args.verbose
    )
    
    print("🔍 Starting VCF Column Documentation Tool")
    print(f"📁 VCF:  {vcf_path}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Sample size: {args.sample_size:,}")
    print(f"📄 Formats: {', '.join(args.formats)}")
    
    if args.max_variants:
        print(f"🧪 Limited to: {args.max_variants:,} variants")
    
    try:
        # Run the documenter
        documenter = VCFColumnDocumenter(config)
        documentation = documenter.analyze_vcf_columns()
        
        # Save documentation
        documenter.save_documentation()
        
        print("\n🎉 VCF Column Documentation completed successfully!")
        print(f"📄 Documentation saved to: {output_dir}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Documentation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
