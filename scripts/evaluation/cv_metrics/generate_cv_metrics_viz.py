#!/usr/bin/env python3
"""
Standalone CV metrics visualization script.

This script can be run independently to generate visualization reports
for existing gene_cv_metrics.csv files.
"""

import sys
from pathlib import Path
import argparse

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import generate_cv_metrics_report

def main():
    """Main entry point for standalone CV metrics visualization."""
    
    parser = argparse.ArgumentParser(description="Generate CV metrics visualization report")
    parser.add_argument("csv_path", help="Path to gene_cv_metrics.csv file")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots and report")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"], 
                       help="Plot format (default: png)")
    parser.add_argument("--dpi", type=int, default=300, help="Plot resolution (default: 300)")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not Path(args.csv_path).exists():
        print(f"Error: CSV file not found at {args.csv_path}")
        return 1
    
    try:
        # Generate report
        result = generate_cv_metrics_report(
            csv_path=args.csv_path,
            out_dir=args.out_dir,
            plot_format=args.plot_format,
            dpi=args.dpi
        )
        
        print(f"✓ CV metrics visualization report generated successfully!")
        print(f"✓ Output directory: {result['visualization_dir']}")
        print(f"✓ Summary report: {result['report_path']}")
        print(f"✓ Generated {len(result['plot_files'])} plots:")
        
        for plot_name, plot_path in result['plot_files'].items():
            print(f"  - {plot_name.replace('_', ' ').title()}: {Path(plot_path).name}")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error generating visualization report: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
