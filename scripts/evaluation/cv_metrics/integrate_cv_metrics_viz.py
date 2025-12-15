#!/usr/bin/env python3
"""
Integration script for CV metrics visualization.

This script modifies the run_gene_cv_sigmoid.py to automatically generate
comprehensive visualization reports for CV metrics.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def integrate_cv_metrics_viz_into_workflow():
    """
    Integrate CV metrics visualization into the main CV workflow.
    
    This function modifies the run_gene_cv_sigmoid.py to include automatic
    CV metrics visualization generation.
    """
    
    cv_script_path = Path("meta_spliceai/splice_engine/meta_models/training/run_gene_cv_sigmoid.py")
    
    if not cv_script_path.exists():
        print(f"Error: CV script not found at {cv_script_path}")
        return False
    
    # Read the current script
    with open(cv_script_path, 'r') as f:
        content = f.read()
    
    # Check if integration is already present
    if 'cv_metrics_viz' in content:
        print("CV metrics visualization integration already present")
        return True
    
    # Add the import statement
    import_line = "from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import generate_cv_metrics_report"
    
    # Find the existing imports section
    import_section_end = content.find("from meta_spliceai.splice_engine.utils_doc import")
    if import_section_end == -1:
        print("Error: Could not find import section")
        return False
    
    # Insert the new import after the existing imports
    import_insertion_point = content.find("\n", import_section_end)
    new_content = (
        content[:import_insertion_point] + 
        "\n" + import_line + 
        content[import_insertion_point:]
    )
    
    # Add the visualization generation code
    viz_code = '''
    # Generate CV metrics visualization report
    try:
        print("\\nGenerating CV metrics visualization report...")
        cv_metrics_csv = out_dir / "gene_cv_metrics.csv"
        if cv_metrics_csv.exists():
            viz_result = generate_cv_metrics_report(
                csv_path=cv_metrics_csv,
                out_dir=out_dir,
                plot_format=args.plot_format,
                dpi=300
            )
            print(f"[INFO] CV metrics visualization completed successfully")
            print(f"[INFO] Visualization directory: {viz_result['visualization_dir']}")
            print(f"[INFO] Generated {len(viz_result['plot_files'])} plots:")
            for plot_name, plot_path in viz_result['plot_files'].items():
                print(f"  - {plot_name.replace('_', ' ').title()}: {Path(plot_path).name}")
        else:
            print(f"[WARNING] CV metrics CSV not found at {cv_metrics_csv}")
    except Exception as e:
        print(f"[WARNING] CV metrics visualization failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
'''
    
    # Find the location to insert the visualization code
    # Insert after the metrics aggregate section
    insertion_point = new_content.find('with open(out_dir / "metrics_aggregate.json", "w") as fh:')
    if insertion_point == -1:
        print("Error: Could not find metrics aggregate section")
        return False
    
    # Find the end of the metrics aggregate section
    insertion_point = new_content.find('\n', new_content.find('json.dump(mean_metrics.to_dict(), fh, indent=2)', insertion_point))
    
    # Insert the visualization code
    new_content = (
        new_content[:insertion_point] + 
        viz_code + 
        new_content[insertion_point:]
    )
    
    # Write the modified content back
    with open(cv_script_path, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully integrated CV metrics visualization into {cv_script_path}")
    print("The CV workflow will now automatically generate visualization reports")
    return True

def create_standalone_cv_viz_script():
    """Create a standalone script for CV metrics visualization."""
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    script_path = Path("scripts/generate_cv_metrics_viz.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    script_path.chmod(0o755)
    
    print(f"Created standalone CV metrics visualization script: {script_path}")
    return script_path

def main():
    """Main entry point for the integration script."""
    
    print("CV Metrics Visualization Integration")
    print("=" * 50)
    
    # Create standalone script
    standalone_script = create_standalone_cv_viz_script()
    
    # Integrate into workflow
    success = integrate_cv_metrics_viz_into_workflow()
    
    if success:
        print("\n✓ Integration completed successfully!")
        print("\nNext steps:")
        print("1. Run your normal CV workflow - it will now automatically generate visualization reports")
        print("2. Or use the standalone script for existing CSV files:")
        print(f"   python {standalone_script} path/to/gene_cv_metrics.csv --out-dir output_dir")
        print("\nExample usage:")
        print("   python scripts/generate_cv_metrics_viz.py models/meta_model_test/gene_cv_metrics.csv --out-dir viz_output")
    else:
        print("\n✗ Integration failed!")
        print("Please check the error messages above and try again")

if __name__ == "__main__":
    main() 