#!/usr/bin/env python3
"""
Generate a gene manifest for an existing training dataset.

This script creates a CSV file showing all genes included in a training dataset,
with their gene names, file locations, and indices for easy lookup.

Usage:
    python scripts/generate_gene_manifest.py /path/to/train_dataset_trimmed
    python scripts/generate_gene_manifest.py /path/to/train_dataset_trimmed --output manifest.csv
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.builder.incremental_builder import (
    generate_gene_manifest,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a gene manifest for an existing training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the training dataset directory (should contain a 'master' subdirectory)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the manifest file. If not specified, creates 'gene_manifest.csv' in the dataset directory.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=1,
        help="Increase verbosity. Use -v for standard output, -vv for detailed output.",
    )
    
    args = parser.parse_args()
    
    try:
        manifest_path = generate_gene_manifest(
            dataset_dir=args.dataset_dir,
            output_path=args.output,
            verbose=args.verbose,
        )
        print(f"\n✅ Gene manifest generated successfully!")
        print(f"📁 Location: {manifest_path}")
        print(f"📊 You can now use this manifest to:")
        print(f"   • Look up which genes are in your training dataset")
        print(f"   • Find the file location of specific genes")
        print(f"   • Get gene names for easier interpretation")
        
    except Exception as e:
        print(f"❌ Error generating gene manifest: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 