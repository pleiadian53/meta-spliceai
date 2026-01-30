#!/usr/bin/env python3
"""
Artifact Validation Script
=========================

Simple script to validate meta-model training artifacts and identify
corrupted or incorrectly formatted files.

Usage:
    python scripts/validate_artifacts.py [--artifacts-dir PATH] [--output-report PATH]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.builder.artifact_validator import ArtifactValidator


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate meta-model training artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="data/ensembl/spliceai_eval/meta_models",
        help="Directory containing artifacts"
    )
    
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Output file for validation report"
    )
    
    parser.add_argument(
        "--list-corrupted",
        action="store_true",
        help="Only list corrupted files (for easy deletion)"
    )
    
    parser.add_argument(
        "--list-wrong-schema",
        action="store_true",
        help="Only list files with wrong schemas"
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    artifacts_dir = Path(args.artifacts_dir)
    validator = ArtifactValidator(artifacts_dir)
    
    print(f"Validating artifacts in: {artifacts_dir}")
    print("=" * 60)
    
    # Run validation
    results = validator.validate_directory()
    
    # Generate report
    report = validator.generate_report(results, args.output_report)
    
    # Print report
    print(report)
    
    # Handle special output modes
    if args.list_corrupted:
        print("\nCORRUPTED FILES TO DELETE:")
        print("-" * 40)
        for result in results["corrupted"]:
            print(f"  {result.file_path}")
        return
    
    if args.list_wrong_schema:
        print("\nFILES WITH WRONG SCHEMA:")
        print("-" * 40)
        for result in results["wrong_schema"]:
            print(f"  {result.file_path}")
        return
    
    # Summary
    total_files = sum(len(results[key]) for key in ["valid", "invalid", "corrupted", "wrong_schema"])
    corrupted_files = len(results["corrupted"])
    wrong_schema_files = len(results["wrong_schema"])
    
    if corrupted_files > 0 or wrong_schema_files > 0:
        print(f"\n⚠️  Found {corrupted_files} corrupted files and {wrong_schema_files} files with wrong schemas.")
        print("   Consider cleaning up these files before proceeding with training data assembly.")
    else:
        print(f"\n✅ All {total_files} files are valid!")


if __name__ == "__main__":
    main() 