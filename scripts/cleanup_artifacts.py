#!/usr/bin/env python3
"""
Artifact Cleanup Script
=======================

This script uses the artifact validator to identify and clean up corrupted
or incorrectly formatted files from the meta_models directory.

Usage:
    python scripts/cleanup_artifacts.py [--dry-run] [--artifacts-dir PATH]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.builder.artifact_validator import ArtifactValidator


def main():
    """Main cleanup function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean up corrupted or wrong-schema artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="data/ensembl/spliceai_eval/meta_models",
        help="Directory containing artifacts to clean up",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--corrupted-only",
        action="store_true",
        help="Only delete corrupted files, not wrong-schema files",
    )
    parser.add_argument(
        "--wrong-schema-only",
        action="store_true",
        help="Only delete wrong-schema files, not corrupted files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=1,
        help="Increase verbosity",
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ArtifactValidator(args.artifacts_dir)
    
    if args.verbose:
        print(f"🔍 Scanning artifacts in: {args.artifacts_dir}")
    
    # Run validation
    validation_results = validator.validate_directory()
    
    # Collect files to delete
    files_to_delete = []
    
    if not args.wrong_schema_only:
        corrupted_files = []
        for result in validation_results.get("corrupted_files", []):
            corrupted_files.append(str(result.file_path))
        files_to_delete.extend(corrupted_files)
        if args.verbose:
            print(f"📁 Found {len(corrupted_files)} corrupted files")
    
    if not args.corrupted_only:
        wrong_schema_files = []
        for result in validation_results.get("wrong_schema", []):
            wrong_schema_files.append(str(result.file_path))
        files_to_delete.extend(wrong_schema_files)
        if args.verbose:
            print(f"📁 Found {len(wrong_schema_files)} files with wrong schemas")
    
    if not files_to_delete:
        print("✅ No files to clean up!")
        return
    
    # Show what will be deleted
    print(f"\n🗑️  Files to {'delete' if not args.dry_run else 'delete (DRY RUN)'}:")
    total_size = 0
    
    for file_path in sorted(files_to_delete):
        size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        total_size += size
        size_mb = size / (1024 * 1024)
        print(f"  • {file_path} ({size_mb:.1f}MB)")
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n📊 Total size: {total_size_mb:.1f}MB")
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to actually delete these files.")
        return
    
    # Confirm deletion
    if args.verbose:
        response = input(f"\n❓ Delete {len(files_to_delete)} files ({total_size_mb:.1f}MB)? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Cancelled.")
            return
    
    # Delete files
    deleted_count = 0
    deleted_size = 0
    
    for file_path in files_to_delete:
        try:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                path.unlink()
                deleted_count += 1
                deleted_size += size
                if args.verbose >= 2:
                    print(f"  ✅ Deleted: {file_path}")
            else:
                if args.verbose >= 2:
                    print(f"  ⚠️  File not found: {file_path}")
        except Exception as e:
            print(f"  ❌ Error deleting {file_path}: {e}")
    
    deleted_size_mb = deleted_size / (1024 * 1024)
    print(f"\n✅ Successfully deleted {deleted_count} files ({deleted_size_mb:.1f}MB)")


if __name__ == "__main__":
    main() 