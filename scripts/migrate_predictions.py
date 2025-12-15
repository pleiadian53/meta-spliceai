#!/usr/bin/env python3
"""
Migrate predictions from old directory structure to new clean structure.

Old structure (redundant, nested):
    predictions/diverse_test_{mode}/predictions/{gene_id}_{mode}/combined_predictions.parquet

New structure (clean, flat):
    predictions/{mode}/{gene_id}/combined_predictions.parquet
    predictions/{mode}/tests/{gene_id}/combined_predictions.parquet

Usage:
    # Dry run (shows what would be done)
    python scripts/migrate_predictions.py

    # Actually perform migration
    python scripts/migrate_predictions.py --execute

    # Clean up old directories after migration
    python scripts/migrate_predictions.py --execute --cleanup

Created: 2025-10-28
"""

import sys
from pathlib import Path
import argparse
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_prediction_files(base_dir: Path):
    """Find all prediction files in old structure."""
    return list(base_dir.rglob("combined_predictions.parquet"))


def parse_old_path(file_path: Path):
    """
    Parse old path structure to extract mode, gene_id, and test status.
    
    Examples:
        predictions/diverse_test_hybrid/predictions/ENSG00000169239_hybrid/combined_predictions.parquet
        → mode='hybrid', gene_id='ENSG00000169239', is_test=True
        
        predictions/hybrid/predictions/ENSG00000169239_hybrid/combined_predictions.parquet
        → mode='hybrid', gene_id='ENSG00000169239', is_test=False
    """
    parts = file_path.parts
    
    mode = None
    gene_id = None
    is_test = False
    
    # Detect mode from directory names
    for part in parts:
        part_lower = part.lower()
        
        # Detect test run
        if 'test' in part_lower or 'diverse' in part_lower:
            is_test = True
        
        # Detect mode
        if 'base_only' in part_lower or 'base-only' in part_lower:
            mode = 'base_only'
        elif 'meta_only' in part_lower or 'meta-only' in part_lower:
            mode = 'meta_only'
        elif 'hybrid' in part_lower:
            mode = 'hybrid'
    
    # Extract gene ID
    for part in parts:
        if part.startswith('ENSG'):
            # Remove mode suffix if present
            gene_id = part.split('_')[0]
            break
    
    return mode, gene_id, is_test


def construct_new_path(base_dir: Path, mode: str, gene_id: str, is_test: bool) -> Path:
    """
    Construct new path according to clean structure.
    
    Examples:
        base_dir='predictions', mode='hybrid', gene_id='ENSG00000169239', is_test=False
        → predictions/hybrid/ENSG00000169239/combined_predictions.parquet
        
        base_dir='predictions', mode='hybrid', gene_id='ENSG00000169239', is_test=True
        → predictions/hybrid/tests/ENSG00000169239/combined_predictions.parquet
    """
    if is_test:
        return base_dir / mode / "tests" / gene_id / "combined_predictions.parquet"
    else:
        return base_dir / mode / gene_id / "combined_predictions.parquet"


def migrate_predictions(
    old_base: Path,
    new_base: Path,
    dry_run: bool = True,
    cleanup_old: bool = False
):
    """
    Migrate predictions from old to new structure.
    
    Parameters
    ----------
    old_base : Path
        Old predictions directory
    new_base : Path
        New predictions directory  
    dry_run : bool
        If True, only show what would be done
    cleanup_old : bool
        If True, remove old directories after successful migration
    """
    print("="*80)
    print("PREDICTIONS DIRECTORY MIGRATION")
    print("="*80)
    print(f"Old base: {old_base}")
    print(f"New base: {new_base}")
    print(f"Mode:     {'DRY RUN (no changes)' if dry_run else 'EXECUTE (will copy files)'}")
    if cleanup_old and not dry_run:
        print(f"Cleanup:  Will remove old directories after migration")
    print("")
    
    # Find all prediction files
    pred_files = find_prediction_files(old_base)
    
    if not pred_files:
        print(f"⚠️  No prediction files found in {old_base}")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    print("")
    
    # Group by old directory for cleanup tracking
    old_dirs = set()
    migrations = []
    skipped = []
    
    for old_file in pred_files:
        # Parse old structure
        mode, gene_id, is_test = parse_old_path(old_file)
        
        if not mode or not gene_id:
            skipped.append((old_file, "Could not parse path"))
            continue
        
        # Construct new path
        new_file = construct_new_path(new_base, mode, gene_id, is_test)
        
        # Track old directory for cleanup
        old_dir_to_remove = None
        for part_idx, part in enumerate(old_file.parts):
            if 'diverse' in part.lower() or (part.endswith('_only') or part == 'hybrid'):
                # This is the top-level dir to remove later
                old_dir_to_remove = Path(*old_file.parts[:part_idx+1])
                break
        
        if old_dir_to_remove:
            old_dirs.add(old_dir_to_remove)
        
        migrations.append({
            'old': old_file,
            'new': new_file,
            'mode': mode,
            'gene_id': gene_id,
            'is_test': is_test,
            'size_mb': old_file.stat().st_size / 1024 / 1024
        })
    
    # Summary
    print("="*80)
    print("MIGRATION PLAN")
    print("="*80)
    
    # Group by mode and test status
    by_mode = {}
    total_size_mb = 0
    
    for m in migrations:
        key = (m['mode'], m['is_test'])
        if key not in by_mode:
            by_mode[key] = []
        by_mode[key].append(m)
        total_size_mb += m['size_mb']
    
    for (mode, is_test), items in sorted(by_mode.items()):
        test_label = "tests" if is_test else "production"
        size_mb = sum(i['size_mb'] for i in items)
        print(f"\n{mode:12s} ({test_label:10s}): {len(items):3d} files, {size_mb:6.1f} MB")
        
        # Show first 3 examples
        for item in items[:3]:
            print(f"  {item['gene_id']}")
        
        if len(items) > 3:
            print(f"  ... and {len(items) - 3} more")
    
    print(f"\nTotal: {len(migrations)} files, {total_size_mb:.1f} MB")
    
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} files:")
        for old_file, reason in skipped[:5]:
            print(f"  {old_file.name}: {reason}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")
    
    print("")
    
    if dry_run:
        print("="*80)
        print("DRY RUN COMPLETE")
        print("="*80)
        print("\nTo execute migration, run:")
        print("  python scripts/migrate_predictions.py --execute")
        print("\nTo execute and cleanup old directories:")
        print("  python scripts/migrate_predictions.py --execute --cleanup")
        return
    
    # Execute migration
    print("="*80)
    print("EXECUTING MIGRATION")
    print("="*80)
    print("")
    
    success_count = 0
    error_count = 0
    
    for item in migrations:
        try:
            # Create parent directory
            item['new'].parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(item['old'], item['new'])
            
            # Verify
            if item['new'].exists() and item['new'].stat().st_size == item['old'].stat().st_size:
                success_count += 1
                print(f"✅ {item['gene_id']:20s} ({item['mode']:10s}) → {item['new'].parent}")
            else:
                error_count += 1
                print(f"❌ {item['gene_id']:20s} ({item['mode']:10s}) - size mismatch")
        
        except Exception as e:
            error_count += 1
            print(f"❌ {item['gene_id']:20s} ({item['mode']:10s}) - {e}")
    
    print("")
    print("="*80)
    print("MIGRATION SUMMARY")
    print("="*80)
    print(f"✅ Success: {success_count}/{len(migrations)}")
    print(f"❌ Errors:  {error_count}/{len(migrations)}")
    
    # Cleanup old directories
    if cleanup_old and success_count > 0:
        print("")
        print("="*80)
        print("CLEANING UP OLD DIRECTORIES")
        print("="*80)
        print(f"\nRemoving {len(old_dirs)} old directory trees...")
        
        for old_dir in sorted(old_dirs):
            try:
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                    print(f"✅ Removed: {old_dir}")
            except Exception as e:
                print(f"❌ Failed to remove {old_dir}: {e}")
        
        print("\n✅ Cleanup complete")
    
    print("")
    print("="*80)
    print("MIGRATION COMPLETE")
    print("="*80)
    
    if not cleanup_old:
        print("\nOld directories are still present.")
        print("To remove them, run:")
        print("  python scripts/migrate_predictions.py --execute --cleanup")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate predictions to new clean directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would happen)
  python scripts/migrate_predictions.py
  
  # Execute migration (copy files)
  python scripts/migrate_predictions.py --execute
  
  # Execute and cleanup old directories
  python scripts/migrate_predictions.py --execute --cleanup
  
  # Custom directories
  python scripts/migrate_predictions.py --old /path/to/old --new /path/to/new --execute
"""
    )
    
    parser.add_argument(
        '--old',
        type=Path,
        default=project_root / 'predictions',
        help='Old predictions directory (default: predictions/)'
    )
    
    parser.add_argument(
        '--new',
        type=Path,
        default=project_root / 'predictions',
        help='New predictions directory (default: predictions/)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute migration (default is dry run)'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove old directories after successful migration'
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.old.exists():
        print(f"❌ Old directory does not exist: {args.old}")
        sys.exit(1)
    
    # Execute
    migrate_predictions(
        old_base=args.old,
        new_base=args.new,
        dry_run=not args.execute,
        cleanup_old=args.cleanup
    )


if __name__ == '__main__':
    main()

