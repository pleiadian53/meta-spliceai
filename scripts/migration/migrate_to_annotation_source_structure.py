#!/usr/bin/env python3
"""
Migrate existing data to new annotation-source-based directory structure.

OLD STRUCTURE:
  data/ensembl/
  ├── GRCh38/
  │   └── ...
  └── GRCh37/
      └── ...

NEW STRUCTURE:
  data/
  ├── ensembl/
  │   ├── GRCh38/
  │   │   └── ...
  │   └── GRCh37/
  │       └── ...
  └── mane/
      └── GRCh38/
          └── ...

The annotation source name (ensembl, mane, gencode) already tells us the data origin,
so no additional "genomic_resources" level is needed.

NOTE: Since the old structure was already data/ensembl/, no migration is actually needed!
This script is kept for reference and future migrations.

Date: November 1, 2025
"""

import shutil
import sys
from pathlib import Path


def migrate_data(dry_run: bool = False, verbose: bool = True):
    """Migrate data to new annotation-source-based directory structure.
    
    Parameters
    ----------
    dry_run : bool
        If True, only print what would be done without actually moving files
    verbose : bool
        If True, print detailed progress information
    """
    
    # Paths
    old_root = Path("data/ensembl")
    new_root = Path("data/genomic_resources/ensembl")
    
    if verbose:
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║     📦 MIGRATING TO ANNOTATION-SOURCE-BASED DIRECTORY STRUCTURE            ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"Old root: {old_root}")
        print(f"New root: {new_root}")
        print()
    
    # Check if old root exists
    if not old_root.exists():
        if verbose:
            print(f"✅ No old data to migrate ('{old_root}' does not exist)")
            print("   This is expected if you're setting up a fresh installation.")
        return True
    
    # Check if old root is already a symlink
    if old_root.is_symlink():
        target = old_root.resolve()
        if verbose:
            print(f"✅ '{old_root}' is already a symlink pointing to: {target}")
            print("   Migration appears to have been completed previously.")
        return True
    
    # Create new directory structure
    if verbose:
        print(f"📁 Creating new directory structure: {new_root}")
    
    if not dry_run:
        new_root.mkdir(parents=True, exist_ok=True)
    else:
        print(f"   [DRY RUN] Would create: {new_root}")
    
    # Move all subdirectories
    moved_count = 0
    skipped_count = 0
    
    if verbose:
        print()
        print("📦 Moving subdirectories:")
    
    for item in old_root.iterdir():
        if item.is_dir():
            dest = new_root / item.name
            
            if dest.exists():
                if verbose:
                    print(f"   ⏭️  Skipping {item.name} (already exists at destination)")
                skipped_count += 1
            else:
                if verbose:
                    print(f"   📦 Moving {item.name} → {dest}")
                
                if not dry_run:
                    shutil.move(str(item), str(dest))
                moved_count += 1
        else:
            # Move files too (e.g., GTF, FASTA at root level)
            dest = new_root / item.name
            if dest.exists():
                if verbose:
                    print(f"   ⏭️  Skipping {item.name} (already exists at destination)")
                skipped_count += 1
            else:
                if verbose:
                    print(f"   📄 Moving {item.name} → {dest}")
                
                if not dry_run:
                    shutil.move(str(item), str(dest))
                moved_count += 1
    
    # Create backward compatibility symlink
    if verbose:
        print()
        print("🔗 Creating backward compatibility symlink:")
    
    # Remove old root if it's now empty (after moving everything)
    if not dry_run and old_root.exists() and not any(old_root.iterdir()):
        old_root.rmdir()
        if verbose:
            print(f"   🗑️  Removed empty directory: {old_root}")
    
    # Create symlink
    symlink_target = Path("genomic_resources/ensembl")  # Relative path
    
    if old_root.exists() and not old_root.is_symlink():
        if verbose:
            print(f"   ⚠️  Cannot create symlink: {old_root} still exists and is not empty")
            print(f"      Please manually review and remove it if safe to do so.")
        return False
    elif old_root.is_symlink():
        if verbose:
            print(f"   ✅ Symlink already exists: {old_root} → {old_root.resolve()}")
    else:
        if verbose:
            print(f"   🔗 Creating symlink: {old_root} → {symlink_target}")
        
        if not dry_run:
            old_root.symlink_to(symlink_target)
        else:
            print(f"      [DRY RUN] Would create symlink")
    
    # Summary
    if verbose:
        print()
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║     📊 MIGRATION SUMMARY                                                    ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")
        print(f"   Moved: {moved_count} items")
        print(f"   Skipped: {skipped_count} items (already existed)")
        print(f"   Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print()
        
        if dry_run:
            print("   ℹ️  This was a dry run. No changes were made.")
            print("      Run with --live to actually perform the migration.")
        else:
            print("   ✅ Migration complete!")
            print()
            print("   Next steps:")
            print("   1. Verify the new structure: ls -la data/genomic_resources/ensembl/")
            print("   2. Test the Registry: python -c 'from meta_spliceai.system.genomic_resources import Registry; r = Registry(); print(r.resolve(\"gtf\"))'")
            print("   3. Run GRCh37 workflow to verify everything works")
    
    return True


def main():
    """Main entry point for migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate to annotation-source-based directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would happen)
  python scripts/migration/migrate_to_annotation_source_structure.py --dry-run
  
  # Actually perform migration
  python scripts/migration/migrate_to_annotation_source_structure.py --live
  
  # Quiet mode (minimal output)
  python scripts/migration/migrate_to_annotation_source_structure.py --live --quiet
"""
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files (default)"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually perform the migration (moves files)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Default to dry run if neither specified
    if not args.live and not args.dry_run:
        args.dry_run = True
    
    # Perform migration
    success = migrate_data(
        dry_run=args.dry_run,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

