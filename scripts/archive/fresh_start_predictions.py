#!/usr/bin/env python3
"""
Fresh Start: Clean Predictions Directory

Remove all old predictions and create clean directory structure.
Essential predictions can be easily regenerated from trained models.

Usage:
    # Dry run (see what would be removed)
    python scripts/fresh_start_predictions.py

    # Execute cleanup
    python scripts/fresh_start_predictions.py --execute

    # Keep specific predictions
    python scripts/fresh_start_predictions.py --execute --keep hybrid/ENSG00000169239

Created: 2025-10-28
"""

import sys
import argparse
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def format_size(bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"


def analyze_predictions_directory(base_dir: Path):
    """Analyze current predictions directory structure."""
    if not base_dir.exists():
        return None
    
    analysis = {
        'total_size': get_directory_size(base_dir),
        'subdirs': [],
        'file_count': 0,
        'dir_count': 0
    }
    
    # Analyze top-level subdirectories
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir():
            size = get_directory_size(subdir)
            file_count = sum(1 for _ in subdir.rglob('*') if _.is_file())
            
            analysis['subdirs'].append({
                'name': subdir.name,
                'size': size,
                'size_str': format_size(size),
                'file_count': file_count
            })
            analysis['file_count'] += file_count
            analysis['dir_count'] += 1
    
    return analysis


def create_clean_structure(base_dir: Path, dry_run: bool = True):
    """Create clean directory structure."""
    modes = ['base_only', 'hybrid', 'meta_only']
    
    paths_to_create = [
        base_dir / 'spliceai_eval' / 'meta_models' / 'analysis_sequences',
        base_dir / 'spliceai_eval' / 'meta_models' / 'complete_base_predictions',
    ]
    
    for mode in modes:
        paths_to_create.append(base_dir / mode)
        paths_to_create.append(base_dir / mode / 'tests')
    
    if dry_run:
        print("\n[DRY RUN] Would create:")
        for path in paths_to_create:
            print(f"  {path}")
    else:
        print("\nCreating clean structure...")
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {path}")


def fresh_start_cleanup(
    base_dir: Path,
    dry_run: bool = True,
    keep_paths: list = None
):
    """
    Perform fresh start cleanup.
    
    Parameters
    ----------
    base_dir : Path
        Base predictions directory
    dry_run : bool
        If True, only show what would be done
    keep_paths : list
        List of relative paths to keep (e.g., ['hybrid/ENSG00000169239'])
    """
    keep_paths = keep_paths or []
    
    print("="*80)
    print("FRESH START: PREDICTIONS CLEANUP")
    print("="*80)
    print(f"Directory: {base_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'EXECUTE (will delete)'}")
    if keep_paths:
        print(f"Keep: {', '.join(keep_paths)}")
    print("")
    
    # Analyze current state
    print("Analyzing current predictions directory...")
    analysis = analyze_predictions_directory(base_dir)
    
    if not analysis:
        print(f"✅ Directory does not exist or is empty: {base_dir}")
        create_clean_structure(base_dir, dry_run)
        return
    
    # Report current state
    print("")
    print("="*80)
    print("CURRENT STATE")
    print("="*80)
    print(f"Total size: {format_size(analysis['total_size'])}")
    print(f"Directories: {analysis['dir_count']}")
    print(f"Files: {analysis['file_count']:,}")
    print("")
    
    if analysis['subdirs']:
        print("Top-level subdirectories:")
        for subdir in analysis['subdirs']:
            print(f"  {subdir['name']:30s} {subdir['size_str']:>10s} ({subdir['file_count']:,} files)")
    
    print("")
    
    # Determine what to keep
    keep_full_paths = []
    if keep_paths:
        print("="*80)
        print("PATHS TO PRESERVE")
        print("="*80)
        for rel_path in keep_paths:
            full_path = base_dir / rel_path
            if full_path.exists():
                size = get_directory_size(full_path) if full_path.is_dir() else full_path.stat().st_size
                keep_full_paths.append(full_path)
                print(f"  ✅ {rel_path:40s} {format_size(size):>10s}")
            else:
                print(f"  ⚠️  {rel_path:40s} NOT FOUND")
        print("")
    
    # Determine what to remove
    print("="*80)
    print("CLEANUP PLAN")
    print("="*80)
    
    if not keep_paths:
        print("🗑️  Will remove EVERYTHING in predictions/")
        print(f"   Total: {format_size(analysis['total_size'])}")
    else:
        print("Will remove all except kept paths:")
        for subdir in analysis['subdirs']:
            subdir_path = base_dir / subdir['name']
            is_kept = any(
                str(kept).startswith(str(subdir_path))
                for kept in keep_full_paths
            )
            status = "  (kept)" if is_kept else ""
            print(f"  {subdir['name']:30s} {subdir['size_str']:>10s}{status}")
    
    print("")
    
    if dry_run:
        print("="*80)
        print("DRY RUN COMPLETE")
        print("="*80)
        print("\nTo execute cleanup, run:")
        print("  python scripts/fresh_start_predictions.py --execute")
        if keep_paths:
            keep_args = ' '.join(f'--keep {p}' for p in keep_paths)
            print(f"  python scripts/fresh_start_predictions.py --execute {keep_args}")
        return
    
    # Execute cleanup
    print("="*80)
    print("EXECUTING CLEANUP")
    print("="*80)
    print("")
    
    if not keep_paths:
        # Remove everything
        print(f"Removing entire directory: {base_dir}")
        if base_dir.exists():
            shutil.rmtree(base_dir)
            print(f"✅ Removed: {format_size(analysis['total_size'])}")
    else:
        # Remove selectively
        removed_size = 0
        removed_count = 0
        
        for subdir in analysis['subdirs']:
            subdir_path = base_dir / subdir['name']
            
            # Check if this should be kept
            is_kept = any(
                str(kept).startswith(str(subdir_path))
                for kept in keep_full_paths
            )
            
            if not is_kept and subdir_path.exists():
                print(f"Removing: {subdir['name']:30s} {subdir['size_str']:>10s}")
                shutil.rmtree(subdir_path)
                removed_size += subdir['size']
                removed_count += 1
                print(f"  ✅ Removed")
        
        print("")
        print(f"Removed {removed_count} directories ({format_size(removed_size)})")
    
    # Create clean structure
    print("")
    create_clean_structure(base_dir, dry_run=False)
    
    # Final report
    print("")
    print("="*80)
    print("CLEANUP COMPLETE")
    print("="*80)
    
    if base_dir.exists():
        final_analysis = analyze_predictions_directory(base_dir)
        print(f"Before: {format_size(analysis['total_size'])}")
        print(f"After:  {format_size(final_analysis['total_size'])}")
        print(f"Freed:  {format_size(analysis['total_size'] - final_analysis['total_size'])}")
    else:
        print(f"Before: {format_size(analysis['total_size'])}")
        print(f"After:  0 B (directory removed)")
        print(f"Freed:  {format_size(analysis['total_size'])}")
    
    print("")
    print("✅ Clean directory structure created")
    print("✅ Ready for new predictions with OutputManager")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fresh start cleanup for predictions directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would happen)
  python scripts/fresh_start_predictions.py
  
  # Execute cleanup (remove everything)
  python scripts/fresh_start_predictions.py --execute
  
  # Keep specific predictions
  python scripts/fresh_start_predictions.py --execute --keep hybrid/ENSG00000169239
  
  # Keep multiple paths
  python scripts/fresh_start_predictions.py --execute \\
    --keep hybrid/ENSG00000169239 \\
    --keep meta_only/ENSG00000171812
"""
    )
    
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        default=project_root / 'predictions',
        help='Predictions directory (default: predictions/)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute cleanup (default is dry run)'
    )
    
    parser.add_argument(
        '--keep',
        action='append',
        help='Path to keep (relative to predictions dir, can be specified multiple times)'
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.predictions_dir.exists():
        print(f"✅ Predictions directory does not exist: {args.predictions_dir}")
        print("Nothing to clean up!")
        sys.exit(0)
    
    # Execute
    fresh_start_cleanup(
        base_dir=args.predictions_dir,
        dry_run=not args.execute,
        keep_paths=args.keep or []
    )


if __name__ == '__main__':
    main()

