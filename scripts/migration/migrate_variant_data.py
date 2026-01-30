#!/usr/bin/env python3
"""
Migrate variant database data to systematic organization.

This script reorganizes variant database data to follow the systematic
path structure defined by the genomic resource management system.

Usage:
------
# Dry run (see what would be moved)
python scripts/migrate_variant_data.py --dry-run

# Execute migration
python scripts/migrate_variant_data.py

# Migrate specific database only
python scripts/migrate_variant_data.py --database splicevardb
python scripts/migrate_variant_data.py --database clinvar
"""

import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)


def get_migration_plan() -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Generate migration plan for all variant databases.
    
    Returns
    -------
    dict
        Mapping from database name to list of (source, target) path tuples
    """
    # Get systematic paths
    manager = CaseStudyResourceManager()
    project_root = Path(__file__).parents[1]
    
    migration_plan = {
        "splicevardb": [],
        "clinvar": [],
        "training_datasets": []
    }
    
    # SpliceVarDB migrations
    old_splicevardb = project_root / "meta_spliceai/splice_engine/case_studies/workflows/splicevardb"
    new_splicevardb = manager.case_study_paths.splicevardb
    
    if old_splicevardb.exists():
        for file in old_splicevardb.glob("*"):
            if file.is_file() and not file.name.startswith("_"):
                # Determine target subdirectory
                if file.suffix in [".tsv", ".csv"]:
                    target = new_splicevardb / "raw" / file.name
                elif file.suffix in [".parquet", ".vcf", ".json"]:
                    target = new_splicevardb / "processed" / file.name
                else:
                    target = new_splicevardb / file.name
                
                migration_plan["splicevardb"].append((file, target))
    
    # ClinVar migrations
    # Check multiple possible old locations
    old_clinvar_code = project_root / "meta_spliceai/splice_engine/case_studies/data_sources/datasets/clinvar_20250831"
    old_clinvar_data = manager.genomic_manager.genome.base_data_dir / "ensembl" / "clinvar"  # Old location
    new_clinvar = manager.case_study_paths.clinvar  # New: ensembl/case_studies/clinvar/
    
    # From code directory
    if old_clinvar_code.exists():
        migration_plan["clinvar"].append((old_clinvar_code, new_clinvar / "clinvar_20250831"))
    
    # From old data location (if different from new)
    if old_clinvar_data.exists() and old_clinvar_data != new_clinvar:
        # Move entire directory
        migration_plan["clinvar"].append((old_clinvar_data, new_clinvar))
    
    # Training datasets migrations
    # NOTE: Training datasets STAY in data/ root (no migration needed for existing ones)
    # Only migrate if they're in the wrong location (under code directories)
    old_datasets = project_root / "meta_spliceai/splice_engine/case_studies/data_sources/datasets"
    new_training = manager.genomic_manager.genome.base_data_dir  # data/ root
    
    if old_datasets.exists():
        for item in old_datasets.iterdir():
            if item.is_dir() and item.name.startswith("train_"):
                # Only migrate if not already in data/ root
                target = new_training / item.name
                if not target.exists():  # Avoid duplicate
                    migration_plan["training_datasets"].append((item, target))
    
    return migration_plan


def print_migration_plan(plan: Dict[str, List[Tuple[Path, Path]]]):
    """Print the migration plan in a readable format."""
    print("\n" + "=" * 70)
    print("MIGRATION PLAN")
    print("=" * 70)
    
    total_moves = sum(len(moves) for moves in plan.values())
    
    if total_moves == 0:
        print("\n✓ No migrations needed - all data is already organized!")
        return
    
    for database, moves in plan.items():
        if not moves:
            continue
        
        print(f"\n{database.upper()}")
        print("-" * 70)
        
        for source, target in moves:
            if source.is_dir():
                print(f"  DIR:  {source}")
                print(f"    →   {target}")
            else:
                print(f"  FILE: {source.name}")
                print(f"    →   {target}")
        
        print(f"\n  Total: {len(moves)} item(s)")
    
    print("\n" + "=" * 70)
    print(f"Total migrations: {total_moves}")
    print("=" * 70)


def execute_migration(
    plan: Dict[str, List[Tuple[Path, Path]]],
    dry_run: bool = False
):
    """
    Execute the migration plan.
    
    Parameters
    ----------
    plan : dict
        Migration plan from get_migration_plan()
    dry_run : bool
        If True, only print what would be done without moving files
    """
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be moved")
        print_migration_plan(plan)
        return
    
    print("\n" + "=" * 70)
    print("EXECUTING MIGRATION")
    print("=" * 70)
    
    # Create backup metadata
    backup_info = {
        "timestamp": datetime.now().isoformat(),
        "migrations": {}
    }
    
    total_success = 0
    total_failed = 0
    
    for database, moves in plan.items():
        if not moves:
            continue
        
        print(f"\n{database.upper()}")
        print("-" * 70)
        
        database_moves = []
        
        for source, target in moves:
            try:
                # Create target directory
                target.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file or directory
                if source.is_dir():
                    print(f"  Moving directory: {source.name}")
                    shutil.move(str(source), str(target))
                else:
                    print(f"  Moving file: {source.name}")
                    shutil.move(str(source), str(target))
                
                database_moves.append({
                    "source": str(source),
                    "target": str(target),
                    "success": True
                })
                total_success += 1
                
            except Exception as e:
                print(f"  ✗ Failed to move {source.name}: {e}")
                database_moves.append({
                    "source": str(source),
                    "target": str(target),
                    "success": False,
                    "error": str(e)
                })
                total_failed += 1
        
        backup_info["migrations"][database] = database_moves
    
    # Save migration log
    log_file = Path("migration_log.json")
    with open(log_file, "w") as f:
        json.dump(backup_info, f, indent=2)
    
    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ Successfully migrated: {total_success}")
    if total_failed > 0:
        print(f"✗ Failed migrations: {total_failed}")
    print(f"\nMigration log saved to: {log_file}")
    
    # Print new locations
    print("\n" + "=" * 70)
    print("NEW DATA LOCATIONS")
    print("=" * 70)
    
    manager = CaseStudyResourceManager()
    print(f"\nSpliceVarDB:        {manager.case_study_paths.splicevardb}")
    print(f"ClinVar:            {manager.case_study_paths.clinvar}")
    print(f"MutSpliceDB:        {manager.case_study_paths.mutsplicedb}")
    print(f"DBASS:              {manager.case_study_paths.dbass}")
    print(f"Training datasets:  {manager.genomic_manager.genome.base_data_dir}/train_*")
    print(f"\nNote: All variant databases are now at same level under case_studies/")


def verify_migration(plan: Dict[str, List[Tuple[Path, Path]]]):
    """Verify that migration was successful."""
    print("\n" + "=" * 70)
    print("VERIFYING MIGRATION")
    print("=" * 70)
    
    all_good = True
    
    for database, moves in plan.items():
        if not moves:
            continue
        
        print(f"\n{database.upper()}")
        
        for source, target in moves:
            if target.exists():
                print(f"  ✓ {target.name}")
            else:
                print(f"  ✗ {target.name} - NOT FOUND")
                all_good = False
            
            if source.exists():
                print(f"    ⚠️  Source still exists: {source}")
    
    if all_good:
        print("\n✓ All migrations verified successfully!")
    else:
        print("\n✗ Some migrations failed verification")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Migrate variant database data to systematic organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without moving files"
    )
    parser.add_argument(
        "--database",
        choices=["splicevardb", "clinvar", "training_datasets", "all"],
        default="all",
        help="Migrate specific database only"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration after completion"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force migration even if target exists"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("VARIANT DATA MIGRATION TO SYSTEMATIC ORGANIZATION")
    print("=" * 70)
    
    # Generate migration plan
    print("\nGenerating migration plan...")
    plan = get_migration_plan()
    
    # Filter by database if specified
    if args.database != "all":
        plan = {args.database: plan[args.database]}
    
    # Check if any migrations needed
    total_moves = sum(len(moves) for moves in plan.values())
    if total_moves == 0:
        print("\n✓ No migrations needed - all data is already organized!")
        return
    
    # Print plan
    print_migration_plan(plan)
    
    # Execute or dry run
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - Run without --dry-run to execute migration")
    else:
        # Confirm before executing
        response = input("\nProceed with migration? [y/N]: ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return
        
        execute_migration(plan, dry_run=False)
        
        if args.verify:
            verify_migration(plan)


if __name__ == "__main__":
    main()

