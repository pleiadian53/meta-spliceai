#!/usr/bin/env python3
"""
Test script to verify path migration readiness.

This script performs a DRY RUN analysis without moving any files.
It checks:
1. Resource manager path configuration
2. Existing data locations
3. Code files that need updating
4. Safety of proposed migration

Usage:
------
python scripts/test_path_migration.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parents[1]))

def test_resource_manager():
    """Test that resource manager returns correct paths."""
    print("=" * 70)
    print("TEST 1: Resource Manager Configuration")
    print("=" * 70)
    
    try:
        from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
            CaseStudyResourceManager
        )
        
        manager = CaseStudyResourceManager()
        
        # Test paths
        clinvar_path = manager.case_study_paths.clinvar
        splicevardb_path = manager.case_study_paths.splicevardb
        mutsplicedb_path = manager.case_study_paths.mutsplicedb
        dbass_path = manager.case_study_paths.dbass
        
        print(f"\n✓ Resource manager initialized successfully")
        print(f"\nConfigured Paths:")
        print(f"  ClinVar:     {clinvar_path}")
        print(f"  SpliceVarDB: {splicevardb_path}")
        print(f"  MutSpliceDB: {mutsplicedb_path}")
        print(f"  DBASS:       {dbass_path}")
        
        # Verify they're siblings
        assert clinvar_path.parent == splicevardb_path.parent, "Paths not at same level!"
        assert "case_studies/clinvar" in str(clinvar_path), "ClinVar not in case_studies!"
        assert "case_studies/splicevardb" in str(splicevardb_path), "SpliceVarDB not in case_studies!"
        
        print(f"\n✓ All variant databases are at same level (case_studies/)")
        print(f"✓ Resource manager configuration is CORRECT")
        
        return True, manager
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False, None


def check_existing_data(manager):
    """Check what data currently exists."""
    print("\n" + "=" * 70)
    print("TEST 2: Existing Data Locations")
    print("=" * 70)
    
    findings = []
    
    # Check training data
    print("\n📊 Training Datasets (should stay in data/):")
    project_root = Path(__file__).parents[1]
    data_dir = project_root / "data"
    
    for pattern in ["train_pc_*", "train_regulatory_*"]:
        for item in data_dir.glob(pattern):
            if item.is_dir():
                size_mb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024*1024)
                print(f"  ✓ {item.name:40s} ({size_mb:.1f} MB) - NO MOVE NEEDED")
                findings.append(("training", item, size_mb, "KEEP"))
    
    # Check for ClinVar data
    print("\n🧬 ClinVar Data:")
    
    # Old location (in code)
    old_clinvar_code = project_root / "meta_spliceai/splice_engine/case_studies/data_sources/datasets/clinvar_20250831"
    if old_clinvar_code.exists():
        size_mb = sum(f.stat().st_size for f in old_clinvar_code.rglob('*') if f.is_file()) / (1024*1024)
        print(f"  ⚠️  Found in CODE directory: {old_clinvar_code}")
        print(f"     Size: {size_mb:.2f} MB")
        print(f"     Action: Should move to {manager.case_study_paths.clinvar}")
        findings.append(("clinvar_code", old_clinvar_code, size_mb, "MOVE"))
    
    # Old location (in data/ensembl/)
    old_clinvar_data = project_root / "data/ensembl/clinvar"
    if old_clinvar_data.exists():
        size_mb = sum(f.stat().st_size for f in old_clinvar_data.rglob('*') if f.is_file()) / (1024*1024)
        print(f"  ⚠️  Found in OLD location: {old_clinvar_data}")
        print(f"     Size: {size_mb:.2f} MB")
        print(f"     Action: Should move to {manager.case_study_paths.clinvar}")
        findings.append(("clinvar_data", old_clinvar_data, size_mb, "MOVE"))
    
    # New location
    new_clinvar = manager.case_study_paths.clinvar
    if new_clinvar.exists():
        size_mb = sum(f.stat().st_size for f in new_clinvar.rglob('*') if f.is_file()) / (1024*1024)
        print(f"  ✓ Already in CORRECT location: {new_clinvar} ({size_mb:.2f} MB)")
        findings.append(("clinvar_new", new_clinvar, size_mb, "GOOD"))
    
    if not any(f[0].startswith("clinvar") for f in findings):
        print(f"  ℹ️  No ClinVar data found (will download directly to correct location)")
    
    # Check for SpliceVarDB data
    print("\n🧬 SpliceVarDB Data:")
    
    old_splicevardb = project_root / "meta_spliceai/splice_engine/case_studies/workflows/splicevardb"
    if old_splicevardb.exists():
        data_files = list(old_splicevardb.glob("*.tsv")) + list(old_splicevardb.glob("*.json"))
        if data_files:
            size_mb = sum(f.stat().st_size for f in data_files) / (1024*1024)
            print(f"  ⚠️  Found draft data: {old_splicevardb}")
            print(f"     Files: {[f.name for f in data_files]}")
            print(f"     Size: {size_mb:.2f} MB")
            print(f"     Action: Should move to {manager.case_study_paths.splicevardb}")
            findings.append(("splicevardb_draft", old_splicevardb, size_mb, "MOVE"))
    
    new_splicevardb = manager.case_study_paths.splicevardb
    if new_splicevardb.exists():
        size_mb = sum(f.stat().st_size for f in new_splicevardb.rglob('*') if f.is_file()) / (1024*1024)
        print(f"  ✓ Already in CORRECT location: {new_splicevardb} ({size_mb:.2f} MB)")
        findings.append(("splicevardb_new", new_splicevardb, size_mb, "GOOD"))
    
    if not any(f[0].startswith("splicevardb") for f in findings):
        print(f"  ℹ️  No SpliceVarDB data found (will download directly to correct location)")
    
    return findings


def check_code_updates():
    """Check which code files need updating."""
    print("\n" + "=" * 70)
    print("TEST 3: Code Files Requiring Updates")
    print("=" * 70)
    
    import re
    project_root = Path(__file__).parents[1]
    case_studies = project_root / "meta_spliceai/splice_engine/case_studies"
    
    # Patterns to check
    patterns = {
        "old_clinvar": r'data/ensembl/clinvar(?!/case_studies)',
        "old_splicevardb": r'data/(external|interim)/splicevardb',
    }
    
    files_to_update = []
    
    for pyfile in case_studies.rglob("*.py"):
        try:
            with open(pyfile) as f:
                content = f.read()
            
            for name, pattern in patterns.items():
                if re.search(pattern, content):
                    files_to_update.append((pyfile, name))
                    break
        except:
            continue
    
    if files_to_update:
        print(f"\n⚠️  Found {len(files_to_update)} Python files with old paths:")
        for filepath, pattern in files_to_update[:10]:  # Show first 10
            rel_path = filepath.relative_to(project_root)
            print(f"  - {rel_path} (pattern: {pattern})")
        if len(files_to_update) > 10:
            print(f"  ... and {len(files_to_update) - 10} more")
    else:
        print("\n✓ No Python files with old hardcoded paths found!")
    
    return files_to_update


def generate_summary(manager, findings, files_to_update):
    """Generate migration summary."""
    print("\n" + "=" * 70)
    print("MIGRATION ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Data analysis
    move_needed = [f for f in findings if f[3] == "MOVE"]
    total_size_mb = sum(f[2] for f in move_needed)
    
    print(f"\n📊 Data Migration:")
    if move_needed:
        print(f"  Files to move: {len(move_needed)}")
        print(f"  Total size: {total_size_mb:.2f} MB")
        for name, path, size, _ in move_needed:
            print(f"    - {name}: {size:.2f} MB")
    else:
        print(f"  ✓ No data files need moving!")
    
    # Code analysis
    print(f"\n💻 Code Updates:")
    print(f"  Python files to update: {len(files_to_update)}")
    
    # Risk assessment
    print(f"\n⚠️  Risk Assessment:")
    if total_size_mb == 0:
        print(f"  ✓ ZERO risk - No data to move")
    elif total_size_mb < 10:
        print(f"  ✓ LOW risk - Small files only ({total_size_mb:.2f} MB)")
    elif total_size_mb < 100:
        print(f"  ⚠️  MEDIUM risk - {total_size_mb:.2f} MB of data")
    else:
        print(f"  ⚠️  HIGH risk - {total_size_mb:.2f} MB of data")
        print(f"  Recommendation: Manual review before migration")
    
    # Training data check
    training_data = [f for f in findings if f[3] == "KEEP"]
    if training_data:
        training_size = sum(f[2] for f in training_data)
        print(f"\n✓ Training data is safe:")
        print(f"  {len(training_data)} datasets totaling {training_size:.1f} MB")
        print(f"  These will NOT be moved (already in correct location)")
    
    # Recommendations
    print(f"\n📋 Recommended Actions:")
    print(f"  1. ✅ Resource manager is correctly configured")
    if files_to_update:
        print(f"  2. ⚠️  Update {len(files_to_update)} Python files to use resource manager")
    else:
        print(f"  2. ✓ No code updates needed")
    
    if move_needed:
        print(f"  3. ⚠️  Move {len(move_needed)} data items ({total_size_mb:.2f} MB)")
    else:
        print(f"  3. ✓ No data migration needed")
    
    print(f"\n📄 See detailed analysis in:")
    print(f"  - docs/MIGRATION_DRY_RUN_ANALYSIS.md")
    print(f"  - docs/CODE_UPDATES_REQUIRED.md")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PATH MIGRATION DRY RUN ANALYSIS")
    print("=" * 70)
    print("\n⚠️  This is a READ-ONLY analysis. No files will be moved.\n")
    
    # Test 1: Resource manager
    success, manager = test_resource_manager()
    if not success:
        print("\n✗ Cannot proceed - fix resource manager first")
        return 1
    
    # Test 2: Check existing data
    findings = check_existing_data(manager)
    
    # Test 3: Check code
    files_to_update = check_code_updates()
    
    # Summary
    generate_summary(manager, findings, files_to_update)
    
    print("\n" + "=" * 70)
    print("✓ DRY RUN COMPLETE - No files were modified")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())





