# Scripts Reorganization Summary

**Date**: January 30, 2026

## Overview
Reorganized the `scripts/` directory to improve maintainability and organization. All root-level scripts have been categorized into appropriate subdirectories based on their purpose.

## New Directory Structure

### Analysis (`scripts/analysis/`)
Scripts for data analysis, inspection, and visualization:
- `analyze_overlapping_genes.py`
- `analyze_splice_positions.py`
- `analyze_training_data.py`
- `f1_pr_analysis_merged.py`
- `inspect_analysis_sequences.py`
- `inspect_meta_model_training_data.py`
- `inspect_parquet.py`
- `plot_splice_sites_hist.py`

### Testing (`scripts/testing/`)
Test scripts, checks, and validation utilities:
- `test_dataset_integrity.py`
- `test_datasets_loading.py`
- `test_gpu_performance.py`
- `test_gpu_xgboost.py`
- `test_leakage_probe.py`
- `test_leakage_probe_debug.py`
- `test_path_migration.py`
- `test_transcript_topk.py`
- `minimal_check.py`
- `check_dataset.py`
- `check_gpu.py`
- `check_polars_version.py`
- `check_versions.py`
- `run_all_tests.sh`

### Data Processing (`scripts/data_processing/`)
Data preparation, transformation, and gene manifest tools:
- `consolidate_sequences.py`
- `enhance_splice_sites.py`
- `patch_gene_type.py`
- `patch_structural_features.py`
- `validate_sequences.py`
- `regenerate_splice_sites.sh`
- `generate_gene_manifest.py`
- `query_gene_manifest.py`

### Migration (`scripts/migration/`)
Data and schema migration scripts:
- `migrate_predictions.py`
- `migrate_splicevardb_data.sh`
- `migrate_splicevardb_to_case_studies.sh`
- `migrate_variant_data.py`

### Maintenance (`scripts/maintenance/`)
Cleanup and maintenance utilities:
- `cleanup_artifacts.py`
- `cleanup_predictions.sh`
- `final_cleanup.sh`
- Scripts moved from root that were in maintenance/

### Installation (`scripts/installation/`)
Setup, installation, and environment configuration:
- `build_for_fabric.sh`
- `deepspeed_patch.py`
- `diagnose_gpu_environment.py`
- `fix_ml_dependencies.py`
- `github_init.sh`

### Training (`scripts/training/`)
Model training workflows and monitoring:
- `post_training_analysis.sh`
- Existing training scripts remain in this directory

### Validation (`scripts/validation/`)
**NEW** - Validation and pre-flight check scripts:
- `validate_artifacts.py`
- `validate_meta_model_training_data.py`
- `pre_flight_checks.py`

### Utilities (`scripts/utilities/`)
**NEW** - Documentation generation and helper tools:
- `meta_model_concept_diagram.py`
- `meta_model_concept_diagram_simplified.py`
- `position_centric_data_repr_diagram.py`
- `manage_scripts.py`

### Archive (`scripts/archive/`)
**NEW** - Outdated or one-time-use scripts:
- `fresh_start_predictions.py` (temporary script)
- `preview_rename.sh` (one-time rename operation)
- `rename_package.sh` (one-time rename operation)
- `rename_package_safe.sh` (one-time rename operation)
- `verify_rename.sh` (one-time rename operation)
- `run_builder_resumable_old.sh` (superseded by newer version in builder/)

## Duplicates Resolved
- `test_incremental_builder.sh` - Root version removed (newer version already in testing/)
- `run_builder_resumable.sh` - Root version archived as `run_builder_resumable_old.sh` (newer version in builder/)

## Entry Points Verification
Verified that no moved scripts are entry points in `pyproject.toml`:
- ✅ `run_base_model` - CLI entry point (not moved)
- ✅ `evaluate_predictions` - CLI entry point (not moved)
- ✅ `annotate_splice_sites` - CLI entry point (not moved)

## Documentation Files
The following documentation files remain in `scripts/` root:
- `README.md` - Main scripts documentation
- `QUICK_REFERENCE.md` - Quick reference guide
- `QUICK_START.md` - Quick start guide
- `README_f1_pr_analysis.md` - F1/PR analysis documentation
- `SCRIPT_INVENTORY.md` - Script inventory
- `REORGANIZATION_SUMMARY.md` - This file

## Benefits
1. **Easier Navigation**: Scripts are now grouped by purpose
2. **Clear Separation**: Test scripts separated from production scripts
3. **Historical Context**: Outdated scripts archived rather than deleted
4. **Maintainability**: Easier to find and update related scripts
5. **Reduced Clutter**: Root directory now contains only documentation

## Next Steps
- Consider adding README.md files to each subdirectory explaining its purpose
- Review archived scripts for potential deletion after validation period
- Update any scripts that reference moved files (check import paths if applicable)
