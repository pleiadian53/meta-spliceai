# Meta-SpliceAI Scripts Directory

**Central location for all operational scripts, tools, and utilities.**

⚠️ **This directory has been reorganized** - See documentation below for navigation

---

## 🚀 Quick Start

**New to this project?** Start here:

1. **Read**: [QUICK_START.md](./QUICK_START.md) - Common tasks and workflows
2. **Browse**: Directory structure below
3. **Search**: Use the index at the bottom of this file

**Need RunPods?** See [../runpods/](../runpods/) - All RunPods utilities now centralized there!

---

## 📁 Current Directory Structure

### Core Operations

| Directory | Purpose | Key Scripts |
|-----------|---------|-------------|
| **training/** | Model training workflows | `run_*_chromosomes.sh`, `start_full_genome_*.sh` |
| **testing/** | Test scripts (85 files) | `test_*.py`, validation scripts |
| **evaluation/** | Model evaluation | CV metrics, feature importance, overfitting analysis |

### Data Management

| Directory | Purpose | Key Scripts |
|-----------|---------|-------------|
| **data/** | Data analysis & processing | Splice site analysis, gene patterns |
| **data_management/** | Data acquisition | `download_clinvar.py` |
| **data_processing/** | Data transformation | `parse_mutsplicedb.py` |
| **analysis/** | Advanced analysis | Splice site visualization, quantification |

### Environment & Setup

| Directory | Purpose | Key Scripts |
|-----------|---------|-------------|
| **setup/** | Setup utilities | General setup scripts |
| **gpu_env_setup/** | GPU environment | GPU installation, testing, performance |
| **installation/** | Installation scripts | Environment migration |

### Operations & Tools

| Directory | Purpose | Key Scripts |
|-----------|---------|-------------|
| **monitoring/** | Training monitoring | `monitor_meta_training.sh`, universal monitor |
| **builder/** | Incremental builder | `run_builder_resumable.sh` |
| **scaling_solutions/** | Scalability tools | Memory optimization, performance monitoring |
| **maintenance/** | Cleanup & maintenance | `cleanup_old_predictions.sh` |

### Infrastructure

| Directory | Purpose | Key Scripts |
|-----------|---------|-------------|
| **notebook/** | Jupyter setup | `setup_jupyter_remote.sh` |
| **mlflow/** | MLflow setup | `setup_mlflow_remote.sh` |
| **migration/** | Data migration | Migration utilities |

### Documentation & Archive

| Directory | Purpose | Contents |
|-----------|---------|----------|
| **docs/** | Documentation | Guides, monitoring docs |
| **archive/** | Deprecated scripts | Old/superseded code |
| **base_model/** | Base model tools | Download OpenSpliceAI models |
| **inference/** | Inference scripts | Inference utilities |

---

## 🎯 Common Tasks

### Training

```bash
# Train on single chromosome
./training/run_single_chromosome.sh 21

# Full genome training
./training/start_full_genome_run.sh

# Monitor training
./monitoring/monitor_meta_training.sh
```

### Testing

```bash
# Run comprehensive test
./testing/test_complete_pipeline.sh

# Test inference modes
./testing/test_inference_modes.sh

# Validate setup
../verify_setup.py
```

### Data Processing

```bash
# Analyze splice sites
./data/quick_splice_analysis.sh

# Generate manifest
./data/generate_splice_manifest.py

# Download data
./data_management/download_clinvar.py
```

### Environment Setup

```bash
# Check GPU
./gpu_env_setup/test_gpu_installation.sh

# Verify environment
./gpu_env_setup/verify_gpu_setup.py

# Check versions
./check_versions.py
```

### RunPods (Centralized)

```bash
# All RunPods utilities moved to dedicated directory
cd ../runpods
./scripts/quick_pod_setup.sh meta-spliceai
```

---

## 📊 Directory Statistics

| Category | Directories | Files |
|----------|-------------|-------|
| Testing | 1 | 85 |
| Training | 1 | 15 |
| Scaling | 1 | 19 |
| Analysis | 1 | 13 |
| Data | 3 | 14 |
| Evaluation | 1 | 11 |
| GPU/Setup | 2 | 8 |
| **Total** | **21** | **~200** |

**Additional**: ~63 loose files at root level

---

## 🔍 Find Scripts By Purpose

### I Want To...

**Train Models**:
- Full genome: `training/start_full_genome_run.sh`
- Single chromosome: `training/run_single_chromosome.sh`
- Multi-GPU: `run_multi_gpu_training.py`
- Monitor: `monitoring/monitor_meta_training.sh`

**Test/Validate**:
- Complete pipeline: `testing/test_complete_pipeline.sh`
- Inference: `testing/test_inference_modes.sh`
- Base model: `testing/test_base_model_validation_run2.py`
- Meta model: `testing/test_meta_modes_comprehensive.py`

**Analyze Data**:
- Splice sites: `data/quick_splice_analysis.sh`
- Gene patterns: `data/analyze_gene_patterns.sh`
- Training data: `analyze_training_data.py`
- Overlaps: `analyze_overlapping_genes.py`

**Setup Environment**:
- GPU: `gpu_env_setup/install_gpu_environment.sh`
- Jupyter: `notebook/setup_jupyter_remote.sh`
- MLflow: `mlflow/setup_mlflow_remote.sh`
- Verify: `pre_flight_checks.py`

**Work with RunPods**:
- Setup: `../runpods/scripts/quick_pod_setup.sh`
- SSH config: `../runpods/scripts/runpod_ssh_manager.sh`
- Documentation: `../runpods/START_HERE.md`

**Debug/Investigate**:
- GPU: `diagnose_gpu_environment.py`
- Dataset: `check_dataset.py`
- Meta model: `inspect_meta_model_training_data.py`
- Parquet: `inspect_parquet.py`

**Maintain/Cleanup**:
- Predictions: `cleanup_predictions.sh`
- Artifacts: `cleanup_artifacts.py`
- Old data: `maintenance/cleanup_old_predictions.sh`

---

## 🗺️ Navigation Tips

### Finding Scripts

1. **By name**: Use `find` or `grep`
   ```bash
   find . -name "*training*"
   grep -r "function_name" .
   ```

2. **By purpose**: Check this README's index

3. **By directory**: See structure above

4. **By recency**: 
   ```bash
   ls -lt | head -20
   ```

### Understanding Scripts

1. **Read the header**: Most scripts have usage comments
2. **Check docs/**: Additional documentation
3. **Look for README**: Some subdirectories have READMEs

---

## ⚠️ Known Issues

### Current State
- Many loose files at root (63 files)
- `testing/` directory has 85 files (needs organization)
- Overlapping categories (data, data_management, data_processing)
- Some scripts may be outdated
- No clear archival strategy

### Ongoing Work
- Moving deprecated scripts to `archive/`
- Consolidating overlapping categories
- Creating per-directory documentation

---

## 📝 Best Practices

### When Adding New Scripts

1. **Choose the right directory**:
   - Training workflows → `training/`
   - Tests → `testing/`
   - Data tools → `data/`
   - Setup → appropriate setup directory

2. **Include header documentation**:
   ```python
   """
   Script Purpose: One-line description
   Usage: python script.py [args]
   Dependencies: list of requirements
   """
   ```

3. **Make it executable** (if shell script):
   ```bash
   chmod +x script.sh
   ```

4. **Update this README** if it's a commonly-used script

### When Modifying Scripts

1. **Check dependencies**: May affect other scripts
2. **Test thoroughly**: Don't break existing workflows
3. **Document changes**: Update headers and docs
4. **Consider versioning**: For major changes

---

## 🔧 Utility Scripts

### Script Management

- **manage_scripts.py**: Script management utility
- **pre_flight_checks.py**: Pre-execution validation

### Verification

- **check_versions.py**: Python package versions
- **check_gpu.py**: GPU availability
- **verify_rename.sh**: Package rename verification

---

## 📚 Documentation

### In scripts/docs/

- Monitoring guides
- Architecture documentation
- Best practices

### In main project

- `../README.md` - Project overview
- `../GETTING_STARTED.md` - Setup guide
- `../PROJECT_SUMMARY.md` - Project summary

### External

- `../runpods/` - RunPods utilities
- `../meta_spliceai/` - Main package
- `../notebooks/` - Jupyter notebooks

---

## 🚧 Future Improvements

Planned enhancements (see REORGANIZATION_PLAN.md):

1. **Reorganize testing/** - Group by type (unit, integration, validation)
2. **Consolidate data dirs** - Merge into single hierarchy
3. **Archive outdated** - Move old scripts to `archive/` with docs
4. **Create QUICK_START** - Common tasks guide
5. **Per-directory READMEs** - Detailed documentation
6. **Script inventory** - Complete catalog with status

---

## 📞 Getting Help

### Quick Reference

1. **This README** - Overview and navigation
2. **QUICK_START.md** - Common tasks
3. **REORGANIZATION_PLAN.md** - Future structure
4. **Directory READMEs** - Specific details

### Finding Documentation

```bash
# Find all README files
find . -name "README.md"

# Search documentation
grep -r "topic" docs/

# List recent changes
ls -lt | head -20
```

---

## 📑 Complete File Index

<details>
<summary>Click to expand complete listing</summary>

### Root Level Scripts (63 files)

Training-related:
- run_multi_gpu_training.py
- run_multi_gpu_training_ig.py

Data-related:
- analyze_training_data.py
- analyze_overlapping_genes.py
- check_dataset.py
- generate_gene_manifest.py
- inspect_analysis_sequences.py
- f1_pr_analysis_merged.py
- plot_splice_sites_hist.py

Testing:
- test_transcript_topk.py
- test_leakage_probe.py
- test_leakage_probe_debug.py
- test_path_migration.py
- test_incremental_builder.sh
- validate_artifacts.py
- validate_meta_model_training_data.py
- test_gpu_xgboost.py

Environment:
- check_gpu.py
- check_versions.py
- check_polars_version.py
- diagnose_gpu_environment.py
- fix_ml_dependencies.py
- deepspeed_patch.py
- pre_flight_checks.py

Maintenance:
- cleanup_predictions.sh
- cleanup_artifacts.py
- fresh_start_predictions.py
- final_cleanup.sh
- inspect_meta_model_training_data.py
- inspect_parquet.py
- post_training_analysis.sh

Migration:
- migrate_splicevardb_to_case_studies.sh
- migrate_splicevardb_data.sh
- migrate_predictions.py
- migrate_variant_data.py
- patch_gene_type.py
- patch_structural_features.py

Deprecated:
- rename_package.sh
- rename_package_safe.sh
- verify_rename.sh

Visualization:
- meta_model_concept_diagram.py
- meta_model_concept_diagram_simplified.py
- position_centric_data_repr_diagram.py

Tools:
- build_for_fabric.sh
- manage_scripts.py

... (See subdirectories for organized scripts)

</details>

---

**Last Updated**: January 27, 2026  
**Status**: In Reorganization - See REORGANIZATION_PLAN.md

For questions or suggestions, see the main project documentation.
