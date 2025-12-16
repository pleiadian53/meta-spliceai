# Artifact Validation and Cleanup

This document describes the utilities for validating and cleaning up artifacts generated during the meta-model training data assembly process.

## Overview

During the incremental build process, various artifacts are generated in the `data/ensembl/spliceai_eval/meta_models/` directory. Sometimes, due to interrupted processes or schema changes, some artifacts may become corrupted or have incorrect schemas. The validation and cleanup utilities help identify and resolve these issues.

## Artifact Types

The validator recognizes three main artifact types used for training data assembly:

1. **`analysis_sequences_*_chunk_*.tsv`** - Detailed sequence analysis data
2. **`splice_positions_enhanced_*_chunk_*.tsv`** - Enhanced splice position data  
3. **`splice_errors_*_chunk_*.tsv`** - Error information

## Artifact Validator Module

### Purpose

The `artifact_validator.py` module provides comprehensive validation of artifact schemas and integrity. It can identify:

- **Corrupted files** - Files that cannot be read or parsed
- **Wrong schema files** - Files with incorrect column structure or data types
- **Invalid files** - Files with validation errors (missing required columns, wrong data types, etc.)

### Usage

#### Command Line Interface

```bash
# Basic validation with full report
python scripts/validate_artifacts.py

# List only corrupted files (for easy deletion)
python scripts/validate_artifacts.py --list-corrupted

# List only wrong schema files
python scripts/validate_artifacts.py --list-wrong-schema

# Generate summary report
python scripts/validate_artifacts.py --summary

# Save detailed report to file
python scripts/validate_artifacts.py --output-report validation_report.txt

# Generate JSON report
python scripts/validate_artifacts.py --json-report validation_results.json
```

#### Programmatic Usage

```python
from meta_spliceai.splice_engine.meta_models.builder.artifact_validator import ArtifactValidator

# Initialize validator
validator = ArtifactValidator()

# Validate all artifacts in the default directory
results = validator.validate_directory()

# Validate specific file
result = validator.validate_file(Path("path/to/artifact.tsv"))

# Generate report
report = validator.generate_report(results, output_file="report.txt")
```

### Validation Criteria

The validator checks each artifact against strict schema definitions:

#### Analysis Sequences Schema
- **Required columns**: 50+ columns including `gene_id`, `position`, `score`, `strand`, etc.
- **Data types**: String, integer, and float columns with specific type requirements
- **File size limit**: 100MB maximum
- **Row count**: Minimum 1 row

#### Splice Positions Enhanced Schema
- **Required columns**: 40+ columns including `gene_id`, `position`, `predicted_position`, etc.
- **Data types**: String, integer, and float columns with specific type requirements
- **File size limit**: 200MB maximum
- **Row count**: Minimum 1 row

#### Splice Errors Schema
- **Required columns**: 8 columns including `position`, `gene_id`, `error_type`, etc.
- **Data types**: String and integer columns
- **File size limit**: 50MB maximum
- **Row count**: Minimum 0 rows (can be empty)

### Validation Results

The validator categorizes files into four types:

1. **Valid** - Files that pass all validation checks
2. **Corrupted** - Files that cannot be read or parsed
3. **Wrong Schema** - Files with incorrect column structure or data types
4. **Invalid** - Files with other validation errors

## Artifact Cleanup Script

### Purpose

The `cleanup_artifacts.py` script uses the artifact validator to identify and safely delete problematic files, helping to reclaim disk space and ensure data integrity.

### Usage

```bash
# Dry run - show what would be deleted without actually deleting
python scripts/cleanup_artifacts.py --dry-run

# Delete only corrupted files
python scripts/cleanup_artifacts.py --corrupted-only

# Delete only wrong schema files
python scripts/cleanup_artifacts.py --wrong-schema-only

# Delete both corrupted and wrong schema files (default)
python scripts/cleanup_artifacts.py

# Use custom artifacts directory
python scripts/cleanup_artifacts.py --artifacts-dir /path/to/artifacts

# Increase verbosity
python scripts/cleanup_artifacts.py -vv
```

### Safety Features

The cleanup script includes several safety features:

- **Dry run mode** - Shows what would be deleted without actually deleting
- **Confirmation prompt** - Asks for confirmation before deletion
- **Size reporting** - Shows total size of files to be deleted
- **Verbose logging** - Detailed output of what's being deleted
- **Error handling** - Continues even if some files can't be deleted

### Example Output

```
🔍 Scanning artifacts in: data/ensembl/spliceai_eval/meta_models
📁 Found 3 corrupted files
📁 Found 6 files with wrong schemas

🗑️  Files to delete (DRY RUN):
  • data/ensembl/spliceai_eval/meta_models/analysis_sequences_8_chunk_501_1000.tsv (45.2MB)
  • data/ensembl/spliceai_eval/meta_models/splice_positions_enhanced_8_chunk_501_1000.tsv (67.8MB)
  • data/ensembl/spliceai_eval/meta_models/splice_errors_8_chunk_501_1000.tsv (12.1MB)
  • data/ensembl/spliceai_eval/meta_models/gene_tx_candidates.tsv (2.3MB)
  • data/ensembl/spliceai_eval/meta_models/overlapping_genes.tsv (1.7MB)
  • data/ensembl/spliceai_eval/meta_models/splice_sites.tsv (8.9MB)
  • data/ensembl/spliceai_eval/meta_models/annotations.tsv (15.4MB)
  • data/ensembl/spliceai_eval/meta_models/genomic_sequences.tsv (23.6MB)
  • data/ensembl/spliceai_eval/meta_models/gene_features.tsv (5.2MB)

📊 Total size: 181.2MB

💡 Run without --dry-run to actually delete these files.
```

## Integration with Incremental Builder

The artifact validation and cleanup utilities are designed to work seamlessly with the incremental builder:

### Pre-Build Validation

Before running a new incremental build, you can validate existing artifacts:

```bash
# Check for corrupted files that might interfere with the build
python scripts/validate_artifacts.py --list-corrupted

# Clean up corrupted files before starting
python scripts/cleanup_artifacts.py --corrupted-only
```

### Post-Build Validation

After completing an incremental build, validate the generated artifacts:

```bash
# Validate all artifacts
python scripts/validate_artifacts.py

# Generate detailed report
python scripts/validate_artifacts.py --output-report build_validation_report.txt
```

### Recovery from Interrupted Builds

If an incremental build is interrupted, you can clean up partial artifacts:

```bash
# Identify and clean up corrupted artifacts from interrupted build
python scripts/cleanup_artifacts.py --corrupted-only

# Resume the build with --overwrite flag
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --output-dir train_pc_1000 \
    --overwrite
```

## Best Practices

### When to Use Validation

- **Before starting a new build** - Ensure no corrupted artifacts will interfere
- **After interrupted builds** - Identify and clean up partial artifacts
- **After schema changes** - Find files with outdated schemas
- **Regular maintenance** - Periodic validation to ensure data integrity

### When to Use Cleanup

- **After interrupted builds** - Remove corrupted partial artifacts
- **After schema updates** - Remove files with outdated schemas
- **Disk space management** - Reclaim space from problematic files
- **Before major rebuilds** - Clean slate for new builds

### File Preservation

The cleanup script is conservative and only deletes files that are clearly problematic:

- **Corrupted files** - Cannot be read or parsed
- **Wrong schema files** - Don't match expected artifact schemas

Files that are valid but may be used for other purposes (like `gene_tx_candidates.tsv`) are preserved unless they have wrong schemas.

### Backup Strategy

Before running cleanup on a large dataset:

```bash
# Create backup of artifacts directory
cp -r data/ensembl/spliceai_eval/meta_models data/ensembl/spliceai_eval/meta_models_backup

# Run cleanup with dry-run first
python scripts/cleanup_artifacts.py --dry-run

# If satisfied, run actual cleanup
python scripts/cleanup_artifacts.py
```

## Troubleshooting

### Common Issues

1. **"File does not exist" errors** - Check that the artifacts directory path is correct
2. **Permission errors** - Ensure write permissions for cleanup operations
3. **Memory errors** - Large files may require more memory for validation
4. **Schema mismatch** - Files may have been generated with different schema versions

### Debugging

For detailed debugging, use verbose mode:

```bash
# Maximum verbosity for debugging
python scripts/validate_artifacts.py -vvv

# Verbose cleanup with detailed output
python scripts/cleanup_artifacts.py -vv
```

### Getting Help

If you encounter issues:

1. Check the validation report for specific error messages
2. Use dry-run mode to see what would be affected
3. Review the artifact schemas in the validator code
4. Check file permissions and disk space

## API Reference

### ArtifactValidator Class

```python
class ArtifactValidator:
    def __init__(self, artifacts_dir: Optional[Path] = None)
    def validate_file(self, file_path: Path) -> ValidationResult
    def validate_directory(self, directory: Optional[Path] = None) -> Dict[str, List[ValidationResult]]
    def generate_report(self, results: Dict[str, List[ValidationResult]], output_file: Optional[Path] = None) -> str
```

### ValidationResult Class

```python
@dataclass
class ValidationResult:
    file_path: Path
    artifact_type: ArtifactType
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    column_count: int
    file_size_mb: float
    schema_matches: bool
    data_types_valid: bool
    has_required_columns: bool
```

For more detailed API documentation, see the source code in `artifact_validator.py`.

---

## Artifact Lifecycle and Overwriting

### Understanding Artifact Behavior

The `data/ensembl/spliceai_eval/meta_models/` directory is a **shared, mutable resource** that stores intermediate artifacts from the base model pass (SpliceAI inference). Understanding how artifacts are created, used, and overwritten is critical for managing multiple training datasets.

### Artifact Generation

Artifacts are generated when the incremental builder is run with the `--run-workflow` flag:

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --run-workflow \              # Triggers base model pass
    --output-dir train_pc_1000
```

**What happens:**
1. **SpliceAI inference** runs on selected genes
2. **Three artifact types** are generated per chromosome/chunk:
   - `analysis_sequences_*_chunk_*.tsv`
   - `splice_positions_enhanced_*_chunk_*.tsv`
   - `splice_errors_*_chunk_*.tsv`
3. **Artifacts saved** to `data/ensembl/spliceai_eval/meta_models/`
4. **Training dataset built** using these artifacts

### Artifact Reuse

Artifacts can be reused in subsequent builds **without** the `--run-workflow` flag:

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 500 \
    --output-dir train_pc_500      # No --run-workflow
```

**What happens:**
1. **Check for existing artifacts** for selected genes
2. **Reuse artifacts** if available
3. **Build training dataset** from existing artifacts
4. **No SpliceAI inference** (faster, uses less compute)

### Overwriting Behavior

**Critical:** When `--run-workflow` is used, **new artifacts overwrite existing ones** for the same genes/chromosomes.

#### Example Timeline

| Time | Action | Genes | Artifacts Result |
|------|--------|-------|------------------|
| T1 | Build dataset A with `--run-workflow` | 100 genes | Artifacts for 100 genes created |
| T2 | Build dataset B without `--run-workflow` | 50 genes (subset of A) | Artifacts reused |
| T3 | Build dataset C with `--run-workflow` | 1000 genes | **Artifacts overwritten** (new 1000 genes) |
| T4 | Cannot rebuild dataset A | 100 genes | ❌ Artifacts no longer available |

#### Implications

- ✅ **Most recent run:** Artifacts always match the latest `--run-workflow` execution
- ❌ **Earlier runs:** Artifacts from previous runs are lost (overwritten)
- ⚠️ **Reproducibility:** Cannot rebuild old datasets without regenerating artifacts

---

## Multi-Dataset Artifact Management

### The Superset Principle

**Ideal state:**
```
Artifacts ⊇ Dataset_1 ∪ Dataset_2 ∪ ... ∪ Dataset_N
```

Artifacts should contain **all genes** from **all training datasets** to enable:
- Building new training datasets from subsets
- Rebuilding existing datasets
- Comparing different gene selections

### Current Limitations

The default artifact directory **does not** maintain the superset principle because:
1. New runs with `--run-workflow` overwrite existing artifacts
2. No accumulation mechanism across runs
3. Artifacts are organized by chromosome/chunk, not by dataset

### Best Practices for Multiple Datasets

#### Option 1: Separate Artifact Directories (Recommended for Production)

```bash
# Dataset 1: 100 genes
python -m ... --run-workflow \
    --output-dir train_100 \
    --eval-dir data/ensembl/spliceai_eval/meta_models_100genes

# Dataset 2: 1000 genes
python -m ... --run-workflow \
    --output-dir train_1000 \
    --eval-dir data/ensembl/spliceai_eval/meta_models_1000genes
```

**Pros:**
- ✅ Complete isolation
- ✅ No overwriting
- ✅ Easy to manage

**Cons:**
- ❌ More disk space
- ❌ Duplicated artifacts for overlapping genes

#### Option 2: Timestamped Snapshots

```bash
# Create snapshots before overwriting
cp -r data/ensembl/spliceai_eval/meta_models \
      data/ensembl/spliceai_eval/meta_models_20251017

# Run new build
python -m ... --run-workflow --output-dir train_1000
```

**Pros:**
- ✅ Preserves historical artifacts
- ✅ Easy versioning

**Cons:**
- ❌ Manual snapshot creation
- ❌ High disk usage

#### Option 3: Accumulate Artifacts (Custom Implementation)

Modify the workflow to **skip** genes that already have artifacts:

```python
# Custom logic (not currently implemented)
existing_genes = scan_artifact_directory()
new_genes = selected_genes - existing_genes
run_workflow_on(new_genes)  # Only process new genes
```

**Pros:**
- ✅ Maintains superset principle
- ✅ Efficient disk usage

**Cons:**
- ❌ Requires custom implementation
- ❌ Complex management

---

## Gene Consistency Verification

### Why Verify Gene Consistency?

Ensuring that training dataset genes have corresponding artifacts is critical for:
- **Data integrity:** Verify all features are available
- **Reproducibility:** Ensure datasets can be rebuilt
- **Debugging:** Identify missing or corrupted artifacts

### Verification Workflow

#### Step 1: Extract Genes from Training Dataset

```bash
python << 'EOF'
import polars as pl
from pathlib import Path

train_dir = Path("data/train_pc_1000_3mers/master")
genes_train = set()

for batch_file in train_dir.glob("*.parquet"):
    df = pl.read_parquet(batch_file)
    genes_train.update(df['gene_id'].unique().to_list())

print(f"Training dataset genes: {len(genes_train):,}")
print(f"Sample: {sorted(list(genes_train))[:10]}")
EOF
```

#### Step 2: Extract Genes from Artifacts

```bash
python << 'EOF'
import polars as pl
from pathlib import Path
from tqdm import tqdm

artifacts_dir = Path("data/ensembl/spliceai_eval/meta_models")
artifact_files = sorted(artifacts_dir.glob("analysis_sequences_*.tsv"))

genes_artifacts = set()
for artifact_file in tqdm(artifact_files, desc="Scanning"):
    df = pl.read_csv(artifact_file, separator='\t', columns=['gene_id'])
    genes_artifacts.update(df['gene_id'].unique().to_list())

print(f"Artifact genes: {len(genes_artifacts):,}")
EOF
```

#### Step 3: Compare and Verify

```python
# Calculate coverage
coverage = genes_train.intersection(genes_artifacts)
missing = genes_train - genes_artifacts

print(f"\n=== Verification Results ===")
print(f"Training genes: {len(genes_train):,}")
print(f"Artifact genes: {len(genes_artifacts):,}")
print(f"Coverage: {len(coverage)}/{len(genes_train)} ({len(coverage)/len(genes_train)*100:.1f}%)")

if missing:
    print(f"\n⚠️  Missing from artifacts: {len(missing)} genes")
    print(f"Sample missing: {sorted(list(missing))[:10]}")
else:
    print(f"\n✅ Perfect coverage: All training genes have artifacts")
```

### Expected Results

#### Perfect Coverage (100%)

```
✅ All training genes have artifacts
```

**Interpretation:** The training dataset was built with `--run-workflow` or immediately after, so all artifacts are present.

#### Partial Coverage (<100%)

```
⚠️  Coverage: 85%
⚠️  Missing from artifacts: 150 genes
```

**Possible causes:**
1. Training dataset built without `--run-workflow` (using older artifacts)
2. Artifacts overwritten by subsequent run
3. Artifacts deleted or corrupted

**Resolution:**
- Rebuild with `--run-workflow` to regenerate artifacts
- Use artifact snapshots (if available)
- Accept limitation if training already completed

#### Zero Coverage (0%)

```
❌ Coverage: 0%
❌ All genes missing from artifacts
```

**Cause:** Artifacts from different gene set (completely overwritten)

**Resolution:**
- Must regenerate artifacts with `--run-workflow`

### Automated Verification Script

Create a reusable verification script:

```python
# scripts/verify_artifact_coverage.py
import polars as pl
from pathlib import Path
from tqdm import tqdm
import sys

def verify_coverage(train_dir: Path, artifacts_dir: Path):
    """Verify artifact coverage for a training dataset."""
    
    # Extract training genes
    genes_train = set()
    for batch_file in train_dir.glob("*.parquet"):
        df = pl.read_parquet(batch_file)
        genes_train.update(df['gene_id'].unique().to_list())
    
    # Extract artifact genes
    genes_artifacts = set()
    for artifact_file in tqdm(artifacts_dir.glob("analysis_sequences_*.tsv")):
        df = pl.read_csv(artifact_file, separator='\t', columns=['gene_id'])
        genes_artifacts.update(df['gene_id'].unique().to_list())
    
    # Calculate coverage
    coverage = genes_train.intersection(genes_artifacts)
    coverage_pct = len(coverage) / len(genes_train) * 100 if genes_train else 0
    
    # Report
    print(f"\n{'='*60}")
    print(f"ARTIFACT COVERAGE REPORT")
    print(f"{'='*60}")
    print(f"Training dataset: {train_dir}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"\nTraining genes: {len(genes_train):,}")
    print(f"Artifact genes: {len(genes_artifacts):,}")
    print(f"Coverage: {len(coverage):,}/{len(genes_train):,} ({coverage_pct:.1f}%)")
    
    if coverage_pct == 100.0:
        print(f"\n✅ PASS: Perfect coverage")
        return 0
    elif coverage_pct >= 95.0:
        print(f"\n⚠️  WARN: High coverage but some genes missing")
        return 1
    else:
        print(f"\n❌ FAIL: Low coverage, many genes missing")
        return 2

if __name__ == "__main__":
    train_dir = Path(sys.argv[1])
    artifacts_dir = Path(sys.argv[2])
    sys.exit(verify_coverage(train_dir, artifacts_dir))
```

**Usage:**

```bash
python scripts/verify_artifact_coverage.py \
    data/train_pc_1000_3mers/master \
    data/ensembl/spliceai_eval/meta_models
```

---

## Gene Manifest Tracking

### Current Gap

The incremental builder currently does **not** maintain a central gene manifest in the artifacts directory showing which genes have artifacts available.

### Proposed Enhancement

Add a `gene_manifest.csv` file to the artifacts directory:

```csv
gene_id,chrom,artifact_date,artifact_version,analysis_sequences,splice_positions,splice_errors
ENSG00000000001,1,2025-10-17,v1.0,TRUE,TRUE,TRUE
ENSG00000000002,2,2025-10-17,v1.0,TRUE,TRUE,TRUE
...
```

**Benefits:**
- ✅ Quick lookup of available genes
- ✅ Track artifact versions
- ✅ Verify artifact completeness
- ✅ Enable smart accumulation

### Implementation (Future Work)

```python
# In splice_prediction_workflow.py, after artifact generation:
def update_gene_manifest(artifacts_dir: Path, genes: List[str]):
    manifest_path = artifacts_dir / "gene_manifest.csv"
    
    # Load existing manifest
    if manifest_path.exists():
        manifest_df = pl.read_csv(manifest_path)
    else:
        manifest_df = pl.DataFrame()
    
    # Add new genes
    new_entries = pl.DataFrame({
        'gene_id': genes,
        'artifact_date': [datetime.now()] * len(genes),
        'artifact_version': ['v1.0'] * len(genes)
    })
    
    # Merge and save
    manifest_df = pl.concat([manifest_df, new_entries])
    manifest_df.write_csv(manifest_path)
```

---

## Summary

### Key Takeaways

1. **Artifacts are mutable:** The `meta_models/` directory is a shared, overwritable resource
2. **Overwriting is normal:** New runs with `--run-workflow` replace existing artifacts
3. **Superset not guaranteed:** Artifacts may not contain all genes from all training datasets
4. **Verification is critical:** Always verify artifact coverage for production datasets
5. **Management strategies exist:** Use separate directories, snapshots, or custom accumulation

### Best Practices

✅ **For development/testing:**
- Use the default shared artifacts directory
- Accept that artifacts may be overwritten
- Rebuild datasets with `--run-workflow` as needed

✅ **For production:**
- Use separate artifact directories per major dataset
- Create snapshots before overwriting
- Verify artifact coverage before training
- Document which artifacts were used for each training run

✅ **For reproducibility:**
- Archive artifacts alongside training datasets
- Version artifact directories
- Maintain gene manifests
- Document the artifact generation pipeline 