#!/bin/bash
# Migration script for SpliceVarDB data to case_studies/data_sources/datasets/
# Per DATA_LAYOUT_MASTER_GUIDE.md: case studies belong in code tree, not data/

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "SpliceVarDB Migration to Case Studies"
echo "========================================="
echo ""
echo "Per DATA_LAYOUT_MASTER_GUIDE.md:"
echo "  - Case study data → code tree"
echo "  - Heavy reference data → data/"
echo ""

# Source and destination
OLD_FILE="meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv"
RELEASE_DATE=$(date +%Y%m%d)
NEW_DIR="meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb/splicevardb_$RELEASE_DATE"
DATASETS_ROOT="meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb"

# Check if source exists
if [ ! -f "$OLD_FILE" ]; then
    echo "✗ Source file not found: $OLD_FILE"
    echo "  Maybe already migrated?"
    exit 0
fi

# Show file info
echo "Found SpliceVarDB download:"
echo "  Location: $OLD_FILE"
echo "  Size: $(du -h "$OLD_FILE" | cut -f1)"
echo "  Lines: $(wc -l < "$OLD_FILE") variants"
echo ""

# Create release-dated directory structure
echo "Creating directory structure..."
mkdir -p "$NEW_DIR"/{raw,processed}
echo "✓ Created: $NEW_DIR/"
echo ""

# Move file
echo "Moving data to case_studies/data_sources/datasets/..."
mv "$OLD_FILE" "$NEW_DIR/raw/splicevardb_download.tsv"
echo "✓ Moved to: $NEW_DIR/raw/"
echo ""

# Create 'latest' symlink
echo "Creating 'latest' symlink..."
cd "$DATASETS_ROOT"
ln -sf "splicevardb_$RELEASE_DATE" latest
cd "$PROJECT_ROOT"
echo "✓ Symlink: $DATASETS_ROOT/latest -> splicevardb_$RELEASE_DATE"
echo ""

# Create README
cat > "$NEW_DIR/README.md" << 'EOF'
# SpliceVarDB Dataset

**Source:** https://splicevardb.org/  
**Downloaded:** Manual download from website  
**Date:** $(date +%Y-%m-%d)  
**Build:** hg38 (GRCh38) and hg19 (GRCh37)  
**Variants:** 50,716 total

## Location Rationale

Per **DATA_LAYOUT_MASTER_GUIDE.md**, case study data belongs in the **code tree**:

```
meta_spliceai/splice_engine/case_studies/data_sources/datasets/
```

**Why?**
- Case study data is relatively small (<10 MB), portable
- Heavy reference data (GTF, FASTA) stays in `data/` 
- Follows established pattern for all variant databases

## Directory Structure

- `raw/` - Raw downloaded data
  - `splicevardb_download.tsv` - Original TSV download (50K variants)
- `processed/` - Processed formats (parquet, VCF, etc.)

## Data Format

The TSV file contains:
- variant_id: Unique identifier
- hg19: Coordinates in GRCh37
- hg38: Coordinates in GRCh38
- gene: Gene symbol
- hgvs: HGVS notation
- method: Experimental method
- classification: Splice effect (splice-altering, normal, etc.)
- location: Exonic/Intronic
- doi: Publication references

## Usage

```python
from pathlib import Path
import pandas as pd

# Case study data is in code tree
splicevardb_dir = Path("meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb/latest")

# Load raw TSV
raw_tsv = splicevardb_dir / "raw/splicevardb_download.tsv"
variants = pd.read_csv(raw_tsv, sep='\t')

# Process and save
processed = splicevardb_dir / "processed/splicevardb_validated_GRCh38.parquet"
variants.to_parquet(processed)
```

## Training with OpenSpliceAI

```python
from meta_spliceai.splice_engine.openspliceai_recalibration import (
    SpliceVarDBTrainingPipeline,
    PipelineConfig
)

config = PipelineConfig(
    # Input: Case study data (in code tree)
    data_dir="meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb/latest",
    
    # Output: Training dataset (in data/ root)
    output_dir="data/splicevardb_meta_training",
    
    # Reference: OpenSpliceAI base model data
    reference_genome="data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna",
    
    genome_build="GRCh38"
)

pipeline = SpliceVarDBTrainingPipeline(config=config)
results = pipeline.run()
```

## References

Sullivan, R.T., et al. (2024). SpliceVarDB: A database of splice-affecting variants.  
https://splicevardb.org/

## Documentation

- **Data Layout**: `docs/data/DATA_LAYOUT_MASTER_GUIDE.md`
- **Package Docs**: `meta_spliceai/splice_engine/openspliceai_recalibration/`
EOF

echo "✓ Created README.md"
echo ""

echo "========================================="
echo "✓ Migration Complete!"
echo "========================================="
echo ""
echo "Data structure:"
tree -L 3 "$DATASETS_ROOT" 2>/dev/null || ls -R "$DATASETS_ROOT"
echo ""
echo "Next steps:"
echo "1. Process TSV into parquet: variants.to_parquet('$NEW_DIR/processed/...')"
echo "2. Train meta-model with OpenSpliceAI"
echo "3. Output goes to: data/splicevardb_meta_training/"





