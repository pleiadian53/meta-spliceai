#!/bin/bash
# Migration script for SpliceVarDB downloaded data
# Moves 50K variants from code directory to systematic location

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "SpliceVarDB Data Migration"
echo "========================================="
echo ""

# Source and destination
OLD_FILE="meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv"
NEW_DIR="data/ensembl/case_studies/splicevardb/raw"
NEW_FILE="$NEW_DIR/splicevardb_download.tsv"

# Check if source exists
if [ ! -f "$OLD_FILE" ]; then
    echo "✗ Source file not found: $OLD_FILE"
    echo "  Maybe already migrated or never downloaded?"
    exit 0
fi

# Show file info
echo "Found SpliceVarDB download:"
echo "  Location: $OLD_FILE"
echo "  Size: $(du -h "$OLD_FILE" | cut -f1)"
echo "  Lines: $(wc -l < "$OLD_FILE") variants"
echo ""

# Create destination
echo "Creating systematic directory structure..."
mkdir -p "$NEW_DIR"
mkdir -p "data/ensembl/case_studies/splicevardb/processed"
mkdir -p "data/ensembl/case_studies/splicevardb/cache"
echo "✓ Created: $NEW_DIR"
echo ""

# Move file
echo "Moving data..."
mv "$OLD_FILE" "$NEW_FILE"
echo "✓ Moved to: $NEW_FILE"
echo ""

# Create README
cat > "data/ensembl/case_studies/splicevardb/README.md" << 'EOF'
# SpliceVarDB Dataset

**Source:** https://splicevardb.org/  
**Downloaded:** Manual download from website  
**Build:** hg38 (GRCh38) and hg19 (GRCh37)  
**Variants:** 50,716 total

## Directory Structure

- `raw/` - Raw downloaded data
  - `splicevardb_download.tsv` - Original TSV download (50K variants)
- `processed/` - Processed formats (parquet, VCF, etc.)
- `cache/` - API cache and intermediate files

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
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)

# Get systematic path
manager = CaseStudyResourceManager()
splicevardb_path = manager.case_study_paths.splicevardb
raw_tsv = splicevardb_path / "raw/splicevardb_download.tsv"

# Or use the loader
from meta_spliceai.splice_engine.openspliceai_recalibration import SpliceVarDBLoader
loader = SpliceVarDBLoader()  # Automatically uses systematic paths
variants = loader.load_validated_variants(build="GRCh38")
```

## References

Sullivan, R.T., et al. (2024). SpliceVarDB: A database of splice-affecting variants.  
https://splicevardb.org/
EOF

echo "✓ Created README.md"
echo ""

echo "========================================="
echo "✓ Migration Complete!"
echo "========================================="
echo ""
echo "Data is now at: $NEW_FILE"
echo "Structure:"
ls -lh "data/ensembl/case_studies/splicevardb/"
echo ""
echo "Next steps:"
echo "1. Update code to use CaseStudyResourceManager"
echo "2. Process TSV into parquet format"
echo "3. Generate training features"





