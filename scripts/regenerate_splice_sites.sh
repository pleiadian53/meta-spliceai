#!/bin/bash
# Regenerate splice_sites_enhanced.tsv with all 14 columns
# This script uses the NEW extraction code to regenerate your files

set -e

echo "=============================================================================="
echo "REGENERATING SPLICE SITES WITH ENHANCED METADATA (14 columns)"
echo "=============================================================================="
echo ""

# Activate environment
source ~/.bash_profile
mamba activate metaspliceai

cd /Users/pleiadian53/work/meta-spliceai

# Function to regenerate for a specific data directory
regenerate_for_build() {
    local data_dir=$1
    local gtf_file=$2
    local build_name=$3
    
    echo "Processing: $build_name"
    echo "Data dir: $data_dir"
    echo "GTF file: $(basename $gtf_file)"
    echo ""
    
    if [ ! -f "$gtf_file" ]; then
        echo "⚠ GTF file not found, skipping: $gtf_file"
        echo ""
        return
    fi
    
    # Check old file columns
    if [ -f "$data_dir/splice_sites_enhanced.tsv" ]; then
        old_cols=$(head -n 1 "$data_dir/splice_sites_enhanced.tsv" | tr '\t' '\n' | wc -l | xargs)
        echo "Old file has $old_cols columns"
        
        # Backup old file
        backup_file="$data_dir/splice_sites_enhanced.tsv.backup_$(date +%Y%m%d_%H%M%S)"
        echo "Creating backup: $(basename $backup_file)"
        cp "$data_dir/splice_sites_enhanced.tsv" "$backup_file"
    else
        echo "No existing file to backup"
    fi
    
    echo ""
    echo "Extracting splice sites with NEW code (14 columns)..."
    echo "----------------------------------------------------------------------"
    
    # Run extraction with NEW code
    python -c "
from meta_spliceai.system.genomic_resources import extract_splice_sites_from_gtf
from pathlib import Path

df = extract_splice_sites_from_gtf(
    gtf_path='$gtf_file',
    consensus_window=2,
    output_file='$data_dir/splice_sites_enhanced.tsv',
    save=True,
    return_df=True,
    verbosity=2
)

print('')
print('✓ Extraction complete!')
print(f'✓ Output: $data_dir/splice_sites_enhanced.tsv')
print(f'✓ Rows: {len(df):,}')
print(f'✓ Columns: {len(df.columns)}')
"
    
    echo ""
    
    # Verify new file
    if [ -f "$data_dir/splice_sites_enhanced.tsv" ]; then
        new_cols=$(head -n 1 "$data_dir/splice_sites_enhanced.tsv" | tr '\t' '\n' | wc -l | xargs)
        echo "✓ New file has $new_cols columns"
        
        if [ "$new_cols" -eq 14 ]; then
            echo "✅ SUCCESS: File regenerated with all 14 columns!"
        else
            echo "⚠ WARNING: Expected 14 columns but got $new_cols"
        fi
        
        echo ""
        echo "Column names:"
        head -n 1 "$data_dir/splice_sites_enhanced.tsv" | tr '\t' '\n' | nl
        echo ""
        
        echo "Sample data (first 3 rows):"
        head -n 4 "$data_dir/splice_sites_enhanced.tsv" | column -t -s $'\t' | head -20
    else
        echo "❌ ERROR: Output file not created!"
    fi
    
    echo ""
    echo "=============================================================================="
    echo ""
}

# Regenerate for MANE/GRCh38
regenerate_for_build \
    "data/mane/GRCh38" \
    "data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf" \
    "MANE GRCh38"

# Regenerate for Ensembl/GRCh37 (if exists)
if [ -d "data/ensembl/GRCh37" ]; then
    regenerate_for_build \
        "data/ensembl/GRCh37" \
        "data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf" \
        "Ensembl GRCh37"
fi

echo "=============================================================================="
echo "REGENERATION COMPLETE"
echo "=============================================================================="
echo ""
echo "Your splice_sites_enhanced.tsv files now have 14 columns!"
echo ""
echo "Backups of old files saved with .backup_* extension"
echo ""


