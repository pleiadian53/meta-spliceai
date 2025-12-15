#!/bin/bash
#
# Extract genomic sequences for GRCh38 MANE
#
# This script extracts gene sequences from the GRCh38 genome assembly
# using MANE annotations, preparing them for OpenSpliceAI predictions.
#

set -e  # Exit on error

echo "========================================================================"
echo "Extract GRCh38 MANE Genomic Sequences"
echo "========================================================================"
echo

# Note: Run this script with the environment already activated:
# source ~/.bash_profile && mamba activate surveyor && ./scripts/setup/extract_grch38_mane_sequences.sh

# Navigate to project root
cd /Users/pleiadian53/work/meta-spliceai

# Check prerequisites
echo "Checking prerequisites..."
if [ ! -f "data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf" ]; then
    echo "❌ MANE GTF not found. Please run download_grch38_mane_data.sh first."
    exit 1
fi

if [ ! -f "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa" ]; then
    echo "❌ GRCh38 FASTA not found. Please run download_grch38_mane_data.sh first."
    exit 1
fi

echo "✅ Prerequisites satisfied"
echo

# Extract sequences using the data preparation workflow
echo "========================================================================"
echo "Extracting genomic sequences..."
echo "========================================================================"
echo

python -c "
import sys
sys.path.insert(0, '/Users/pleiadian53/work/meta-spliceai')

from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import prepare_genomic_sequences
from meta_spliceai.system.genomic_resources import Registry

# Initialize registry for GRCh38 MANE
registry = Registry(build='GRCh38_MANE', release='1.3')

print(f'Data directory: {registry.data_dir}')
print(f'GTF: {registry.get_gtf_path()}')
print(f'FASTA: {registry.get_fasta_path()}')
print()

# Extract sequences
result = prepare_genomic_sequences(
    local_dir=str(registry.data_dir),
    gtf_file=str(registry.get_gtf_path()),
    genome_fasta=str(registry.get_fasta_path()),
    mode='gene',
    seq_type='full',
    do_extract=True,  # Force extraction
    chromosomes=None,  # All chromosomes
    test_mode=False,   # Full extraction
    seq_format='parquet',
    verbosity=2
)

if result['success']:
    print()
    print('✅ Sequence extraction complete!')
    print(f'Main sequence file: {result.get(\"main_sequence_file\", \"N/A\")}')
    if 'chromosome_files' in result:
        print(f'Chromosome-specific files: {len(result[\"chromosome_files\"])}')
else:
    print('❌ Sequence extraction failed!')
    sys.exit(1)
"

echo
echo "========================================================================"
echo "✅ GRCh38 MANE sequence extraction complete!"
echo "========================================================================"
echo
echo "Next steps:"
echo "  1. Run OpenSpliceAI predictions"
echo "  2. Test with: python scripts/testing/test_openspliceai_gene_categories.py"
echo

