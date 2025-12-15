#!/bin/bash
#
# Download GRCh38 MANE Data for OpenSpliceAI Compatibility
#
# This script downloads GRCh38 genome and MANE annotations to match
# OpenSpliceAI's training data (GRCh38, MANE v1.3).
#
# Date: 2025-11-06
# Reason: OpenSpliceAI trained on GRCh38/MANE, we need matching resources
#

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║           Download GRCh38 MANE Data for OpenSpliceAI Compatibility          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
SPECIES="homo_sapiens"
BUILD="GRCh38_MANE"
MANE_RELEASE="1.3"  # MANE release version
DATA_ROOT="data/mane"

echo "Configuration:"
echo "  Species: $SPECIES"
echo "  Build: GRCh38 (MANE annotations)"
echo "  MANE Release: $MANE_RELEASE"
echo "  Data Root: $DATA_ROOT"
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "surveyor"; then
    echo "❌ Error: 'surveyor' conda environment not found"
    echo "   Please create it first: conda env create -f environment.yml"
    exit 1
fi

echo "✅ Found 'surveyor' conda environment"
echo ""

# Step 1: Download MANE GFF and GRCh38 FASTA
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Download GRCh38 FASTA and MANE Annotations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Expected files
MANE_DIR="$DATA_ROOT/GRCh38"
GFF_FILE="$MANE_DIR/MANE.GRCh38.v${MANE_RELEASE}.refseq_genomic.gff"
FASTA_FILE="$MANE_DIR/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

# Check if files already exist
if [ -f "$GFF_FILE" ] && [ -f "$FASTA_FILE" ]; then
    echo "⚠️  Files already exist:"
    echo "   GFF: $GFF_FILE"
    echo "   FASTA: $FASTA_FILE"
    echo ""
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download..."
        SKIP_DOWNLOAD=true
    fi
fi

if [ "$SKIP_DOWNLOAD" != "true" ]; then
    echo "Downloading GRCh38 MANE data..."
    echo ""
    
    # Create directory
    mkdir -p "$MANE_DIR"
    
    # Download MANE GFF
    echo "📥 Downloading MANE v${MANE_RELEASE} annotations..."
    MANE_URL="https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_${MANE_RELEASE}/MANE.GRCh38.v${MANE_RELEASE}.refseq_genomic.gff.gz"
    
    if curl -L -f "$MANE_URL" -o "${GFF_FILE}.gz"; then
        echo "✅ Downloaded MANE GFF"
        gunzip -f "${GFF_FILE}.gz"
        echo "✅ Decompressed MANE GFF"
    else
        echo "❌ Failed to download MANE GFF"
        echo "   URL: $MANE_URL"
        exit 1
    fi
    
    echo ""
    
    # Download GRCh38 FASTA
    echo "📥 Downloading GRCh38 reference genome..."
    echo "   Note: Using Ensembl GRCh38 primary assembly (standard for splice site analysis)"
    
    # Use Ensembl's GRCh38 FASTA (standard, well-maintained)
    FASTA_URL="https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
    
    if curl -L -f "$FASTA_URL" -o "${FASTA_FILE}.gz"; then
        echo "✅ Downloaded GRCh38 FASTA"
        gunzip -f "${FASTA_FILE}.gz"
        echo "✅ Decompressed GRCh38 FASTA"
    else
        echo "❌ Failed to download GRCh38 FASTA"
        echo "   URL: $FASTA_URL"
        echo ""
        echo "Alternative: Use existing Ensembl GRCh38 FASTA if available"
        echo "   ln -s ../../ensembl/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa $FASTA_FILE"
        exit 1
    fi
    
    echo ""
    echo "✅ Download complete"
else
    echo "Using existing files"
fi

echo ""

# Step 2: Index FASTA (if needed)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Index FASTA File"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

FASTA_INDEX="${FASTA_FILE}.fai"

if [ -f "$FASTA_INDEX" ]; then
    echo "✅ FASTA index already exists: $FASTA_INDEX"
else
    echo "Creating FASTA index..."
    
    if command -v samtools &> /dev/null; then
        samtools faidx "$FASTA_FILE"
        echo "✅ FASTA index created"
    else
        echo "⚠️  samtools not found, skipping FASTA indexing"
        echo "   Install samtools for faster FASTA access:"
        echo "   conda install -c bioconda samtools"
    fi
fi

echo ""

# Step 3: Convert GFF to GTF (optional, for compatibility)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Convert GFF to GTF (Optional)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

GTF_FILE="${GFF_FILE%.gff}.gtf"

if [ -f "$GTF_FILE" ]; then
    echo "✅ GTF file already exists: $GTF_FILE"
else
    echo "Converting GFF to GTF format..."
    
    if command -v gffread &> /dev/null; then
        gffread "$GFF_FILE" -T -o "$GTF_FILE"
        echo "✅ GTF conversion complete"
    else
        echo "⚠️  gffread not found, skipping GTF conversion"
        echo "   The system can work with GFF3 format directly"
        echo "   To install gffread: conda install -c bioconda gffread"
    fi
fi

echo ""

# Step 4: Derive splice sites
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Derive GRCh38 Splice Sites from MANE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SPLICE_SITES_FILE="$MANE_DIR/splice_sites_enhanced.tsv"

if [ -f "$SPLICE_SITES_FILE" ]; then
    echo "⚠️  Splice sites file already exists: $SPLICE_SITES_FILE"
    echo ""
    read -p "Do you want to regenerate? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping derivation..."
        SKIP_DERIVE=true
    fi
fi

if [ "$SKIP_DERIVE" != "true" ]; then
    echo "Deriving splice sites from MANE annotations..."
    echo ""
    echo "⚠️  Note: GFF3 format support is experimental"
    echo "   If derivation fails, please convert to GTF first"
    echo ""
    
    # Try with GFF first, fall back to GTF if available
    ANNOTATION_FILE="$GFF_FILE"
    if [ -f "$GTF_FILE" ]; then
        ANNOTATION_FILE="$GTF_FILE"
        echo "Using GTF format: $GTF_FILE"
    else
        echo "Using GFF3 format: $GFF_FILE"
    fi
    
    # Use the already-activated environment
    # Activate environment and run derivation
    source ~/.bash_profile 2>/dev/null || true
    mamba activate surveyor 2>/dev/null || conda activate surveyor
    
    python -m meta_spliceai.system.genomic_resources.cli derive \
        --build GRCh38_MANE \
        --release "$MANE_RELEASE" \
        --splice-sites \
        --consensus-window 2 \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Derivation complete"
    else
        echo ""
        echo "❌ Derivation failed"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Ensure GTF conversion succeeded"
        echo "  2. Check that FASTA file is indexed"
        echo "  3. Verify MANE GFF format compatibility"
        exit 1
    fi
else
    echo "Using existing splice sites file"
fi

echo ""

# Step 5: Verify files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Verify Downloaded Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Checking files..."
echo ""

# Check GFF
if [ -f "$GFF_FILE" ]; then
    GFF_SIZE=$(du -h "$GFF_FILE" | cut -f1)
    GFF_LINES=$(wc -l < "$GFF_FILE")
    echo "✅ GFF file exists: $GFF_FILE"
    echo "   Size: $GFF_SIZE"
    echo "   Lines: $GFF_LINES"
else
    echo "❌ GFF file missing: $GFF_FILE"
    exit 1
fi

echo ""

# Check FASTA
if [ -f "$FASTA_FILE" ]; then
    FASTA_SIZE=$(du -h "$FASTA_FILE" | cut -f1)
    echo "✅ FASTA file exists: $FASTA_FILE"
    echo "   Size: $FASTA_SIZE"
else
    echo "❌ FASTA file missing: $FASTA_FILE"
    exit 1
fi

echo ""

# Check splice sites
if [ -f "$SPLICE_SITES_FILE" ]; then
    SPLICE_SIZE=$(du -h "$SPLICE_SITES_FILE" | cut -f1)
    SPLICE_LINES=$(wc -l < "$SPLICE_SITES_FILE")
    echo "✅ Splice sites file exists: $SPLICE_SITES_FILE"
    echo "   Size: $SPLICE_SIZE"
    echo "   Lines: $SPLICE_LINES"
else
    echo "⚠️  Splice sites file missing: $SPLICE_SITES_FILE"
    echo "   You may need to derive it manually"
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ GRCh38 MANE data successfully downloaded and processed"
echo ""
echo "Files created:"
echo "  1. $GFF_FILE"
echo "  2. $FASTA_FILE"
if [ -f "$GTF_FILE" ]; then
    echo "  3. $GTF_FILE"
fi
if [ -f "$SPLICE_SITES_FILE" ]; then
    echo "  4. $SPLICE_SITES_FILE"
fi
echo ""
echo "Next steps:"
echo "  1. Test OpenSpliceAI with sample genes:"
echo "     python -c \"from meta_spliceai import run_base_model_predictions; \\"
echo "                results = run_base_model_predictions(base_model='openspliceai', \\"
echo "                                                      target_genes=['BRCA1'], \\"
echo "                                                      mode='test')\""
echo ""
echo "  2. Run full OpenSpliceAI evaluation:"
echo "     python scripts/testing/test_openspliceai_predictions.py"
echo ""
echo "  3. Compare with SpliceAI (after coordinate liftover):"
echo "     python scripts/analysis/compare_base_models.py"
echo ""
echo "Expected performance:"
echo "  OpenSpliceAI on GRCh38/MANE should achieve ~90-95% PR-AUC"
echo "  (similar to SpliceAI on GRCh37/Ensembl)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

