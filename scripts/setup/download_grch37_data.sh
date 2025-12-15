#!/bin/bash
#
# Download GRCh37 Data for SpliceAI Compatibility
#
# This script downloads GRCh37 genome and annotations to match
# SpliceAI's training data (GRCh37/hg19, GENCODE V24lift37).
#
# Date: 2025-10-31
# Reason: SpliceAI trained on GRCh37, we were using GRCh38
# Impact: 44% performance drop due to coordinate mismatch
#

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              Download GRCh37 Data for SpliceAI Compatibility                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
SPECIES="homo_sapiens"
BUILD="GRCh37"
RELEASE="87"  # Last Ensembl release for GRCh37
DATA_ROOT="data/ensembl"

echo "Configuration:"
echo "  Species: $SPECIES"
echo "  Build: $BUILD"
echo "  Release: $RELEASE"
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

# Step 1: Download GTF and FASTA
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Download GRCh37 GTF and FASTA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if files already exist
GTF_FILE="$DATA_ROOT/$BUILD/Homo_sapiens.GRCh37.$RELEASE.gtf"
FASTA_FILE="$DATA_ROOT/$BUILD/Homo_sapiens.GRCh37.dna.primary_assembly.fa"

if [ -f "$GTF_FILE" ] && [ -f "$FASTA_FILE" ]; then
    echo "⚠️  Files already exist:"
    echo "   GTF: $GTF_FILE"
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
    echo "Downloading GRCh37 data..."
    echo ""
    
    conda run -n surveyor --no-capture-output \
        python -m meta_spliceai.system.genomic_resources.cli bootstrap \
        --build $BUILD \
        --release $RELEASE \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Download complete"
    else
        echo ""
        echo "❌ Download failed"
        exit 1
    fi
else
    echo "Using existing files"
fi

echo ""

# Step 2: Derive splice sites
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Derive GRCh37 Splice Sites"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SPLICE_SITES_FILE="$DATA_ROOT/$BUILD/splice_sites_enhanced.tsv"

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
    echo "Deriving splice sites from GRCh37 GTF..."
    echo ""
    
    conda run -n surveyor --no-capture-output \
        python -m meta_spliceai.system.genomic_resources.cli derive \
        --build $BUILD \
        --release $RELEASE \
        --splice-sites \
        --consensus-window 2 \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Derivation complete"
    else
        echo ""
        echo "❌ Derivation failed"
        exit 1
    fi
else
    echo "Using existing splice sites file"
fi

echo ""

# Step 3: Verify files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Verify Downloaded Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Checking files..."
echo ""

# Check GTF
if [ -f "$GTF_FILE" ]; then
    GTF_SIZE=$(du -h "$GTF_FILE" | cut -f1)
    GTF_LINES=$(wc -l < "$GTF_FILE")
    echo "✅ GTF file exists: $GTF_FILE"
    echo "   Size: $GTF_SIZE"
    echo "   Lines: $GTF_LINES"
else
    echo "❌ GTF file missing: $GTF_FILE"
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
    echo "❌ Splice sites file missing: $SPLICE_SITES_FILE"
    exit 1
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ GRCh37 data successfully downloaded and processed"
echo ""
echo "Files created:"
echo "  1. $GTF_FILE"
echo "  2. $FASTA_FILE"
echo "  3. $SPLICE_SITES_FILE"
echo ""
echo "Next steps:"
echo "  1. Re-run evaluation on GRCh37:"
echo "     python scripts/testing/comprehensive_spliceai_evaluation.py --build GRCh37"
echo ""
echo "  2. Re-run adjustment detection on GRCh37:"
echo "     python scripts/testing/test_score_adjustment_detection.py --build GRCh37"
echo ""
echo "  3. Update workflows to use GRCh37:"
echo "     Set genome_build='GRCh37' in workflow configuration"
echo ""
echo "Expected performance improvement:"
echo "  PR-AUC: 0.54 → 0.80-0.90"
echo "  Top-k Accuracy: 0.55 → 0.75-0.85"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

