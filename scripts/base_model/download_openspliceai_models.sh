#!/bin/bash
# Download OpenSpliceAI Pre-trained Models
# 
# Source: https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane/10000nt
# 
# Models: OpenSpliceAI-MANE with 10,000nt context window (best performance)
# Training: MANE annotations + GRCh38
# Context: 10,000 nucleotides (optimal for splice site prediction)

set -e

# Configuration
MODEL_DIR="data/models/openspliceai"
GITHUB_BASE="https://github.com/Kuanhao-Chao/OpenSpliceAI/raw/main/models/openspliceai-mane/10000nt"

echo "=============================================================================="
echo "DOWNLOADING OPENSPLICEAI MODELS"
echo "=============================================================================="
echo ""
echo "Source: OpenSpliceAI-MANE (10,000nt context)"
echo "URL: https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane/10000nt"
echo "Training: MANE annotations + GRCh38"
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"
echo "📁 Created directory: $MODEL_DIR"
echo ""

# Download 5 ensemble models (rs10-rs14)
echo "📥 Downloading 5 ensemble models..."
echo ""

for i in {10..14}; do
    MODEL_FILE="model_10000nt_rs${i}.pt"
    MODEL_URL="$GITHUB_BASE/$MODEL_FILE"
    OUTPUT_PATH="$MODEL_DIR/$MODEL_FILE"
    
    echo "  ⏳ Downloading $MODEL_FILE..."
    echo "     URL: $MODEL_URL"
    
    if curl -L -f "$MODEL_URL" -o "$OUTPUT_PATH" 2>/dev/null; then
        FILE_SIZE=$(ls -lh "$OUTPUT_PATH" | awk '{print $5}')
        echo "     ✅ Downloaded ($FILE_SIZE)"
    else
        echo "     ❌ Failed to download $MODEL_FILE"
        echo "     Please check URL: $MODEL_URL"
        exit 1
    fi
    echo ""
done

# Create metadata file
METADATA_FILE="$MODEL_DIR/metadata.json"
cat > "$METADATA_FILE" << EOF
{
  "model_name": "OpenSpliceAI-MANE",
  "version": "10000nt",
  "source": "https://github.com/Kuanhao-Chao/OpenSpliceAI",
  "training_data": {
    "annotations": "MANE",
    "genome_build": "GRCh38",
    "context_window": 10000
  },
  "models": [
    "model_10000nt_rs10.pt",
    "model_10000nt_rs11.pt",
    "model_10000nt_rs12.pt",
    "model_10000nt_rs13.pt",
    "model_10000nt_rs14.pt"
  ],
  "download_date": "$(date -u +"%Y-%m-%d %H:%M:%S UTC")",
  "notes": "Ensemble of 5 models for robust predictions"
}
EOF

echo "✅ Created metadata: $METADATA_FILE"
echo ""

# Summary
echo "=============================================================================="
echo "DOWNLOAD COMPLETE"
echo "=============================================================================="
echo ""
echo "📊 Summary:"
ls -lh "$MODEL_DIR"/*.pt | awk '{printf "  • %s (%s)\n", $9, $5}'
echo ""
echo "📁 Location: $MODEL_DIR"
echo "📄 Metadata: $METADATA_FILE"
echo ""
echo "✅ Ready to use OpenSpliceAI!"
echo ""
echo "Next steps:"
echo "  1. Test loading: python -c \"from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_openspliceai_ensemble; models = load_openspliceai_ensemble(); print(f'✅ {len(models)} models loaded')\""
echo "  2. Run predictions: python -c \"from meta_spliceai import run_base_model_predictions; r = run_base_model_predictions(base_model='openspliceai', target_genes=['BRCA1'], mode='test')\""
echo ""
echo "=============================================================================="


