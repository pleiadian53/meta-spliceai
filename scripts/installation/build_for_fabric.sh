#!/bin/bash

# Build MetaSpliceAI for Microsoft Fabric deployment
set -e

echo "🏗️  Building MetaSpliceAI for Microsoft Fabric"
echo "================================================="

# Check if we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "surveyor" ]]; then
    echo "❌ Please activate the surveyor environment first:"
    echo "   mamba activate surveyor"
    exit 1
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Installing..."
    pip install poetry
fi

# Configure Poetry for the current environment
echo "🔧 Configuring Poetry..."
poetry config virtualenvs.create false
poetry config virtualenvs.in-project false

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build wheel
echo "🛠️  Building wheel..."
poetry build --format wheel

# Verify build
echo "✅ Build completed! Generated files:"
ls -la dist/

# Get wheel filename
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo ""
echo "📦 Wheel file: $(basename "$WHEEL_FILE")"

# Test wheel installation (in a temporary environment)
echo ""
echo "🧪 Testing wheel installation..."
TEMP_ENV="test-wheel-$$"
python -m venv "$TEMP_ENV"
source "$TEMP_ENV/bin/activate"

echo "   Installing wheel in test environment..."
pip install "$WHEEL_FILE" --quiet

echo "   Testing import..."
if python -c "import meta_spliceai; print(f'✅ Successfully imported meta_spliceai v{meta_spliceai.__version__}')" 2>/dev/null; then
    echo "   ✅ Wheel installation test passed!"
else
    echo "   ❌ Wheel installation test failed!"
    deactivate
    rm -rf "$TEMP_ENV"
    exit 1
fi

deactivate
rm -rf "$TEMP_ENV"

echo ""
echo "🎉 Microsoft Fabric wheel build completed successfully!"
echo ""
echo "📋 Next steps for Microsoft Fabric deployment:"
echo "   1. Upload $(basename "$WHEEL_FILE") to your Fabric workspace"
echo "   2. In a Fabric notebook, run:"
echo "      %pip install /path/to/$(basename "$WHEEL_FILE")"
echo "   3. Test the installation:"
echo "      import meta_spliceai"
echo "      print(f'MetaSpliceAI v{meta_spliceai.__version__} loaded!')"
echo ""
echo "🔗 Alternative installation methods:"
echo "   • From private PyPI: %pip install meta-spliceai --extra-index-url <your-index>"
echo "   • With Fabric extras: %pip install meta-spliceai[fabric]"
echo ""
echo "✅ Ready for Microsoft Fabric deployment!" 