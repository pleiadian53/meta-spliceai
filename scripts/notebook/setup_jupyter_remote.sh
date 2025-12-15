#!/bin/bash
# Setup script for Jupyter on remote VM with VSCode integration

set -e

echo "🚀 Setting up Jupyter for remote development..."

JUPYTER_PORT=8888
CONDA_ENV="surveyor"

# Check if environment exists (prefer mamba if available)
if command -v mamba >/dev/null 2>&1; then
    ENV_LIST_CMD="mamba env list"
else
    ENV_LIST_CMD="conda env list"
fi

if ! bash -lc "$ENV_LIST_CMD" | grep -q "^$CONDA_ENV "; then
    echo "❌ Conda environment '$CONDA_ENV' not found!"
    echo "Please create it first with: mamba env create -f environment.yml"
    exit 1
fi

# Activate environment (prefer mamba)
if command -v mamba >/dev/null 2>&1; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate "$CONDA_ENV"
else
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# Generate Jupyter config if not exists
if [ ! -f "$HOME/.jupyter/jupyter_lab_config.py" ]; then
    echo "📝 Generating Jupyter configuration..."
    jupyter lab --generate-config
fi

# Configure Jupyter for remote access
cat >> "$HOME/.jupyter/jupyter_lab_config.py" <<EOF

# Remote access configuration (added by setup script)
c.ServerApp.ip = '127.0.0.1'  # Only localhost (use SSH tunnel)
c.ServerApp.port = $JUPYTER_PORT
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = False
c.ServerApp.token = ''  # No token for convenience (secure with SSH)
c.ServerApp.password = ''  # No password (secure with SSH)
EOF

# Install kernel for the environment
echo "📦 Installing Jupyter kernel for $CONDA_ENV..."
python -m ipykernel install --user --name=$CONDA_ENV --display-name="Python ($CONDA_ENV)"

# Create start script
cat > "$HOME/work/meta-spliceai/start_jupyter.sh" <<'EOF'
#!/bin/bash
# Start Jupyter Lab for development

JUPYTER_PORT=8888

# Activate environment (prefer mamba)
if command -v mamba >/dev/null 2>&1; then
  eval "$(mamba shell hook --shell bash)"
  mamba activate surveyor
else
  eval "$(conda shell.bash hook)"
  conda activate surveyor
fi

echo "Starting Jupyter Lab on port $JUPYTER_PORT..."
echo "Access UI at: http://localhost:$JUPYTER_PORT"
echo "Press Ctrl+C to stop"

cd "$HOME/work/meta-spliceai"
jupyter lab --ip=127.0.0.1 --port=$JUPYTER_PORT --no-browser
EOF

chmod +x "$HOME/work/meta-spliceai/start_jupyter.sh"

echo ""
echo "🎉 Jupyter setup complete!"
echo ""
echo "📋 Quick Start Guide:"
echo "-------------------"
echo ""
echo "1. Start Jupyter Lab:"
echo "   ./start_jupyter.sh"
echo ""
echo "2. SSH with port forwarding from your local machine:"
echo "   ssh -L 8888:localhost:8888 -L 5000:localhost:5000 $USER@$(hostname -I | awk '{print $1}')"
echo ""
echo "3. Access Jupyter Lab in your browser:"
echo "   http://localhost:8888"
echo ""
echo "🔧 VSCode: Will auto-forward ports when you open notebooks!"




