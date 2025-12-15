#!/bin/bash
# Setup script for MLflow on remote VM with VSCode integration

set -e

echo "🚀 Setting up MLflow for remote development..."

# Configuration
MLFLOW_PORT=5000
MLFLOW_DATA_DIR="$HOME/output/mlflow_data"
MLFLOW_DB="$MLFLOW_DATA_DIR/db/mlflow.db"
MLFLOW_ARTIFACTS="$MLFLOW_DATA_DIR/artifacts"
# Allow override via environment (e.g., CONDA_ENV=myenv bash scripts/mlflow/setup_mlflow_remote.sh)
CONDA_ENV="${CONDA_ENV:-surveyor}"

# Create directories
echo "📁 Creating MLflow directories..."
mkdir -p "$MLFLOW_DATA_DIR/db"
mkdir -p "$MLFLOW_DATA_DIR/artifacts"
mkdir -p "$HOME/work/meta-spliceai/mlruns"

# Check if environment exists (robust; prefer mamba if available)
if command -v mamba >/dev/null 2>&1; then
    CHECK_ENV_CMD="mamba env export -n \"$CONDA_ENV\" >/dev/null 2>&1"
else
    CHECK_ENV_CMD="conda env export -n \"$CONDA_ENV\" >/dev/null 2>&1"
fi

if ! bash -lc "$CHECK_ENV_CMD"; then
    echo "❌ Conda/Mamba environment '$CONDA_ENV' not found!"
    echo "Create it first, for example:"
    echo "  mamba env create -f $HOME/work/meta-spliceai/environment.yml -n $CONDA_ENV"
    echo "Or activate an existing env and re-run with: CONDA_ENV=<name> bash scripts/mlflow/setup_mlflow_remote.sh"
    exit 1
fi

# Activate environment and install/update MLflow (prefer mamba)
echo "📦 Installing/updating MLflow in $CONDA_ENV environment..."
if command -v mamba >/dev/null 2>&1; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate "$CONDA_ENV"
else
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# Install MLflow if not present or update
# pip install --upgrade mlflow protobuf alembic psutil plotly python-dotenv

# Create systemd service file (optional - for persistent server)
SERVICE_FILE="$HOME/.config/systemd/user/mlflow.service"
mkdir -p "$HOME/.config/systemd/user"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$HOME/work/meta-spliceai
Environment="PATH=$HOME/.local/share/mamba/envs/$CONDA_ENV/bin:$HOME/miniconda3/envs/$CONDA_ENV/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$HOME/.local/share/mamba/envs/$CONDA_ENV/bin/mlflow server \
    --host 127.0.0.1 \
    --port $MLFLOW_PORT \
    --backend-store-uri sqlite:///$MLFLOW_DB \
    --default-artifact-root file://$MLFLOW_ARTIFACTS \
    --serve-artifacts
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

echo "✅ Systemd service file created at: $SERVICE_FILE"

# Create convenience scripts
cat > "$HOME/work/meta-spliceai/start_mlflow.sh" <<EOF
#!/bin/bash
# Start MLflow server for development

MLFLOW_PORT=$MLFLOW_PORT
MLFLOW_DATA_DIR="$MLFLOW_DATA_DIR"

# Activate environment (prefer mamba)
if command -v mamba >/dev/null 2>&1; then
  eval "\$(mamba shell hook --shell bash)"
  mamba activate "$CONDA_ENV"
else
  eval "\$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

echo "Starting MLflow server on port \$MLFLOW_PORT..."
echo "Access UI at: http://localhost:\$MLFLOW_PORT"
echo "Press Ctrl+C to stop"

mlflow server \
    --host 127.0.0.1 \
    --port \$MLFLOW_PORT \
    --backend-store-uri sqlite:///\$MLFLOW_DATA_DIR/db/mlflow.db \
    --default-artifact-root file://\$MLFLOW_DATA_DIR/artifacts \
    --serve-artifacts
EOF

chmod +x "$HOME/work/meta-spliceai/start_mlflow.sh"

# Create environment file
cat > "$HOME/work/meta-spliceai/.env.example" <<EOF
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=surveyor-inference
MLFLOW_ARTIFACT_ROOT=$MLFLOW_ARTIFACTS

# Jupyter Configuration
JUPYTER_PORT=8888
JUPYTER_ENABLE_LAB=true

# Python Path Configuration
PYTHONPATH=$HOME/work/meta-spliceai:\$PYTHONPATH
EOF

# Add to bashrc if not already present
if ! grep -q "MLFLOW_TRACKING_URI" "$HOME/.bashrc"; then
    echo "" >> "$HOME/.bashrc"
    echo "# MLflow Configuration" >> "$HOME/.bashrc"
    echo "export MLFLOW_TRACKING_URI=http://localhost:$MLFLOW_PORT" >> "$HOME/.bashrc"
    echo "✅ Added MLFLOW_TRACKING_URI to ~/.bashrc"
fi

echo ""
echo "🎉 MLflow setup complete!"
echo ""
echo "📋 Quick Start Guide:"
echo "-------------------"
echo ""
echo "1. Start MLflow server (choose one):"
echo "   a) Quick start:     ./start_mlflow.sh"
echo "   b) Background:      nohup ./start_mlflow.sh > mlflow.log 2>&1 &"
echo "   c) Systemd:         systemctl --user enable --now mlflow"
echo ""
echo "2. SSH with port forwarding from your local machine:"
echo "   ssh -L 5000:localhost:5000 $USER@$(hostname -I | awk '{print $1}')"
echo ""
echo "3. Access MLflow UI in your browser:"
echo "   http://localhost:5000"
echo ""
echo "4. Run inference with MLflow tracking:"
echo "   python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
       --mlflow-enable \
       --mlflow-experiment meta-spliceai \
       [other args...]"
echo ""
echo "📁 Data locations:"
echo "   Database:  $MLFLOW_DB"
echo "   Artifacts: $MLFLOW_ARTIFACTS"
echo ""
echo "🔧 VSCode: Settings already configured for auto port-forwarding!"




