#!/bin/bash

# Tmux Runner for Flexible Build Script
# Usage: ./run_flexible_build_tmux.sh [--log-dir LOG_DIR] [session_name] [flexible_build_args...]
#
# Options:
#   --log-dir DIR     Directory for log files (default: logs/)
#   session_name      Name for tmux session (default: auto-generated)
#
# Examples:
#   ./run_flexible_build_tmux.sh "my_build" --n-genes 1000 --overwrite
#   ./run_flexible_build_tmux.sh --log-dir /tmp/logs "my_build" --n-genes 1000
#   ./run_flexible_build_tmux.sh --log-dir=./custom_logs --n-genes 1000

set -e

# Default values
DEFAULT_SESSION_NAME="flexible_builder_$(date +%Y%m%d_%H%M%S)"
DEFAULT_LOG_DIR="logs"

# Parse arguments - check if first argument is a log directory override
if [[ $# -gt 0 && "$1" == --log-dir=* ]]; then
    LOG_DIR="${1#--log-dir=}"
    shift
elif [[ $# -gt 1 && "$1" == "--log-dir" ]]; then
    LOG_DIR="$2"
    shift 2
else
    LOG_DIR="$DEFAULT_LOG_DIR"
fi

# Now parse session name
SESSION_NAME=${1:-$DEFAULT_SESSION_NAME}
if [[ $# -gt 0 ]]; then
    shift  # Remove session name from arguments
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "❌ Session '$SESSION_NAME' already exists!"
    echo "Choose a different name or kill the existing session:"
    echo "   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Check if mamba environment exists
if ! mamba env list | grep -q "surveyor"; then
    echo "❌ Mamba environment 'surveyor' not found!"
    echo "Please create the surveyor environment first."
    exit 1
fi

# Build the flexible build command
FLEXIBLE_BUILD_CMD="./scripts/scaling_solutions/memory_optimization/flexible_build.sh $@"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Create log file name with proper path
LOG_FILE="$LOG_DIR/flexible_builder_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Tmux Flexible Builder ==="
echo "Session name: $SESSION_NAME"
echo "Command: $FLEXIBLE_BUILD_CMD"
echo "Log directory: $LOG_DIR/"
echo "Log file: $LOG_FILE"
echo "============================="

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"

# Send the command to the session
tmux send-keys -t "$SESSION_NAME" "mamba activate surveyor" Enter
tmux send-keys -t "$SESSION_NAME" "$FLEXIBLE_BUILD_CMD 2>&1 | tee $LOG_FILE" Enter

echo ""
echo "✅ Tmux session '$SESSION_NAME' created successfully!"
echo ""
echo "Useful commands:"
echo "  tmux attach -t $SESSION_NAME    # Attach to session"
echo "  tmux list-sessions              # List all sessions"
echo "  tmux kill-session -t $SESSION_NAME  # Kill session"
echo "  tail -f $LOG_FILE               # Monitor log file"
echo ""
echo "Session is now running in the background."
echo "You can detach from the session with: Ctrl+B, then D" 