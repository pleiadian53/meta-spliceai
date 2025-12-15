#!/bin/bash

# Comprehensive performance monitoring for incremental builder
# Monitors CPU, memory, disk I/O, and swap usage

set -e

# Configuration
LOG_FILE="build_performance.log"
INTERVAL=10  # seconds between measurements
PID_FILE="/tmp/incremental_builder.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize log file
echo "# Performance Monitor Log" > "$LOG_FILE"
echo "# Timestamp,CPU%,Memory_GB,Swap_GB,Disk_IO_MBps,Process_RSS_GB" >> "$LOG_FILE"

log_message() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warning_message() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error_message() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success_message() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to get process memory usage
get_process_memory() {
    local pid=$1
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
        local rss_kb=$(cat "/proc/$pid/status" | grep VmRSS | awk '{print $2}')
        echo "scale=2; $rss_kb / 1024 / 1024" | bc -l 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Function to get disk I/O
get_disk_io() {
    # Get disk I/O in MB/s
    local io_stats=$(iostat -d 1 2 | tail -1 | awk '{print $3}')
    echo "$io_stats" 2>/dev/null || echo "0"
}

# Function to check for OOM conditions
check_oom_risk() {
    local available_mem=$(free -g | grep Mem | awk '{print $7}')
    local swap_used=$(free -g | grep Swap | awk '{print $3}')
    
    if [[ $available_mem -lt 2 ]]; then
        warning_message "Low available memory: ${available_mem}GB"
    fi
    
    if [[ $swap_used -gt 0 ]]; then
        warning_message "Swap is being used: ${swap_used}GB"
    fi
}

# Function to monitor system resources
monitor_system() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Memory usage
    local mem_total=$(free -g | grep Mem | awk '{print $2}')
    local mem_used=$(free -g | grep Mem | awk '{print $3}')
    local mem_available=$(free -g | grep Mem | awk '{print $7}')
    
    # Swap usage
    local swap_total=$(free -g | grep Swap | awk '{print $2}')
    local swap_used=$(free -g | grep Swap | awk '{print $3}')
    
    # Disk I/O
    local disk_io=$(get_disk_io)
    
    # Process memory (if PID is available)
    local process_rss="0"
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        process_rss=$(get_process_memory "$pid")
    fi
    
    # Log to file
    echo "$timestamp,$cpu_usage,$mem_used,$swap_used,$disk_io,$process_rss" >> "$LOG_FILE"
    
    # Display current status
    echo -e "\n${BLUE}=== System Status ===${NC}"
    echo "CPU Usage: ${cpu_usage}%"
    echo "Memory: ${mem_used}GB / ${mem_total}GB (${mem_available}GB available)"
    echo "Swap: ${swap_used}GB / ${swap_total}GB"
    echo "Disk I/O: ${disk_io} MB/s"
    echo "Process RSS: ${process_rss}GB"
    
    # Check for warnings
    check_oom_risk
    
    echo -e "${BLUE}========================${NC}\n"
}

# Function to start monitoring
start_monitoring() {
    log_message "Starting performance monitoring..."
    log_message "Log file: $LOG_FILE"
    log_message "Monitoring interval: ${INTERVAL}s"
    
    # Create PID file for the incremental builder process
    if [[ -n "$1" ]]; then
        echo "$1" > "$PID_FILE"
        log_message "Monitoring process PID: $1"
    fi
    
    # Start monitoring loop
    while true; do
        monitor_system
        sleep "$INTERVAL"
    done
}

# Function to analyze log file
analyze_log() {
    if [[ ! -f "$LOG_FILE" ]]; then
        error_message "Log file not found: $LOG_FILE"
        return 1
    fi
    
    echo -e "\n${BLUE}=== Performance Analysis ===${NC}"
    
    # Calculate statistics
    local total_lines=$(wc -l < "$LOG_FILE")
    local data_lines=$((total_lines - 2))  # Subtract header lines
    
    if [[ $data_lines -le 0 ]]; then
        error_message "No data found in log file"
        return 1
    fi
    
    echo "Data points: $data_lines"
    
    # Extract numeric columns and calculate stats
    local cpu_stats=$(tail -n +3 "$LOG_FILE" | cut -d',' -f2 | sort -n)
    local mem_stats=$(tail -n +3 "$LOG_FILE" | cut -d',' -f3 | sort -n)
    local swap_stats=$(tail -n +3 "$LOG_FILE" | cut -d',' -f4 | sort -n)
    
    # Calculate averages
    local cpu_avg=$(echo "$cpu_stats" | awk '{sum+=$1} END {print sum/NR}')
    local mem_avg=$(echo "$mem_stats" | awk '{sum+=$1} END {print sum/NR}')
    local swap_avg=$(echo "$swap_stats" | awk '{sum+=$1} END {print sum/NR}')
    
    # Calculate max values
    local cpu_max=$(echo "$cpu_stats" | tail -1)
    local mem_max=$(echo "$mem_stats" | tail -1)
    local swap_max=$(echo "$swap_stats" | tail -1)
    
    echo -e "\n${GREEN}Performance Summary:${NC}"
    echo "CPU Usage: Avg=${cpu_avg}%, Max=${cpu_max}%"
    echo "Memory Usage: Avg=${mem_avg}GB, Max=${mem_max}GB"
    echo "Swap Usage: Avg=${swap_avg}GB, Max=${swap_max}GB"
    
    # Check for potential issues
    if (( $(echo "$mem_max > 40" | bc -l) )); then
        warning_message "High memory usage detected (max: ${mem_max}GB)"
    fi
    
    if (( $(echo "$swap_max > 0" | bc -l) )); then
        warning_message "Swap was used during build (max: ${swap_max}GB)"
    fi
    
    if (( $(echo "$cpu_avg > 80" | bc -l) )); then
        warning_message "High CPU usage detected (avg: ${cpu_avg}%)"
    fi
}

# Main script logic
case "${1:-}" in
    "start")
        if [[ -n "$2" ]]; then
            start_monitoring "$2"
        else
            start_monitoring
        fi
        ;;
    "analyze")
        analyze_log
        ;;
    "monitor")
        # Start monitoring in background
        start_monitoring "$2" &
        MONITOR_PID=$!
        echo "Monitor PID: $MONITOR_PID"
        echo "To stop monitoring: kill $MONITOR_PID"
        ;;
    *)
        echo "Usage: $0 {start|analyze|monitor} [PID]"
        echo ""
        echo "Commands:"
        echo "  start [PID]    - Start monitoring with optional process PID"
        echo "  analyze        - Analyze existing log file"
        echo "  monitor [PID]  - Start monitoring in background"
        echo ""
        echo "Examples:"
        echo "  $0 start 12345                    # Monitor specific process"
        echo "  $0 monitor 12345 &                # Background monitoring"
        echo "  $0 analyze                        # Analyze performance log"
        exit 1
        ;;
esac 