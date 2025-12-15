#!/usr/bin/env python3
"""
Memory profiling wrapper for incremental builder to identify peak usage patterns.
"""
import subprocess
import psutil
import time
import sys
from pathlib import Path

def monitor_memory(pid, log_file="memory_usage.log"):
    """Monitor memory usage of a process and its children."""
    try:
        process = psutil.Process(pid)
        with open(log_file, "w") as f:
            f.write("timestamp,rss_mb,vms_mb,percent,children_rss_mb\n")
            
            while process.is_running():
                try:
                    # Main process memory
                    memory_info = process.memory_info()
                    rss_mb = memory_info.rss / 1024 / 1024
                    vms_mb = memory_info.vms / 1024 / 1024
                    percent = process.memory_percent()
                    
                    # Children memory
                    children_rss = 0
                    try:
                        for child in process.children(recursive=True):
                            children_rss += child.memory_info().rss / 1024 / 1024
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    # Log
                    timestamp = time.time()
                    f.write(f"{timestamp},{rss_mb:.1f},{vms_mb:.1f},{percent:.1f},{children_rss:.1f}\n")
                    f.flush()
                    
                    print(f"Memory: {rss_mb:.1f}MB RSS, {children_rss:.1f}MB children", end="\r")
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                    
                time.sleep(2)
                
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        print("Process not found or access denied")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memory_profile_builder.py <command> [args...]")
        sys.exit(1)
    
    # Start the command
    cmd = sys.argv[1:]
    print(f"Starting command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    
    # Monitor memory
    print(f"Monitoring PID {process.pid}")
    monitor_memory(process.pid)
    
    # Wait for completion
    process.wait()
    print(f"\nCommand finished with exit code: {process.returncode}")
    print("Memory log saved to memory_usage.log") 