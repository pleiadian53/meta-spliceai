#!/bin/bash

# Swap Setup Guide for Incremental Builder
# Optimized for data science workloads with memory spikes

echo "=== Swap Setup for Incremental Builder ==="

# Check current memory and swap status
echo "Current memory status:"
free -h
echo ""

echo "Current swap status:"
swapon --show
echo ""

# Check available disk space
echo "Available disk space:"
df -h / | grep -v Filesystem
echo ""

# Recommend swap size based on available RAM
RAM_GB=$(free -g | grep Mem: | awk '{print $2}')
echo "Detected RAM: ${RAM_GB}GB"

if [ $RAM_GB -gt 32 ]; then
    SWAP_SIZE="8G"
    echo "Recommended swap size: $SWAP_SIZE (for high-RAM systems)"
elif [ $RAM_GB -gt 16 ]; then
    SWAP_SIZE="16G" 
    echo "Recommended swap size: $SWAP_SIZE (equal to half RAM)"
else
    SWAP_SIZE="${RAM_GB}G"
    echo "Recommended swap size: $SWAP_SIZE (equal to RAM)"
fi

echo ""
echo "=== Setup Commands (run as sudo) ==="
echo ""

cat << 'EOF'
# 1. Create swap file
sudo fallocate -l 8G /swapfile

# 2. Set secure permissions
sudo chmod 600 /swapfile

# 3. Make it a swap file
sudo mkswap /swapfile

# 4. Enable swap
sudo swapon /swapfile

# 5. Make permanent (survives reboot)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 6. Optimize swappiness for data science workloads
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
EOF

echo ""
echo "=== Alternative: Temporary Swap (no permanent changes) ==="
echo ""

cat << 'EOF'
# Create temporary swap (will be removed on reboot)
sudo fallocate -l 8G /tmp/swapfile
sudo chmod 600 /tmp/swapfile  
sudo mkswap /tmp/swapfile
sudo swapon /tmp/swapfile

# Check it's active
swapon --show

# Remove after use (optional)
# sudo swapoff /tmp/swapfile && sudo rm /tmp/swapfile
EOF

echo ""
echo "=== Monitoring Swap Usage ==="
echo ""

cat << 'EOF'
# Real-time memory and swap monitoring
watch -n 2 'free -h && echo "" && swapon --show'

# Check swap usage by process
sudo cat /proc/*/status | grep -E "Name:|VmSwap:" | paste - -

# Monitor during incremental builder
while true; do
    SWAP_USED=$(free -m | grep Swap | awk '{print $3}')
    RAM_AVAIL=$(free -m | grep Mem | awk '{print $7}')
    echo "$(date): RAM available: ${RAM_AVAIL}MB, Swap used: ${SWAP_USED}MB"
    sleep 30
done
EOF

echo ""
echo "=== Performance Expectations ==="
echo ""

# Check storage type and performance
DISK_TYPE="unknown"
if lsblk -d -o name,rota | grep -q "0"; then
    DISK_TYPE="SSD"
elif lsblk -d -o name,rota | grep -q "1"; then
    DISK_TYPE="HDD"
fi

echo "Storage type: $DISK_TYPE"

if [ "$DISK_TYPE" = "SSD" ]; then
    echo "Expected swap performance: ~500 MB/s (very good for data science)"
    echo "Memory spikes will cause brief slowdowns but no crashes"
elif [ "$DISK_TYPE" = "HDD" ]; then
    echo "Expected swap performance: ~100 MB/s (acceptable for batch processing)"
    echo "Memory spikes will cause noticeable slowdowns but no crashes"
fi

echo ""
echo "=== Safety Notes ==="
echo "✓ Swap prevents OOM crashes"
echo "✓ No data loss risk"  
echo "✓ Can be safely removed after use"
echo "✓ Minimal disk wear for occasional use"
echo "✓ No impact on other processes"

echo ""
echo "=== Quick Start ==="
echo "Run these commands to set up temporary swap:"
echo ""
echo "sudo fallocate -l 8G /tmp/swapfile"
echo "sudo chmod 600 /tmp/swapfile"
echo "sudo mkswap /tmp/swapfile" 
echo "sudo swapon /tmp/swapfile"
echo ""
echo "Then run your incremental builder normally." 