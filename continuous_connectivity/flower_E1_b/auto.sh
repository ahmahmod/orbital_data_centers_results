#!/bin/bash

# Define the steps/intervals to iterate through
values=(8 16 64)

# 1. Run DiLoCo mode with varying LOCAL_STEPS
echo "=== Starting DiLoCo Runs ==="
for steps in "${values[@]}"; do
    echo "Running diloco with LOCAL_STEPS=${steps}..."
    METHOD=diloco \
    LOCAL_STEPS=${steps} \
    MAX_SECONDS=3600 \
    ./run.sh
done

# 2. Run Gossip mode with varying GOSSIP_EVERY
echo "=== Starting Gossip Runs ==="
for interval in "${values[@]}"; do
    echo "Running gossip with GOSSIP_EVERY=${interval}..."
    METHOD=gossip \
    GOSSIP_EVERY=${interval} \
    GOSSIP_ALPHA=0.5 \
    MAX_SECONDS=3600 \
    ./run.sh
done

echo "=== All runs completed! ==="
