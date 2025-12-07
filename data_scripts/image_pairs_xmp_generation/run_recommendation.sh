#!/bin/bash

# Array to store all process PIDs
declare -a PIDS

# Configuration
BASE_SOURCE_PATH="./source_data"
BASE_SAVE_PATH="./recommendations"
BASE_LOCAL_PATH="./source_data/portrait_local"
TYPE_SOURCE="portrait_local"
LOG_DIR="./log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to start a process and record its PID
start_process() {
    local start_id=$1
    local end_id=$2
    local log_file=$3
    
    python vllm_api_recommendation.py \
        --base_source_path "$BASE_SOURCE_PATH" \
        --base_save_path "$BASE_SAVE_PATH" \
        --base_local_path "$BASE_LOCAL_PATH" \
        --type_source "$TYPE_SOURCE" \
        --start_id "$start_id" \
        --end_id "$end_id" \
        > "$LOG_DIR/$log_file" 2>&1 &
    
    PIDS+=($!)  # $! is the PID of the last background process
    echo "Started process for IDs $start_id-$end_id with PID $! (log: $log_file)"
}

# Start multiple processes in parallel
# Batch 1: 0-2000 (5 processes)
start_process 0 200 "portrait1.txt"
start_process 200 400 "portrait2.txt"
start_process 400 600 "portrait3.txt"
start_process 600 800 "portrait4.txt"
start_process 800 1000 "portrait5.txt"

# Batch 2: 1000-2000 (5 processes)
start_process 1000 1200 "portrait6.txt"
start_process 1200 1400 "portrait7.txt"
start_process 1400 1600 "portrait8.txt"
start_process 1600 1800 "portrait9.txt"
start_process 1800 2000 "portrait10.txt"

# Batch 3: 2000-3000 (5 processes)
start_process 2000 2200 "portrait11.txt"
start_process 2200 2400 "portrait12.txt"
start_process 2400 2600 "portrait13.txt"
start_process 2600 2800 "portrait14.txt"
start_process 2800 3000 "portrait15.txt"

# Batch 4: 3000-3600 (3 processes)
start_process 3000 3200 "portrait16.txt"
start_process 3200 3400 "portrait17.txt"
start_process 3400 3600 "portrait18.txt"

# Final batch: 3600 to end
start_process 3600 -1 "portrait19.txt"

# Save all PIDs to file
echo "Started ${#PIDS[@]} processes with PIDs: ${PIDS[@]}" > "$LOG_DIR/pids.txt"
echo "All processes started. Check $LOG_DIR/pids.txt for process IDs."