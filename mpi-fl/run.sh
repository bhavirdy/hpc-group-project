#!/bin/bash

# run.sh - Script to run Federated K-Means Clustering

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_file> <k_clusters> [num_processes]"
    echo "Example: $0 data.csv 3 4"
    exit 1
fi

DATA_FILE=$1
K_CLUSTERS=$2
NUM_PROCESSES=${3:-4}  # Default to 4 processes if not specified

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file '$DATA_FILE' not found!"
    exit 1
fi

# Compile the program
echo "Compiling federated k-means..."
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Running Federated K-Means with:"
echo "  Data file: $DATA_FILE"
echo "  K clusters: $K_CLUSTERS"
echo "  Processes: $NUM_PROCESSES"
echo ""

mpirun -np $NUM_PROCESSES ./federated_kmeans $DATA_FILE $K_CLUSTERS

echo ""
echo "Execution completed!"