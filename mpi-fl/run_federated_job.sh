#!/bin/bash
#SBATCH --job-name=fed_workers
#SBATCH --output=logs/fed_workers.out
#SBATCH --error=logs/fed_workers.err

#SBATCH --cpus-per-task=1        # One CPU per MPI process
#SBATCH --time=00:05:00
#SBATCH --partition=stampede
#SBATCH --nodes=1-40             # Allow up to 40 nodes, depending on $1

make

NUM_PROCS=$(( $1 + 1 ))
K_CLUSTERS=6

echo "Running Federated K-Means with $NUM_PROCS processes and $K_CLUSTERS clusters"

mpirun -np $NUM_PROCS ./fed_kmeans $K_CLUSTERS

# echo ""
# echo "=== Calculating Metrics ==="

# python3 ./fed_cluster_assignments/metrics.py