#!/bin/bash
#SBATCH --job-name=fed_workers_$1
#SBATCH --output=logs/fed_workers_$1.out
#SBATCH --error=logs/fed_workers_$1.err
#SBATCH --ntasks=$1              # Total number of MPI processes
#SBATCH --cpus-per-task=1        # One CPU per MPI process
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --partition=stampede
#SBATCH --nodes=1-40             # Allow up to 40 nodes, depending on $1

NUM_PROCS=$1
K_CLUSTERS=6

echo "Running Federated K-Means with $NUM_PROCS processes and $K_CLUSTERS clusters"

mpirun -np $NUM_PROCS ./fed_kmeans $K_CLUSTERS

echo ""
echo "=== Calculating Metrics ==="

python3 ./fed_cluster_assignments/metrics.py