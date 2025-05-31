#!/bin/bash
#SBATCH --job-name=centralised
#SBATCH --output=logs/centralised.out
#SBATCH --error=logs/centralised.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=stampede

echo "=== Job started on $(hostname) at $(date) ==="

make || { echo "Make failed"; exit 1; }

K_CLUSTERS=6
./cent_kmeans $K_CLUSTERS || { echo "K-means failed"; exit 1; }

# echo "=== Calculating Metrics ==="
# python3 ./cent_cluster_assignments/metrics.py || { echo "Metrics script failed"; exit 1; }

# echo "=== Job completed at $(date) ==="
