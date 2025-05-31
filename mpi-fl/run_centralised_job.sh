#!/bin/bash
#SBATCH --job-name=centralised
#SBATCH --output=logs/centralised.out
#SBATCH --error=logs/centralised.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:05:00
#SBATCH --partition=stampede

K_CLUSTERS=2

echo "Running Centralised K-Means with $K_CLUSTERS clusters"

# Run centralised training script
./cent_kmeans $K_CLUSTERS
