#!/bin/bash

# List of worker configurations for federated training
workers=(4 8 16 21)

# Submit federated jobs
for m in "${workers[@]}"
do
    sbatch run_federated_job.sh $m
done

# Submit centralised job
sbatch run_centralised_job.sh
