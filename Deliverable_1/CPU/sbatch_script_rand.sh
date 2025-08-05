#!/bin/bash
#Script for a single random matrix sbatch: $1 (number of rows), $2 (number of columns), $3 (number of non zeros), $4 (seed for the randomization)
#SBATCH --partition=edu-short
#SBATCH --nodelist=edu01
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=outputs/test-%j.out
#SBATCH --error=errors/test-%j.err
./bin/SpMV $1 $2 $3 $4
