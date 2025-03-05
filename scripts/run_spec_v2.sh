#!/bin/bash
#SBATCH --job-name=spec_v2  # Job name
#SBATCH --time=12:00:00  # Time limit
#SBATCH --partition=cpuqueue  # Partition to use
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=32  # Number of CPUs per task
#SBATCH --mem=200G  # Memory per node
#SBATCH --output=/data/peer/rsaha/supervised-spectra/scripts/logs/stdout_%j.out 
#SBATCH --error=/data/peer/rsaha/supervised-spectra/scripts/logs/stderr_%j.err  

# Load bashrc 
source ~/.bashrc

cd /data/peer/rsaha/supervised-spectra
poetry shell


# Run 
cd /data/peer/rsaha/supervised-spectra/scripts
poetry run python spec-v2.py