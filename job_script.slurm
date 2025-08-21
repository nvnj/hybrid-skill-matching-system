#!/bin/bash  
#----------------------------------------------------  
# Sample Slurm job script for Arc nodes  
#SBATCH -J courseJobExtractionld # Job Name  
#SBATCH -o courseJobExtractionOutputllamaDeepseek.txt
#SBATCH -p compute1 # Queue (partition) name  
#SBATCH -N 1 # Total # of nodes (must be 1 for serial)  
#SBATCH -n 1 # Total # of mpi tasks (should be 1 for serial)  
#SBATCH -t 71:30:00 # Run time (hh:mm:ss)  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=naveen.john@utsa.edu
module load anaconda3
conda activate medrag1

# Run with LLaMA
echo "Starting extraction with LLaMA..."
python3 extraction.py --both --provider llama --save-performance llama_performance.json

# Run with DeepSeek
echo "Starting extraction with DeepSeek..."
python3 extraction.py --both --provider deepseek --save-performance deepseek_performance.json
