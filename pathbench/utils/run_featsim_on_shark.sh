#!/bin/bash
# This script is used to run the program on the shark cluster

#SBATCH -J FEAT_SIM
#SBATCH --mem=50G
#SBATCH --partition=all,highmem
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=s.brussee@lumc.nl
#SBATCH --mail-type=BEGIN,END,FAIL

#Clear environment
module purge > /dev/null 2>&1

#Load modules
module load library/cuda/11.6.1/gcc.8.3.1
module load library/openslide/3.4.1/gcc-8.3.1
module load system/python/3.9.17
module load tools/miniconda/python3.9/4.12.0

source ../../venv/bin/activate

python3 calculate_feature_similarity.py
