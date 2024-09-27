#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=sample-%j.out

module purge
module load mambaforge

# create conda environments needed for the project

# loop through all environment .yaml files in this directory
for file in $(ls -1 *.yaml); do

    # create conda environment from .yaml file
    if mamba env create -f $file; then
        echo "Environment created successfully"
    else if mamba env update -f $file; then
        echo "Environment updated successfully"
    else
        echo "Error creating or updating environment"
    fi
    fi
done
