#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=alpine_std_out_std_err-%j.out

module purge
module load anaconda

# this script runs the file and pathing pre-processing

# change directory to the scripts folder
cd scripts/

# run the pre-processing scripts
conda run -n timelapse_env python 0.fix_pathing.py
conda run -n timelapse_env python 1.generate_platemap.py

# revert back to the main directory
cd ../

echo "Pre-processing complete"
