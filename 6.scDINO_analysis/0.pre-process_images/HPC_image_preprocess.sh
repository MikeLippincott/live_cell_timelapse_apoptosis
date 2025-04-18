#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nstasks=64
#SBATCH --partition=al40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=24:00:00
#SBATCH --output=child_featurize-%j.out

module load miniforge
conda init bash
# activate the correct env
conda activate scDINO_env

# convert notebooks into scripts
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# change to the correct directory
cd scripts || exit

# run the script
time python 0.pre-process_images.py
time python 1.calculate_mean_std_per_channel.py

# revert to the original directory
cd .. || exit

# deactivate the env
conda deactivate

# Complete
echo "Image pre-processing complete"
