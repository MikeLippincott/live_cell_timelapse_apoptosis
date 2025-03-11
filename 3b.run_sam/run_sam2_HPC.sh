#!/bin/bash

# run SAM2
echo "Starting SAM2 pipe for object detection..."

# run SAM2
module load cuda
module load anaconda
module load miniforge
conda init bash

mamba activate stardist_env
model=$1
downscale_factor=$2

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/ || exit

# run the pipelines
python 1.run_stardist.py

mamba deavtivate

mamba activate sam2_env
python 2.run_sam2_microscopy.py --model_to_use "$model" --downscale_factor "$downscale_factor" --downscale


cd ../ | exit

mamba deactivate
