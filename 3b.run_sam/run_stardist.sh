#!/bin/bash

# run SAM2
echo "Starting stardist pipe for object detection..."

downscale_factors=( 1 2 4 5 10 )

mamba activate sam2_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/ || exit

for downscale_factor in "${downscale_factors[@]}"; do
    echo "Running stardist pipe for downscale factor: $downscale_factor"
    python 2.run_stardist.py --downscale_factor "$downscale_factor"
done


cd ../ || exit

mamba deactivate
