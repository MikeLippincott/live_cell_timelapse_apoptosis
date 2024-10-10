#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --output=alpine_std_out_std_err-%j.out
#SBATCH --array=1-20

# run SAM2
echo "Starting SAM2 pipe for object detection..."

# run SAM2
module load cuda
module load anaconda
module load mambaforge
conda init bash
mamba activate sam2_env

models=( "tiny" "small" "base" "large" )
downscale_factors=( 2 4 5 10 19 )

# get the number of jobs for the array
n_models=${#models[@]}
n_downscale_factors=${#downscale_factors[@]}
n_jobs=$((n_models * n_downscale_factors))

# get the model index
model_index=$((($SLURM_ARRAY_TASK_ID-1) / $n_downscale_factors))
downscale_index=$((($SLURM_ARRAY_TASK_ID-1) % $n_downscale_factors))

model=${models[$model_index]}
downscale_factor=${downscale_factors[$downscale_index]}


jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb
cd scripts/

# run the pipelines
python 0.create_db_for_pipe.py

command="python 1.run_sam2_microscopy.py"

$command --model_to_use "$model" --downscale "True" --downscale_factor "$downscale_factor"

cd ../../

mamba deactivate
