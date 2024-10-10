#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --output=alpine_parent_std_out_std_err-%j.out

models=( "tiny" "small" "base" "large" )
downscale_factors=( 2 4 5 10 19 )
for model in "${models[@]}" ; do
    for downscale_factor in "${downscale_factors[@]}" ; do
        echo "Model: $model"
        echo "Downscale factor: $downscale_factor"

        # get the current number of jobs in the queue on aa100 partition
        num_jobs=$(squeue -p aa100 | wc -l )
        # subtract 1 to account for the header
        num_jobs=$((num_jobs-1))

        while num_jobs -lt 6; do
            sleep 60
            num_jobs=$(squeue -p aa100 | wc -l )
            num_jobs=$((num_jobs-1))
        done

        if [ "$downscale_factor" -lt 10 ]; then
            sbatch --nodes=1 --mem=100G --time=6:00:00 --partition=aa100 --gres=gpu:1 --constraint=gpu80 --output=alpine_std_out_std_err-%j.out run_sam2_HPC.sh "$model" "$downscale_factor"
            echo "Submitted job for downscale factor $downscale_factor with model $model on the 80 GPU partition."
        else
            sbatch --nodes=1 --mem=100G --time=6:00:00 --partition=aa100 --gres=gpu:1 --output=alpine_std_out_std_err-%j.out run_sam2_HPC.sh "$model" "$downscale_factor"
            echo "Submitted job for downscale factor $downscale_factor with model $model on the 40 GPU partition."
        fi
    done
done
