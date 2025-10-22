#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=fuzzy_match_parent-%j.out

module load miniforge
conda init bash
conda activate cellprofiler_timelapse_env

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb


well_fovs_dir="../4.cellprofiler_analysis/well_fov_loading/well_fov_dirs.csv"

mapfile -t well_fovs < "$well_fovs_dir"

for well_fov in "${well_fovs[@]}"; do
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done
    sbatch 2b.child_fuzzy_match.sh "$well_fov"
done

conda deactivate

echo "All merging sc jobs submitted."
