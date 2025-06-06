#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=merge_child-%j.out

module load miniforge
conda init bash
conda activate cellprofiler_timelapse_env


well_fov=$1

cd scripts/ || exit

python 0.merge_sc.py --well_fov $well_fov

cd ../ || exit

conda deactivate

echo "Merging sc complete."
