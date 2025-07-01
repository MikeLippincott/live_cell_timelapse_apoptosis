#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --partition=amilan
#SBATCH --output=cell_tracking-%j.out

module load miniforge
conda init bash
conda activate cell_tracking_env


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

well_fov=$1
python 0.nuclei_tracking.py --well_fov "$well_fov"

cd ../ || exit

conda deactivate

echo "Cell tracking script completed"
