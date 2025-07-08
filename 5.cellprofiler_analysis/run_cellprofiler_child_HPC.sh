#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=2:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=CP_child-%j.out

module purge
module load anaconda

well_fov=$1

# This script is used to run CellProfiler analysis on the timelapse images.
conda activate cellprofiler_timelapse_env

cd scripts || exit

# run the python script
python 1.run_cellprofiler_analysis_timelapse.py --well_fov "$well_fov"
python 2.copy_cell_mask_over.py --well_fov "$well_fov"
python 3.endpoint_manual_alignment.py --well_fov "$well_fov"
python 6.run_cellprofiler_analysis_endpoint.py --well_fov "$well_fov"

# change the directory back to the original directory
cd ../ || exit

conda deactivate

# End of the script
echo "CellProfiler analysis is completed."
