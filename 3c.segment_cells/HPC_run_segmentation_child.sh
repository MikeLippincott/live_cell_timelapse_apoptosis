#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=00:60:00
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --output=segment_child-%j.out

conda activate timelapse_segmentation_env

main_dir=$1
terminal_dir=$2

cd scripts/ || exit


python 2.nuclei_segmentation.py --input_dir "$main_dir" --clip_limit 0.6 --diameter 70
python 2.nuclei_segmentation.py --input_dir "$terminal_dir" --clip_limit 0.6 --diameter 70

python 3.cell_segmentation.py --input_dir "$main_dir" --clip_limit 0.6 --diameter 100
python 3.cell_segmentation.py --input_dir "$terminal_dir" --clip_limit 0.6 --diameter 100


cd ../ || exit

conda deactivate

echo "Segmentation and tracking done."
