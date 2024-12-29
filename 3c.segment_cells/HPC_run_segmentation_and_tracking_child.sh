#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --output=segment_n_track_child-%j.out

conda activate timelapse_segmentation_env

main_dir=$1
terminal_dir=$2

cd scripts/ || exit

python 0.stardist_segment_every_frame.py --input_dir_main "$main_dir" --input_dir_terminal "$terminal_dir"

cd ../ || exit

conda deactivate

echo "Segmentation and tracking done."
