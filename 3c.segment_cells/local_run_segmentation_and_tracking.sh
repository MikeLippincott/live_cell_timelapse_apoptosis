#!/bin/bash

# This script will work on a local machine that has enough VRAM to actually run the segmentation and tracking.
# Mine does not so we shall run this on the cluster on a NVIDIA a100 40GB VRAM GPU.

conda activate timelapse_segmentation_env

# run the segmentation and tracking
echo "Starting segmentation and tracking..."

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# get the list of dirs in path
mapfile -t main_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory_test/*)
# mapfile -t terminal_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory_test/**  re-optimize segmentation

# remove dirs that contain "4ch" from terminal
for i in "${!terminal_dirs[@]}"; do
    if [[ ${terminal_dirs[$i]} == *"4ch"* ]]; then
        unset 'terminal_dirs[$i]'
    fi
done

# remove empty elements
terminal_dirs=("${terminal_dirs[@]}")
main_dirs=("${main_dirs[@]}")

if [ ${#main_dirs[@]} -ne ${#terminal_dirs[@]} ]; then
    echo "Error: The number of main directories and terminal directories do not match."
    exit 1
fi

# run the pipelines
for i in "${!main_dirs[@]}"; do
    main_dir="${main_dirs[$i]}"
    echo "Processing main directory: $main_dir with terminal directory: $terminal_dir"
    papermill 0.nuclei_segmentation_optimization.ipynb 0.nuclei_segmentation_optimization.ipynb
    sleep 2 # give time for cuda to release memory


done

cd ../ || exit

conda deactivate

echo "Segmentation and tracking done."
