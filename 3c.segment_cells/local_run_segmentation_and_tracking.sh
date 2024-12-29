#!/bin/bash

conda activate timelapse_segmentation_env

# run the segmentation and tracking
echo "Starting segmentation and tracking..."

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# get the list of dirs in path
mapfile -t main_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory_test/*)
mapfile -t terminal_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory_test/*/)

# remove dirs that contain "Annexin" from main
for i in "${!main_dirs[@]}"; do
    if [[ ${main_dirs[$i]} == *"Annexin"* ]]; then
        unset 'main_dirs[$i]'
    fi
done

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
    terminal_dir="${terminal_dirs[$i]}"
    echo "Processing main directory: $main_dir with terminal directory: $terminal_dir"
    python 0.stardist_segment_every_frame.py --input_dir_main "$main_dir" --input_dir_terminal "$terminal_dir"
done

cd ../ || exit

conda deactivate

echo "Segmentation and tracking done."
