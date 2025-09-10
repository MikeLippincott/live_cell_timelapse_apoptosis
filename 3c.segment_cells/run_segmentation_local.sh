#!/bin/bash


conda activate timelapse_segmentation_env

# run the segmentation and tracking
echo "Submitting GPU jobs to segment objects."


cd notebooks/ || exit

# get the list of dirs in path
mapfile -t main_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory/timelapse/*)
mapfile -t terminal_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory/endpoint/*)
# chekc the number of directories are the same
if [ ${#main_dirs[@]} -ne ${#terminal_dirs[@]} ]; then
    echo "Error: The number of main directories and terminal directories do not match."
    exit 1
fi

# run the pipelines
for i in "${!main_dirs[@]}"; do
    main_dir="${main_dirs[$i]}"
    terminal_dir="${terminal_dirs[$i]}"
    echo "Processing main directory: $main_dir"
    papermill 1.nuclei_segmentation.ipynb 1.nuclei_segmentation.ipynb \
        -p input_dir "$main_dir" \
        -p diameter 70 \
        -p clip_limit 0.3
    papermill 1.nuclei_segmentation.ipynb 1.nuclei_segmentation.ipynb \
        -p input_dir "$terminal_dir" \
        -p diameter 70 \
        -p clip_limit 0.3

done

cd ../ || exit

jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb

conda deactivate
