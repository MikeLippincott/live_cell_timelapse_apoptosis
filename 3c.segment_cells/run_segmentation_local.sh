#!/bin/bash


conda activate timelapse_segmentation_env

# run the segmentation and tracking
echo "Submitting GPU jobs to segment objects."


cd scripts/ || exit

# get the list of dirs in path
mapfile -t main_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory/timelapse/*)




# run the pipelines
for i in "${!main_dirs[@]}"; do
    cd ../notebooks || exit
    main_dir="${main_dirs[$i]}"
    echo "Processing main directory: $main_dir"
    papermill 2.nuclei_segmentation.ipynb 2.nuclei_segmentation.ipynb \
        -p input_dir "$main_dir" \
        -p diameter 70 \
        -p clip_limit 0.3
    cd ../scripts || exit
    python 3.cell_segmentation.py --input_dir "$main_dir" --clip_limit 0.3

done

jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb

conda deactivate
