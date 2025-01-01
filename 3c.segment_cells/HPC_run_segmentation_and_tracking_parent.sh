#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=segment_n_track_child-%j.out

# This script will work on a local machine that has enough VRAM to actually run the segmentation and tracking.
# Mine does not so we shall run this on the cluster on a NVIDIA a100 40GB VRAM GPU.

conda activate timelapse_segmentation_env

# run the segmentation and tracking
echo "Submitting GPU jobs to segment and track objects in the images..."

jupyter nbconvert --to python --output-dir=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# get the list of dirs in path
mapfile -t main_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory_test/*)
mapfile -t terminal_dirs < <(ls -d ../../2.cellprofiler_ic_processing/illum_directory_test/*)

cd ../ || exit

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

touch job_ids.txt
jobs_submitted_counter=0
# run the pipelines
for i in "${!main_dirs[@]}"; do
    main_dir="${main_dirs[$i]}"
    terminal_dir="${terminal_dirs[$i]}"
    echo "Processing main directory: $main_dir with terminal directory: $terminal_dir"
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 4 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    echo " '$job_id' '$main_dir' '$terminal_dir' "
    echo " '$job_id' 'i' '$terminal_dir' " >> job_ids.txt
    job_id=$(sbatch HPC_run_segmentation_and_tracking_child.sh "$main_dir" "$terminal_dir")
    # append the job id to the file
    job_id=$(echo $job_id | awk '{print $4}')
    let jobs_submitted_counter++
done


conda deactivate

echo "Submitted all jobs. $jobs_submitted_counter jobs submitted."
