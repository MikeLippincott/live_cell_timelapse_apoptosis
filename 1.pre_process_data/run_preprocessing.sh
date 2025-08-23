#!/bin/bash

# this script runs the file and pathing pre-processing

# change directory to the scripts folder
cd scripts/ || return 1

# run the pre-processing scripts
conda run -n timelapse_env python 0.fix_pathing.py
conda run -n timelapse_env python 1.generate_platemap.py
conda run -n timelapse_env python 2.sort_files_properly.py

# revert back to the main directory
cd ../ || return 1

echo "Pre-processing complete"
