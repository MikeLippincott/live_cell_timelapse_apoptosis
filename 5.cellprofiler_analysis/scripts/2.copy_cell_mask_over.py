#!/usr/bin/env python
# coding: utf-8

# This notebooks copies over the cell mask files from the CellProfiler output directory to the main data directory for easier access during analysis.

# In[1]:


import argparse
import pathlib
import shutil

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")
    parser.add_argument(
        "--well_fov",
        type=str,
        help="Well and field of view in the format 'E-11_F0001'",
        required=True,
    )

    args = parser.parse_args()
    well_fov = args.well_fov

else:
    well_fov = "C-02_F0004"


# In[ ]:


final_timepoint_cell_mask_path = pathlib.Path(
    f"../analysis_output/timelapse/{well_fov}/{well_fov}_T0013_Z0001_C04_illumcorrect.tiff"
).resolve(strict=True)
copied_cell_mask_path = pathlib.Path(
    f"../../2.cellprofiler_ic_processing/illum_directory/endpoint/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_{well_fov}/{well_fov}_T0014_Z0001_cell_mask.tiff"
).resolve()

# copy the cell mask to the terminal timepoint directory
shutil.copy(final_timepoint_cell_mask_path, copied_cell_mask_path)
