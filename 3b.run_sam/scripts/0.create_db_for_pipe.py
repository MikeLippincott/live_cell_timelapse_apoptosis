#!/usr/bin/env python
# coding: utf-8

# This notebook creates a lance database for storing image/single-cell metadata for tracking single cells through time. 
# This lance db will be called in the next notebook in this analysis.

# In[1]:


import os
import pathlib

import lance
import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
from tqdm import tqdm


# In[2]:


# create the database object
uri = pathlib.Path("../../data/objects_db").resolve()
# delete the database directory if it exists
if uri.exists():
    os.system(f"rm -rf {uri}")
db = lancedb.connect(uri)


# In[3]:


# set the path to the videos
tiff_dir = pathlib.Path(
    "../../2.cellprofiler_ic_processing/illum_directory/20231017ChromaLive_6hr_4ch_MaxIP_test_small/"
).resolve(strict=True)

# set the path to the terminal data
terminal_data = pathlib.Path(
    "../../2.cellprofiler_ic_processing/illum_directory/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_test_small"
).resolve(strict=True)


# ### Get data formatted correctly

# In[4]:


# get the list of tiff files in the directory
tiff_files = list(tiff_dir.glob("*.tiff"))
tiff_files = tiff_files + list(terminal_data.glob("*.tiff"))
tiff_file_names = [file.stem for file in tiff_files]
# files to df
tiff_df = pd.DataFrame({"file_name": tiff_file_names, "file_path": tiff_files})

# split the file_path column by _ but keep the original column
tiff_df["file_name"] = tiff_df["file_name"].astype(str)
tiff_df[["Well", "FOV", "Timepoint", "Z-slice", "Channel", "illum"]] = tiff_df[
    "file_name"
].str.split("_", expand=True)
tiff_df["Well_FOV"] = tiff_df["Well"] + "_" + tiff_df["FOV"]
# drop all channels except for the first one
# this is so there is one row per cell
# specifically the first channel is the nuclei channel
# and I will be tracking the obj=ects through the nuclei channel
tiff_df = tiff_df[tiff_df["Channel"] == "C01"]
tiff_df = tiff_df.drop(columns=["Channel", "illum"])

# cast all types to string
tiff_df = tiff_df.astype(str)
# load binary data into the df of each image
tiff_df["image"] = tiff_df["file_path"].apply(lambda x: tifffile.imread(x).flatten())
tiff_df["binary_image"] = tiff_df["image"].apply(lambda x: x.tobytes())
# sort the df by the well, fov, timepoint, z-slice
tiff_df = tiff_df.sort_values(["Well", "FOV", "Timepoint", "Z-slice"])
tiff_df.reset_index(drop=True, inplace=True)
tiff_df.head(1)


# In[5]:


# create the schema for the table in the database
schema = pa.schema(
    [
        pa.field("file_name", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("Well", pa.string()),
        pa.field("FOV", pa.string()),
        pa.field("Timepoint", pa.string()),
        pa.field("Z-slice", pa.string()),
        pa.field("Well_FOV", pa.string()),
        pa.field("image", pa.list_(pa.int16())),
        # add binary data
        pa.field("binary_image", pa.binary()),
    ]
)
# create the table in the database following the schema
tbl = db.create_table(
    "0.original_images", mode="overwrite", data=tiff_df, schema=schema
)

