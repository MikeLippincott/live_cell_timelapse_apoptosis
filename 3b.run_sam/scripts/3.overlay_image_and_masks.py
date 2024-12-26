#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import lancedb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import skimage
import tifffile
from PIL import Image

# In[2]:


# get the data paths
model_paths = pathlib.Path("../sam2_processing_dir/").glob("*model*")
model_paths = [f for f in model_paths if f.is_dir()][0]
model_paths


# In[3]:


tiffs = model_paths.glob("*tiffs*")
masks = model_paths.glob("*masks*")
tiffs = [f for f in tiffs if f.is_dir()]
masks = [f for f in masks if f.is_dir()]
tiff = tiffs[0]
mask = masks[0]

list_of_tiffs = list(tiff.rglob("*.tiff"))
list_of_masks = list(mask.glob("*.png"))
list_of_masks = [f for f in list_of_masks if f.is_file()]
list_of_tiffs = [f for f in list_of_tiffs if f.is_file()]
list_of_tiffs.sort()
list_of_masks.sort()

# make a df of the tiffs and masks
df_tiff = pd.DataFrame(
    {
        "tiff_filename": [
            f.stem.replace("_C01_illumcorrect", "") for f in list_of_tiffs
        ],
        "tiff_path": list_of_tiffs,
        "tiff_parent_path": [f.parent for f in list_of_tiffs],
    }
)
df_mask = pd.DataFrame(
    {
        "mask_filename": [f.stem.replace("_mask", "") for f in list_of_masks],
        "mask_path": list_of_masks,
        "mask_parent_path": [f.parent for f in list_of_masks],
    }
)
# merge the two dfs on the filename
df = df_tiff.merge(df_mask, left_on="tiff_filename", right_on="mask_filename")
print(len(df_tiff), len(df_mask), len(df))
# split the group and get the first two items of the list
df["group"] = df["tiff_filename"].str.split("_").str[:2].str.join("_")
unique_groups = df["group"].unique()
df.head()


# In[4]:


# set custom color map
colors = [(0, 0, 0), (1, 0, 0)]  # Black, Red
n_bins = 2  # Number of bins for the colormap
cmap_name = "custom_cmap"
custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    cmap_name, colors, N=n_bins
)


# In[5]:


# remove groups that contain F0005 and F0006
unique_groups = [g for g in unique_groups if "F0005" not in g and "F0006" not in g]


# In[6]:


for group in unique_groups:

    if "F0005" or "F0006" in group:
        pass
    print(group)
    loop_tmp_df = df[df["group"] == group]
    # iterate though each row of the df
    list_of_overlay = []
    overlay_path = pathlib.Path(
        str(df["mask_parent_path"].iloc[0]).replace("masks", "overlays")
    )
    overlay_path.mkdir(parents=True, exist_ok=True)
    for i, row in loop_tmp_df.iterrows():
        # load the tiff and mask
        tiff = tifffile.imread(row.tiff_path)
        mask = skimage.io.imread(row.mask_path)
        tiff = skimage.exposure.adjust_gamma(tiff, gamma=0.2)
        # plot the image
        plt.imshow(tiff, cmap="gray")
        plt.imshow(mask, alpha=0.3, cmap=custom_cmap)
        plt.axis("off")
        plt.title(row.tiff_filename)
        tmp_file_name = f"../sam2_processing_dir/{row.tiff_filename}_tmp.png"
        plt.savefig(tmp_file_name)
        plt.close()
        img = Image.open(tmp_file_name)
        list_of_overlay.append(img)
    # save the list of overlay images as a gif
    fig_path = pathlib.Path(f"{overlay_path}/{group}_overlay.gif").resolve()
    # save the frames as a gif
    list_of_overlay[0].save(
        fig_path, save_all=True, append_images=list_of_overlay[1:], duration=5, loop=0
    )

    # get all files that have tmp in the name
    tmp_files = list(pathlib.Path("../sam2_processing_dir/").glob("*tmp*.png"))
    # delete all the tmp files
    [f.unlink() for f in tmp_files]
