#!/usr/bin/env python
# coding: utf-8

# This notebook verifies that the FOV mismatch correction is working as expected.
# It loads a timelapse image and a reference image, applies the FOV mismatch correction, and visualizes the results.

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tifffile

# In[2]:


raw_data_timelapse_path = pathlib.Path(
    "../../data/raw_data/20231017ChromaLive_6hr_4ch_MaxIP"
).resolve(strict=True)
raw_data_terminal_path = pathlib.Path(
    "../../data/raw_data/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP"
).resolve(strict=True)
# get the list of dirs in the raw_data_path
timelapse_fovs = [
    x for x in raw_data_timelapse_path.iterdir() if x.is_file() and x.suffix == ".tif"
]
wells = [x.name.split("_")[0] for x in timelapse_fovs]
wells.sort()
wells = np.unique(wells)


# In[3]:


for well in wells:
    timelapse_fov1 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_timelapse_path}/{well}_F0001_T0013_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    timelapse_fov2 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_timelapse_path}/{well}_F0002_T0013_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    timelapse_fov3 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_timelapse_path}/{well}_F0003_T0013_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    timelapse_fov4 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_timelapse_path}/{well}_F0004_T0013_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    terminal_fov1 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_terminal_path}/{well}_F0001_T0001_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    terminal_fov2 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_terminal_path}/{well}_F0002_T0001_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    terminal_fov3 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_terminal_path}/{well}_F0003_T0001_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    terminal_fov4 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_terminal_path}/{well}_F0004_T0001_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    terminal_fov5 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_terminal_path}/{well}_F0005_T0001_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    terminal_fov6 = tifffile.imread(
        pathlib.Path(
            f"{raw_data_terminal_path}/{well}_F0006_T0001_Z0001_C01.tif"
        ).resolve(strict=True)
    )
    blank_image = np.zeros_like(timelapse_fov1)

    # plot the timelapse image and the terminal image overlaid

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Well {well} FOV Mismatch Check", fontsize=16)
    plt.subplot(2, 6, 1)
    plt.imshow(timelapse_fov1, cmap="nipy_spectral")
    plt.title("Timelapse FOV1")
    plt.axis("off")
    plt.subplot(2, 6, 2)
    plt.imshow(timelapse_fov2, cmap="nipy_spectral")
    plt.title("Timelapse FOV2")
    plt.axis("off")
    plt.subplot(2, 6, 3)
    plt.imshow(blank_image, cmap="nipy_spectral")
    plt.title("Blank")
    plt.axis("off")
    plt.subplot(2, 6, 4)
    plt.imshow(timelapse_fov3, cmap="nipy_spectral")
    plt.title("Timelapse FOV3")
    plt.axis("off")
    plt.subplot(2, 6, 5)
    plt.imshow(timelapse_fov4, cmap="nipy_spectral")
    plt.title("Timelapse FOV4")
    plt.axis("off")
    plt.subplot(2, 6, 6)
    plt.imshow(blank_image, cmap="nipy_spectral")
    plt.title("Timelapse FOV5")
    plt.axis("off")
    plt.subplot(2, 6, 7)
    plt.imshow(terminal_fov1, cmap="nipy_spectral")
    plt.title("Terminal FOV1")
    plt.axis("off")
    plt.subplot(2, 6, 8)
    plt.imshow(terminal_fov2, cmap="nipy_spectral")
    plt.title("Terminal FOV2")
    plt.axis("off")
    plt.subplot(2, 6, 9)
    plt.imshow(terminal_fov3, cmap="nipy_spectral")
    plt.title("Terminal FOV3")
    plt.axis("off")
    plt.subplot(2, 6, 10)
    plt.imshow(terminal_fov4, cmap="nipy_spectral")
    plt.title("Terminal FOV4")
    plt.axis("off")
    plt.subplot(2, 6, 11)
    plt.imshow(terminal_fov5, cmap="nipy_spectral")
    plt.title("Terminal FOV5")
    plt.axis("off")
    plt.subplot(2, 6, 12)
    plt.imshow(terminal_fov6, cmap="nipy_spectral")
    plt.title("Terminal FOV6")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
