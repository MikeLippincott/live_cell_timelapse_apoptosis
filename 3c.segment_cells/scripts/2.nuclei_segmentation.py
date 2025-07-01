#!/usr/bin/env python
# coding: utf-8

# This notebook focuses on trying to find a way to segment cells within organoids properly.
# The end goals is to segment cell and extract morphology features from cellprofiler.
# These masks must be imported into cellprofiler to extract features.

# In[ ]:


input_dir = "../../2.cellprofiler_ic_processing/illum_directory/test_data/timelapse/20231017ChromaLive_6hr_4ch_MaxIP_E-11_F0001"
clip_limit = 0.3
diameter = 70


# In[1]:


# set the gpu via OS environment variable
import os
import pathlib

import cellpose
import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import skimage
import tifffile
import torch
from cellpose import core, models
from csbdeep.utils import normalize
from PIL import Image
from stardist.plot import render_label

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

print(in_notebook)
# check if we have a GPU
use_gpu = torch.cuda.is_available()
print("GPU activated:", use_gpu)


# In[ ]:


input_dir = pathlib.Path(input_dir).resolve(strict=True)


# In[3]:


# set up memory profiler for GPU
device = torch.device("cuda:0")
free_before, total_before = torch.cuda.mem_get_info(device)
starting_level_GPU_RAM = (total_before - free_before) / 1024**2
print("Starting level of GPU RAM available (MB): ", starting_level_GPU_RAM)


# ## Set up images, paths and functions

# In[4]:


image_extensions = {".tif", ".tiff"}
files = sorted(input_dir.glob("*"))
files = [str(x) for x in files if x.suffix in image_extensions]


# In[5]:


image_dict = {
    "nuclei_file_paths": [],
    "nuclei": [],
}


# In[6]:


# split files by channel
for file in files:
    if "C01" in file.split("/")[-1]:
        image_dict["nuclei_file_paths"].append(file)
        image_dict["nuclei"].append(tifffile.imread(file).astype(np.float32))
nuclei_image_list = [np.array(nuclei) for nuclei in image_dict["nuclei"]]

nuclei = np.array(nuclei_image_list).astype(np.int16)

nuclei = skimage.exposure.equalize_adapthist(nuclei, clip_limit=clip_limit)

nuclei_image_list = [np.array(nuclei_image) for nuclei_image in nuclei]


# ## Cellpose

# Weird errors occur when running this converted notebook in the command line.
# This cell helps the python interpreter figure out where it is...somehow.
# Wrap the loop in a function to avoid the error...

# In[7]:


def perform_segmentation(nuclei_image_list):
    use_GPU = core.use_gpu()
    use_GPU = use_GPU and torch.cuda.is_available()
    print("Using GPU:", use_GPU)
    model = models.CellposeModel(gpu=use_GPU)
    masks_all_dict = {"masks": [], "imgs": []}
    for img in nuclei_image_list:
        normalized_img = normalize(img)
        mask, _, _ = model.eval(normalized_img, diameter=diameter)

        masks_all_dict["masks"].append(mask)
        masks_all_dict["imgs"].append(img)
    return masks_all_dict


masks_all_dict = perform_segmentation(nuclei_image_list)

for frame_index, frame in enumerate(image_dict["nuclei_file_paths"]):
    tifffile.imwrite(
        pathlib.Path(
            input_dir / f"{str(frame).split('/')[-1].split('_C01')[0]}_nuclei_mask.tiff"
        ),
        masks_all_dict["masks"][frame_index].astype(np.uint16),
    )


# In[8]:


if in_notebook:
    for timepoint in range(len(masks_all_dict["masks"])):
        plt.figure(figsize=(20, 10))
        plt.title(f"z: {timepoint}")
        plt.axis("off")
        plt.subplot(1, 2, 1)
        plt.imshow(nuclei[timepoint], cmap="gray")
        plt.title("Nuclei")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(render_label(masks_all_dict["masks"][timepoint]))
        plt.title("Nuclei masks")
        plt.axis("off")
        plt.show()


# In[9]:


# set up memory profiler for GPU
device = torch.device("cuda:0")
free_after, total_after = torch.cuda.mem_get_info(device)
amount_used = ((total_after - free_after)) / 1024**2
print(f"Used: {amount_used} MB or {amount_used / 1024} GB of GPU RAM")
