#!/usr/bin/env python
# coding: utf-8

# This notebook runs stardist 2D.
# This is to establish a segmentation ground truth for the data.
# This methods does not track cells though however.

# ## 1. Imports

# In[1]:


# top level imports
import argparse
import gc  # garbage collector
import pathlib  # path handling
import shutil  # file handling

import matplotlib.pyplot as plt  # plotting
import numpy as np  # numerical python
import pandas as pd  # data handling
import pyarrow as pa  # pyarrow for parquet
import tqdm  # progress bar
from csbdeep.utils import Path, normalize  # dependecy for stardist
from PIL import Image  # image handling
from skimage import io  # image handling
from skimage.measure import label, regionprops  # coordinate handling
from skimage.transform import resize  # image handling
from stardist.models import StarDist2D  # stardist
from stardist.plot import render_label  # stardist

# In[ ]:


# import the arguments
parser = argparse.ArgumentParser(description="Process timelapse images.")
parser.add_argument(
    "--downscale_factor", type=int, default=1, help="Downsample factor for images"
)

# get the arguments
args = parser.parse_args()

downscale_factor = args.downscale_factor


# ## 2. Import data

# ### Download the model(s)

# In[3]:


# set the path to the videos
stardist_processing_dir = pathlib.Path(
    f"../stardist_processing_dir/{downscale_factor}x_factor/"
).resolve()

tiff_dir = pathlib.Path(
    "../../2.cellprofiler_ic_processing/illum_directory/timelapse/"
).resolve(strict=True)

stardist_processing_dir.mkdir(parents=True, exist_ok=True)
ordered_tiffs = pathlib.Path(stardist_processing_dir / "tiffs/").resolve()
converted_to_video_dir = pathlib.Path(stardist_processing_dir / "jpegs/").resolve()
if converted_to_video_dir.exists():
    shutil.rmtree(converted_to_video_dir)

ordered_tiffs.mkdir(parents=True, exist_ok=True)
converted_to_video_dir.mkdir(parents=True, exist_ok=True)


# ### Get data formatted correctly

# In[4]:


# get the list of tiff files in the directory
tiff_files = list(tiff_dir.rglob("*.tiff"))
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
tiff_df = tiff_df[tiff_df["Channel"] == "C01"]
tiff_df = tiff_df.drop(columns=["Channel", "illum"])
tiff_df["new_path"] = (
    str(ordered_tiffs)
    + "/"
    + tiff_df["Well_FOV"]
    + "/"
    + tiff_df["file_name"]
    + ".tiff"
)
# remove any file name that contain "F0005" or "F0006"
print(f"{tiff_df.shape[0]} prior to removing F0005 and F0006")
tiff_df = tiff_df[~tiff_df["file_name"].str.contains("F0005")]
tiff_df = tiff_df[~tiff_df["file_name"].str.contains("F0006")]
tiff_df.reset_index(drop=True, inplace=True)
print(f"{tiff_df.shape[0]} after removing F0005 and F0006")
tiff_df.head()


# In[5]:


# copy the files to the new directory
# from file path to new path
for index, row in tiff_df.iterrows():
    new_path = pathlib.Path(row["new_path"])
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(row["file_path"], new_path)


# In[6]:


# get the list of directories in the ordered tiffs directory
ordered_tiff_dirs = list(ordered_tiffs.glob("*"))
ordered_tiff_dir_names = [dir for dir in ordered_tiff_dirs]


# In[7]:


for dir in ordered_tiff_dir_names:
    out_dir = converted_to_video_dir / dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for tiff_file in dir.glob("*.tiff"):
        jpeg_file = pathlib.Path(f"{out_dir}/{tiff_file.stem}.jpeg")

        if not jpeg_file.exists():
            try:
                with Image.open(tiff_file) as img:
                    # Convert the image to 8-bit per channel
                    img = img.convert("L")
                    img.save(jpeg_file)
            except Exception as e:
                print(f"Failed to convert {tiff_file}: {e}")


# In[8]:


# get list of dirs in the converted to video dir
converted_dirs = list(converted_to_video_dir.glob("*"))
converted_dir_names = [dir for dir in converted_dirs]
for dir in converted_dir_names:
    dir = sorted(dir.glob("*.jpeg"))
    for i in enumerate(dir):
        # rename the files to be in order
        i[1].rename(f"{dir[0].parent}/{str(i[0] + 1).zfill(3)}.jpeg")


# ### Donwsample each frame to fit the images on the GPU - overwrite the copies JPEGs

# In[9]:


# get files in the directory
converted_dirs_list = list(converted_to_video_dir.rglob("*"))
converted_dirs_list = [f for f in converted_dirs_list if f.is_file()]
# posix path to string
files = [str(f) for f in converted_dirs_list]


# In[10]:


# need to downscale to fit the model and images on the GPU
# note that this is an arbitrary number and can be changed
# sort the files by name
# downsample the image
for f in files:
    img = io.imread(f)
    # downsample the image
    downsampled_img = img[::downscale_factor, ::downscale_factor]
    # save the downsampled image in place of the original image
    io.imsave(f, downsampled_img)


# #### Get the stardist ground truth for each frame and save it

# In[11]:


# where one image set here is a single well and fov over all timepoints
all_images_set_dict = {
    "image_set_name": [],  # e.g. well_fov
    "image_set_path": [],  # path to the directory
    "images": [],  # path to the first frame
    "number_of_objects": [],  # list of x,y coordinates
}

# get the list of directories in the ordered tiffs directory
dirs = list(converted_to_video_dir.glob("*"))
dirs = [dir for dir in dirs if dir.is_dir()]
dirs = sorted(dirs)
for dir in dirs:
    # get the files in the directory
    files = sorted(dir.glob("*.jpeg"))
    all_images_set_dict["image_set_name"].append(dir.name)
    all_images_set_dict["image_set_path"].append(str(dir))
    all_images_set_dict["images"].append(files)


# In[12]:


model = StarDist2D.from_pretrained("2D_versatile_fluo")

for i in tqdm.tqdm(range(len(all_images_set_dict["image_set_name"]))):
    for image in enumerate(all_images_set_dict["images"][i][:3]):
        img = io.imread(image[1])
        labels, _ = model.predict_instances(normalize(img))

        # convert the labels into position coordinates
        regions = regionprops(label(labels))
        coords = np.array([r.centroid for r in regions])
        # save the coordinates to a file in the image set directory
        coords_path = pathlib.Path(
            f"{str(stardist_processing_dir)}/star_dist_coords/{all_images_set_dict['image_set_name'][i]}/"
        ).resolve()
        coords_path.mkdir(parents=True, exist_ok=True)
        coords_path = (
            coords_path
            / f"{all_images_set_dict['images'][i][image[0]].stem}_coords.parquet"
        )
        coords_df = pd.DataFrame(coords, columns=["y", "x"])
        # rescale the coordinates to the original image size
        # coords_df["x"] = coords_df["x"] * downscale_factor
        # coords_df["y"] = coords_df["y"] * downscale_factor
        coords_df.to_parquet(coords_path)
        # save the mask image generated by stardist
        mask = render_label(labels, img=img)
        all_images_set_dict["number_of_objects"].append(len(coords))
        # upscale the mask using the downscale factor
        # mask = resize(
        #     mask, (mask.shape[0] * downscale_factor, mask.shape[1] * downscale_factor)
        # )
        # save the mask
        mask_path = pathlib.Path(
            f"{str(stardist_processing_dir)}/star_dist_masks/{all_images_set_dict['image_set_name'][i]}/"
        ).resolve()
        mask_path.mkdir(parents=True, exist_ok=True)
        mask_path = (
            mask_path / f"{all_images_set_dict['images'][i][image[0]].stem}.jpeg"
        )
        plt.imsave(mask_path, mask)
