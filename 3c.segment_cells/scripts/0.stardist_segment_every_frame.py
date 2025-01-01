#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import tifffile
import torch
import tqdm
from PIL import Image
from rich.pretty import pprint
from stardist.models import StarDist2D
from ultrack import to_tracks_layer, track, tracks_to_zarr
from ultrack.config import MainConfig
from ultrack.imgproc import normalize
from ultrack.utils import estimate_parameters_from_labels, labels_to_contours

# check if in a jupyter notebook

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

print(f"Running in notebook: {in_notebook}")

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# check gpu
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if not gpu_devices:
    print("No GPU found")
else:
    print("GPU found")


# tensorflow clear gpu memory
def clear_gpu_memory():
    from numba import cuda

    cuda.select_device(0)
    cuda.close()


clear_gpu_memory()


# In[2]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--input_dir_main",
        type=str,
        help="Path to the input directory containing the tiff images",
    )
    parser.add_argument(
        "--input_dir_terminal",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    input_dir_main = pathlib.Path(args.input_dir_main).resolve(strict=True)
    input_dir_terminal = pathlib.Path(args.input_dir_terminal).resolve(strict=True)
else:
    print("Running in a notebook")
    input_dir_main = pathlib.Path(
        "../../2.cellprofiler_ic_processing/illum_directory_test/20231017ChromaLive_6hr_4ch_MaxIP_C-02_F0001"
    ).resolve(strict=True)
    input_dir_terminal = pathlib.Path(
        f"../../2.cellprofiler_ic_processing/illum_directory_test/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_{str(input_dir_main).split('MaxIP_')[1]}"
    ).resolve(strict=True)

temporary_output_dir = pathlib.Path("../tmp_output").resolve()
figures_output_dir = pathlib.Path("../figures").resolve()
results_output_dir = pathlib.Path("../results").resolve()
temporary_output_dir.mkdir(exist_ok=True)
figures_output_dir.mkdir(exist_ok=True)
results_output_dir.mkdir(exist_ok=True)


# In[3]:


file_extensions = {".tif", ".tiff"}
# get all the tiff files
tiff_files = list(input_dir_main.glob("*"))
tiff_files = [f for f in tiff_files if f.suffix in file_extensions]
tiff_files = sorted(tiff_files)

tiff_files_terminal = list(input_dir_terminal.glob("*"))
tiff_files_terminal = [f for f in tiff_files_terminal if f.suffix in file_extensions]
tiff_files_terminal = sorted(tiff_files_terminal)

tiff_files = tiff_files + tiff_files_terminal
tiff_files = [f for f in tiff_files if "C01" in f.name]

print(f"Found {len(tiff_files)} tiff files in the input directory")


# In[4]:


model = StarDist2D.from_pretrained("2D_versatile_fluo")


# In[5]:


image_dims = tifffile.imread(tiff_files[0]).shape
timelapse_raw = np.zeros(
    (len(tiff_files), image_dims[0], image_dims[1]), dtype=np.uint16
)
timelapse_raw_visualize = np.zeros(
    (len(tiff_files), image_dims[0], image_dims[1]), dtype=np.uint16
)
stardist_labels = np.zeros(
    (len(tiff_files), image_dims[0], image_dims[1]), dtype=np.uint16
)


# In[6]:


# import stardist
for image_index, image_file_path in tqdm.tqdm(enumerate(tiff_files)):
    image = tifffile.imread(image_file_path)
    timelapse_raw_visualize[image_index, :, :] = image
    image = normalize(image, gamma=1.0)
    timelapse_raw[image_index, :, :] = image

    segmented_image, _ = model.predict_instances(image)
    stardist_labels[image_index, :, :] = segmented_image
# concat all the images into one array
print(stardist_labels.shape)


# In[7]:


detections = np.zeros((len(tiff_files), image_dims[0], image_dims[1]), dtype=np.uint16)
edges = np.zeros((len(tiff_files), image_dims[0], image_dims[1]), dtype=np.uint16)
for frame_index, frame in enumerate(stardist_labels):
    detections[frame_index, :, :], edges[frame_index, :, :] = labels_to_contours(frame)
print(detections.shape, edges.shape)
tifffile.imwrite(f"{temporary_output_dir}/stardist_labels.tif", stardist_labels)
tifffile.imwrite(f"{temporary_output_dir}/timelapse_raw.tif", timelapse_raw)
tifffile.imwrite(f"{temporary_output_dir}/detections.tif", detections)
tifffile.imwrite(f"{temporary_output_dir}/edges.tif", edges)

clear_gpu_memory()


# In[8]:


params_df = estimate_parameters_from_labels(stardist_labels, is_timelapse=True)
if in_notebook:
    params_df["area"].plot(kind="hist", bins=100, title="Area histogram")


# In[9]:


config = MainConfig()
config.segmentation_config.min_area = 30
config.segmentation_config.max_area = 1250
config.segmentation_config.n_workers = 8
config.linking_config.max_distance = 45
config.linking_config.n_workers = 8

config.tracking_config.appear_weight = -1
config.tracking_config.disappear_weight = -1
config.tracking_config.division_weight = -0.1
config.tracking_config.power = 4
config.tracking_config.bias = -0.001
config.tracking_config.solution_gap = 0.0
config.tracking_config.solver_name = "CBC"
pprint(config.dict())
# write the config to a file for reference later
with open(f"{results_output_dir}/config.json", "w") as f:
    f.write(config.json())


# In[10]:


track(
    foreground=detections,
    edges=edges,
    config=config,
    overwrite=True,
)


# In[11]:


tracks_df, graph = to_tracks_layer(config)
labels = tracks_to_zarr(config, tracks_df)
tracks_df.to_parquet(
    f"{results_output_dir}/{str(input_dir_main).split('MaxIP_')[1]}_tracks.parquet"
)
tracks_df.head()


# In[12]:


# save the tracks as parquet
tracks_df.reset_index(drop=True, inplace=True)
tracks = np.zeros((len(tiff_files), image_dims[0], image_dims[1]), dtype=np.uint16)
cum_tracks_df = tracks_df.copy()
timepoints = tracks_df["t"].unique()

# zero out the track_df
cum_tracks_df = cum_tracks_df.loc[cum_tracks_df["t"] == -1]


# In[13]:


if in_notebook:
    for frame_index, _ in enumerate(timelapse_raw):
        tmp_df = tracks_df.loc[tracks_df["t"] == frame_index]
        cum_tracks_df = pd.concat([cum_tracks_df, tmp_df])
        plt.figure(figsize=(6, 5))
        plt.subplot(2, 3, 1)
        # rescale tbe intensity of the raw image
        raw_image = timelapse_raw_visualize[frame_index, :, :]
        raw_image = raw_image * 4096
        plt.imshow(raw_image, cmap="gray")
        plt.title("Raw")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(stardist_labels[frame_index, :, :], cmap="gray")
        plt.title("Masks")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        sns.lineplot(data=cum_tracks_df, x="x", y="y", hue="track_id", legend=False)
        plt.imshow(labels[frame_index, :, :], cmap="gray", alpha=0.5)
        plt.title(f"Frame {frame_index}")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        edge_image = skimage.exposure.adjust_gamma(
            edges[frame_index, :, :], gamma=0.0001
        )
        # make the outline brighter
        edge_image = edge_image * 1000
        plt.imshow(edge_image, cmap="gray")
        plt.title("Edges")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(detections[frame_index, :, :], cmap="gray")
        plt.title("Detections")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        sns.lineplot(data=cum_tracks_df, x="x", y="y", hue="track_id", legend=False)
        plt.imshow(detections[frame_index, :, :], cmap="gray", alpha=0.5)
        plt.title(f"Frame {frame_index}")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{temporary_output_dir}/tracks_{frame_index}.png")
    if in_notebook:
        plt.show()


# In[14]:


# load each image
files = [f for f in temporary_output_dir.glob("*.png")]
files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))
frames = [Image.open(f) for f in files]
fig_path = figures_output_dir / f"{str(input_dir_main).split('MaxIP_')[1]}_tracks.gif"
# plot the line of each track in matplotlib over a gif
# get the tracks
# save the frames as a gif
frames[0].save(fig_path, save_all=True, append_images=frames[1:], duration=3, loop=0)


# In[15]:


# clean up tracking files
# remvoe temporary_output_dir
shutil.rmtree(temporary_output_dir)

track_db_path = pathlib.Path("data.db").resolve()
metadata_toml_path = pathlib.Path("metadata.toml").resolve()
if track_db_path.exists():
    track_db_path.unlink()
if metadata_toml_path.exists():
    metadata_toml_path.unlink()


# In[16]:


clear_gpu_memory()

