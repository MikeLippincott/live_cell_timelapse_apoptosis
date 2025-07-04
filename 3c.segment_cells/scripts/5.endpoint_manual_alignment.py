#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

sys.path.append("../alignment_utils")
from alignment_utils import align_cross_correlation, apply_alignment

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
        "--final_timepoint_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )
    parser.add_argument(
        "--terminal_timepoint_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )
    args = parser.parse_args()
    final_timepoint_dir = pathlib.Path(args.final_timepoint_dir).resolve(strict=True)
    terminal_timepoint_dir = pathlib.Path(args.terminal_timepoint_dir).resolve(
        strict=True
    )


else:
    final_timepoint_dir = pathlib.Path(
        "../../2.cellprofiler_ic_processing/illum_directory/timelapse/20231017ChromaLive_6hr_4ch_MaxIP_C-06_F0001"
    ).resolve(strict=True)
    terminal_timepoint_dir = pathlib.Path(
        "../../2.cellprofiler_ic_processing/illum_directory/endpoint/20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP_C-06_F0001"
    ).resolve(strict=True)

well_fov = final_timepoint_dir.name
well_fov = well_fov.split("_")[4] + "_" + well_fov.split("_")[5]
print(f"Processing well_fov: {well_fov}")
offset_file_path = pathlib.Path(
    f"../results/{final_timepoint_dir.stem.split('MaxIP_')[1]}_offsets.parquet"
).resolve()
offset_file_path.parent.mkdir(parents=True, exist_ok=True)


# In[ ]:


final_timepoint_dna_path = pathlib.Path(
    final_timepoint_dir / f"{well_fov}_T0013_Z0001_C01_illumcorrect.tiff"
).resolve(strict=True)


terminal_timepoint_dna_path = pathlib.Path(
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_C01_illumcorrect.tiff"
).resolve(strict=True)

terminal_timepoint_annexin_path = pathlib.Path(
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_C05_illumcorrect.tiff"
).resolve(strict=True)

terminal_timepoint_nuclei_mask_path = pathlib.Path(
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_nuclei_mask.tiff"
).resolve(strict=True)

terminal_timepoint_cell_mask_path = pathlib.Path(
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_cell_mask.tiff"
).resolve(strict=True)

final_timepoint_dna = tifffile.imread(str(final_timepoint_dna_path))
terminal_timepoint_dna = tifffile.imread(str(terminal_timepoint_dna_path))
terminal_timepoint_annexin = tifffile.imread(str(terminal_timepoint_annexin_path))
terminal_timepoint_nuclei_mask = tifffile.imread(
    str(terminal_timepoint_nuclei_mask_path)
)
terminal_timepoint_cell_mask = tifffile.imread(str(terminal_timepoint_cell_mask_path))


# Align the dna stained images bewteen the two timepoints and apply the same transformation to all terminal images

# In[4]:


# get offsets
offsets = align_cross_correlation(
    pixels1=final_timepoint_dna,
    pixels2=terminal_timepoint_dna,
)
print("Offsets: ", offsets)


# In[5]:


# apply the offsets
aligned_terminal_timepoint_dna = apply_alignment(
    input_image=terminal_timepoint_dna,
    off_x=offsets[0],
    off_y=offsets[1],
    shape=terminal_timepoint_dna.shape,
)
aligned_terminal_timepoint_annexin = apply_alignment(
    input_image=terminal_timepoint_annexin,
    off_x=offsets[0],
    off_y=offsets[1],
    shape=terminal_timepoint_annexin.shape,
)
aligned_terminal_timepoint_nuclei_mask = apply_alignment(
    input_image=terminal_timepoint_nuclei_mask,
    off_x=offsets[0],
    off_y=offsets[1],
    shape=terminal_timepoint_nuclei_mask.shape,
)
aligned_terminal_timepoint_cell_mask = apply_alignment(
    input_image=terminal_timepoint_cell_mask,
    off_x=offsets[0],
    off_y=offsets[1],
    shape=terminal_timepoint_cell_mask.shape,
)


# In[6]:


# add offsets to the dataframe
offsets_df = pd.DataFrame(
    {
        "well_fov": [well_fov],
        "x_offset": [offsets[0]],
        "y_offset": [offsets[1]],
    }
)
offsets_df.to_parquet(
    offset_file_path,
    index=False,
)


# In[ ]:


# save the aligned images
aligned_terminal_timepoint_dna_path = pathlib.Path(
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_C01_illumcorrect_aligned.tiff"
).resolve()
aligned_terminal_timepoint_annexin_path = pathlib.Path(
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_C05_illumcorrect_aligned.tiff"
).resolve()
aligned_terminal_timepoint_nuclei_mask_path = pathlib.Path(
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_nuclei_mask_aligned.tiff"
).resolve()
aligned_terminal_timepoint_cell_mask_path = pathlib.Path(
    terminal_timepoint_dir / f"{well_fov}_T0014_Z0001_cell_mask_aligned.tiff"
).resolve()


tifffile.imwrite(
    str(aligned_terminal_timepoint_dna_path), aligned_terminal_timepoint_dna
)
tifffile.imwrite(
    str(aligned_terminal_timepoint_annexin_path), aligned_terminal_timepoint_annexin
)
tifffile.imwrite(
    str(aligned_terminal_timepoint_nuclei_mask_path),
    aligned_terminal_timepoint_nuclei_mask,
)
tifffile.imwrite(
    str(aligned_terminal_timepoint_cell_mask_path), aligned_terminal_timepoint_cell_mask
)


# In[8]:


if in_notebook:
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 5, 1)
    plt.imshow(final_timepoint_dna, cmap="gray", vmin=0, vmax=255)
    plt.title("Final timepoint\nDNA")
    plt.axis("off")
    plt.subplot(2, 5, 2)
    plt.imshow(terminal_timepoint_dna, cmap="gray", vmin=0, vmax=255)
    plt.title("Terminal timepoint\nDNA")
    plt.axis("off")
    plt.subplot(2, 5, 3)
    plt.imshow(terminal_timepoint_annexin, cmap="gray", vmin=0, vmax=255)
    plt.title("Terminal timepoint\nAnnexin")
    plt.axis("off")
    plt.subplot(2, 5, 4)
    plt.imshow(terminal_timepoint_nuclei_mask, cmap="gray", vmin=0, vmax=255)
    plt.title("Terminal timepoint\nnuclei mask")
    plt.axis("off")
    plt.subplot(2, 5, 5)
    plt.imshow(terminal_timepoint_cell_mask, cmap="gray", vmin=0, vmax=255)
    plt.title("Terminal timepoint\ncell mask")
    plt.axis("off")
    plt.subplot(2, 5, 6)
    plt.imshow(np.zeros_like(final_timepoint_dna), cmap="gray", vmin=0, vmax=255)
    plt.title("Blank")
    plt.axis("off")
    plt.subplot(2, 5, 7)
    plt.imshow(aligned_terminal_timepoint_dna, cmap="gray", vmin=0, vmax=255)
    plt.title("Aligned terminal\ntimepoint DNA")
    plt.axis("off")
    plt.subplot(2, 5, 8)
    plt.imshow(aligned_terminal_timepoint_annexin, cmap="gray", vmin=0, vmax=255)
    plt.title("Aligned terminal\ntimepoint Annexin")
    plt.axis("off")
    plt.subplot(2, 5, 9)
    plt.imshow(aligned_terminal_timepoint_nuclei_mask, cmap="gray", vmin=0, vmax=255)
    plt.title("Aligned terminal\ntimepoint nuclei mask")
    plt.axis("off")
    plt.subplot(2, 5, 10)
    plt.imshow(aligned_terminal_timepoint_cell_mask, cmap="gray", vmin=0, vmax=255)
    plt.title("Aligned terminal\ntimepoint cell mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
