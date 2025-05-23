#!/usr/bin/env python
# coding: utf-8

# # Aggregate feature selected profiles

# ## Import libraries

# In[1]:


import gc
import pathlib

import pandas as pd
from pycytominer import aggregate

# ## Set paths and variables

# In[2]:


# set paths
paths_dict = {
    "timelapse_profiles": {
        "input_dir": pathlib.Path(
            "../data/5.feature_select/profiles/features_selected_profile.parquet"
        ).resolve(strict=True),
        "output_file_dir": pathlib.Path(
            "../data/6.aggregated/profiles/aggregated_profile.parquet"
        ).resolve(),
    },
    "endpoint_data": {
        "input_dir": pathlib.Path(
            "../data/5.feature_select/endpoints/features_selected_profile.parquet"
        ).resolve(strict=True),
        "output_file_dir": pathlib.Path(
            "../data/6.aggregated/endpoints/aggregated_profile.parquet"
        ).resolve(),
    },
}


# ## Perform aggregation

# In[3]:


for data_set in paths_dict:
    paths_dict[data_set]["output_file_dir"].parent.mkdir(exist_ok=True, parents=True)
    # read in the annotated file
    fs_df = pd.read_parquet(paths_dict[data_set]["input_dir"])
    metadata_cols = fs_df.columns[fs_df.columns.str.contains("Metadata")].to_list()
    feature_cols = fs_df.columns[~fs_df.columns.str.contains("Metadata")].to_list()
    if data_set not in "endpoint_data":
        aggregated_df = aggregate(
            fs_df,
            features=feature_cols,
            strata=["Metadata_Well", "Metadata_Time"],
            operation="median",
        )
        aggregated_df = pd.merge(
            aggregated_df,
            fs_df[metadata_cols],
            how="left",
            on=["Metadata_Well", "Metadata_Time"],
        )
    else:
        aggregated_df = aggregate(
            fs_df,
            features=feature_cols,
            strata=["Metadata_Well"],
            operation="median",
        )
        aggregated_df = pd.merge(
            aggregated_df,
            fs_df[metadata_cols],
            how="left",
            on=["Metadata_Well"],
        )

    # rearrange the columns such that the metadata columns are first
    for col in reversed(aggregated_df.columns):
        if col.startswith("Metadata_"):
            tmp_pop = aggregated_df.pop(col)
            aggregated_df.insert(0, col, tmp_pop)
    print(aggregated_df.shape)
    aggregated_df.to_parquet(paths_dict[data_set]["output_file_dir"])
