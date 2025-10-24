#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:07:19 2024

@author: okohl

SubRsn Project - HMM Inference 1

    Prepare preprocessed date by
    time-delay embedding, 
    dimensionality reduction via PCA,
    standardisation.
    
    The prepared datasets are save as .npy files to be quickly loaded prior to HMM fitting.

"""

import numpy as np
from glob import glob
import pathlib
from osl_dynamics.data import Data
from osl_dynamics.utils import misc

def save(arr, ind, n_files, output_dir="."):
    """
    !!! Adapted from Osl-dynamics to allow for coherent naming even when running files one
    after another to avoid memory issues!!!
    Saves (prepared) data to numpy files.

    Parameters
    ----------
    output_dir : str
        Path to save data files to. Default is the current working
        directory.
    """
    # Create output directory
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Function to save a single array
    padded_number = misc.leading_zeros(ind, n_files)
    np.save(f"{output_dir}/array{padded_number}.npy", arr)

# Dirs where things are
norm_model_dir = ".../data/hmm/8_states_norm_model/public"
preproc_dir = '.../data/preprocessed/npy'
out_dir = '.../data/hmm/data_in'

# Get file names - sorted by condition
files = []
for condition in ['periMedOff','periMedOn','HC']:
    files.extend(sorted(glob(f"{preproc_dir}/*{condition}*")))

n_files = len(files)

# Prepare the data: 
#    1) Time-delay embedding and PCA dimensionality reduction
#    2) Standardise time reduced time courses

for ind, file in enumerate(files):
    data = Data(file, n_jobs=4)
    pca_components = np.load(f"{norm_model_dir}/pca_components.npy") # Load PCA components from canonical dataset
    data = data.prepare({
        "tde_pca": {"n_embeddings": 15, "pca_components": pca_components},
        "standardize": {},
    })

    save(data[0], ind, n_files, out_dir)


data.save(out_dir)
data.delete_dir()