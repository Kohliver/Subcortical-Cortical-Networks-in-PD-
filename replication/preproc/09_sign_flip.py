#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:50:36 2024

@author: okohl

Rassoulou Replicttion - Preproc 7
    Align the sign of each parcel time course across subjects
    and save the data as a vanilla numpy file.
    Corrects for sign ambigouity of beamforming and potential out canceling of
    phase relationships across participants.
"""

import os
import mne
import numpy as np
from glob import glob
from osl.source_recon.sign_flipping import (
    load_covariances,
    find_flips,
    )

def load(filename):
    """Load data without bad segments."""
    raw = mne.io.read_raw_fif(filename, verbose=False)
    raw = raw.pick("misc")
    data = raw.get_data(reject_by_annotation="omit", verbose=False)
    return data.T


#%% Sign Flipping for meg data only files that go into HMMs

# Dirs
root_dir = '.../Rassoulou2024'
indir = '.../Rassoulou2024/src_new/orthogonalised'
outdir = '.../Rassoulou2024/src_new/sign_flipped'

# Get path and filenames for PD patients in peri, HC and pre conditions
files = sorted(glob(f'{indir}/*_parc-raw.fif'))
    
# Get covariance matrices
covs = load_covariances(
    files,
    n_embeddings=15,
    standardize=True,
    loader=load,
)

# Load template covariance - use template cov from dataset on which canonical staes were trained
template_cov = np.load(".../data/hmm/norm_model/public/template_cov.npy")

# Output directory
os.makedirs("data/npy", exist_ok=True)

# Do sign flipping
for i in range(len(files)):
    print("Sign flipping", files[i])

    # Find the channels to flip
    flips, metrics = find_flips(
        covs[i],
        template_cov,
        n_embeddings=15,
        n_init=3,
        n_iter=2500,
        max_flips=20,
    )
        
    # Apply flips to the parcellated data and save
    parc_data = load(files[i])
    parc_data *= flips[np.newaxis, ...]
    
    subject = files[i].split('/')[-1].split('_tsss')[0]
    np.save(f"{outdir}/{subject}_sflip.npy", parc_data)