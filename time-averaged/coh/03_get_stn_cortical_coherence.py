#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:57:38 2024

@author: okohl

SubRSN Project - STN Power and Coherence 3
    Calculate time-averaged STN-Cortical Coherence with multitaper.

"""
print("Importing packages")
import numpy as np
from glob import glob 
from osl_dynamics.data import Data
from osl_dynamics.analysis import static


# Directories
preproc_dir = '.../data/static/stn/bcc/stn_ctx/tcs_noLC'
output_dir = '.../data/static/stn/bcc/stn_ctx'
demo_dir = '.../data/demographics'
 
# Get file names - sorted by condition
files = []
for condition in ['periMedOff','periMedOn' ]:
    files.extend(sorted(glob(f"{preproc_dir}/*{condition}*")))

# Load data
data = Data(files, n_jobs=10)

# Calculate multitaper spectra
f, psd, coh, w = static.multitaper_spectra(
    data=data.time_series(),
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    standardize=True,
    calc_coh=True,
    return_weights=True,
    n_jobs=8,
)

np.save(f"{output_dir}/f.npy", f)
np.save(f"{output_dir}/psd.npy", psd)
np.save(f"{output_dir}/coh.npy", coh)
np.save(f"{output_dir}/w.npy", w)

# Delete temporary directory
data.delete_dir()