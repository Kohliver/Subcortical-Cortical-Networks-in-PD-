#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:22:23 2024

@author: okohl

SubRsn Project - STN-Cortical Coherence
    Calculate State-specific Power and Coherence on combined MEG and LFP data.
"""

print("Importing packages")

import pickle
import numpy as np
from glob import glob
import pandas as pd
from itertools import compress
from osl_dynamics.data import Data
from osl_dynamics.analysis import spectral, power, connectivity

# Set dirs to data
demo_dir = '.../data/demographics'
preproc_dir = '.../data/static/stn/bcc/stn_ctx/tcs_wSTN'
hmm_dir = '.../data/hmm/post_hoc/8_states_norm/inf'
output_dir = '.../data/hmm/post_hoc/8_states_norm/stn_coh/bcc'

# Load Session Data
df = pd.read_csv( f'{demo_dir}/behavioural_all.csv')
mask = (df["Session" ].values < 3) & (df['withinMed'] == 1)

# Get file names - sorted by condition
files = []
for condition in ['periMedOff','periMedOn']:
    files.extend(sorted(glob(f"{preproc_dir}/*{condition}*")))
  
# Load data
data = Data(files, n_jobs=10)
x = data.trim_time_series(n_embeddings=15, sequence_length=400)

# Load inferred state probabilities
a = pickle.load(open(f"{hmm_dir}/alp.pkl", "rb"))
a = list(compress(a,mask))

# Calculate multitaper spectra
f, psd, coh, w = spectral.multitaper_spectra(
    data=x,
    alpha=a,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    standardize=True,
    return_weights=True,
    n_jobs=8,
)
np.save(f"{output_dir}/f.npy", f)
np.save(f"{output_dir}/psd.npy", psd)
np.save(f"{output_dir}/coh.npy", coh)
np.save(f"{output_dir}/w.npy", w)

# Calculate power maps and coherence networks
p = power.variance_from_spectra(f, psd)
c = connectivity.mean_coherence_from_spectra(f, coh)
np.save(f"{output_dir}/pow_maps.npy", p)
np.save(f"{output_dir}/coh_nets.npy", c)

# Delete temporary directory
data.delete_dir()