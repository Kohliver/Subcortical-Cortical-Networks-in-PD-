#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:15:46 2024

@author: okohl

Sub-RSN Project - Preprocessing 3

Run OSL Preprocessing Pipeline:
    Broadband (0.5 to 125Hz) and notch filters (50 & 100Hz)
    Reasmple to 250Hz
    Bad Segment detection in Mag, Grad, and STN channels
    Bad channel detection in Mag and Grad channels
    Interpolate bad channels

Curcially, preproc is run separately for peri condition because
in peri condition we have STN recordings in EEG channels. Running the peri
condition separately allows to also identify bad segments based on bad_segments
in STN-channels. This may be helpful to get cleaner STN signals.    
"""

from glob import glob
from dask.distributed import Client
from osl import preprocessing, utils
import os

import sys
sys.path.append(".../helper/")
from params import data_root

# Input dir 
maxfilter_root = '.../data'

# Outdir
preproc_dir = f"{data_root}/preproc"  # output directory containing the preprocess files

#%% Run Preprocessing for HCs

# Get File names
inputs = []
for condition in ['HC']:
    inputs.extend(sorted(glob(f'{maxfilter_root}/maxfilter/*/*{condition}*/*_task-RestingState_*_raw_tsss.fif')))

# Settings
config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 500, picks: mag}
    - bad_segments: {segment_len: 500, picks: grad}
    - bad_segments: {segment_len: 500, picks: mag, mode: diff}
    - bad_segments: {segment_len: 500, picks: grad, mode: diff}
    - bad_channels: {picks: mag}
    - bad_channels: {picks: grad}
    #- ica_raw: {picks: meg, n_components: 40}
    #- ica_autoreject: {apply: False}
    - interpolate_bads: {}
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=12, threads_per_worker=1)

    # Main preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=preproc_dir,
        overwrite=True,
        dask_client=True,
    )
    
    
#%% Run preprocessing for peri condition separately to allow for additional bad
#   segment detection on STN channels

# Get File Names
inputs = []
for condition in ['peri']:
    inputs.extend(sorted(glob(f'{maxfilter_root}/maxfilter/*/*{condition}*/*_task-RestingState_*_raw_tsss.fif')))

# Outdir
preproc_dir = f"{data_root}/preproc"  # output directory containing the preprocess files

# Settings
config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 500, picks: mag}
    - bad_segments: {segment_len: 500, picks: grad}
    - bad_segments: {segment_len: 500, picks: eeg}
    - bad_segments: {segment_len: 500, picks: mag, mode: diff}
    - bad_segments: {segment_len: 500, picks: grad, mode: diff}
    - bad_segments: {segment_len: 500, picks: eeg, mode: diff}
    - bad_channels: {picks: mag}
    - bad_channels: {picks: grad}
    - ica_raw: {picks: meg, n_components: 40}
    - ica_autoreject: {apply: False}
    - interpolate_bads: {}
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=12, threads_per_worker=1)

    # Main preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=preproc_dir,
        overwrite=True,
        dask_client=True,
    )
    