#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:22:16 2024

@author: okohl

Rassoulou Replicttion - Preproc 4

Run SSP and remove manually selected projections (previous step) from all
files. Power Spectra of cleaned data are plotted.
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

from osl import preprocessing
from glob import glob

# Directories
preproc_dir = ".../Rassoulou2024/preproc"
ssp_preproc_dir = ".../Rassoulou2024/preproc_ssp"
report_dir = ".../Rassoulou2024/preproc_ssp/report/proj1"

os.makedirs(ssp_preproc_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Paths to files
preproc_files = sorted(glob(f"{preproc_dir}/*/*_preproc-raw.fif")) 

# Load csv with nproj per participant (make sure that order matches file order in preproc_files)
ecg_nprojs = np.genfromtxt('.../results/preproc/rassoulou_nproj.csv', delimiter=',', dtype=int)

# Exclude a few participants marked in ecg_nprojs as np.nan or -1 - most excluded because 
# needed to be cleaned with manual ICA because bad ECG projections
# 1 Dataset and 1 complete participant because of bad data quality
# Curcially, to loop through indices of bad data sets backwards and remove
bads =  np.where(ecg_nprojs == -1)[0]
for bad in reversed(bads):
    del preproc_files[bad]
    ecg_nprojs = np.delete(ecg_nprojs,bad)
            
#  Outfile names & Sub names
ssp_preproc_files = []
plot_files = []
for file in preproc_files:
        fname = os.path.basename(file).split('.')[0]
        subject = fname.split('_')[0]
        
        ssp_preproc_files.append(f"{ssp_preproc_dir}/{subject}/{fname}.fif")
        plot_files.append(f"{report_dir}/{subject}/{fname}")

for preproc_file, output_raw_file, plot_dir, ecg_nproj in zip(preproc_files, ssp_preproc_files, plot_files, ecg_nprojs):

    # Make output directory
    os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)
    os.makedirs(os.path.dirname(plot_dir), exist_ok=True)

    # Load preprocessed fif and ICA
    dataset = preprocessing.read_dataset(preproc_file, preload=True)
    raw = dataset["raw"]

    # Only keep MEG, ECG, EOG, EMG
    raw = raw.pick_types(meg=True, ecg=True, eog=True, emg=True, eeg=True)

    # Create a Raw object without any channels marked as bad
    raw_no_bad_channels = raw.copy()
    raw_no_bad_channels.load_bad_channels()

    #  Calculate SSP using ECG
    n_proj = ecg_nproj
    ecg_epochs = mne.preprocessing.create_ecg_epochs(
        raw_no_bad_channels, picks="all"
    ).average(picks="all")
    ecg_projs, events = mne.preprocessing.compute_proj_ecg(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=12,
    )

    # Add ECG SSPs to Raw object
    raw_ssp = raw.copy()
    raw_ssp.add_proj(ecg_projs.copy())

    # Calculate SSP using EOG
    n_proj = 1
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw_no_bad_channels, picks="all"
    ).average()
    eog_projs, events = mne.preprocessing.compute_proj_eog(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=12,
    )

    # Add EOG SSPs to Raw object
    raw_ssp.add_proj(eog_projs.copy())

    # Apply SSPs
    raw_ssp.apply_proj()

    # Plot power spectrum of cleaned data
    raw_ssp.plot_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4))
    plt.savefig(f"{plot_dir}_psd.png", bbox_inches="tight")
    plt.close()

    # if len(ecg_projs) > 0:
    #     fig = mne.viz.plot_projs_joint(ecg_projs, ecg_epochs, show=False)
    #     plt.savefig(f"{plot_dir}_proj_ecg.png", bbox_inches="tight")
    #     plt.close()

    # if len(eog_projs) > 0:
    #     fig = mne.viz.plot_projs_joint(eog_projs, eog_epochs, show=False)
    #     plt.savefig(f"{plot_dir}_proj_eog.png", bbox_inches="tight")
    #     plt.close()

    # Save cleaned data
    raw_ssp.save(output_raw_file, overwrite=True)