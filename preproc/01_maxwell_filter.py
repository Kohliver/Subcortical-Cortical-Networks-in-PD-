#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:22:16 2024

@author: okohl

Sub-RSN Project - Preprocessing 1

    Maxfilter Raw Data
    Movement Compensation was not used because no continuuos HPIs recorded.
    
    Code is modefied version from here:
    https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html
"""

import os
from glob import glob
import matplotlib.pyplot as plt
import pickle

from mne.io import read_raw_fif
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter

import sys
sys.path.append("/home/esther/Research/sub_rsn/scripts/helper/")
from params import ct_file, get_calibration_file, get_outfile_dir, data_root

# Get Names for loading and saving data
files = []
for condition in ['peri','HC']:
    files.extend(sorted(glob(f'.../BIDS/BIDS_Output_3/*/*{condition}*/meg/*_task-RestingState_run*_meg.fif')))

out_root = '.../sub_rsn/'

# Get filenames for saving maxfiltered data
out_files = []
for file in files:
    out_files.append(get_outfile_dir(file,
                                     out_root=out_root))
         
# Loop over scans to maxfilter data
noisy_chs = []
flat_chs = []
for fname, outname in zip(files, out_files):       # 104
    
    print(f'Maxfiltering {fname}')
    
    # Load Raw file 
    raw = read_raw_fif(fname, on_split_missing="ignore").copy()
    
    # Maxfilter Preperation
    fine_cal_file = get_calibration_file(raw.info)
    crosstalk_file = ct_file
    
    # Bad Channel Detection
    raw.info["bads"] = []
    raw_check = raw.copy()
    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw_check,
        cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        return_scores=True,
        verbose=True,
    )
    
    noisy_chs.append(len(auto_noisy_chs) ) 
    flat_chs.append(len(auto_flat_chs) ) 
    
    # Update List of Bad Channels
    bads = raw.info["bads"] + auto_noisy_chs + auto_flat_chs
    raw.info["bads"] = bads
    
    # Run Maxfilter
    raw_tsss = maxwell_filter(raw,
                             st_duration=10, # Maxfilter default
                             cross_talk=crosstalk_file, 
                             calibration=fine_cal_file,
                             verbose=True)
    
    # Create Folder Structure
    path = os.path.dirname(outname)
    if not os.path.isdir(path):
        os.makedirs(path)
    
    # Save Maxfiltere raw file
    raw_tsss.save(outname, overwrite=True)
    
    # Plot comparing pre vs post maxfiltering
    fig, axes = plt.subplots(2, 2, sharey=True, layout="constrained", figsize=(10, 6))
    raw.compute_psd(fmax=45,n_fft=4800).plot(
        average=False, amplitude=False, picks="meg", 
        exclude="bads",axes=axes[0])
    raw_tsss.compute_psd(fmax=45,n_fft=4800).plot(
        average=False, amplitude=False, picks="meg", 
        exclude="bads",axes=axes[1])
    
    # Save Plot
    outdir = outname.split('.')[0]
    plt.savefig(f'{outdir}_tsss_check.png')
    plt.close()


# Save results from automatic bad channel detection
outpath = f'{data_root}/maxfilter'
with open(f'{outpath}/noisy_chs','wb') as fp:
    pickle.dump(noisy_chs, fp)
    
with open(f'{outpath}/flat_chs','wb') as fp:
    pickle.dump(flat_chs, fp)
    
    
# Movement Comp not used because no continuous HPI recording
# chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
# chpi_locs = mne.chpi.compute_chpi_locs(raw.info,chpi_amplitudes)
# head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)

# head_pos = mne.chpi.read_head_pos(head_pos_file)
# mne.viz.plot_head_positions(head_pos, mode="traces")
