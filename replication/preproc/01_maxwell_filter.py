 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:22:16 2024

@author: okohl

Rassoulou Replication - Preproc 1
Maxfilter Raw Data.

Participants with to little head shape points for automated head sphere estimation:
    sub-AbzsOg
    sub-FYbcap

"""

import os
from glob import glob
import matplotlib.pyplot as plt
import pickle

from mne.io import read_raw_fif
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter
from mne.bem import fit_sphere_to_headshape

import sys
sys.path.append(".../scripts/helper/")
from params import ct_file, get_calibration_file, get_outfile_dir_Hirschmann

    
# Get Filenames 
files = sorted(glob('.../Rassoulou2024/raw_data/*/*Rest_acq*.fif'))
 
# Get filenames for saving maxfiltered data
out_root = '.../Rassoulou2024/maxfiltered'

out_files = []
for file in files:
    out_files.append(get_outfile_dir_Hirschmann(file,
                                     out_root=out_root))
         
# Loop over scans to maxfilter data
noisy_chs = []
flat_chs = []
manual_sphere_fit = []
for fname, outname in zip(files, out_files):       # 104
    
    print(f'Maxfiltering {fname}')
    
    # Load Raw file 
    raw = read_raw_fif(fname, on_split_missing="ignore").copy()
    
    # Maxfilter Preperation
    fine_cal_file = get_calibration_file(raw.info)
    crosstalk_file = ct_file
    
    # Check whether extra headshape points were collected.
    # If yes run normal maxfilter
    # If not run head_sphere estimation based on HPI and Cardinal Points
    # This is not optimal but the best option I see, if I want to apply the maxfilter.
    dig_dict = raw.info['dig']
    dig_kinds = ['FIFFV_POINT_EEG', 'FIFFV_POINT_EXTRA']
    is_fitting_points = any([p["kind"]._name in dig_kinds for p in dig_dict])
    
    # Normal Maxfiltering if Headshape Points
    if is_fitting_points:

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
        
    # Fit head sphere manually and give to maxfilter separately, if not extra headshape points were collected
    else:

        # Fit Head Sphere from cardinal and HPI points
        radius, origin_head, origin_device = fit_sphere_to_headshape(raw.info, dig_kinds=('cardinal', 'hpi'))
        
        # Bad Channel Detection
        raw.info["bads"] = []
        raw_check = raw.copy()
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check,
            cross_talk=crosstalk_file,
            calibration=fine_cal_file,
            origin=origin_head,
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
                                 origin=origin_head,
                                 verbose=True)
        
        # Make not of Participants for which head shape fitting was performed
        manual_sphere_fit.append(fname.split('/')[6])
    
    
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
outpath = f'{out_root}/'
with open(f'{outpath}/noisy_chs','wb') as fp:
    pickle.dump(noisy_chs, fp)
    
with open(f'{outpath}/flat_chs','wb') as fp:
    pickle.dump(flat_chs, fp)
    
with open(f'{outpath}/manual_sphere_fit_rTasks','wb') as fp:
    pickle.dump(manual_sphere_fit, fp)
    
    
    # Movement Comp not used because no continuous HPI recording
    # chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    # chpi_locs = mne.chpi.compute_chpi_locs(raw.info,chpi_amplitudes)
    # head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
    
    # head_pos = mne.chpi.read_head_pos(head_pos_file)
    # mne.viz.plot_head_positions(head_pos, mode="traces")
 
    
 
# Get List with Participant IDS
from os.path import basename
import pandas as pd  

name = []
for f in files:
    # Extract information from bids filename
    fname = basename(f)
    split = fname.split('_')
    sub = split[0]
    session = split[1]
    task = split[2]
    acq = split[3]
    run = split[4]
    
    name.append(f'{sub}_{session}_{task}_{acq}_{run}')
 
sub_dict = {'ID': name}
df = pd.DataFrame(sub_dict)

df.to_csv('/home/esther/Research/sub_rsn/results/preproc/SubIDs_Rassoulou.csv')

