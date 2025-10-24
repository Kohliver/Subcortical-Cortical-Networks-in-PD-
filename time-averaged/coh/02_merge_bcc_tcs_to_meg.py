#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:22:16 2024

@author: okohl

SubRSN Project - STN Power and Coherence 2
    Extract best clinical STN contact and merge to MEG signal to later calculate STN-cortical coherence.
"""

import mne
import numpy as np
from osl.preprocessing.osl_wrappers import detect_badchannels
from glob import glob
import pandas as pd

# Directories
preproc_dir = '.../data/preprocessed/fif_noLC'
out_dir = '.../data/static/stn/bcc/stn_ctx/tcs_noLC'
demo_dir = '.../sub_rsn/data/demographics'
 
# Run demographic information
df = pd.read_csv( f'{demo_dir}/behavioural_all.csv')
bcc  = df[["BCC_right","BCC_left"]] # 1 = Off; 2 = On

# Get file names - sorted by condition
files = []
for condition in ['periMedOff','periMedOn' ]:
    files.extend(sorted(glob(f"{preproc_dir}/*{condition}*"))) 

# run loop concatenating files of participants and calculating psds of STN channels
r = []
l = []
for iSub, file in enumerate(files):
    
    chan_right = ['EEG001','EEG002','EEG003','EEG004','EEG005','EEG006','EEG007','EEG008']   
    chan_left = ['EEG033','EEG034','EEG035','EEG036','EEG037','EEG038','EEG039','EEG040']
    
    # Grab Best Clinical Contact Names
    bccs = bcc.iloc[iSub].to_list()
    
    # --- Load Data ---
    raw = mne.io.read_raw_fif(file).pick('eeg')
    meas_date = raw.info["meas_date"] 
    annotations = raw.annotations 
      
    # Bad Channel detection
    raw = detect_badchannels(raw, picks='eeg',significance_level=0.01)
   
    # Adjust STN channel names so that bads are excluded
    bads = raw.info['bads']
   
    # Select only good channels in on recordings
    chan_right = chan_right
    chan_left = chan_left
    if len(bads) > 0:
        for bad in bads:
            if bad not in bccs: # Make sure not to remove BCCs
                while (bad in chan_right):
                    chan_right.remove(bad)
                while (bad in chan_left):
                    chan_left.remove(bad)
       
    # Re-reference Right & Left (against the mean across all contacts) 
    stn_right = raw.copy().pick_channels(chan_right).load_data().set_eeg_reference(ref_channels="average",ch_type='eeg')   
    stn_left = raw.copy().pick_channels(chan_left).load_data().set_eeg_reference(ref_channels="average",ch_type='eeg')
    
    # pick channels as specified in sure et al. 2021 https://doi.org/10.3389/fnins.2021.724334
    included_chan_r = [bccs[0]]
    stn_right = stn_right.pick(included_chan_r)
    r.append(stn_right.ch_names)
    
    included_chan_l = [bccs[1]]
    stn_left = stn_left.pick(included_chan_l)
    l.append(stn_left.ch_names)
      
    # Merge Right and left channels
    raw_in = stn_right.copy().add_channels([stn_left],force_update_info=True)
    
    # Merg STN channel with max beta power alongside meg data
    meg = mne.io.read_raw_fif(file).pick('misc').get_data(reject_by_annotation='omit')
    stn = raw_in.get_data(reject_by_annotation='omit')
    merged = np.vstack([meg,stn]).T
 
    # Create Filename
    fname = file.split("/")[-1]  # Get File Name
    fname = " ".join(fname.split(".")[:-1])  # Remove Split information from file name
    outfile = f"{out_dir}/{fname}.npy"
    
    # Save off Condition File as .npy
    print(f"saving: {outfile}")
    np.save(outfile, merged)
    