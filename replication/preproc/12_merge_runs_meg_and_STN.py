#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:19:42 2025

@author: okohl

Rassoulou Replicttion - Preproc 10
    Bring MEG x STN .npy files into fif format and merge all blocks of respective
    participant. Data is the input for simultaneous MEG x LFP analyses.
"""

from glob import glob
import numpy as np
import mne

# ------------------------------------------------------------------------------

def convert2mne_raw(parc_data, raw, parcel_names=None, extra_chans="stim"):
    """Create and returns an MNE raw object that contains parcellated data.

    Parameters
    ----------
    parc_data : np.ndarray
        (nparcels x ntpts) parcel data.
    raw : mne.Raw
        mne.io.raw object that produced parc_data via source recon and parcellation. Info such as timings and bad segments will be copied from this to parc_raw.
    parcel_names : list of str
        List of strings indicating names of parcels. If None then names are set to be parcel_0,...,parcel_{n_parcels-1}.
    extra_chans : str or list of str
        Extra channels, e.g. 'stim' or 'emg', to include in the parc_raw object. Defaults to 'stim'. stim channels are always added to parc_raw if they are present in raw.

    Returns
    -------
    parc_raw : mne.Raw
        Generated parcellation in mne.Raw format.
    """
    # What extra channels should we add to the parc_raw object?
    if isinstance(extra_chans, str):
        extra_chans = [extra_chans]
    extra_chans = np.unique(["stim"] + extra_chans)

    # Create Info object
    info = raw.info
    if parcel_names is None:
        parcel_names = [f"parcel_{i}" for i in range(parc_data.shape[0])]
    parc_info = mne.create_info(ch_names=parcel_names, ch_types="misc", sfreq=info["sfreq"])

    # Create Raw object
    parc_raw = mne.io.RawArray(parc_data, parc_info)

    # Copy timing info
    parc_raw.set_meas_date(raw.info["meas_date"])
    parc_raw.__dict__["_first_samps"] = raw.__dict__["_first_samps"]
    parc_raw.__dict__["_last_samps"] = raw.__dict__["_last_samps"]
    parc_raw.__dict__["_cropped_samp"] = raw.__dict__["_cropped_samp"]

    # Copy annotations from raw
    parc_raw.set_annotations(raw._annotations)

    # Add extra channels
    for extra_chan in extra_chans:
        if extra_chan in raw:
            chan_raw = raw.copy().pick(extra_chan)
            chan_data = chan_raw.get_data()
            chan_info = mne.create_info(chan_raw.ch_names, raw.info["sfreq"], [extra_chan] * chan_data.shape[0])
            chan_raw = mne.io.RawArray(chan_data, chan_info)
            parc_raw.add_channels([chan_raw], force_update_info=True)

    # Copy the description from the sensor-level Raw object
    parc_raw.info["description"] = raw.info["description"]

    return parc_raw

# ------------------------------------------------------------------------------

# Set dirs
src_dir = ".../Rassoulou2024/src_new/sign_flipped"
out_dir = ".../Rassoulou2024/src_new/npy"
stn_dir = ".../Rassoulou2024/src_new/stn"

# Select Subjects to include in all analyses
subs = ['0cGdk9','2IhVOz','2IU8mi','8RgPiG','AB2PeX',
        'AbzsOg','BYJoWR','dCsWjQ','FIyfdR','FYbcap',
        'gNX5yb','hnetKS','i4oK0F','iDpl28','jyC0j3',
        'oLNpHd','PuPVlx','VopvKx',] # Exclude IDPL28 because only 1 good Med Off resting State scan

glasser52 = '.../osl/osl/source_recon/parcellation/files/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz'

#%% Get also merged .fif files including 

# Make sure to only include subs and run for main analysis
excluded_files = [f'{src_dir}/sub-jyC0j3_ses-PeriOp_task-HRest_acq-MedOff_run-1_sflip.npy',]
excluded_stn_files = [f'{stn_dir}/sub-jyC0j3_ses-PeriOp_task-HRest_acq-MedOff_run-1_tsss_parc-raw.fif',
                      f'{stn_dir}/sub-8RgPiG_ses-PeriOp_task-Rest_acq-MedOff_run-1_split-01_tsss-1_parc-raw.fif']

# Here the excluding of data sets is happening
for iSub, sub in enumerate(subs):
    
    # --- Select Files of interest ---
    
    # Get Path and Filenames for on and off conditio of participant
    on_files = sorted(glob(f"{src_dir}/sub-{sub}*MedOn*_sflip.npy"))
    off_files = sorted(glob(f"{src_dir}/sub-{sub}*MedOff*_sflip.npy"))
    
    stn_on_files = sorted(glob(f"{stn_dir}/sub-{sub}*MedOn*_parc-raw.fif"))
    stn_off_files = sorted(glob(f"{stn_dir}/sub-{sub}*MedOff*_parc-raw.fif"))
    
    # Remove files that should be excluded
    for on_file, off_file, stn_on_file, stn_off_file in zip(on_files, off_files, stn_on_files, stn_off_files):
        
        # remove off file if part of excluded Files
        if on_file in excluded_files:
            on_files.remove(on_file)
            
        if stn_on_file in excluded_stn_files:
            stn_on_files.remove(stn_on_file)
         
        # remove off file if part of excluded Files
        if off_file in excluded_files:
            off_files.remove(off_file)
            
        if stn_off_file in excluded_stn_files:
            stn_off_files.remove(stn_off_file)
    
    # --- Load Data, convert to LFP-MEG fif and save ---
    
    if sub == 'iDpl28': # Only off resting state scan available
           
         # Load and Merge On Files
         off_raws = [] 
         for off_file, stn_off_file in zip(off_files, stn_off_files):   
             
             # Load Fif file - w STN information
             tmp_raw = mne.io.read_raw_fif(stn_off_file).pick(['misc','eeg']) # Load files
             
             # Load Npy file - w sign-flipped parcel tcs
             tmp_npy = np.load(off_file).T
             
             # creat new fif with STN channels and cortical data from npy file
             off_raws.append(convert2mne_raw(tmp_npy, tmp_raw, parcel_names=None, extra_chans=["stim","eeg"]))
             
         # merge raw files off condition
         off_raw = mne.concatenate_raws(off_raws, on_mismatch='ignore') # concatenat
         
         # Get Filename for saving
         fname = off_files[0].split("/")[-1]  # Get File Name
         fname = fname.split('_')[0]  # Remove Split information from file name
         outfile = f"{out_dir}/{fname}_rest_MedOff.fif"
     
         # Save off Condition File as .npy
         print(f"saving: {outfile}")
         off_raw.save(outfile, overwrite=True)
        
         
    elif sub == 'sub-6m9kB5':
        print('sub-6m9kB5 excluded because of bad data quality.')
    
    else:
    
        # Load On Files
        on_raws = [] 
        for on_file, stn_on_file in zip(on_files, stn_on_files):   
            
            # Load Fif file - w STN information
            tmp_raw = mne.io.read_raw_fif(stn_on_file).pick(['misc','eeg'])   # Load files
            
            # Load Npy file - w sign-flipped parcel tcs
            tmp_npy = np.load(on_file).T
            
            # creat new fif with STN channels and cortical data from npy file
            on_raws.append(convert2mne_raw(tmp_npy, tmp_raw, parcel_names=None, extra_chans=["stim","eeg"]))
            
            
        # Load Off Files
        off_raws = [] 
        for off_file, stn_off_file in zip(off_files, stn_off_files):   
            
            # Load Fif file - w STN information
            tmp_raw = mne.io.read_raw_fif(stn_off_file).pick(['misc','eeg']) # Load files
            
            # Load Npy file - w sign-flipped parcel tcs
            tmp_npy = np.load(off_file).T
            
            # creat new fif with STN channels and cortical data from npy file
            off_raws.append(convert2mne_raw(tmp_npy, tmp_raw, parcel_names=None, extra_chans=["stim","eeg"]))
            
    
        # merge raw files off on & on condition
        on_raw = mne.concatenate_raws(on_raws, on_mismatch='ignore') # concatenat
        off_raw = mne.concatenate_raws(off_raws, on_mismatch='ignore') # concatenat
    
        # Get Filename for saving
        fname = on_files[0].split("/")[-1]  # Get File Name
        fname = fname.split('_')[0]  # Remove Split information from file name
        outfile = f"{out_dir}/{fname}_rest_MedOn.fif"
    
        # Save on Condition File as .fif
        print(f"saving: {outfile}")
        on_raw.save(outfile, overwrite=True)
        
        # Get Filename for saving
        fname = off_files[0].split("/")[-1]  # Get File Name
        fname = fname.split('_')[0]  # Remove Split information from file name
        outfile = f"{out_dir}/{fname}_rest_MedOff.fif"
    
        # Save off Condition File as .npy
        print(f"saving: {outfile}")
        off_raw.save(outfile, overwrite=True)
        