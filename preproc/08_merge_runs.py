#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:54:51 2024

@author: okohl

Sub-RSN Project - Preprocessing 7

Prepare data for subsequent analyses.
For each participant data of different runs is loaded, merged and saved as .fif
and .npy file.

.npy files contain only meg data and are carried forward to HMM analysis.
.fif files contain meg and stn data for STN x MEG coherence analysis.

Procedure:
1) Load all blocks of a participant and merge them.
2) Bring into npy array format with correct orientation.
3) Save 1 file per participant and condition in preprocessed folder.

"""
from glob import glob
import numpy as np 

# Set dirs
src_dir = ".../data/src"
 
#%% Peri Condition: Export .npy files with MEG data only for HMMs

subs = ['S001','S002','S003','S004','S008','S009','S010','S011','S014',
        'S016','S020','S021','S022','S025','S027','S028','S029',
        'S032','S033','S034','S036','S037','S038','S043','S048']

# Dir where data is saved
out_dir = ".../data/preprocessed/npy"

# For subs with both conditions
for iSub, sub in enumerate(subs):

    # Get Path and Filenames for on and off conditio of participant
    on_files = sorted(glob(f"{src_dir}/*/sub-{sub}*periMedOn*_sflip.npy"))
    off_files = sorted(glob(f"{src_dir}/*/sub-{sub}*periMedOff*_sflip.npy"))
    
    # Remove recording blocks with poor data quality
    if sub == 'S022':
        # remove Med On Run 2
        del on_files[1]
    elif sub == 'S043':
        # remove Med Off run 2 and 3; remove Med On run 1 and 3
        del off_files[1:3]
        del on_files[2]
        del on_files[0]

    # Load and merge raw files off on & on condition
    on_raw = np.vstack([np.load(on) for on in on_files])  # Load files & merge
    off_raw = np.vstack([np.load(off) for off in off_files])  # Load files & merge

    # Get Filename for saving
    fname = on_files[0].split("/")[-2]  # Get File Name
    fname = "_".join(fname.split("_")[:-1])  # Remove Split information from file name
    outfile = f"{out_dir}/{fname}.npy"

    # Save on Condition File as .npy
    print(f"saving: {outfile}")
    np.save(outfile, on_raw)

    # Get Filename for saving
    fname = off_files[0].split("/")[-2]  # Get File Name
    fname = "_".join(fname.split("_")[:-1])  # Remove Split information from file name
    outfile = f"{out_dir}/{fname}.npy"

    # Save off Condition File as .npy
    print(f"saving: {outfile}")
    np.save(outfile, off_raw)
   
# for subs with only off condition
subs_off = ['S012']

for iSub, sub in enumerate(subs_off):

    # Get Path and Filenames for on and off conditio of participant
    off_files = sorted(glob(f"{src_dir}/*/sub-{sub}*periMedOff*_sflip.npy"))
        
    # Load and merge raw files off on & on condition
    off_raw = np.vstack([np.load(off) for off in off_files])  # Load files & merge

    # Get Filename for saving
    fname = off_files[0].split("/")[-2]  # Get File Name
    fname = "_".join(fname.split("_")[:-1])  # Remove Split information from file name
    outfile = f"{out_dir}/{fname}.npy"

    # Save off Condition File as .npy
    print(f"saving: {outfile}")
    np.save(outfile, off_raw)
    
# For subs with only on condition
subs_on = ['S013']

for iSub, sub in enumerate(subs_on):

    # Get Path and Filenames for on and off conditio of participant
    on_files = sorted(glob(f"{src_dir}/*/sub-{sub}*periMedOn*_sflip.npy"))
    
    # Load and merge raw files off on & on condition
    on_raw = np.vstack([np.load(on) for on in on_files])  # Load files & merge

    # Get Filename for saving
    fname = on_files[0].split("/")[-2]  # Get File Name
    fname = "_".join(fname.split("_")[:-1])  # Remove Split information from file name
    outfile = f"{out_dir}/{fname}.npy"

    # Save on Condition File as .npy
    print(f"saving: {outfile}")
    np.save(outfile, on_raw)
   

#%% Export .npy files with MEG data only of HCs for HMMs

HCs = ['S101','S102','S103','S104','S105','S106','S107','S108','S109',
       'S110','S111','S112','S113','S114','S115','S116','S117','S118',
       'S119','S120','S121','S122','S123','S124','S125']

# Dir where data is saved
out_dir = "/home/esther/Research/sub_rsn/data/preprocessed/npy"

for iSub, sub in enumerate(HCs):

    # Get Path and Filenames for on and off conditio of participant
    files = sorted(glob(f"{src_dir}/*/sub-{sub}*HC*_sflip.npy"))

    # Load and merge raw files off on & on condition
    merged = np.vstack([np.load(f) for f in files])  # Load files & merge
    
    # Get Filename for saving
    fname = files[0].split("/")[-2]  # Get File Name
    fname = "_".join(fname.split("_")[:-1])  # Remove Split information from file name
    outfile = f"{out_dir}/{fname}.npy"

    # Save on Condition File as .fif
    print(f"saving: {outfile}")
    np.save(outfile, merged)
    
#%% Export .fif files with MEG and STN data for simultaneous LFP x MEG analyses
import mne

# Select Subjects to include in all analyses
subs =  ['S001','S002','S003','S004','S008','S009','S010','S011',
         'S014','S016','S020','S021','S022','S025','S027','S028',
         'S029','S032','S033','S034','S036','S037','S038',
         'S043','S048'] 


out_dir = ".../data/preprocessed/fif_wSTN"

for iSub, sub in enumerate(subs):

    # Get Path and Filenames for on and off conditio of participant
    on_files = sorted(glob(f"{src_dir}/src/*{sub}*periMedOn*/parc/parc_wSTN-raw.fif"))
    off_files = sorted(glob(f"{src_dir}/src/*{sub}*periMedOff*/parc/parc_wSTN-raw.fif"))
    
    # Remove block with poor data quality
    if sub == 'S022':
        # remove Med On Run 2
        del on_files[1]
    elif sub == 'S043':
        # remove Med Off run 2 and 3; remove Med On run 1 and 3
        del off_files[1:3]
        del on_files[2]
        del on_files[0]
                 
    # Load an merge raw files off on condition
    on_raws = [mne.io.read_raw_fif(on).pick(['misc','eeg']) for on in on_files] # Load files
    on_raw = mne.concatenate_raws(on_raws, on_mismatch='ignore') # concatenat
          
    # Get Filename for saving
    fname = on_files[0].split("/")[-3] # Get File Name
    fname = '_'.join(fname.split('_')[:-1]) # Remove Split information from file name    
    outfile = f"{out_dir}/{fname}_raw.fif"
    
    # Save on Condition File as .fif
    print(f"saving: {outfile}")
    on_raw.save(outfile, overwrite=True)
    
    # Load an merge raw files off on condition
    off_raws = [mne.io.read_raw_fif(off).pick(['misc','eeg']) for off in off_files] # Load files
    off_raw = mne.concatenate_raws(off_raws, on_mismatch='ignore') # concatenat
    
    # Get Filename for saving
    fname = off_files[0].split("/")[-3] # Get File Name
    fname = '_'.join(fname.split('_')[:-1]) # Remove Split information from file name    
    outfile = f"{out_dir}/{fname}_raw.fif"
    
    # Save off Condition File as .fif
    print(f"saving: {outfile}")
