#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:22:16 2024

@author: okohl

Rassoulou Replicttion - Preproc 9
    For each data file:
        for each electrode select contact with the largest beta power 
        and merge to MEG data.

    All Participants have 4 ring contacts in each electrode.
    Since no best clinical contact information is available, channel with highest
    beta power is selected.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from osl.preprocessing.osl_wrappers import detect_badchannels
from glob import glob
import pandas as pd
from itertools import compress

# Directories
preproc_dir = ".../Rassoulou2024/src_new/npy"
demo_dir = '.../Rassoulou2024/demographic'
plot_dir = '.../Rassoulou2024/results/static_new/stn/medication_contrast'
out_dir = '.../Rassoulou2024/src_new/npy_wSTN'
stn_out_dir = '.../Rassoulou2024/static/stn/spectra_new/'
plot_dir = '.../Rassoulou2024/results/static_new/stn/psd_perSub'

# Get file names - sorted by condition
files = []
for condition in ['MedOff',"MedOn"]:
    files.extend(sorted(glob(f"{preproc_dir}/*{condition}*.fif"))) 

#%% run loop through off files to identify STN channel with highest beta power in Off Condition
chans = []
chan_inds = []
for iSub, file in enumerate(files):
    
    # --- Load Channel Names from Montage ---
    
    # Dir to montage csv
    folder = file.split('/')[-1].split('_')[0]
    mname = sorted(glob(f'../Rassoulou2024/{folder}/ses-PeriOp/montage/*.tsv'))
    
    # read the file using pandas
    df = pd.read_csv(mname[0], sep='\t')    
    chan_right = list(df['right_contacts_new'])
    chan_left = list(df['left_contacts_new']  ) 
    
    # --- Load Data ---
    raw = mne.io.read_raw_fif(file).pick('eeg')
    meas_date = raw.info["meas_date"] 
    annotations = raw.annotations 
      
    # --- Bad Channel detection ---
    raw = detect_badchannels(raw, picks='eeg',significance_level=0.01)
   
    # Adjust STN channel names so that bads are excluded
    bads = raw.info['bads']
   
    # Select only good channels in on recordings
    chan_right = chan_right
    chan_left = chan_left
    if len(bads) > 0:
        for bad in bads:
            while (bad in chan_right):
                chan_right.remove(bad)
            while (bad in chan_left):
                chan_left.remove(bad)
       
    # --- Re-reference Right & Left ---
    stn_right = raw.copy().pick_channels(chan_right).load_data().set_eeg_reference(ref_channels="average",ch_type='eeg')   
    stn_left = raw.copy().pick_channels(chan_left).load_data().set_eeg_reference(ref_channels="average",ch_type='eeg')
    
    #%% --- Pick channel with highest beta power ---
    
    # --- Normalised time-courses ---
    raw_norm = raw.load_data().apply_function(zscore, n_jobs=10, picks='eeg')
    
    # --- Calculate Power Spectra ---
    # Parameter for PowerSpectrum calculation
    n_fft=500    
    n_overlap= int(n_fft/2)
    
    psd = raw_norm.compute_psd(method='welch',
                        fmin=5,
                        fmax=35,
                        n_fft=n_fft,
                        n_overlap=n_overlap,
                        n_per_seg=n_fft,
                        picks='eeg',
                        reject_by_annotation=True)
    
    def get_beta_power(psd):
        pow_in = psd.get_data() 
        fmask = np.logical_and(psd.freqs>13,psd.freqs<30) # Get Beta Mask          
        return np.mean(pow_in[:,fmask],axis=1) # Calculate mean Beta Power
       
    betas = get_beta_power(psd)

    n_chan_right = len(chan_right)

    # Get Channel Names of channels with max beta power (per Hemisphere)    
    r_chan_ind = np.argmax(betas[:n_chan_right])
    l_chan_ind = np.argmax(betas[n_chan_right:])
    
    r_chan = chan_right[r_chan_ind]
    l_chan = chan_left[l_chan_ind]
    
    #%% Make plot of PSDs for respective participant   
                
    # Grab PSDs and Freqs
    pow_in = psd.get_data() 
    f = psd.freqs
    
    n_r_chan = len(chan_right) # Get Number of right channels
    
    # Plot on vs off medication
    fig, axs = plt.subplots(ncols=1, nrows=2, dpi=300)
    plt.subplots_adjust(hspace=.5)
    
    axs[0].plot(f, pow_in[:n_r_chan].T, color='#9CBCD9', linewidth=1)
    axs[0].plot(f, pow_in[r_chan_ind].T, color='#586F8C', linewidth=2)
    
    axs[0].tick_params(axis='both', which='major',labelsize=8)
    axs[0].locator_params(nbins=5)
    
    axs[0].set_title(f'Right Hemisphere - {r_chan}')
    axs[0].set_ylabel('Power (dB)')
    axs[0].legend(handlelength=1,
                  handletextpad=.4,
                  labelspacing=.4,
                  frameon=False)
    
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False) 
    
    axs[1].plot(f, pow_in[n_r_chan:].T, color='#9CBCD9', linewidth=1)
    axs[1].plot(f, pow_in[l_chan_ind + n_r_chan].T, color='#586F8C', linewidth=2)
    
    axs[1].tick_params(axis='both', which='major',labelsize=8)
    axs[1].locator_params(nbins=5)
    
    axs[1].set_title(f'Left Hemisphere - {l_chan}')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Power (dB)')
    
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    
    fname = file.split("/")[-1]  # Get File Name
    fname = " ".join(fname.split(".")[:-1])  # Remove Split information from file name
    plt.savefig(f'{plot_dir}/{fname}_STN_psd.png')
    
    # Store Vars for Saving    
    chans.append([r_chan, l_chan])
    chan_inds.append([r_chan_ind, l_chan_ind])

#%% Grab Selected STNs Tcs, rereference them and merge them to MEG

# Get file names - sorted by condition
files = []
for condition in ['MedOff','MedOn' ]:
    files.extend(sorted(glob(f"{preproc_dir}/*{condition}*.fif"))) 

# Load Group Data
df = pd.read_csv( f'{demo_dir}/demographic_data.csv')
mask = (df["both_conditions" ].values == 1)[df['condition'] == 1]

chans_append = list(compress(chans, mask))
chan_inds_append = list(compress(chan_inds, mask))

# Dublicate results from off files
chans = chans + chans_append
chan_inds = chan_inds + chan_inds_append
   
for iSub, file in enumerate(files):
    
    # --- Load Channel Names from Montage ---
    r_chan_ind, l_chan_ind = chan_inds[iSub]
    
    # --- Load Channel Names from Montage ---
    
    # Dir to montage csv
    folder = file.split('/')[-1].split('_')[0]
    mname = sorted(glob(f'.../Rassoulou2024/{folder}/ses-PeriOp/montage/*.tsv'))
    
    # read the file using pandas
    df = pd.read_csv(mname[0], sep='\t')    
    chan_right = list(df['right_contacts_new'])
    chan_left = list(df['left_contacts_new']  ) 
    
    # --- Load Data ---
    raw = mne.io.read_raw_fif(file).pick('eeg')
      
    # --- Bad Channel detection ---
    raw = detect_badchannels(raw, picks='eeg',significance_level=0.01)
   
    # Adjust STN channel names so that bads are excluded
    bads = raw.info['bads']
   
    # Select only good channels in on recordings
    chan_right = chan_right
    chan_left = chan_left
    if len(bads) > 0:
        for bad in bads:
            while (bad in chan_right):
                chan_right.remove(bad)
            while (bad in chan_left):
                chan_left.remove(bad)
       
    # --- Re-reference Right & Left ---
    stn_right = raw.copy().pick_channels(chan_right).load_data().set_eeg_reference(ref_channels="average",ch_type='eeg')   
    stn_left = raw.copy().pick_channels(chan_left).load_data().set_eeg_reference(ref_channels="average",ch_type='eeg')
       
    # --- Merg STN channel with max beta power alongide meg data ---
    meg = mne.io.read_raw_fif(file).pick('misc').get_data(reject_by_annotation='omit')
    stn_r = stn_right.get_data(reject_by_annotation='omit')[[r_chan_ind]]
    stn_l = stn_left.get_data(reject_by_annotation='omit')[[l_chan_ind]]
    merged = np.vstack([meg, stn_r, stn_l]).T
 
    # --- Create Filename And Save ---
    fname = file.split("/")[-1]  # Get File Name
    fname = " ".join(fname.split(".")[:-1])  # Remove Split information from file name
    outfile = f"{out_dir}/{fname}.npy"
    
    # Save off Condition File as .npy
    print(f"saving: {outfile}")
    np.save(outfile, merged)
    
# Save STN channel informationm    
np.save(f"{stn_out_dir}/stn_channels.npy", chans)
np.save(f"{stn_out_dir}/stn_channel_inds.npy", chan_inds)


#%% Calculate PSD for selected STN channels
psd_all = np.zeros([len(files),2,61])
for iSub, file in enumerate(files):
    
    # Load newly created npy
    fname = file.split("/")[-1]  # Get File Name
    fname = " ".join(fname.split(".")[:-1])  # Remove Split information from file name
    infile = f"{out_dir}/{fname}.npy"
    
    # Load Data and zscore
    tc = np.load(infile)[:,-2:]
    tc = zscore(tc,axis=0)
    
    # Calculate ppsd
    # Parameter for PowerSpectrum calculation
    n_fft=500    
    n_overlap= int(n_fft/2)
    
    psd, f = mne.time_frequency.psd_array_welch(
                        tc.T,
                        sfreq=250,
                        fmin=5,
                        fmax=35,
                        n_fft=n_fft,
                        n_overlap=n_overlap,
                        n_per_seg=n_fft)
    
    psd_all[iSub] = psd
    
np.save(f"{stn_out_dir}/stn_psd.npy", psd_all)
np.save(f"{stn_out_dir}/stn_f.npy", f)
    