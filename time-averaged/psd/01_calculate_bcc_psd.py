#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:22:16 2024

@author: okohl

SubRSN Project - STN Power and Coherence 1

    Best Clinical (BCC) Channel of each electrode is selected and
    STN power spectrum is calculated.
    BCC time courses are save for later burst analyses.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from osl.preprocessing.osl_wrappers import detect_badchannels
from glob import glob
import pandas as pd

# PSD plot per Sub?
make_plots = True

# Path to files
preproc_dir = '.../data/preprocessed/fif_noLC'
out_dir = '.../data/static/stn/bcc/psd/'
plot_dir = ".../results/static/stn/bcc/spectra_perSub/"
demo_dir = '.../sub_rsn/data/demographics'

# Set Channel Names
channels_right = ['EEG001','EEG002','EEG003','EEG004','EEG005','EEG006','EEG007','EEG008']   
channels_left = ['EEG033','EEG034','EEG035','EEG036','EEG037','EEG038','EEG039','EEG040']

# Run demographic information
df = pd.read_csv( f'{demo_dir}/behavioural_all.csv')
bcc  = df[["BCC_right","BCC_left"]] # 1 = Off; 2 = On

# Get file names - sorted by condition
files = []
for condition in ['periMedOff','periMedOn' ]:
    files.extend(sorted(glob(f"{preproc_dir}/*{condition}*")))

# run loop concatenating files of participants and calculating psds of STN channels
chans = []
chan_inds = []
sub = []
psd_all = np.zeros([len(files),2,87]) 
for iFile, file in enumerate(files):
            
    #%% Load Data
    raw = mne.io.read_raw_fif(file).pick('eeg')
    meas_date = raw.info["meas_date"] 
    annotations = raw.annotations 
    
    # Grab Best Clinical Contact Names
    bccs = bcc.iloc[iFile].to_list()
      
    #%% Remove Bad Contacts and re-reference against the mean across all good contacts
    
    # Bad Channel detection
    raw = detect_badchannels(raw, picks='eeg',significance_level=0.01)
   
    # Adjust STN channel names so that bads are excluded
    bads = raw.info['bads']
   
    # Select only good channels in on recordings
    chan_right = channels_right.copy()
    chan_left = channels_left.copy()
    if len(bads) > 0:
        for bad in bads:
            if bad not in bccs: # Make sure not to remove BCCs
                while (bad in chan_right):
                    chan_right.remove(bad)
                while (bad in chan_left):
                    chan_left.remove(bad)
   
    # Re-reference Right & Left (average across all contacts on electrode) 
    stn_right = raw.copy().pick_channels(chan_right).load_data().set_eeg_reference(ref_channels="average",ch_type='eeg')   
    stn_left = raw.copy().pick_channels(chan_left).load_data().set_eeg_reference(ref_channels="average",ch_type='eeg')
    
    #%% Select bcc per hemisphere, merge, and save
    
    # pick channels as specified in sure et al. 2021 https://doi.org/10.3389/fnins.2021.724334
    included_chan_r = [bccs[0]]
    bcc_r_ind = mne.pick_channels(included_chan_r,stn_right.ch_names)
    stn_right_bcc = stn_right.copy().pick(included_chan_r)
    
    included_chan_l = [bccs[1]]
    bcc_l_ind = mne.pick_channels(included_chan_l,stn_left.ch_names)
    stn_left_bcc = stn_left.copy().pick(included_chan_l)
      
    # Merge Right and left channels
    raw_merged = stn_right_bcc.copy().add_channels([stn_left_bcc],force_update_info=True)
    
    # FIF file to npy
    stn = raw_merged.get_data(reject_by_annotation='omit')
    
    # Create Filename and Save
    fname = file.split("/")[-1]  # Get File Name
    fname = " ".join(fname.split(".")[:-1])  # Remove Split information from file name
    np.save(f"{out_dir}/tcs/{fname}.npy", stn)

    # Grab Contract information
    chans.append(raw_merged.ch_names)
    sub.append(file.split("/")[-1])
    chan_inds.append([bcc_r_ind,bcc_l_ind])

    #%% Calculate PSDs
    
    # Merge all contacts
    raw_in = stn_right.copy().add_channels([stn_left],force_update_info=True)
    
    # Normalised time-courses
    raw_norm = raw_in.apply_function(zscore, n_jobs=10)
    
    # Parameter for PowerSpectrum calculation
    n_fft=500    
    n_overlap= int(n_fft/2)
    
    # Calculate Power Spectra
    psd = raw_norm.compute_psd(method='welch',
                        fmin=2,
                        fmax=45,
                        n_fft=n_fft,
                        n_overlap=n_overlap,
                        n_per_seg=n_fft,
                        picks='eeg',
                        reject_by_annotation=True)
        
    # Grab PSDs of selected contacts
    psd_all[iFile] = psd.get_data()[[bcc_r_ind,bcc_l_ind]].squeeze()
    
    
    #%% Make plot of PSDs for respective participant   
    if make_plots == True:
        
        # Grab PSDs and Freqs
        pow_in = psd.get_data() 
        f = psd.freqs
        
        n_r_chan = len(chan_right) # Get Number of right channels
        
        # Plot on vs off medication
        fig, axs = plt.subplots(ncols=1, nrows=2, dpi=300)
        plt.subplots_adjust(hspace=.5)
        
        axs[0].plot(f, pow_in[:n_r_chan].T, color='#9CBCD9', linewidth=1)
        axs[0].plot(f, pow_in[bcc_r_ind].T, color='#586F8C', linewidth=2)
        
        axs[0].tick_params(axis='both', which='major',labelsize=8)
        axs[0].locator_params(nbins=5)
        
        axs[0].set_title(f'Right Hemisphere - {included_chan_r}')
        axs[0].set_ylabel('Power (dB)')
        axs[0].legend(handlelength=1,
                      handletextpad=.4,
                      labelspacing=.4,
                      frameon=False)
        
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False) 
        
        axs[1].plot(f, pow_in[n_r_chan:].T, color='#9CBCD9', linewidth=1)
        axs[1].plot(f, pow_in[bcc_l_ind].T, color='#586F8C', linewidth=2)
        
        axs[1].tick_params(axis='both', which='major',labelsize=8)
        axs[1].locator_params(nbins=5)
        
        axs[1].set_title(f'Left Hemisphere - {included_chan_l}')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Power (dB)')
        
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
            
        plt.savefig(f'{plot_dir}/{fname}_STN_psd.png')
    
# Save STN channel informationm    
np.save(f"{out_dir}/stn_channels.npy", chans)
np.save(f"{out_dir}/stn_channel_inds.npy", chan_inds)
np.save(f"{out_dir}/sub_names.npy", sub) 
np.save(f"{out_dir}/stn_psd.npy", psd_all)
np.save(f"{out_dir}/stn_f.npy", f)
    
    