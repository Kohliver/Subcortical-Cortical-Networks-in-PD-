#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:19:42 2025

@author: okohl

Rassoulou Replicttion - Preproc 8
    Merge recording blocks for further analyses.
"""

from glob import glob
import numpy as np
from osl.source_recon.parcellation import parcel_centers, plot_markers
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.stats import zscore

# ------------------------------------------------------------------------------

# This function is copied from osl and sligthly modified to have better frequency resolution:
def plot_psd(parc_ts, fs, parcellation_file, filename, freq_range=None):
    """Plot PSD of each parcel time course.

    Parameters
    ---------- 
    parc_ts : np.ndarray
        (parcels, time) or (parcels, time, epochs) time series.
    fs : float
        Sampling frequency in Hz.
    parcellation_file : str
        Path to parcellation file.
    filename : str
        Output filename.
    freq_range : list of len 2
        Low and high frequency in Hz.
    """
    
    # 
    parc_ts = zscore(parc_ts, axis=1)
    
    # Calcualte PSD of continuous data
    f, psd = welch(parc_ts, fs=fs, nperseg=fs*4, nfft=fs*4)

    n_parcels = psd.shape[0]

    if freq_range is None:
        freq_range = [f[0], f[-1]]

    # Re-order to use colour to indicate anterior->posterior location
    parc_centers = parcel_centers(parcellation_file)
    order = np.argsort(parc_centers[:, 1])
    parc_centers = parc_centers[order]
    psd = psd[order]

    # Plot PSD
    fig, ax = plt.subplots(dpi=300)
    cmap = plt.cm.viridis_r
    for i in reversed(range(n_parcels)):
        ax.plot(f, psd[i], c=cmap(i / n_parcels))
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (a.u.)", fontsize=14)
    ax.set_xlim(freq_range[0], freq_range[1])
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()

    # Plot parcel topomap
    inside_ax = ax.inset_axes([0.45, 0.55, 0.5, 0.55])
    plot_markers(np.arange(n_parcels), parc_centers, node_size=12, colorbar=False, axes=inside_ax)

    # Save
    plt.savefig(filename)
    plt.close()   

# ------------------------------------------------------------------------------

# Set dirs
src_dir = ".../Rassoulou2024/src_new/sign_flipped"
out_dir = ".../Rassoulou2024/src_new/npy"

# Select Subjects to include in all analyses
subs = ['0cGdk9','2IhVOz','2IU8mi','8RgPiG','AB2PeX',
        'AbzsOg','BYJoWR','dCsWjQ','FIyfdR','FYbcap',
        'gNX5yb','hnetKS','i4oK0F','iDpl28','jyC0j3',
        'oLNpHd','PuPVlx','VopvKx',] # Exclude IDPL28 because only 1 good Med Off resting State scan


glasser52 = '.../osl/osl/source_recon/parcellation/files/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz'

#%% Export .npy files with MEG data only of HCs for HMMs

# Make sure to only include subs and run for main analysis
excluded_files = [f'{src_dir}/sub-jyC0j3_ses-PeriOp_task-HRest_acq-MedOff_run-1_sflip.npy',]

# Here the excluding of data sets is happening
for iSub, sub in enumerate(subs):
    
    # Get Path and Filenames for on and off conditio of participant
    on_files = sorted(glob(f"{src_dir}/sub-{sub}*MedOn*_sflip.npy"))
    off_files = sorted(glob(f"{src_dir}/sub-{sub}*MedOff*_sflip.npy"))
    
    # Remove files that should be excluded
    for on_file, off_file in zip(on_files, off_files):
        
        # remove off file if part of excluded Files
        if on_file in excluded_files:
            on_files.remove(on_file)
         
        # remove off file if part of excluded Files
        if off_file in excluded_files:
            off_files.remove(off_file)
    
    if sub == 'iDpl28': # Only off resting state scan available
            
        # Load and merge raw files off on & on condition
        off_raw = np.vstack([np.load(off) for off in off_files])  # Load files & merge
        
        # Get Filename for saving
        fname = off_files[0].split("/")[-1]  # Get File Name
        fname = fname.split('_')[0]  # Remove Split information from file name
        outfile = f"{out_dir}/{fname}_rest_MedOff.npy"

        # Save off Condition File as .npy
        print(f"saving: {outfile}")
        np.save(outfile, off_raw)
        
        # Plot power spectrum
        plot_psd(off_raw.T, 
                  fs=250, 
                  parcellation_file=glasser52, 
                  freq_range=[1,45],
                  filename=f'/.../Rassoulou2024/src_new/npy/spectra/{fname}_rest_MedOff.png')
        
    elif sub == 'sub-6m9kB5':
        print('sub-6m9kB5 excluded because of bad data quality.')
    
    else:
    
        # Load and merge raw files off on & on condition
        on_raw = np.vstack([np.load(on) for on in on_files])  # Load files & merge
        off_raw = np.vstack([np.load(off) for off in off_files])  # Load files & merge
    
        # Get Filename for saving
        fname = on_files[0].split("/")[-1]  # Get File Name
        fname = fname.split('_')[0]  # Remove Split information from file name
        outfile = f"{out_dir}/{fname}_rest_MedOn.npy"
    
        # Save on Condition File as .fif
        print(f"saving: {outfile}")
        np.save(outfile, on_raw)
        
        # Plot power spectrum
        plot_psd(on_raw.T, 
                 fs=250, 
                 parcellation_file=glasser52, 
                 freq_range=[1,45],
                 filename=f'.../Rassoulou2024/src_new/npy/spectra/{fname}_rest_MedOff.png')
    
        # Get Filename for saving
        fname = off_files[0].split("/")[-1]  # Get File Name
        fname = fname.split('_')[0]  # Remove Split information from file name
        outfile = f"{out_dir}/{fname}_rest_MedOff.npy"
    
        # Save off Condition File as .npy
        print(f"saving: {outfile}")
        np.save(outfile, off_raw)
        
        # Plot power spectrum
        plot_psd(off_raw.T, 
                 fs=250,
                 freq_range=[1,45],
                 parcellation_file=glasser52, 
                 filename=f'{out_dir}/spectra/{fname}_rest_MedOff.png')
        
