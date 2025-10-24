#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:35:27 2025

@author: okohl

Rassoulou Replicttion - Preproc 6
    Data was source reconstructed and parcellated with Fieldtrip because
    no T1 MRIs but Fieldtrip Head- and Sourfacemodels were provided.
    
    Parcellated data is imported and leakage correction is applied before the
    data is saved as a .fif file for further preprocessing.
    
"""

from scipy.io import loadmat
from glob import glob
from osl.source_recon.parcellation import symmetric_orthogonalise
import mne 
import os
import numpy as np

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

# Dirs
root_dir = '.../Rassoulou2024'
preproc_dir = '.../Rassoulou2024/preproc_new/'
outdir = '.../Rassoulou2024/src_new/orthogonalised'

os.chdir(outdir)

# Load parc file - data was parcellated in Field Trip
parc_files = sorted(glob('.../Rassoulou2024/src_new/parcellated/*')) # excluded because of rank issues: [12,14,17,18]

# Mask mit indices of preproc files to include
preproc_files = []
for pfile in parc_files:
    folder_name = pfile.split('/')[-1].split('.mat')[0]
    idx = folder_name.index('_tsss')
    folder_name = folder_name[:idx] + '_raw' + folder_name[idx:]      
    preproc_files.append(glob(f'{preproc_dir}/{folder_name}/*raw.fif')[0])
    
for parc_file, preproc_file in zip(parc_files, preproc_files):
    
    # Grab File Name
    fname = parc_file.split('/')[-1].split('.mat')[0]

    # Load Data
    parc = loadmat(parc_file)['tc']
    preproc = mne.io.read_raw_fif(preproc_file)
       
    # leakage coorrection [nparcel x time]
    parc = symmetric_orthogonalise(parc, maintain_magnitudes=True, compute_weights=False)
    
    # To fif 
    #if os.path.isfile(outdir + f'{fname}_parc-raw.fif'):
    print(f'Saving {fname}')
    
    parc_raw = convert2mne_raw(parc, preproc, parcel_names=None, extra_chans=["stim","eeg"])
    parc_raw.save(f'{fname}_parc-raw.fif') 
    
