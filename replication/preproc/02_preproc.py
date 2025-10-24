 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:15:46 2024

@author: okohl

Rassoulou Replicttion - Preproc 3

Run OSL Preprocessing Pipeline:
    Broadband (0.5 to 125Hz) and notch filters (50 & 100Hz)
    Reasmple to 250Hz
    Bad Segment detection in Mag, Grad, and STN channels
    Bad channel detection in Mag and Grad channels
    Interpolate bad channels
    
"""

from glob import glob
from dask.distributed import Client
from osl import preprocessing, utils
import os
from mne.bem import fit_sphere_to_headshape
 
# --- Define Custom Function ---

def custom_interpolate_bads(dataset, userargs):
   
   # Check whether extra headshape points were collected.
   dig_dict = dataset['raw'].info['dig']
   dig_kinds = ['FIFFV_POINT_EEG', 'FIFFV_POINT_EXTRA']
   is_fitting_points = any([p["kind"]._name in dig_kinds for p in dig_dict])
   
   # If extra head shape points were collected standard bad_channel interpolation
   if is_fitting_points:   
       # if a logger was set up, e.g., by run_proc_chain, log_or_print will write to the logfile
       utils.logger.log_or_print("Stander interpolation of bad channels in the raw data.")
    
       # Interpolate Bads
       dataset["raw"] = dataset['raw'].interpolate_bads()
       
    # Fit head sphere manually and give to interpolate_bads, if not extra headshape points were collected
   else:
       # Fit Head Sphere from cardinal and HPI points
       _ , origin_head, _ = fit_sphere_to_headshape(dataset['raw'].info, dig_kinds=('cardinal', 'hpi'))
       
       # if a logger was set up, e.g., by run_proc_chain, log_or_print will write to the logfile
       utils.logger.log_or_print("No extra headshape points collected\nInterpolation of bad channels in the raw data based on head sphere fitted from cardinal and hpi points.")
    
       # Interpolate Bads
       dataset["raw"] = dataset['raw'].interpolate_bads(origin=origin_head)    
   return dataset

# -----------

# Get filenames
data_root = ".../Rassoulou2024/" 
os.chdir(data_root)
 
inputs = []
for condition in ['Rest','MRest','HRest']:
    inputs.extend(sorted(glob(f'{data_root}/maxfilter/*/*/*-{condition}*.fif')))

# Outdir
preproc_dir = f"{data_root}/preproc_new"  # output directory containing the preprocess files

# Settings
config = """
    preproc:
    - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
    - bad_segments: {segment_len: 500, picks: grad, significance_level: 0.1}
    - bad_segments: {segment_len: 500, picks: eeg, significance_level: 0.1}
    - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
    - bad_segments: {segment_len: 500, picks: grad, mode: diff, significance_level: 0.1}
    - bad_segments: {segment_len: 500, picks: eeg, mode: diff, significance_level: 0.1}
    - bad_channels: {picks: mag}
    - bad_channels: {picks: grad}
    - ica_raw: {picks: meg, n_components: 40}
    - ica_autoreject: {apply: False}
    - custom_interpolate_bads: {}
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=8, threads_per_worker=1)

    # Main preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=preproc_dir,
        extra_funcs=[custom_interpolate_bads],
        overwrite=True,
        dask_client=True,
    )
    