#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:52:55 2022

@author: okohl

SubRsn Project - State-specific STN Power 1

Script Calculates Burst Metrics and burst on vs. off time courses
that are used for overlap analysis.

Steps:
    1. Load Data
    2. Burst analysis for left and right motor cortex
        a. Filter Power to 18 to 25Hz range. (Sig Cluster in Static Analysis)
        b. Amplitude Envelopes calculated with Hilbert Transform.
        c. Amplitude Envelopes thresholded at the 75th Percentile.
        d. Just keep instances with duration longer than 100ms      
    3. Combining of on vs off burst vectors of both motor parcels.
    4. Calculation of Burst Metrics
    5. Is burst vector is stored for calculation of power of bursts with GLM-Periodogram.
    6. Burst Metrics are saved.
"""

import numpy as np
import pickle
from scipy.stats import zscore
from glob import glob
from neurodsp.timefrequency.hilbert import amp_by_time
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats

# ----------------------------------------------------------------------------------

def percentile_to_thresh(tc, percentile=75):
    """
    Calculate a normalized threshold for a time series based on a specified percentile.

    Parameters:
        tc (array-like): The time series data.
        percentile (float, optional): The percentile of the time series to use for the threshold. 
                                      Defaults to 75 (75th percentile).

    Returns:
        float: The normalized threshold value, where the percentile value is divided 
               by the average magnitude of the time series.
    """
   
    # Calculate the average magnitude (mean of the absolute values of the time series)
    average_magnitude = np.median(np.abs(tc))

    # Compute the specified percentile of the time series
    percentile = np.percentile(tc, percentile)

    # Normalize the computed percentile by the average magnitude and return the result
    return percentile / average_magnitude

# ------------------------------------------------------------------------------------------

# --- Set Parameters and Dirs ----
in_dir = '.../data/static/stn/bcc/psd/tcs'
out_dir = '.../data/burst/stn/beta'
demo_dir = '.../data/demographics'

# Get file names - sorted by condition
files = []
for condition in ['periMedOff','periMedOn' ]:
    files.extend(sorted(glob(f"{in_dir}/*{condition}*")))

# --- Start Loop through Participants to save Copes and Tstats ---
nFiles = len(files)

is_burst_all = []
burst_lifetimes = []
burst_FOs = np.zeros(nFiles)
burst_rates = np.zeros(nFiles)
burst_meanLTs = np.zeros(nFiles)
for ind, file in enumerate(files):
    
    print('Loading Data for Subject' + str(ind))
    
    # --- Load Data ---
    data_in = np.load(file) 
    data_in = zscore(data_in,axis=1) # Normalise
    
    tc = data_in.mean(axis=0)
 
    # Parameters for burst extraction
    freq_range = [13, 30]
    fsample = 250
 
    # Compute thresh corresponding to 75th percentile
    tc_magnitude = amp_by_time(tc, fsample, freq_range, remove_edges=False)
    thresh = percentile_to_thresh(tc_magnitude,75)
    
    # Find bursts and extract burst metrics
    is_burst = detect_bursts_dual_threshold(tc, fs=fsample, dual_thresh=(thresh, thresh), f_range=freq_range, min_n_cycles = None, min_burst_duration=.1)
    burst_dict = compute_burst_stats(is_burst, fs=fsample)
   
    # ----- Collect Burst Metrics -----
     
    # Get Average Measures
    burst_FOs[ind] = burst_dict['percent_burst']
    burst_rates[ind] = burst_dict['bursts_per_second']
    burst_meanLTs[ind] = burst_dict['duration_mean']
    
    # Burst on vs Off vector
    is_burst_all.append(is_burst) # store for export later
            
# --- Save Burst Metrics with Amp Envelope as Beta Power measure ---
np.save(f'{out_dir}/fo.npy', burst_FOs)
np.save(f'{out_dir}/rate.npy', burst_rates)
np.save(f'{out_dir}/lts.npy', burst_meanLTs)
pickle.dump(is_burst_all, open(f"{out_dir}/is_burst.pkl", "wb"))

