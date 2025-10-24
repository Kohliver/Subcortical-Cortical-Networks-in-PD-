# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:52:55 2022

@author: okohl

SubRsn Project - State-specific STN Power 2

Script compares burst overlap with HMM state overlap.

Steps:
    1. Load Data
    2. Gamma time course is binarised
    3. isBurst vector loaded
    4. Overlap between isBurst and Gammas and overlap is calculated
    5. Overlap between isBurst and shifted Gammas and overlap is calculated
    6. Save Overlap TCs and Overlap Metric
    
    
Overlap is calculated as the sum of co-occurrences devided by the sum of respective state
occurences. This means the overlap will be expressed relative to state occurences.
"In XX% of a time state X is present a burst co-occures".


Overlap calculated between is Burst Vector and rolled versions of HMM-State
on-vs-of array to construct null distributions for later significance testing.
Procedure is repeated for 1000 roles with a randome number rolling number.
Rolling numbers of 1000 roles are the same for each particpant.
"""

import numpy as np
import pickle
import pandas as pd
from osl_dynamics.data import Data
from osl_dynamics.inference import modes
import random
from itertools import compress
 
    
# ------------------------------------------------------------------------

def calculate_overlaps(x, y_matrix, num_permutations=1000):
    """
    Calculate observed and permuted overlaps using rolling, for multiple y time courses.

    Args:
        x: Binary time course 1 (array-like, shape = (T,)).
        y_matrix: Binary time courses 2 (matrix-like, shape = (N, T)).
        num_permutations: Number of permutations for the null distribution.

    Returns:
        observed_overlaps: Observed overlaps for each time course in y_matrix (array of shape (N,)).
        permuted_overlaps: Permuted overlaps for each time course in y_matrix 
                           (array of shape (N, num_permutations)).
    """
    num_time_points = x.shape[0]
    num_time_courses = y_matrix.shape[0]
    num_mat_1s = np.sum(y_matrix,axis=1)

    # Calculate observed overlaps (vectorized for all time courses)
    observed_overlaps = np.sum(np.logical_and(y_matrix, x), axis=1) / num_mat_1s

    # Preallocate matrix for permuted overlaps
    permuted_overlaps = np.zeros((num_time_courses, num_permutations))

    # Generate permuted overlaps
    for p in range(num_permutations):
        # Roll each row (time course) in y_matrix by a random shift
        random_shifts = np.random.randint(0, num_time_points, size=num_time_courses)
        y_rolled = np.array([
            np.roll(y_matrix[i], random_shifts[i]) for i in range(num_time_courses)
        ])
        permuted_overlaps[:, p] = np.sum(np.logical_and(y_rolled, x), axis=1) / num_mat_1s

    return observed_overlaps, permuted_overlaps


def calculate_overlaps_shuffle(x, y_matrix, num_permutations=1000):
    """
    Calculate observed and permuted overlaps using random shuffling, for multiple y time courses.

    Args:
        x: Binary time course 1 (array-like, shape = (T,)).
        y_matrix: Binary time courses 2 (matrix-like, shape = (N, T)).
        num_permutations: Number of permutations for the null distribution.

    Returns:
        observed_overlaps: Observed overlaps for each time course in y_matrix (array of shape (N,)).
        permuted_overlaps: Permuted overlaps for each time course in y_matrix 
                           (array of shape (N, num_permutations)).
    """
    num_time_points = x.shape[0]
    num_time_courses = y_matrix.shape[0]
    num_mat_1s = np.sum(y_matrix, axis=1)  # Number of 1's in each row of y_matrix
    #num_mat_1s = np.sum(x)

    # Calculate observed overlaps (vectorized for all time courses)
    observed_overlaps = np.sum(np.logical_and(y_matrix, x), axis=1) / num_mat_1s

    # Preallocate matrix for permuted overlaps
    permuted_overlaps = np.zeros((num_time_courses, num_permutations))

    # Generate permuted overlaps
    for p in range(num_permutations):
        # Randomly shuffle each row of y_matrix
        y_shuffled = np.array([np.random.permutation(y_matrix[i]) for i in range(num_time_courses)])
        permuted_overlaps[:, p] = np.sum(np.logical_and(y_shuffled, x), axis=1) / num_mat_1s

    return observed_overlaps, permuted_overlaps


def calculate_statistics_two_tailed(observed_overlaps, permuted_overlaps):
    """
    Calculate two-tailed p-values for observed overlaps against permuted overlaps.

    Args:
        observed_overlaps: Observed overlaps (array of shape (N,)).
        permuted_overlaps: Permuted overlaps (array of shape (N, num_permutations)).

    Returns:
        p_values: Two-tailed p-values for each time course (array of shape (N,)).
    """
    # One-tailed p-values for greater and less directions
    p_greater = np.mean(permuted_overlaps >= observed_overlaps[:, None], axis=1)
    p_less = np.mean(permuted_overlaps <= observed_overlaps[:, None], axis=1)
    
    # Two-tailed p-value
    p_values = 2 * np.minimum(p_greater, p_less)
    return p_values
 
# --------------------------------------------------------------------------------
    
# --- Set Parameters and Dirs ----
demo_dir = '.../data/demographics'
burst_dir = '.../data/burst/stn/beta'
alp_dir = '.../data/hmm/post_hoc/8_states_norm/inf'
out_dir = ".../data/hmm/post_hoc/8_states_norm/overlap/beta"

# Load Session Data
df = pd.read_csv( f'{demo_dir}/behavioural_all.csv')
mask = (df["Session" ].values < 3) & (df['withinMed'] == 1)

# Load and prepare state time courses  
is_burst_all = pickle.load(open(f"{burst_dir}/is_burst_comb.pkl", "rb"))
is_burst_all = [is_burst[:,np.newaxis] for is_burst in is_burst_all] # Add first same first dimension to all tcs to pretend "channels"
is_burst_trimmed = Data(is_burst_all, n_jobs=16) # lOd a data
is_burst_trimmed = is_burst_trimmed.trim_time_series(n_embeddings=15, sequence_length=400) # Trim to length of burst state tie course inferred by hmm

# Load State Time Courses
alps = pickle.load(open(f"{alp_dir}/alp.pkl", "rb"))
alps = list(compress(alps,mask))

# --- Start Loop through Participants to save Copes and Tstats ---
nFiles = len(alps) 
K = alps[0].shape[1]
nPerm = 1000

overlap_all = np.zeros([nFiles,4])
overlap_nulls_all = np.zeros([nFiles,4,nPerm])
ps = np.zeros([nFiles,4 ])
for ind in range(nFiles):
 
    print('File' + str(ind))
    
    # Load HMM Data Data and get binary state time course
    alp = alps[ind]
    stc = modes.argmax_time_courses(alp)
    
    # Adjust State Time course to State 5 vs all other states and state 1 vs all other state time courses
    is_state1 = stc[:,0] == 1
    is_state6 = stc[:,5] == 1
    stc = np.vstack([is_state1,~is_state1, is_state6, ~is_state6])
    
    # Load Burst On Off Set Data
    is_burst = is_burst_trimmed[ind].squeeze()
     
    # Get Overlap Metrics
    overlap, overlap_nulls  = calculate_overlaps_shuffle(is_burst, stc, num_permutations=nPerm)
    ps[ind,:] = calculate_statistics_two_tailed(overlap, overlap_nulls) 
 
    # Store data for saving
    overlap_all[ind] = overlap
    overlap_nulls_all[ind] = overlap_nulls 
    
  
# --- Save Data ---
np.save(f'{out_dir}/normalised_overlap_stateNormCont_ps_comb.npy' ,  ps)  
np.save(f'{out_dir}/normalised_overlap_stateNormCont_comb.npy' ,  overlap_all) 
np.save(f'{out_dir}/normalised_overlap_stateNormCont_nulls_comb.npy' ,  overlap_nulls_all)