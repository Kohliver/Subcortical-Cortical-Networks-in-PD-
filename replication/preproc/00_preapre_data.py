#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:29:50 2025

@author: okohl

Rassoulou Replication - Preproc 1
Prepare Data for preprocessing.
In most of the files task and resting-state data are stored in the same fif file.
This scripts separates the two conditions and saves them as different files.
"""

import os
import re
import math
import numpy as np
from glob import glob
import pandas as pd
from mne.io import read_raw_fif

# Rounding
def round_down(number, decimals):
    return math.floor(number * 10**decimals) / 10**decimals

# Splitting function from rassoulou2024: 
# https://github.com/Fayed-Rsl/RHM_preprocessing/blob/master/utils/technical_validation_utils.py#L47
def get_raw_condition(raw, conds):
    '''    
    Parameters
    ----------
    raw : mne.io.Raw
        raw object
    conds : list
        list of conditions
    Returns
    -------
    tuple of rest and task segments
    if no event found in the raw.annotations --> the value will be None.
    '''    
    # create a copy of the list so we can modify it without changing the original list 
    conditions = conds.copy() 
    
    # available conditions in the bids dataset
    allowed_conditions = [['rest', 'HoldL'], ['rest', 'MoveL'], ['rest', 'HoldR'], ['rest', 'MoveR'], ['rest']]
    assert conditions in allowed_conditions, f'Conditions should be in {allowed_conditions}'
    
    # initialise the segments by None
    task_segments = None 
    rest_segments = None
    
    # check that the raw actually have resting state and not only task [e.g., files run-2]
    if 'rest' not in raw.annotations.description:
        conditions.remove('rest')

    for task in conditions:
        # get the onset and the duration of the event 
        segments_onset = raw.annotations.onset[raw.annotations.description == task]
        segments_duration = raw.annotations.duration[raw.annotations.description == task]
        
        # substract the first_sample delay in the onset
        segments_onset = segments_onset - (raw.first_samp / raw.info['sfreq'])
        
        # Make sure that tmax is not longer than data length - if so round down at 4th decimal
        segments_offset = segments_onset + segments_duration
        recording_dur = (len(raw)-1) / raw.info['sfreq']
        
        mask = segments_offset >= recording_dur # Find if offset values are larger than duration of recording
        if any(mask):
            for i, m in enumerate(mask): # If tmax larger than 
                if m:
                   segments_offset[i] = round_down(segments_offset[i], 4)
            
        
        # loop trough the onset and duration to get only part of the raw that are in the task
        for i, (onset, offset) in enumerate(zip(segments_onset, segments_offset)):
            # if it is not resting state
            if task != 'rest':
                # if it is the first onset, initialise the raw object storing all of the task segments
                if i == 0:
                    task_segments = raw.copy().crop(tmin=onset, tmax=offset)
                # otherwise, append the segments to the existing raw object
                else:
                    task_segments.append([raw.copy().crop(tmin=onset, tmax=offset)])
            # do the same for rest
            else:
                if i == 0:
                    rest_segments = raw.copy().crop(tmin=onset, tmax=offset)
                else:
                    rest_segments.append([raw.copy().crop(tmin=onset, tmax=offset)])
    return rest_segments, task_segments 

# --- Get Started ---

# Get Filenames if .fif files
files = []
for condition in ['Rest', 'MoveL','HoldL','MoveR','HoldR']:
    files.extend(sorted(glob(f'.../Rassoulou2024/*/*/meg/*task-{condition}_acq*.fif')))

# Get file names of montages
montage_files = []
for file in files:
    sub_base, fname = file.split('meg/')
    new_fname = fname.split('task')[0] + 'montage.tsv'
    montage_files.append(sub_base + 'montage/' + new_fname) # List with paths to montages
    
# Get filenames for saving data
out_root = '.../Rassoulou2024/raw_data'

out_files_task = []
out_files_rest = []
for file in files:
    
    # Define file names for saving later
    fname_task = file.split('meg/')[1]
    condition = fname_task.split('_')[2]
    
    if condition == 'task-Rest':
        fname_rest = re.sub(r'(?<=task-)[^_]+','Rest',fname_task, flags=re.DOTALL)
    else:
        fname_rest = re.sub(r'(?<=task-)[^_]+',(condition[5] + 'Rest'),fname_task, flags=re.DOTALL)
    folder_name = fname_task.split('_')[0]
    
    # Add to list for later saving
    out_files_task.append(os.path.join(out_root, folder_name, fname_task))
    out_files_rest.append(os.path.join(out_root, folder_name, fname_rest))
    
         
# --- Loop over scans to split rest and task data and store them in preproc dir ---
noisy_chs = []
flat_chs = []
for fname, mname, outTask, outRest in zip(files, montage_files, out_files_task, out_files_rest):
    
    print(f'Processing {fname}')
    
    # Load Raw file 
    raw = read_raw_fif(fname, on_split_missing="ignore").copy()
   
    # read the file using pandas
    df = pd.read_csv(mname, sep='\t')
    
    # create a dictionary mapping old names to new names for right and left channels
    montage_mapping = {row['right_contacts_old']: row['right_contacts_new'] for idx, row in df.iterrows()} # right
    montage_mapping.update({row['left_contacts_old']: row['left_contacts_new'] for idx, row in df.iterrows()}) # left
    
    # remove in the montage mapping the channels that are not in the raw anymore (because they were marked as bads)
    montage_mapping  = {key: value for key, value in montage_mapping.items() if key in raw.ch_names }
    print(montage_mapping)
    
    # rename the channels using the montage mapping scheme
    raw.rename_channels(montage_mapping)
    
    # prepare the conditions to crop the data and get the rest_segments and task_segments
    task = fname.split('_')[2].split('-')[1] # Get condition name
    conditions = ['rest'] + [task] # rest + task (hold or move)

    # if the task is only 'Rest' then keep the raw segments as it is and ignore the task_segments
    if task == 'Rest': # only true for 2 subject
        rest_segments, task_segments = get_raw_condition(raw, ['rest'])
    # otherwise, crop the data and get the rest_segments and the task_segments
    else:
        rest_segments, task_segments = get_raw_condition(raw, conditions)
        
    # Create Folder Structure for saving
    path = os.path.dirname(outTask)
    if not os.path.isdir(path):
        os.makedirs(path)
    
    # Save data
    if task_segments is not None:
        task_segments.save(outTask)
    if rest_segments is not None:
        rest_segments.save(outRest)
    