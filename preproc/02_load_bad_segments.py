#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:55:11 2024

@author: okohl

Sub-RSN Project -  Preprocessing 2

    Load Maxfiltered data and add bad segment and channel information from previous
    studies on this data set.
    
    Bad segments were manually marked by two researchers.
    
    Importantly this script has to be run from the MNE bids conda environment.
    (OSL and MNE BIDS are not compatible because they require differnt versions of MNE).
    
    Before this script can be run, the convert_annotation_files.py script in the
    helper folder needs to be run.
"""

import mne
from glob import glob
import pickle

import sys
sys.path.append(".../helper/")
from params import data_root


# Get filenames of inputs
with open(f'{data_root}/annotations/fif_wAnnotation','rb') as fp:
    fif_files = pickle.load(fp)
    
txt_files = sorted(glob('.../annotations/sub-*/*/*.txt'))


# Loop through all files:
for fif_file, txt_file in zip(fif_files, txt_files):

    # Load Fif to extract duration between onset of hardwere acquisition system and start of scan.
    raw = mne.io.read_raw_fif(fif_file, preload=True,on_split_missing="ignore")
    
    # Load Annotations        
    annotations = mne.read_annotations(txt_file)
    
    # Add annotations to raw object
    raw = raw.set_annotations(annotations)
    
    # Overwrite Fif file with fif file & added annotations
    raw.save(fif_file, overwrite=True)
    