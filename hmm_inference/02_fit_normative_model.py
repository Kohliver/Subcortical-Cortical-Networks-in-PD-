#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:10:25 2024

@author: okohl

SubRsn Project - HMM Inference 2
    Run HMM fitting canonical states to our data.
    I.e., state descriptions obtained from CamCan data set are loaded and used
    to identify their state occurences in our dataset.
    
    This procedure assuress that good state descriptions are fitted to the data.
    
    See Gohil in prep. (https://github.com/OHBA-analysis/Canonical-HMM-Networks) 
"""

import os
import pickle
import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model

# Dirs where things are
norm_model_dir = ".../data/hmm/norm_model/public"
data_dir = '.../data/hmm/data_in'
output_dir = '.../data/hmm/post_hoc/12_states_norm/inf'

os.makedirs(output_dir, exist_ok=True)

# Load prepared data
data = Data(data_dir, n_jobs=16)

# Load cannonical model
means = np.load(f"{norm_model_dir}/12_states/means.npy")
covs = np.load(f"{norm_model_dir}/12_states/covs.npy")
trans_prob = np.load(f"{norm_model_dir}/12_states/trans_prob.npy")

n_states, n_channels = means.shape

config = Config(
    n_states=n_states,
    n_channels=n_channels,
    sequence_length=400,
    learn_means=False,
    learn_covariances=True,
    learn_trans_prob=True,
    initial_means=means,
    initial_covariances=covs,
    initial_trans_prob=trans_prob,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=20,
)
model = Model(config)

# Get inferred state probabilities and calculate argmax
alp = model.get_alpha(data)
pickle.dump(alp, open(f"{output_dir}/alp.pkl", "wb"))

# Get free energy
free_energy = model.free_energy(data)
with open(f"{output_dir}/loss.dat", "w") as file:
    file.write(f"free_energy = {free_energy}\n")