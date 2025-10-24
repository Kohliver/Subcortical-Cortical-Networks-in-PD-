#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:30:05 2024

@author: okohl

SubRsn Project - HMM Inference 4
    Post-hoc estimation of state dynamics from state-probability time courses

"""

print("Importing packages")

import pickle
import numpy as np

from osl_dynamics.inference import modes

output_dir = '.../data/hmm/post_hoc/8_states_norm/inf'

# Load inferred state probabilities
alp = pickle.load(open(f"{output_dir}/alp.pkl", "rb"))

# Calculate state time course - Hard Classified
stc = modes.argmax_time_courses(alp)

# Calculate summary stats
print("Calculating summary stats")
fo = modes.fractional_occupancies(stc)
lt = modes.mean_lifetimes(stc, sampling_frequency=250) * 1e3
intv = modes.mean_intervals(stc, sampling_frequency=250)
sr = modes.switching_rates(stc, sampling_frequency=250)

# Save
np.save(f"{output_dir}/fo.npy", fo)
np.save(f"{output_dir}/lt.npy", lt)
np.save(f"{output_dir}/intv.npy", intv)
np.save(f"{output_dir}/sr.npy", sr)