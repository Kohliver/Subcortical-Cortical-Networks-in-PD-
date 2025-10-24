#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:41:47 2024

@author: okohl

SubRsn Project - HMM Inference 5

    - Make surface plot with my plotting seetings.
    - Get state rates and beta change
    - Calculate Group Contrasts and make plot

    - Calculate Group Metrics
    - Burst Metric Group Contrast and Plot

    - Overlap analysis comparison.
    
"""

import os
import numpy as np
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
  
# ----------------------------
# Define Plotting Function
def tsplot(ax, data, mean_data,time, color_data = 'blue', color_mean = 'k', linestyle = 'solid', linewidth = 2):
    x = time
    
    # Data Line
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    se = sd/np.sqrt(len(data))
    cis = (est - se, est + se)
    ax.fill_between(x,cis[0],cis[1],alpha = 0.2, facecolor = color_data)
    ax.plot(x,est,color = color_data, linestyle = linestyle, linewidth = 2.5)
    ax.margins(x=0)
    
    # Mean Data Line
    est = np.mean(mean_data, axis=0)
    sd = np.std(mean_data, axis=0)
    se = sd/np.sqrt(len(mean_data))
    cis = (est - se, est + se)
    ax.fill_between(x,cis[0],cis[1],alpha = 0.2, facecolor = color_mean)
    ax.plot(x,est,color = color_mean, linestyle = linestyle , linewidth = 1.5)
    ax.margins(x=0)
    
    # Make Axes pretty
    ax.set_xlim([2, 30])
    ax.set_ylim([0,0.12]) # 6 = .041, 8= .063, 10 = .06
    ax.set_xlabel('Frequency (Hz)',fontsize=20, labelpad=12)
    ax.set_ylabel('Power (a.u.)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14) 
    ax.ticklabel_format(scilimits=(-1,1))

    # Set x-Ticks
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter('{x:.0f}')    
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    # Set y-ticks
    ax.yaxis.set_major_locator(MultipleLocator(.02))
    #ax.yaxis.set_major_formatter('{x:.0f}')    
    ax.yaxis.set_minor_locator(MultipleLocator(.01))

    # Despine
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

# ----------------------------------------------------


K = 8

# Set Dirs
indir = '.../hmm/post_hoc/8_states_norm/inf' 
output_dir ='.../results/hmm/8_states_norm/state_descriptions'
demo_dir = '.../data/demographics'

# Make output directory
os.makedirs(output_dir, exist_ok=True)

# Load Group Data
df = pd.read_csv( f'{demo_dir}/behavioural_all.csv')

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = ".../osl/osl/source_recon/parcellation/files/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"
 
# Load spectra
f = np.load(indir + "/f.npy")
psd = np.load(indir + "/psd.npy")
coh = np.load(indir + "/coh.npy")
w = np.load(indir + "/w.npy")
fo = np.load(indir + '/fo.npy')

# Subtract mean across State
mean_psd = np.empty([psd.shape[0],psd.shape[2],psd.shape[3]])
for iSub in range(psd.shape[0]):
    mean_psd[iSub] = np.average(psd[iSub], axis=0, weights=fo[iSub])
    psd[iSub] = psd[iSub] - mean_psd[iSub]
         

#%% Average across all conditions
# --- Plot power spectra ---

# # Calculate the group average power spectrum for each state
# gpsd = np.average(psd, axis=0, weights=w)
# m_gpsd = np.average(mean_psd, axis=0, weights=w)
gpsd = psd
m_gpsd = mean_psd

# Pick Frequency Range
f_in = np.logical_and(f>=3,f<=30)
gpsd = gpsd[:,:,:,f_in]
m_gpsd = m_gpsd[:,:,f_in]
freqs = f[f_in]

# Combine mean and state specific changes
comb_gpsd = gpsd + m_gpsd[:,np.newaxis]

# --- Plot Motor Parcel Average ---
col =  [ '#727272','#3d3d3d']

for i in range(gpsd.shape[1]):
    
    # Make Plot
    fig, ax = plt.subplots()
    tsplot(ax, comb_gpsd[:,i,17:19].mean(axis=1), m_gpsd.mean(axis=1), time=freqs, color_data=col[1], color_mean='grey')
    
    #Save Figure
    plt.savefig(output_dir +  f"/all/motor_psd_{i}.svg",
                transparent = True,bbox_inches="tight",format="svg")


# --- Plot power maps ---

# Calculate the group average power spectrum for each state
gpsd = np.average(psd, axis=0, weights=w)

# Calculate the power map by integrating the power spectra over a frequency range
p = power.variance_from_spectra(f, gpsd, frequency_range = [2,30])
 
for i in range(p.shape[0]):
    
    # Get ymax
    ymax = abs(p[i].max()); 
    
    # Plot
    power.save(
        p[i],
        parcellation_file=parcellation_file,
        mask_file=mask_file,
        component=0,
        subtract_mean=True,
        plot_kwargs={"cmap": "RdBu_r", "bg_on_data": 1, "darkness": 0.8, "alpha": 1,"vmin": -ymax, "vmax": ymax},
        filename=f"{output_dir}/all/pow_{i}.png",
    )
    

# --- Plot coherence networks ---

# Calculate the group average
gcoh = np.average(coh, axis=0, weights=w)

# Calculate the coherence network by averaging over a frequency range
c = connectivity.mean_coherence_from_spectra(f, gcoh, frequency_range = [2,30])

# Threshold the top 2% of connections
c = connectivity.threshold(c, percentile=98, subtract_mean=True)

# Plot
connectivity.save(
    c,
    parcellation_file=parcellation_file,
    component=0,
    plot_kwargs={"edge_cmap": "Reds"},
    filename=output_dir + "/all/coh_.svg",
)



#%% Per conditions

conditions = ['periMedOff','periMedOn','HC']
for ind, condition in enumerate(conditions):
    
    mask = (df["Session"].values == ind + 1) #& (df['withinMed'].values == 1) 

    # --- Plot power spectra ---
    
    # # Calculate the group average power spectrum for each state
    # gpsd = np.average(psd, axis=0, weights=w)
    # m_gpsd = np.average(mean_psd, axis=0, weights=w)
    gpsd = psd[mask]
    m_gpsd = mean_psd[mask]
    
    # Pick Frequency Range
    f_in = np.logical_and(f>=3,f<=30)
    gpsd = gpsd[:,:,:,f_in]
    m_gpsd = m_gpsd[:,:,f_in]
    freqs = f[f_in]
    
    # Combine mean and state specific changes
    comb_gpsd = gpsd + m_gpsd[:,np.newaxis]
    
    # --- Plot Grand Average ---
    col =  [ '#727272','#3d3d3d']
    
    for i in range(gpsd.shape[1]):
        
        # Make Plot
        fig, ax = plt.subplots()
        tsplot(ax, comb_gpsd[:,i].mean(axis=1), m_gpsd.mean(axis=1), time=freqs, color_data=col[1], color_mean='grey')
        
        #Save Figure
        plt.savefig(output_dir +  f"/{condition}/psd_{i}.svg",
                    transparent = True,bbox_inches="tight",format="svg")
    
    
    # --- Plot Motor Parcel Average ---
    for i in range(gpsd.shape[1]):
        
        # Make Plot
        fig, ax = plt.subplots()
        tsplot(ax, comb_gpsd[:,i,17:19].mean(axis=1), m_gpsd.mean(axis=1), time=freqs, color_data=col[1], color_mean='grey')
        
        #Save Figure
        plt.savefig(output_dir +  f"/{condition}/motor_psd_{i}.svg",
                    transparent = True,bbox_inches="tight",format="svg")
    
    
    # --- Plot power maps ---
    
    # Calculate the group average power spectrum for each state
    gpsd = np.average(psd[mask], axis=0, weights=w[mask])
    
    # Calculate the power map by integrating the power spectra over a frequency range
    p = power.variance_from_spectra(f, gpsd, frequency_range = [2,30])
     
    for i in range(p.shape[0]):
        
        # Get ymax
        ymax = abs(p[i].max()); 
        
        # Plot
        power.save(
            p[i],
            parcellation_file=parcellation_file,
            mask_file=mask_file,
            component=0,
            subtract_mean=True,
            plot_kwargs={"cmap": "RdBu_r", "bg_on_data": 1, "darkness": 0.8, "alpha": 1,"vmin": -ymax, "vmax": ymax},
            filename=f"{output_dir}/{condition}/pow_{i}.png",
        )
        
    
    # --- Plot coherence networks ---
    
    # Calculate the group average
    gcoh = np.average(coh[mask], axis=0, weights=w[mask])
    
    # Calculate the coherence network by averaging over a frequency range
    c = connectivity.mean_coherence_from_spectra(f, gcoh, frequency_range = [2,30])
    
    # Threshold the top 2% of connections
    c = connectivity.threshold(c, percentile=98, subtract_mean=True)
    
    # Plot
    connectivity.save(
        c,
        parcellation_file=parcellation_file,
        component=0,
        plot_kwargs={"edge_cmap": "Reds"},
        filename=output_dir + f"/{condition}/coh_.svg",
    )
