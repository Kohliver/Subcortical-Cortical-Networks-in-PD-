#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:49:29 2025

@author: okohl

SubRSN Project - STN Power and Coherence 4

    Figure for Static Analyses.
    1) STN Power Spectrum Contrast
    2) STN - SMA Coherence Spectrum

"""

import numpy as np
import glmtools as glm
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------

# Gret Cluster forming threshold from model degrees of freedome
def get_cluster_forming_threshold(dof_model, alpha=0.05):
    return stats.t.ppf(1 - alpha/2, dof_model)
  

# Function identifying cluster start and end points
def find_cluster_intervals(binary_vector):
    # Find indices where transitions from 0 to 1 occur
    start_indices = np.where(np.diff(binary_vector) > 0)[0]
    # Find indices where transitions from 1 to 0 occur and adjust for the end of intervals
    end_indices = np.where(np.diff(binary_vector) < 0)[0] - 1

    # Handle the case where the binary vector starts or ends with a true value
    if binary_vector[0] != 0:
        start_indices = np.insert(start_indices, 0, 0)
    if binary_vector[-1] != 0:
        end_indices = np.append(end_indices, len(binary_vector) - 1)

    # Combine start and end indices to form intervals
    intervals = list(zip(start_indices, end_indices))

    return intervals
  

# Calculate Cluster Permutation tests of State TCs
def tc_ClusterPermutation_test(data, covariates, 
                               nPerm=10000,
                               metric='tstats',
                               cluster_forming_threshold=[],
                               pooled_dims=[1],
                               n_jobs=4):
    
    # --- Define Dataset for GLM -----    
    data = glm.data.TrialGLMData(data=data,
                                 **covariates)
                                 
    # ----- Specify regressors and Contrasts in GLM Model -----
    DC = glm.design.DesignConfig()
    DC.add_regressor(name='Constant',rtype='Constant')
    for name in covariates:
        DC.add_regressor(name=name, rtype="Parametric", datainfo=name, preproc="z")
    
    DC.add_contrast(name="On > Off", values=[1] + [0] * len(covariates))

    
    #  ---- Create design martix and fit model and grab tstats ----
    des = DC.design_from_datainfo(data.info)
    #des.plot_summary(savepath='/home/esther/Research/sub_rsn/results/static/stn/bcc/spectra_condition_contrast/contrast_summary.png')
    #des.plot_efficiency(savepath='/home/esther/Research/sub_rsn/results/static/stn/spectra_condition_contrast/contrast_efficiency.png')
    model = glm.fit.OLSModel(des,data)
    
    # -------------------------------------
    # Permutation Test Pooling Across States
    # ---------------------------------------
    
    # Calculate Cluster forming threshold from Mode Dof if not specified
    if not cluster_forming_threshold:
        cluster_forming_threshold = get_cluster_forming_threshold(model.dof_model)
      
    # Run Permutation Tests
    contrast = 0
    CP = glm.permutations.ClusterPermutation(des, data, contrast, nPerm,
                                            metric=metric,
                                            tail=0,
                                            cluster_forming_threshold=cluster_forming_threshold,
                                            pooled_dims=pooled_dims,
                                            nprocesses=n_jobs)
   
    # Get Cluster inndices and pvalues
    cluster_masks, cluster_stats = CP.get_sig_clusters(data, 90)    
    
    # Set Empty p and cluster inds in case no significant clusters
    if cluster_stats is None:
        cluster_inds = []
        pvalues = []
    
    elif len(cluster_stats) > 0: #len
    
        # get cluster inds
        cluster_inds = find_cluster_intervals(cluster_masks)
    
        # get pvalues
        nulls = CP.nulls
        percentiles = stats.percentileofscore(nulls,abs(cluster_stats))
        pvalues = 1 - percentiles/100
        
    return model.tstats, pvalues, cluster_inds, cluster_forming_threshold

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
    ax.plot(x,est,color = color_mean, linestyle = linestyle , linewidth = 2.5)
    ax.margins(x=0)
    
    # Make Axes pretty
    ax.set_xlim([2, 45])
    ax.set_ylim([0,0.05]) # 6 = .041, 8= .063, 10 = .06
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

# --------------------------------------------------------------------------

# Dirs
psd_dir = '.../data/static/stn/bcc/psd'
spect_dir = '.../data/static/stn/bcc/stn_ctx'
demo_dir = '.../data/demographics'
plot_dir = '.../results/static/stn/bcc/spectra_condition_contrast/'

# Load Group Data
df = pd.read_csv( f'{demo_dir}/behavioural_all.csv')
mask = (df["Session" ].values < 3) & (df['withinMed'] == 1)
condition  = df["Session"].values[mask] # 1 = Off; 2 = On

#%% Set Up Figure

# Create a figure
fig = plt.figure(figsize=(10, 6), dpi=600)

# Add three subplots with manually specified positions [left, bottom, width, height]
ax2 = fig.add_axes([0.05, 0.035, 0.35, 0.4]) # Subplot 2 #
ax1 = fig.add_axes([0.1, 0.55, 0.55, 0.4])  # Subplot 1

#%% PSD Plot

# Load PSDs
f = np.load(f'{psd_dir}/stn_f.npy')
psd = np.load(f'{psd_dir}/stn_psd.npy')

# Mask for Freq range of interest
mask = np.logical_and(f >= 5, f <= 45)
f = f[mask]

# Calculate Difference between conditions
diff = psd[condition == 2] - psd[condition == 1]
diff = diff.mean(axis=1)

diff = diff[:,mask]

# --- Cluster-based Permutation test ---

# set covariates
covariates = {}  

# Cluster-Based Permutation GLMs
ts, ps, cluster_inds, thresh = tc_ClusterPermutation_test(diff,  
                                                          covariates)

# # Get peak frequency and mean t across cluster
mean_t = ts[:,cluster_inds[2][0]:cluster_inds[2][1]].mean()
peak_freq = f[np.argmax(abs(ts))]

print(f'Cluster p: {ps}')
print(f'Cluster meanT: {mean_t}')
print(f'Cluster peak freq: {peak_freq}')
        

# --- Plot Grand Average ---

# Prepare Data
off = psd[condition==1].mean(axis = 1)[:,mask]
on = psd[condition==2].mean(axis = 1)[:,mask]

# Group Colors
col =  [ '#586F8C','#9CBCD9']
    
# Make Plot
#fig, ax = plt.subplots(dpi=300)
tsplot(ax1, off, on, time=f, color_data=col[0], color_mean=col[1])
ax1.set_xlim([4.5, 45])

custom_lines = [Line2D([0], [0], color=col[0], lw=4),
                Line2D([0], [0], color=col[1], lw=4),]

ax1.set_ylabel('Relative Power (a.u.)', fontsize=20)
ax1.legend(custom_lines, ['Off-Medication', 'On-Medication'],
          frameon=False,
          fontsize=14,
          handletextpad=0.5,
          labelspacing=0.3)

# Mark Significant Clusters
for i_c, c in enumerate(cluster_inds):
    if ps[i_c] <= 0.05:
        # h = ax.axvspan(f[c[0]], f[c[1] - 1],
        #                   color='r', alpha=0.3)
        ax1.plot((f[c[0]], f[c[1]] - 1), (.022, .022), color='grey', linewidth=3)

#%% Coherence Plot

# Set motor Parcels
sma_inds = [6,32]
motor_labels = ['SMA']

# Load Files and grab STN-CTX Coh
f = np.load(f"{spect_dir}/f.npy")
coh = np.load(f"{spect_dir}/coh.npy")[:,-2:,sma_inds]

# Mask for Freq range of interest
mask = np.logical_and(f >= 3, f <= 45)
f = f[mask]

# Grab Motor Parcels and average ipsilateral coherence for both hemispheres
gc_impaired = np.stack([coh[:,0,0], coh[:,1,1]]).mean(axis=0)

gc_impaired = gc_impaired[:,mask]

# --- Cluster-based Permutation test ---

# Get diff between hemispjeres
diff = gc_impaired[condition==1] - gc_impaired[condition==2]

# set covariates
covariates = {}  

# Cluster-Based Permutation GLMs
ts, ps, cluster_inds, thresh = tc_ClusterPermutation_test(diff,  
                                                          covariates)    

print(ps)

# --- Plot Grand Average ---
col =  [ '#586F8C','#9CBCD9']
    
# Make Plot
#fig, ax = plt.subplots(dpi=300)
tsplot(ax2, gc_impaired[condition==1], gc_impaired[condition==2], time=f, color_data=col[0], color_mean=col[1])
ax2.set_ylim([0.012,0.065])

custom_lines = [Line2D([0], [0], color=col[0], lw=4),
                Line2D([0], [0], color=col[1], lw=4),]

ax2.set_ylabel('STN-SMA\nCoherence', fontsize=20)

# Mark Significant Clusters -- This needs fixing!!!!
ymax = np.hstack([gc_impaired[condition==1].mean(axis=0),gc_impaired[condition==2].mean(axis=0)])
yhight = np.max(ymax) * 1.1

for i_c, cl in enumerate(cluster_inds):
    if ps[i_c] <= 0.05:
        ax2.plot((f[cl[0]], f[cl[1]] - 1), (yhight, yhight), color='grey', linewidth=3)

#Save Figure
plt.savefig(f"{plot_dir}/Figure1_coh_sma_STN_mean_HQ.svg",
            transparent = True,bbox_inches="tight",format="svg")
