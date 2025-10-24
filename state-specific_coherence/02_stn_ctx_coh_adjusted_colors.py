#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:46:50 2024

@author: okohl

SubRsn Project - STN-Cortical Coherence
    Plot state-specific STN-Cortical coherence.
    Assess whether it is significantly different from time-averaged STN-cortical coherence
    with cluster-permutation tests.

"""

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting
import glmtools as glm
from scipy import stats
import pandas as pd


# ---------------------------------------------------------

# Gret Cluster forming threshold from model degrees of freedome
def get_cluster_forming_threshold(dof_error, alpha=0.05):
    return stats.t.ppf(1 - alpha/2, dof_error)
  

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
                               nPerm=1000,
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
    #des.plot_summary(savepath='/home/esther/Research/sub_rsn/results/static/stn/spectra_condition_contrast/contrast_summary.png')
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


# run glms with max tstatistic/copes permutation tests
def condition_contrast_glms(data, metric='tstats',pooled_dims=(1),n_jobs=4):

    # Define Dataset for GLM    
    data = glm.data.TrialGLMData(data=data)
                                 
    # Specify regressors and Contrasts in GLM Model 
    DC = glm.design.DesignConfig()
    DC.add_regressor(name='Constant',rtype='Constant')
    
    DC.add_contrast(name="On > Off", values=[1])
    
    # Create design martix and fit model and grab tstats 
    des = DC.design_from_datainfo(data.info)
    model = glm.fit.OLSModel(des,data)
     
    # Permutation Test Pooling Across States 
    perm = glm.permutations.MaxStatPermutation(des, data, 0, nperms=10000,
                                            metric=metric, nprocesses=n_jobs,
                                            pooled_dims=pooled_dims)
       
    thresh = perm.get_thresh([95])
     
    # Get p-values and significance mask
    if pooled_dims:
        if metric == "tstats":
            metrics = model.tstats[0]
            percentiles = stats.percentileofscore(perm.nulls, abs(metrics))            
            mask = abs(metrics) > thresh[0]
        elif metric == "copes":
            metrics = model.copes[0]
            percentiles = stats.percentileofscore(perm.nulls, abs(metrics))
            mask = abs(metrics) > thresh[0]
        pvalues = 1 - percentiles / 100 
    else:
        if metric == "tstats":
            metrics = model.tstats[0]
            percentiles = [stats.percentileofscore(perm.nulls[:,i], abs(metrics[i])) for i in range(perm.nulls.shape[1])] 
            percentiles = np.array(percentiles)    
            mask = abs(metrics) > thresh[0]
        elif metric == "copes":
            metrics = model.copes[0]
            percentiles = [stats.percentileofscore(perm.nulls[:,i], abs(metrics[i])) for i in range(perm.nulls.shape[1])] 
            percentiles = np.array(percentiles)  
            mask = abs(metrics) > thresh[0]
        pvalues = 1 - percentiles / 100
    
    return metrics, pvalues, model.dof_model, thresh, mask

def tsplot(ax, data, mean_data,time, color_data = 'blue', color_mean = 'k', linestyle = 'solid', linewidth = 2):
    x = time
    
    # Data Line
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    se = sd/np.sqrt(len(data))
    cis = (est - se, est + se)
    ax.fill_between(x,cis[0],cis[1],alpha = 0.2, facecolor = color_data)
    ax.plot(x,est,color = color_data, linestyle = linestyle, linewidth = 3.5)
    ax.margins(x=0)
    
    # Mean Data Line
    est = np.mean(mean_data, axis=0)
    sd = np.std(mean_data, axis=0)
    se = sd/np.sqrt(len(mean_data))
    cis = (est - se, est + se)
    ax.fill_between(x,cis[0],cis[1],alpha = 0.2, facecolor = color_mean)
    ax.plot(x,est,color = color_mean, linestyle = linestyle , linewidth = 3)
    ax.margins(x=0)
    
    # Make Axes pretty
    ax.set_xlim([5, 30])
    ax.set_ylim([0.015,0.08]) # 6 = .041, 8= .063, 10 = .06
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

# --------------------------------------------------------------------------------

# Dir where data is
spect_dir = '.../data/hmm/post_hoc/8_states_norm/stn_coh/bcc'
hmm_dir = '.../data/hmm/post_hoc/8_states_norm/inf'
out_dir = '.../results/hmm/post_hoc/8_states_norm/stn_ctx_coh/bcc'
surf_dir = '.../results/hmm/post_hoc/8_states_norm/stn_ctx_coh/bcc/surf'
demo_dir = '.../data/demographics'

# Load Group Data
df = pd.read_csv( f'{demo_dir}/behavioural_all.csv')
mask = (df["Session" ].values < 3) & (df['withinMed'] == 1) 
condition  = df["Session"].values[mask] # 1 = Off; 2 = On

# Load Files
f = np.load(f"{spect_dir}/f.npy")
psd = np.load(f"{spect_dir}/psd.npy")
coh = np.load(f"{spect_dir}/coh.npy")
w = np.load(f"{spect_dir}/w.npy")
fo = np.load(f"{hmm_dir}/fo.npy")

f_mask = np.logical_and(f >=5, f <= 30)
f = f[f_mask]

# Grab STN Channels Cortex Coh
c = coh[condition==1][:,:,-2:,:-2].squeeze()
 
# Subtract mean across State
mean_c = np.empty([c.shape[0],c.shape[2],c.shape[3],c.shape[4]])
for iSub in range(c.shape[0]):
    mean_c[iSub] = np.average(c[iSub], axis=0, weights=fo[iSub])
    c[iSub] = c[iSub] - mean_c[iSub]


# Group Colors
col = ["#313B48", "#8D8B88", "#9B9692"]
colors =  plt.cm.tab20( np.arange(20).astype(int) )
cols1 = [colors[2],colors[6]]
 
lab = 'SMA'
inds = [6,32] # parcel idx of sma parcels

# Grab Motor Parcels and average across them as well as STN channels  - only consider ipsilateral connections   
gc = np.stack([c[:25,:,0,inds[0]],c[:25,:,1,inds[1]]]).mean(axis=(0))
mean_gc = np.stack([mean_c[:25,0,inds[0]],mean_c[:25,1,inds[1]]]).mean(axis=(0))
    
# Add Mean and State specific spectra
comb_gc = gc + mean_gc[:,np.newaxis]

# Overview Plot
fig = plt.figure(dpi=600,figsize=(16,8))
gs = fig.add_gridspec(2, 4)
ax = np.zeros(8,dtype=object)
ax[0] = fig.add_subplot(gs[0, 0])
ax[1] = fig.add_subplot(gs[0, 1])
ax[2] = fig.add_subplot(gs[0, 2])
ax[3] = fig.add_subplot(gs[0, 3])
ax[4] = fig.add_subplot(gs[1, 0])
ax[5] = fig.add_subplot(gs[1, 1])
ax[6] = fig.add_subplot(gs[1, 2])
ax[7] = fig.add_subplot(gs[1, 3])

for iK in range(gc.shape[1]):
    
    # --- Cluster-based Permutation test ---
    
    # set covariates
    covariates = {}  
    
    # Cluster-Based Permutation GLMs
    ts, ps, cluster_inds, thresh = tc_ClusterPermutation_test(gc[:,iK, f_mask],  
                                                              covariates)

    # --- Plotting ----
    
    if iK == 0:
        tsplot(ax[iK], comb_gc[:,iK,f_mask], mean_gc[:,f_mask], time=f, color_data=cols1[0], color_mean=col[2])
    elif iK == 5:
        tsplot(ax[iK], comb_gc[:,iK,f_mask], mean_gc[:,f_mask], time=f, color_data=cols1[1], color_mean=col[2])
    else:
        tsplot(ax[iK], comb_gc[:,iK,f_mask], mean_gc[:,f_mask], time=f, color_data=col[0], color_mean=col[2])
    
    if lab == 'Auditory Association':
        ax[iK].set_ylim([0.015,0.085])
    
    custom_lines = [Line2D([0], [0], color=col[0], lw=4.5),
                    Line2D([0], [0], color=col[1], lw=4),]
    
    if iK == 0 or iK == 4:
        ax[iK].set_ylabel(f'STN-{lab}\nCoherence', fontsize=20,labelpad=8)
    else:
        ax[iK].set_ylabel('')
        
    if iK < 4:
        ax[iK].set_xlabel('')
        
    if iK == 7:
        ax[iK].legend(custom_lines, ['State-specific Coherence', 'Mean Coherence'],
                  frameon=False,
                  fontsize=12,
                  handletextpad=0.5,
                  labelspacing=0.3)
        
    # Mark Significant Clusters
    concatenated_spectra = np.hstack([comb_gc[:,iK].mean(axis=0),mean_gc.mean(axis=0)])
    if (lab == 'SMA') and (iK == 0):
        y = np.max(concatenated_spectra)
    else:
        y = np.max(concatenated_spectra) * 1.2
    
    for i_c, cl in enumerate(cluster_inds):
        if ps[i_c] < (0.05/8):
            ax[iK].plot((f[cl[0]], f[cl[1]] - 1), (y, y), color='grey', linewidth=3.5)
   
# #Save Figure
plt.savefig(f"{out_dir}/coh_{lab}_STN_overview_bonf_adjustedColors_mean_off_HQ.svg",
            transparent = True,bbox_inches="tight",format="svg")
