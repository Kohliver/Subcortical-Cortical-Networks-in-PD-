#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:38:59 2025

@author: okohl


SubRsn Project - State-specific STN Power 3
    Load State-specific STN power and state x STN beta burst overlaps, calculate
    stats and plot.
"""

import numpy as np
from osl_dynamics.analysis import spectral, power, connectivity
import pandas as pd 
from scipy.stats import percentileofscore
import glmtools as glm
from scipy import stats
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

#%% --- Define a few functions

def get_subject_level_covs(subject_ids):
    
    # Unique subject IDs in the dataset
    unique_ids = np.unique(subject_ids)
    
    # Create dummy variables
    dummy_dict = {}
    for uid in unique_ids:
        dummy_vector = (subject_ids == uid).astype(int)
        dummy_dict[f"sub{uid}"] = dummy_vector
        
    return dummy_dict

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
        DC.add_regressor(name=name, rtype="Parametric", datainfo=name, preproc="None")
    
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
        cluster_forming_threshold = get_cluster_forming_threshold(model.dof_error)
      
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
def within_glms(data, covariates={}, metric='tstats',pooled_dims=(1),n_jobs=4):

    # Define Dataset for GLM    
    data = glm.data.TrialGLMData(data=data,
                                 **covariates)
                                 
    # Specify regressors and Contrasts in GLM Model 
    DC = glm.design.DesignConfig()
    DC.add_regressor(name='Constant',rtype='Constant')
    for name in covariates:
        DC.add_regressor(name=name, rtype="Parametric", datainfo=name, preproc="None")
    
    DC.add_contrast(name="On > Off", values=[1] + [0] * len(covariates))
    
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
    
    return metrics, pvalues, model.dof_error, thresh, mask


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
    ax.set_ylim([0,0.16]) # 6 = .041, 8= .063, 10 = .06
    ax.set_xlabel('Frequency (Hz)',fontsize=18, labelpad=12)
    ax.set_ylabel('Power (a.u.)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14) 
    ax.ticklabel_format(scilimits=(-1,1))

    # Set x-Ticks
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter('{x:.0f}')    
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    # Set y-ticks
    ax.yaxis.set_major_locator(MultipleLocator(.05))
    #ax.yaxis.set_major_formatter('{x:.0f}')    
    ax.yaxis.set_minor_locator(MultipleLocator(.025))

    # Despine
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    



#%% Prepare Plotting 

K = 8

# Dir where data is
spect_dir = '.../data/hmm/post_hoc/8_states_norm/stn_psd'
hmm_dir = '.../data/hmm/post_hoc/8_states_norm/inf'
overlap_dir = ".../data/hmm/post_hoc/8_states_norm/overlap/beta"
demo_dir = '.../data/demographics'
out_dir = '.../results/hmm/post_hoc/8_states_norm/stn_psd' 

# Load Group Data
df = pd.read_csv( f'{demo_dir}/behavioural_all.csv')
mask = (df["Session" ].values < 3) & (df['withinMed'] == 1) 
condition  = df["Session"].values[mask] # 1 = Off; 2 = On
subject_ids = df["SubNo"].values[mask]

#%% Set up plot

# Set Colors
greys = ['#757575','#9e9e9e']
col = ["#313B48", "#8D8B88", "#9B9692"]
colors =  plt.cm.tab20( np.arange(20).astype(int) )
cols1 = [colors[2],colors[6]]
cols2 = [colors[2],colors[3],colors[6],colors[7]]


# --- Set up Overview Plot ---
fig = plt.figure(dpi=600,figsize=(10,8))
gs = fig.add_gridspec(2, 2,wspace=.3, hspace=.6)
ax = np.zeros(K,dtype=object)
ax[0] = fig.add_subplot(gs[0, 0])
ax[1] = fig.add_subplot(gs[0, 1])
ax[2] = fig.add_subplot(gs[1, 0])
ax[3] = fig.add_subplot(gs[1, 1])


#%% --- Load State Specific STN Beta Power --- 

# Load Files
f = np.load(f"{spect_dir}/f.npy")
psd = np.load(f"{spect_dir}/psd.npy")
w = np.load(f"{spect_dir}/w.npy")
fo = np.load(f"{hmm_dir}/fo.npy")

# Grab PSD data of impaired hemisphere
p = psd.mean(axis=2)

# Subtract mean across State
mean_p = np.empty([50,50])
for iSub in range(p.shape[0]):
    mean_p[iSub] = np.average(p[iSub], axis=0, weights=fo[iSub])
    p[iSub] = p[iSub] - mean_p[iSub]
         
# Add Mean and State specific spectra
comb_p = p + mean_p[:,np.newaxis]

comb_p = comb_p[condition==1]
mean_p = mean_p[condition==1]
p = p[condition==1]

# Filter to range of interest
freq = f[f<=30]
p = p[:,:,f<=30]

#%% Plot State-specific vs. mean STN power (off condition)

# Loop Across States
for ind, iK in enumerate([0,7]):

    # --- Cluster-based Permutation test ---
       
    # set covariates
    covariates = {}
    
    # Cluster-Based Permutation GLMs
    ts, ps, cluster_inds, thresh = tc_ClusterPermutation_test(p[:,iK],  
                                                              covariates)
    
    # --- Plotting ----
    
    tsplot(ax[ind], comb_p[:,iK], mean_p[:], time=f, color_data=cols1[ind], color_mean=col[2])
    
    custom_lines = [Line2D([0], [0], color=cols1[1], lw=4.5),
                    Line2D([0], [0], color=col[1], lw=4),]
    
    if ind == 0:
        ax[ind].set_ylabel('Relative Power (a.u.)', fontsize=18,labelpad=8)
    else:
        ax[ind].set_ylabel('')
        
    #Mark Significant Clusters
    concatenated_spectra = np.hstack([comb_p[:,iK].mean(axis=0),mean_p.mean(axis=0)])
    y = np.max(concatenated_spectra) * 1.2
    
    for i_c, cl in enumerate(cluster_inds):
        if ps[i_c] <= (0.05/8):
            ax[ind].plot((f[cl[0]], f[cl[1]] - 1), (y, y), color=greys[1], linewidth=3.5)
   
        

  
#%% Load Empirically observed overlaps and prepare for plotting

# Load data of Off Med Condition
empirical = np.load(f'{overlap_dir}/normalised_overlap_stateNormCont.npy')[:25] * 100

# Calculate GLMs with max Tstatistik Permutation Tests
covariates = {}

# Get Differences and merge
diff1 = empirical[:,0] - empirical[:,1]
diff2 = empirical[:,2] - empirical[:,3]
diff = np.vstack([diff1,diff2]).T

# Compare against 0
ts, ps, _, _, _ = within_glms(diff, covariates)

#  Loop across state 1 and 6
for ind, iK in enumerate([0,2]):

    # Data Frame For Stats
    df = pd.DataFrame({
        "Participant": np.tile(np.arange(1, len(empirical)+1), 2),
        "State": np.repeat(["State 1", "Other States"], len(empirical)),
        "Overlap": np.hstack([empirical[:,iK], empirical[:,iK+1]])
    })

    # --- Add Burst x State Overlap Figure ----
        
    # Add plots
    bplot = sns.boxplot(x="State", y='Overlap', data=df, 
                        palette=['white','white'], width=.7, legend=False, 
                        showfliers=False,ax = ax[ind+2])
    points = sns.stripplot(x="State", y='Overlap', data=df, 
                  palette=[cols2[ind]], size=6, 
                  legend=False, ax = ax[ind+2]) 
    
    # adding transparency to colors
    for i, patch in enumerate(bplot.patches):
     patch.set_edgecolor(greys[1])
    
    # adding transparency to colors
    for i, patch in enumerate(points.collections):
        if ind + 2 == 2:
            patch.set_facecolor(cols2[i])
        else:
            patch.set_facecolor(cols2[i+2])
     
    # Make Axis pretty
    #ax[ind+2].xaxis.labelpad = 10
    ax[ind+2].set_ylim([0,30])
    ax[ind+2].set_xlabel('', fontsize=16) 
    ax[ind+2].locator_params(axis="y", nbins=5)
    ax[ind+2].tick_params(axis='both', which='major', labelsize=14) 
    
    if ind + 2 == 2:
        ax[ind+2].set_xticklabels(['State 1', 'Other States'], fontsize=14)
    else:
        ax[ind+2].set_xticklabels(['State 6', 'Other States'], fontsize=14)
    
    
    if ind + 2 == 2:
        ax[ind+2].set_ylabel('Overlap (%)', fontsize = 18, labelpad = 8)
    else:
        ax[ind+2].set_ylabel('')
    
    # Remove Box Around Subplot
    sns.despine(ax=ax[ind+2] , top=True, right=True, left=False,
            bottom=False, offset=None, trim=False)
    
    
    # Significance of Group Contrast Peri vs Hc
    metric = df['Overlap'].values
    iCond = .45
    
    # Get Significance stars 
    if ps[ind] < .01:      
        p = '**'  
        x = iCond
        
        y = np.max(metric) * 1.05 
        ax[ind+2].text(x=x , y=y , s=p , zorder=10, size=20)
        ax[ind+2].plot([0, 1], [y, y], 'k-', lw=1) 
    elif ps[ind] < .05:
        p = "*"   
        x = iCond
        
        y = np.max(metric) * 1.05 
        ax[ind+2].text(x=x , y=y , s=p , zorder=10, size=20)
        ax[ind+2].plot([0, 1], [y, y], 'k-', lw=1) 
    else:
        p = ""
    
# # Save Figure
plt.savefig( f"{out_dir}/Figure4_stateSpecific_STN_power_noLegend_mean_off_HQ.svg",
            transparent = True,bbox_inches="tight",format="svg")            

    
   
 