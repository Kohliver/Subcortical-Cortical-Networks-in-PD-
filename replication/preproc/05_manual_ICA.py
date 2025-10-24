#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:24:14 2025

@author: okohl

Rassoulou Replicttion - Preproc 5

Manual ICA run for participants for which no good ECG was recorded because of what
automated SSP eye and heart beat event detection performed poorly.
"""

import os
import matplotlib.pyplot as plt

from osl import preprocessing, utils
from dask.distributed import Client

get_ipython().run_line_magic("matplotlib", "qt")

def check_ica(raw, ica, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Find EOG and ECG correlations
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
    ica_scores = ecg_scores + eog_scores

    # Barplot of ICA component "EOG match" and "ECG match" scores
    ica.plot_scores(ica_scores)
    plt.savefig(f"{save_dir}/correl_plot.png", bbox_inches="tight")
    plt.close()

    # Plot bad components
    ica.plot_components(ica.exclude)
    plt.savefig(f"{save_dir}/bad_components.png", bbox_inches="tight")
    plt.close()

def plot_psd(raw, save_dir):
    raw.compute_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4)).plot()
    plt.savefig(f"{save_dir}/powspec.png", bbox_inches="tight")
    plt.close()

#%% Setup paths

# Directories
preproc_dir = ".../Rassoulou2024/preproc"
output_dir = ".../Rassoulou2024/preproc_ica"

# Subjects
subjects = ['sub-PuPVlx_ses-PeriOp_task-MRest_acq-MedOn_run-1_raw_tsss',
            'sub-PuPVlx_ses-PeriOp_task-HRest_acq-MedOff_run-1_split-01_raw_tsss',
            'sub-PuPVlx_ses-PeriOp_task-HRest_acq-MedOn_run-1_split-01_raw_tsss',
            'sub-PuPVlx_ses-PeriOp_task-MRest_acq-MedOff_run-1_split-01_raw_tsss',
            'sub-dCsWjQ_ses-PeriOp_task-HRest_acq-MedOff_run-1_raw_tsss',
            'sub-dCsWjQ_ses-PeriOp_task-HRest_acq-MedOn_run-1_split-01_tsss',
            'sub-dCsWjQ_ses-PeriOp_task-MRest_acq-MedOff_run-1_raw_tsss',
            'sub-dCsWjQ_ses-PeriOp_task-MRest_acq-MedOn_run-1_raw_tsss',
            'sub-gNX5yb_ses-PeriOp_task-HRest_acq-MedOff_run-1_tsss',
            'sub-gNX5yb_ses-PeriOp_task-HRest_acq-MedOn_run-1_raw_tsss',
            'sub-gNX5yb_ses-PeriOp_task-MRest_acq-MedOff_run-1_split-01_raw_tsss',
            'sub-gNX5yb_ses-PeriOp_task-MRest_acq-MedOn_run-1_split-01_tsss',
            'sub-AB2PeX_ses-PeriOp_task-HRest_acq-MedOff_run-1_raw_tsss',
            'sub-AB2PeX_ses-PeriOp_task-HRest_acq-MedOn_run-1_raw_tsss',
            'sub-AB2PeX_ses-PeriOp_task-MRest_acq-MedOff_run-1_split-01_raw_tsss',
            'sub-AB2PeX_ses-PeriOp_task-MRest_acq-MedOn_run-1_raw_tsss',
            'sub-BYJoWR_ses-PeriOp_task-HRest_acq-MedOff_run-1_split-01_raw_tsss',
            'sub-BYJoWR_ses-PeriOp_task-HRest_acq-MedOn_run-1_split-01_raw_tsss',
            'sub-BYJoWR_ses-PeriOp_task-MRest_acq-MedOff_run-1_raw_tsss',
            ]

# Paths to files
preproc_files = []
automatic_raw_files = []
automatic_ica_files = []
manual_raw_files = []
manual_ica_files = []
report_dirs = []
for subject in subjects:
    fname = subject.replace('_raw','')    
    fname = f'{fname}_preproc-raw'
    
    preproc_files.append(f"{preproc_dir}/{subject}/{fname}.fif")
    automatic_raw_files.append(f"{output_dir}/automatic/{fname}/{fname}.fif")
    automatic_ica_files.append(f"{output_dir}/automatic/{fname}/{fname}-ica.fif")
    manual_raw_files.append(f"{output_dir}/manual/{fname}/{fname}_preproc-raw.fif")
    manual_ica_files.append(f"{output_dir}/manual/{fname}/{fname}-ica.fif")
    report_dirs.append(f"{output_dir}/manual/report/{fname}")

#%% Automate ICA

# Settings
config = """
    preproc:
    - ica_raw: {picks: meg, n_components: 40}
    - ica_autoreject: {apply: False}
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=8, threads_per_worker=1)

    # Main preprocessing
    preprocessing.run_proc_batch(
        config,
        preproc_files,
        outdir=f'{output_dir}/automatic/',
        overwrite=True,
        dask_client=True,
    )


#%% Manual ICA ICA artefact rejection

# Index for the preprocessed data file we want to do ICA for
index = 0
subject = subjects[index]
print("Doing", subject)

# Files for the corresponding index
preproc_file = automatic_raw_files[index]
output_raw_file = manual_raw_files[index]
output_ica_file = manual_ica_files[index]
report_dir = report_dirs[index]

# Create output directories
os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Load preprocessed fif and ICA
dataset = preprocessing.read_dataset(preproc_file, preload=True)
raw = dataset["raw"]
ica = dataset["ica"]

# Mark bad ICA components interactively
preprocessing.plot_ica(ica, raw)

#%% Create figures to check ICA with
check_ica(raw, ica, report_dir)
plt.close("all")

# Apply ICA
print()
raw = ica.apply(raw, exclude=ica.exclude)

print()
print("Removed components:", ica.exclude)
print()

# Plot power spectrum of cleaned data
plot_psd(raw, report_dir)

#%% Save cleaned data
dataset["raw"].save(output_raw_file, overwrite=True)
dataset["ica"].save(output_ica_file, overwrite=True)