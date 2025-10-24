"""Coregisteration.

Created on Mon Jul 15 14:22:16 2024

@author: okohl

Sub-RSN Project - Preprocessing 5
    Run Coregistration of MEG and MRI Scans

    Note, these scripts do not include/use the nose in the coregistration.
    dcm2nii import performed with functions implemented in SPM.
    
    First coregistration is run for all participants with good MRI scans.
    For participants in which coregt with original T1 did not yield acceptable
    results (N=8), coreg was run with template t1 scans.
    
"""

from glob import glob
import os
from dask.distributed import Client

from osl import source_recon, utils

# Directories
preproc_dir = ".../data/preproc_ssp"
anat_dir = ".../raw_data/nifti"
coreg_dir = ".../data/correg"

os.makedirs(coreg_dir, exist_ok=True)

# Subs for which corregistration needs to be run in template brains
template_subs = ['S004','S009','S010','S011','S012','S013','S014','S103']

# Set up Files
# Meg Files
# preproc_files = sorted(glob(f"{preproc_dir}/*/*_preproc_raw.fif"))

preproc_files = []
for condition in ['HC','peri']:
    preproc_files.extend(sorted(glob(f"{preproc_dir}/*/*{condition}*_preproc_raw.fif")))


# Remove subs for which brain template is used - because we need to allow MRI scaling for them
for ind, file in reversed(list(enumerate(preproc_files))):
    sub = file.split('/')[-2]
    if any(sub in f'sub-{out_sub}' for out_sub in template_subs):
        del preproc_files[ind]
   
#  Select T1 Files
subjects = []
smri_files = []
for file in preproc_files:
        base = os.path.basename(file).split('_')
        subject = f'{base[0]}_{base[1]}_{base[2]}'
        subjects.append(subject)
        sub = subject.split('-')[1].split('_')[0]
        smri_files.append(f"{anat_dir}/{sub}/anat_t1.nii")     

# Settings
config = """
    source_recon:
    - extract_polhemus_from_info: {}
    - remove_stray_headshape_points: {}
    - compute_surfaces:
        include_nose: False
    - coregister:
        use_nose: False
        use_headshape: True
        allow_smri_scaling: False
        n_init: 1
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")


    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=2, threads_per_worker=1)

    # Run coregistration
    source_recon.run_src_batch(
        config,
        outdir=coreg_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
    

#%% Now run coregistrations for participants whos MRI scans are erroranous & do not allow for good coregistration
# Instead of participants T1 scans a template T1 scan is used for coreg.    

# Grab meg files
preproc_files = []
for sub in template_subs:
    preproc_files.extend(sorted(glob(f"{preproc_dir}/sub-{sub}/*_preproc_raw.fif")))

#  T1 Files
subjects = []
smri_files = []
for file in preproc_files:
        base = os.path.basename(file).split('_')
        subject = f'{base[0]}_{base[1]}_{base[2]}'
        subjects.append(subject)
        smri_files.append(".../osl/osl/source_recon/parcellation/files/MNI152_T1_2mm_brain.nii.gz")

# Settings
config = """
    source_recon:
    - extract_polhemus_from_info: {}
    - remove_stray_headshape_points: {}
    - compute_surfaces:
        include_nose: False
    - coregister:
        use_nose: False
        use_headshape: True
        allow_smri_scaling: True
        n_init: 1
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")


    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=2, threads_per_worker=1)

    # Run coregistration
    source_recon.run_src_batch(
        config,
        outdir=coreg_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
    
