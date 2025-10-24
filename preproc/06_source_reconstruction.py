"""Source reconstruction.

Sub-RSN Project - Preprocessing 6
    This includes beamforming, parcellation and orthogonalisation.

    Note, before this script is run the /coreg directory created by 3_coregister.py
    must be copied and renamed to /src.
"""

from dask.distributed import Client
from osl import source_recon, utils
import os
from glob import glob

# Directories
preproc_dir = ".../data/preproc_ssp"
coreg_dir = ".../data/correg"
src_dir = ".../data/src"

os.makedirs(coreg_dir, exist_ok=True)

# Files
preproc_files = []
for condition in ['peri','HC']:
    preproc_files.extend(sorted(glob(f"{preproc_dir}/*/*{condition}*_preproc_raw.fif")))

# Get Sub IDs
subjects = []
for file in preproc_files:
        base = os.path.basename(file).split('_')
        subject = f'{base[0]}_{base[1]}_{base[2]}'
        subjects.append(subject)

# Settings
config = """
    source_recon:
    - forward_model:
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 60}
        parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Copy directory containing the coregistration
    if not os.path.exists(src_dir):
        cmd = f"cp -r {coreg_dir} {src_dir}"
        print(cmd)
        os.system(cmd)

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=2, threads_per_worker=1)

    # Source reconstruction
    source_recon.run_src_batch(
        config,
        outdir=src_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        dask_client=True,
    )
