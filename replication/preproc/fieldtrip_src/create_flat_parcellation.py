#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:23:04 2025

@author: okohl

Create Parcellation in flat label format.
This means each voxel contains number of parcel it belongs to as value.
This should enable usage with FieldTrip and niilearn.
I.e., when using MaskLabelAtlas etc.

"""

import numpy as np
from nilearn import image
import nibabel as nib

# Load Glasser52 Parcellation
parc_fname = '.../data/Glasser52/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz'
img = nib.load(parc_fname)

# Extract Data
parc = img.get_fdata() # get data
affine = img.affine  # get affine

# Check that no voxel is part of several parcels
if (parc.sum(axis=3) > 1).any():
    print('1 or more voxels are allocated to 2 or more parcels. This will cause issues further down the script.')
    
# Flatten Parcellation
flat_parc = np.zeros(parc.shape[:3])
for ind in range(52):
    flat_parc[parc[:,:,:,ind] == 1] = int(ind+1)

# Create a new Nifti image with the flat parcellation
new_img = nib.Nifti1Image(flat_parc, affine=affine)

# Save the new image
output_path = ".../Glasser52/flat_parcellation/Glasser52_flat-MNI152NLin6_res-8x8x8.nii.gz"
nib.save(new_img, output_path)

