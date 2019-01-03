# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:00:56 2018

Creating a 2MM MNI mask

1. Create a subject-composite mask (intersection of all subject-brain spaces)
2. Resample MNI 152 T1 image to the subject mask
3. ???
4. Profit

@author: ixa080020
"""

import os
import nilearn
import nibabel as nib
from nilearn import masking
from nilearn.image import resample_to_img
nilearn.EXPAND_PATH_WILDCARDS = False

import proj_utils as pu
pdir = pu._get_proj_dir_v2()
scanDir = os.path.join(pdir,"hcp_extracted_scan_data/REST1_scans/denoised_python")

#scanDir = r"/petastore/ganymede/Project/clint/denoised_python"
scan = [os.path.join(scanDir,f) for f in os.listdir(scanDir) if f.endswith(".nii")][0]

brainMask = "mni152T1mask.nii"
mni2mm = resample_to_img(brainMask,os.path.join(scanDir,scan))

dataFor2mm = mni2mm.get_data()
dataFor2mm[dataFor2mm<1] = 0
dataFor2mm[dataFor2mm>1] = 1

mni2mmZeroed = nib.Nifti1Image(dataFor2mm, mni2mm.affine)

nib.save(mni2mmZeroed,"mni152T1mask2mm.nii")