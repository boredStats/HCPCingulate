# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:22:33 2018

2nd level analysis for cingulate voxel-voxel analysis

@author: ixa080020
"""

import os
import numpy as np
import h5py
import re
import datetime
from scipy.stats import ttest_1samp

#--- Funtions for getting column data ---#
def chunk_getter(maxcol, chunk_size=1000):
    """
    Calculate number of chunks to divide x_cols into
    Default chunk_size is 1000 variables per chunk
    """
    chunks = 1
    while(maxcol/chunks) > chunk_size:
        chunks += 1
    return chunks

def colrange_getter(maxcol, chunk, chunk_size=1000):
    """
    Get the range of x_cols to grab
    """
    colrange = range(chunk*chunk_size, (chunk + 1)*chunk_size)
    if max(colrange) >= maxcol:
        colrange = range(chunk*chunk_size, maxcol)
    return colrange

#--- Functions for doing natural sorting ---#
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

with open("secondLevPathsPrivate.txt") as f:
    paths = ["r"+k.replace("\n", "") for k in f]
    
chunkDir =  paths[0]
dataDir = paths[1]
outDir = paths[2]

subjects = os.listdir(dataDir)
aal2CingMask = sorted(os.listdir(chunkDir))[0] #Edit cingulum chunk here!!!
maskName = aal2CingMask.replace(".nii.gz","")

print("%s Getting metadata for %s" % (datetime.datetime.now(), maskName))
hf = h5py.File(os.path.join(dataDir,subjects[0]), 'r')
keynames = list(hf.keys())
keynames.sort(key=alphanum_key)
brainSize = len(hf[keynames[0]][:,0])

maxColsInMask = int(keynames[-1].split()[-1]) #Last key in key names, last value in key
hf.close()

print("%s Running ttests for %s" % (datetime.datetime.now(), maskName))

tmaskBrainData = np.ndarray(shape=[maxColsInMask, brainSize])
pmaskBrainData = np.ndarray(shape=[maxColsInMask, brainSize])

chunkSize = 10 #doing ttests on small number of voxels at a time

#vox = 0
for key in keynames:
    f = "%s_%s_secondLev.hdf5" % (maskName, key)
    fname = os.path.join(outDir, f)
    if os.path.isfile(fname):
        continue
    else:
        file = h5py.File(fname, "a")
        file.close()
        
    a = int(key.split()[-1])
    b = int(key.split()[-3])
    
    if (a - b + 1) == 1000:
        maxColsInKey = 1000
    else:
        maxColsInKey = int(a - b)
        #int(key.split()[-1])
    chunks = chunk_getter(maxColsInKey, chunkSize)
    vox = 0
    for chunk in range(chunks):
        colrange = colrange_getter(maxColsInKey, chunk, chunkSize)
        #--- Running ttests on each voxel of a chunk of hdf5 dataset ---#
        for voxel in list(colrange):
            #--- initialize second level matrix ---#
            secondLevelData = np.ndarray(shape=[len(subjects), brainSize])
            for s,subj in enumerate(subjects):
                subjFile = os.path.join(dataDir, subj)
                hf = h5py.File(subjFile, 'r')
                data = hf[key][:, int(voxel)]
                hf.close()
                
                np.nan_to_num(data, copy=False)
                data[data>=1.0] = 1 - 1e-6
                data[data<=-1.0] = -1 + 1e-6

                zData = np.arctanh(data) #fisher transform

                np.nan_to_num(zData, copy=False)
                secondLevelData[s, :] = zData 
                del data

            #--- whole brain t-test ---#    
            popmean = np.zeros(shape=[1, brainSize])

            tBrain, pBrain = ttest_1samp(secondLevelData, popmean, axis=0)

            tmaskBrainData[vox, :] = tBrain
            pmaskBrainData[vox, :] = pBrain
            del secondLevelData
            vox += 1
    
    file = h5py.File(fname, "a")
    file.create_dataset("tBrains",
                        shape=tmaskBrainData.shape,
                        dtype="f",
                        data=tmaskBrainData)
    file.create_dataset("pBrains",
                        shape=pmaskBrainData.shape,
                        dtype="f",
                        data=pmaskBrainData)
    file.close()
    
    del tmaskBrainData
    del pmaskBrainData