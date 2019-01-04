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
import time
import random
from scipy.stats import ttest_1samp

def sleeper(tc=30):
    """Quick sleep function to prevent overlap of cluster jobs"""
    t = random.randint(1, 9)
    time.sleep(t*tc)

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

def secondLevel():
    with open("secondLevPathsPrivate.txt") as f:
        paths = [k.replace("\n", "") for k in f]
        
    chunkDir = paths[0]
    dataDir = paths[1]
    outDir = paths[2]
    
    allSubjects = os.listdir(dataDir)
    chunkSize = 10 #extracts "small" amounts of data at a time
    
    aalChunks = sorted(os.listdir(chunkDir))[1:]
    for aal2 in aalChunks:
        sleeper()
        
        maskName = aal2.replace(".nii.gz","")
        subjects = [s for s in allSubjects if maskName in s]
        
        hf = h5py.File(os.path.join(dataDir,subjects[0]), 'r')
        keynames = list(hf.keys())
        keynames.sort(key=alphanum_key)
        brainSize = len(hf[keynames[0]][:,0])
        hf.close()
    
        for key in keynames:
            sleeper()
            
            b = int(key.split()[-1])
            a = int(key.split()[-3])
            if (b - a + 1) == 1000:
                maxColsInKey = 1000
            else:
                maxColsInKey = int(b - a)    
        
            fname = "%s_cols_%d_to_%d.hdf5" % (maskName, a, b)
            fpath = os.path.join(outDir, fname)
            if os.path.isfile(fpath):
                continue
            else:
                file = h5py.File(fpath, "a")
                file.close()
                
            tmaskBrainData = np.ndarray(shape=[maxColsInKey, brainSize])
            pmaskBrainData = np.ndarray(shape=[maxColsInKey, brainSize])
            
            chunks = chunk_getter(maxColsInKey, chunkSize)
            vox = 0
            for chunk in range(chunks):
                colrange = colrange_getter(maxColsInKey, chunk, chunkSize)
                for voxel in list(colrange):
                    secondLev = np.ndarray(shape=[len(subjects), brainSize])
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
                        secondLev[s, :] = zData 
                        del data
        
                    #--- whole brain t-test ---#    
                    popmean = np.zeros(shape=[1, brainSize])
        
                    tBrain, pBrain = ttest_1samp(secondLev, popmean, axis=0)
        
                    tmaskBrainData[vox, :] = tBrain
                    pmaskBrainData[vox, :] = pBrain
                    del secondLev
                    vox += 1
            
            file = h5py.File(fpath, "a")
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
if __name__ == "__main__":
    secondLevel()