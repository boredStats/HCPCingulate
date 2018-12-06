# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:10:43 2018

@author: ixa080020
"""
import numpy as np

def convertToHSV(factorOne, factorTwo):
    r = np.sqrt(factorOne**2 + factorTwo**2)
    theta = np.arctan2(factorTwo, factorOne)
    
    hue = theta-np.min(theta)
    hue = hue/np.max(hue)
    sat = r/np.max(r)
    
    return hue, sat