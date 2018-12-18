# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:22:48 2018

Utilites for hierarchical clustering analyses

@author: ixa080020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram as dendro
from nilearn.plotting import plot_connectome

def create_dendrogram_figs(h, d=None, leaves=None, thresh=None, fname=None):
    """
    Create custom dendrograms
    
    Parameters
    ----------
    hclust : scipy.cluster.linkage object
    
    d : string    
        String to describe distance metric used in linkage clustering
        When d = None, Euclidean distance is used    
    
    leaves : list 
        Default is None
        list of names to assign to each leaf in hierarchy
    
    threshold : float 
        Default is None
        Threshold used to assign leaves to a node
        When threshold = None, optimal clustering is automatically calculated
        
    fname : string
        Path to save dendrogram
    """
    
    if d is None:
        d = "Euclidean distance"
        
    dendroFig, dendroAx = plt.subplots(figsize=(12,20))
    dendro(h, orientation='left', labels=leaves, above_threshold_color='k',
            color_threshold=thresh, ax=dendroAx)
    dendroAx.axvline(x=thresh,color='k',linestyle=':')
    dendroAx.set_xlabel(d,fontsize=18)
    dendroAx.tick_params(axis='both',labelsize=14)
    plt.show()
    
    if fname is not None:
        dendroFig.savefig(fname, bbox_inches='tight')

def plot_brains(coords, colors, fname):
    plt.clf()
    adjMat = np.zeros(shape=(len(colors),len(colors)))
    
    bf = plt.figure(figsize=[24,20])
    plot_connectome(adjMat, coords, colors, display_mode='lr', node_size=900,
                    figure=bf, output_file=fname)
    
def save_sol(clust_labels,leaf_names,fname):
    save_df = pd.DataFrame(clust_labels,index=leaf_names)
    save_df.to_csv(fname, header=False)
    return save_df

def get_subcluster_labels(sol_df):
    sols = len(np.unique(sol_df.values))
    cluster_list = []
    for c in range(sols):
        cl = []
        for i,s in enumerate(list(sol_df.index)):
            if int(sol_df.values[i])==c+1:
                cl.append(s)
        cluster_list.append(cl)
    return cluster_list