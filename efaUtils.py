# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:56:52 2018

Utilities for exploratory factor analysis
"""
import pylab
import numpy as np
import sklearn.decomposition as decomp
import matplotlib.pyplot as plt
from sklearn.utils import resample
from matplotlib.colors import hsv_to_rgb
from adjustText import adjust_text #gg_repel for python
from nilearn.plotting import plot_connectome
from copy import deepcopy

def centerMat(data):
    #Remove means from each column in a matrix
    data -= np.mean(data, axis=0)
    return data

def permPCA(data, nIters=1000):
    """
    Permutation tests for eigenvalues
    
    Parameters
    ----------
    data : numpy array
        N x M array where N = # of observations and M = # of variables
    
    nIters : int
        Optional. Recommend high value for better power
        
    Returns
    -------
    eigs : numpy array
        A vector of eigenvalues
    
    p: numy array
        A vector of p-values corresponding to the eigen values. 
        
        p-values are calculated as:
            P(eig_perm > eig_obs)+1 / niters+1 
        see Phipson & Smyth 2010 for more information
    """
    permData = deepcopy(data)
    pca = decomp.PCA(n_components=min(data.shape), svd_solver='full')
    pca.fit(data)
    eigs = pca.explained_variance_
    
    permEigs = []
    ns=0
    while ns != nIters:
        ns += 1
        #Permuting columns
        for i in range(permData.shape[1]):
            np.random.shuffle(permData[:, i])
        pca.fit(permData)
        permEigs.append(pca.explained_variance_)
    permArray = np.asarray(permEigs)
    
    p = []
    for j in range(permArray.shape[1]):
        nHit = np.where(permArray[:, j] >= eigs[j])
        pval = (len(nHit[0]) + 1)/ (ns+1)
        p.append(pval)
    return eigs, p

def plotScree(eigenvalues, eigenPvals=None, kaiser=False, fname=None):
    """
    Create a scree plot for factor analysis using matplotlib
    
    Parameters
    ----------
    eigenvalues : numpy array
        A vector of eigenvalues
    
    eigenPvals : numpy array
        A vector of p-values corresponding to a permutation test
    
    kaiser : bool
        Plot the Kaiser criterion on the scree 
        
    Optional
    --------
    fname : filepath
        filepath for saving the image
        
    Returns
    -------
    fig, ax, ax2 : matplotib objects
        Figure, axes handles for the scree plot
    """
    
    percentVar = (np.multiply(100, eigenvalues)) / np.sum(eigenvalues)
    cumulativeVar = np.zeros(shape=[len(percentVar)])
    c = 0
    for i,p in enumerate(percentVar):
        c = c+p
        cumulativeVar[i] = c
    
    fig,ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Scree plot", fontsize='xx-large')
    ax.plot(np.arange(1,len(percentVar)+1), eigenvalues, '-k')
    ax.set_ylim([0,(max(eigenvalues)*1.2)])
    ax.set_ylabel('Eigenvalues', fontsize='xx-large')
    ax.set_xlabel('Factors', fontsize='xx-large')
#    ax.set_xticklabels(fontsize='xx-large') #TO-DO: make tick labels bigger
    
    ax2 = ax.twinx()
    ax2.plot(np.arange(1,len(percentVar)+1), percentVar,'ok')
    ax2.set_ylim(0,max(percentVar)*1.2)
    ax2.set_ylabel('Percentage of variance explained', fontsize='xx-large')

    if eigenPvals is not None and len(eigenPvals) == len(eigenvalues):
        #TO-DO: add p<.05 legend?
        pvalueCheck = [i for i,t in enumerate(eigenPvals) if t<.05]
        eigenCheck = [e for i,e in enumerate(eigenvalues) for j in pvalueCheck if i==j]
        ax.plot(np.add(pvalueCheck,1), eigenCheck, 'ob', markersize=10)
    
    if kaiser:
        ax.axhline(1, color='k', linestyle=':', linewidth=2)
    
#    cumulative=False #TO-DO: Check cumulative variance aesthetics
#    if cumulative:    
#        ax3 = ax.twinx()
#        ax3.bar(np.arange(1, len(percentVar)+1), cumulativeVar, width=1.0, alpha=.2)
#        ax3.set_yticklabels([])
#        ax3.set_yticks([])

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    return fig, ax, ax2

def plotLoadings(fpair, flabel=[1,2], colors=None, text=None, fname=None):
    load1 = fpair[:, 0]
    load2 = fpair[:, 1]
    fig,ax = plt.subplots(figsize=(10,10))
    ax.set_title("Factor Loadings", fontsize='xx-large')
    ax.scatter(load1, load2, c=colors)
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    ax.set_xlabel('Factor %d' % int(flabel[0]), fontsize='x-large')
    ax.set_ylabel('Factor %d' % int(flabel[1]), fontsize='x-large')
    
    if text is not None and len(text) == len(load1):
        texts = []
        for i,s in enumerate(text):
            texts.append(ax.text(load1[i], load2[i], s, fontsize=10))
        adjust_text(texts)
        
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    return fig, ax

def createColorSpace(loadingsPair):
    factorOne = loadingsPair[:,0]
    factorTwo = loadingsPair[:,1]
    R = np.sqrt(factorOne**2 + factorTwo**2)
    theta = np.arctan2(factorTwo, factorOne)
    
    theta[theta<0] = 2*np.pi + theta[theta<0]
    hue = theta/(2*np.pi)
    val = R
    
    hsv = np.ones([len(factorOne),3])
    hsv[:,0] = hue
    hsv[:,2] = val
    
    rgb = np.zeros([len(factorOne),3])
    rgb = hsv_to_rgb(hsv)
    
    return R, theta, rgb

def createColorLegend(R, theta, fname=None):
    figsize = 5
    #labs example
    #labs = ['+ F1','','+ F2','','- F1','','- F2','']

    fig = plt.figure(figsize=(figsize, figsize))
    fig.patch.set_facecolor('w')
    ax = fig.add_subplot(111, projection='polar')
    plt.ylim(0, 1)
    
    res = 100   # Colormesh resolution. 100 suggested.
    
    theta2, R2 = np.meshgrid(
    np.linspace(2*np.pi, 0, res),
    np.linspace(0, 1, res))
    
    t,r = np.meshgrid(
        np.linspace(1, 0, res-1),
        np.linspace(0, 1, res))
    
    image = np.dstack((t, np.ones_like(r), r))
    
    color = image.reshape((image.shape[0]*image.shape[1],image.shape[2]))
    color = hsv_to_rgb(color)
    
    p = ax.pcolormesh(theta2, R2, np.zeros_like(R2), color=color)
    ax.set_xticklabels(['+ F1','','+ F2','','- F1','','- F2',''])
    ax.tick_params(axis='x',colors='k')
    p.set_array(None)
    
    plt.grid(True,c='w')
    ax.scatter(theta, R, c='w',s=75, edgecolors='none', alpha=.75)
    
    if fname is not None:
        plt.savefig(fname, dpi=300, facecolor='w')

def plotBrains(nodeCoords, nodeColors, fname=None):
    numc = len(nodeColors)
    adjacency_mat = np.zeros(shape=(numc, numc))
    
    brain_fig = plt.figure(figsize=[24,12])
    plot_connectome(
            adjacency_matrix=adjacency_mat,
            node_coords=nodeCoords,
            node_color=nodeColors,
            display_mode='lr',
            node_size=900,
            figure=brain_fig,
            output_file=fname,
            black_bg=False)

def normalizeR(x, mi=0, mx=1):
    def nfunc(i, mi, mx):
        return (i - mi) / (mx - mi)
    j = [nfunc(i, mi, mx) for i in x]
    return np.asarray(j)

def convertToCmap(x, mi=0, mx=1, cmap='seismic'):
    y = normalizeR(x, mi, mx)
    cm = pylab.get_cmap(cmap)
    return [cm(i) for i in y]

def assignColPair(cpair, x):
    rgb = [cpair[0]] * len(x)
    for i in range(len(x)):
        if x[i] < 0:
            rgb[i] = cpair[1]
    return np.asarray(rgb)

def rgbaGen(rgbT, x):
    rgba = np.ndarray(shape=[len(x), 4])
    y = np.abs(x)
    for i in range(rgbT.shape[1]):
        rgba[:, i] = rgbT[:, i]
    rgba[:, 3] = y
    return rgba
    
def generateLoadColors(cpair, x):
    rgb = assignColPair(cpair, x)
    return rgbaGen(rgb, x)

### Needs testing, do not use
def _contributions(data, eigs, nFactors):
    fad = decomp.FactorAnalysis(n_components=nFactors)
    
    numObs = data.shape[0]
    scores = fad.fit_transform(data)
    sqscores = scores**2
    w = 1/numObs #assumes sklearn uses equal masses for each observation
    # See Abdi & Williams 2010 [PCA], Abdi, Williams, & Valentin 2013 [MFA]
    scoresWeighted = sqscores * w
    
    eigRep = np.repeat(eigs[:nFactors], numObs)
    eigRes = np.reshape(eigRep, scores.shape, order="F")
    return scoresWeighted / eigRes

def _boostrapcontributions(data, eigs, nFactors, nIters=100):
    fad = decomp.FactorAnalysis(n_components=nFactors)
    
    clist = []
    ns = 0
    while ns != nIters:
        ns += 1
        bootSample = resample(data)
        scores = fad.fit_transform(bootSample)
        scoresWeighted = (scores ** 2) * (1/data.shape[0])
        
        eigRep = np.reshape(np.repeat(eigs[:nFactors], data.shape[0]), scores.shape, order="F")
        
        cont = scoresWeighted / eigRep
        clist.append(cont)
    return clist

def _plotCont(data,labels=None,colors=None,fname=None,meanCheck=True):
    if colors is None:
        colors = ['tab:blue'] * len(data)
    if meanCheck==True:
        boolCheck = np.full(data.shape, True, dtype='bool')
        mu = np.mean(abs(data))
        for i,d in enumerate(data):
            if d < mu:
                colors[i]= "tab:grey"
        boolCheck[data < mu] = False
    
    fig, ax = plt.subplots(figsize=(10,10))

    x = np.arange(1, len(data)+1)
    ax.bar(x, data, color=colors)    
    ax.axhline(0, color='k') #origin line
    ax.set_xticks(np.arange(1, len(data) + 1))
    if labels is not None:
        ax.set_xticklabels(labels, rotation=90, fontsize=14)
#    ax.set_ylabel("Column loadings", fontsize=16)
#    ax.set_title("Column loadings for Dimension %d" % factorNum, fontsize='xx-large')
    
    if meanCheck:    
        ax.axhline(mu, color='k', linestyle=':')
        if np.any([data<0]):
            ax.axhline(-mu, color='k', linestyle=':')
#        greyIndices = [i for i,x in enumerate(boolCheck) if x == False]
#        for i in greyIndices:
#            ax.get_xticklabels()[i].set_color('tab:grey')
        
    if fname:
        fig.savefig(fname, bbox_inches='tight')
        return fig, ax
    return fig, ax, mu