# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:15:38 2016

@author: aa
"""
import numpy as np
def estimateGaussian(X):
    m=X.shape[0]
    mu=np.sum(X,axis=0)/m
    sigmav=np.sum((X-mu)**2,axis=0)/m
    return (mu,sigmav)
    
    