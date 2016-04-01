# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:17:04 2016

@author: aa
"""
import numpy as np
from numpy import linalg

def multivariateGaussian(X,mu,sigmav):
    k=len(mu)
    if len(sigmav.shape) == 1:
        sigmav=np.diag(sigmav) 
    X=X-mu

    ex=np.exp(-0.5*np.sum(np.dot(X,linalg.pinv(sigmav))*X,axis=1))
    
    p=(2*np.pi)**(-k/2.0)*linalg.det(sigmav)**(-0.5)*ex
    return p
