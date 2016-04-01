# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:39:22 2016

@author: aa
"""
import numpy as np

def selectThreshold(yval,pval):
    ma=pval.max()
    mi=pval.min()
    stepsize=(ma-mi)/1000.0
    m=np.arange(mi,ma+stepsize,stepsize)
    bestEpsilon=0
    bestF1=0  
    yval=yval.flatten()
    
    for epsilon in m:
        yp=pval < epsilon
        pos=np.where(yp == True)
        neg=np.where(yp == False)
        tp=(yval[pos]==1).sum()
        fp=(yval[pos]==0).sum()
        fn=(yval[neg]==1).sum()
        if (tp+fp) == 0 or (tp+fn)==0:
            continue
        prec=1.0*tp/(tp+fp)
        rec=1.0*tp/(tp+fn)
        F1=2.0*prec*rec/(prec+rec)        
        if F1 > bestF1:
            bestEpsilon=epsilon
            bestF1=F1
        
    return (bestEpsilon,bestF1)
        
        
    