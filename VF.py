# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:18:14 2016

@author: aa
"""
import numpy as np
import MG
import matplotlib.pyplot as plt

def visualizeFit(X,mu,sigmav):
    datas=np.linspace(0,35,num=71)
    x=np.meshgrid(datas,datas)
    X1=x[0]
    X2=x[1]
    x=np.vstack((X1.flatten(),X2.flatten())).transpose()
    Z=MG.multivariateGaussian(x,mu,sigmav)
    Z=Z.reshape(X1.shape)
    plt.plot(X[:,0],X[:,1],'bx')
    
    flag=(np.isinf(Z)).sum()

    pos=np.linspace(-20,0,num=7)
    lev=10**pos
    if ~flag:
        plt.contour(X1,X2,Z,levels=lev)
