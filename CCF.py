# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:19:51 2016

@author: aa
"""
import numpy as np

def cofiCostFunc(params,Y,R,num_users,num_movies,num_features,Lambda):
    X=params[0:num_movies*num_features].reshape(num_movies,num_features)
    Theta=params[num_movies*num_features:].reshape(num_users,num_features)
    
    X_grad=np.zeros(X.shape)
    Theta_grad=np.zeros(Theta.shape)
    
    tmp=(np.dot(X,Theta.transpose())-Y)**2
    J=1.0/2*(R*tmp).sum()+Lambda/2.0*(Theta*Theta).sum()+Lambda/2.0*(X*X).sum()

    X_grad=np.dot((np.dot(X,Theta.transpose())-Y)*R,Theta)+Lambda*X
    Theta_grad=np.dot(((np.dot(X,Theta.transpose())-Y)*R).transpose(),X)+Lambda*Theta
    grad=np.hstack((X_grad.flatten(),Theta_grad.flatten()))
    return grad
