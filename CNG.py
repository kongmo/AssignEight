import numpy as np

def computeNumericalGradient(func,params):
    numgrad=np.zeros(params.shape)
    perturb=np.zeros(params.shape)

    e=1e-4
    m=params.size
    for i in range(m):
        perturb[i]=e
        loss1=func(params-perturb)[0]
        loss2=func(params+perturb)[0]
        numgrad[i]=(loss2-loss1)/(2.0*e)
        perturb[i]=0
    return numgrad
    
