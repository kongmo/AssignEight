import numpy as np

def normalizeRatings(Y,R):
    m=Y.shape[0]
    Ymean=np.zeros((m,1))
    Ynorm=np.zeros(Y.shape)
    for i in range(m):
        idx=np.where(R[i,:]==1)
        Ymean[i]=Y[i,idx].mean()
        Ynorm[i,idx]=Y[i,idx]-Ymean[i]
    return (Ynorm,Ymean)
