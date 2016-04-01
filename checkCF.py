import numpy as np
import CCF
import CNG
from numpy import linalg

def checkCostFunction(Lambda=None):
    if Lambda==None:
        Lambda=0

    X_t=np.random.random((4,3))
    Theta_t=np.random.random((5,3))

    Y=np.dot(X_t,Theta_t.transpose())
    Y[np.random.random(Y.shape)>0.5]=0
    R=np.zeros(Y.shape)
    R[np.where(Y != 0)]=1
    shape=X_t.shape
    X=np.random.randn(shape[0],shape[1])
    shape=Theta_t.shape
    Theta=np.random.randn(shape[0],shape[1])
    num_users=Y.shape[1]
    num_movies=Y.shape[0]
    num_features=Theta_t.shape[1]

    params=np.hstack((X.flatten(),Theta.flatten()))
    func=lambda x: CCF.cofiCostFunc(x,Y,R,num_users,num_movies,num_features,Lambda)
    numgrad=CNG.computeNumericalGradient(func,params)

    res=CCF.cofiCostFunc(params,Y,R,num_users,num_movies,num_features,Lambda)

    grad=res[1]
    m=grad.size
    for i in range(m):
        print '%6.3f   %6.3f' % (grad[i],numgrad[i])

    diff=linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)

    print '''If your backpropagation implementation is correct, then\nthe relative difference will be small (less than 1e-9).'''
    print 'Relative Difference: %g' % diff
