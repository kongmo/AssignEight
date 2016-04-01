# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:55:33 2016

@author: aa
"""
import scipy.io as sio
import matplotlib.pyplot as plt
import EG
import MG
import VF
import ST
import numpy as np

#Part One: Load Example Dataset
print 'One: ========== Visualizing Example Dataset for Outlier Detection...'
data=sio.loadmat('ex8data1')
X=data['X']
plt.plot(X[:,0],X[:,1],'bx')
plt.axis(xmin=0,xmax=30,ymin=0,ymax=30)
plt.xlabel('Latency (ms)')

plt.ylabel('Throughput (mb/s)')
plt.title('The First Dataset')
plt.show()

#Part Two: Estimate the Dataset Statistics
print 'Two: ============== Estimate the Dataset Statistics ...'
result=EG.estimateGaussian(X)
mu=result[0]
sigmav=result[1]
p=MG.multivariateGaussian(X,mu,sigmav)
VF.visualizeFit(X,mu,sigmav)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('The Gaussian distribution contours of the distribution fit to the dataset.')
plt.axis(xmin=0,xmax=30,ymin=0,ymax=30)
#plt.show()

#Part Three: Find Outliers
print 'Three: ============ Find Outliers...'
Xval=data['Xval']
yval=data['yval']
pval=MG.multivariateGaussian(Xval,mu,sigmav)
res=ST.selectThreshold(yval,pval)
epsilon=res[0]
F1=res[1]
print 'Best epsilon found using cross-validataion: %8.2e' % epsilon
print 'Best F1 on Cross Validation Set: %5.3f ' % F1
print '(you should see a value epsilon of about 8.99e-05)'

outliners=np.where(p < epsilon)
plt.plot(X[outliners,0],X[outliners,1],'ro',linewidth=2,markersize=10)
plt.show()

#Part Four: Multidimensional Outliers
print 'Four: =========== Multidimensional Outliers...'
data=sio.loadmat('ex8data2')
X=data['X']
Xval=data['Xval']
yval=data['yval']
res=EG.estimateGaussian(X)
mu=res[0]
sigmav=res[1]
p=MG.multivariateGaussian(X,mu,sigmav)
pval=MG.multivariateGaussian(Xval,mu,sigmav)
result=ST.selectThreshold(yval,pval)
epsilon=result[0]
F1=result[1]
print 'Best epsilon found using cross-validation : %8.3e' % epsilon
print 'Best F1 on Cross Validation Set: %f ' % F1
print 'Outliers found: %d ' % (p < epsilon).sum()
print '(you should see a value epsilon of about 1.38e-18)'
