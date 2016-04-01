# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:04:04 2016

@author: aa
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import CCF
import checkCF
import NR
import scipy.optimize as sop
from scipy.optimize import fmin

data=sio.loadmat('ex8_movies')
Y=data['Y']
R=data['R']
#Part One: Loading Movie Ratings Dataset
##print 'One: ============== Loading movie ratings dataset...'
##tmp=Y[0,np.where(R[0,:] !=0)]
##print 'Average rating for movie 1 (Toy Story): %f' % tmp.mean()
##print '(This value should be about 3.878319)'
#plt.imshow(Y)
#plt.xlabel('Users')
#plt.ylabel('Movies')

#Part Two: Collaborative Filtering Cost Function
print 'Two: =================== Collaborative Filtering Cost Function...'
data=sio.loadmat('ex8_movieParams')
X=data['X']
Theta=data['Theta']
num_users=4
num_movies=5
num_features=3
X=X[0:num_movies,0:num_features]
Theta=Theta[0:num_users,0:num_features]
Y=Y[0:num_movies,0:num_users]
R=R[0:num_movies,0:num_users]

##params=np.hstack((X.flatten(),Theta.flatten()))
##J=CCF.cofiCostFunc(params,Y,R,num_users,num_movies,num_features,0)
##print 'Cost at loaded parameters: %f ' % J[0]
##print '(This value should be about 22.22)'


#Part Three: collaborative Filtering Gradient
print 'Three: =============== Checking Gradients (without regularization)...'

##checkCF.checkCostFunction()


#Part Four: Collaborative Filtering Cost Regularization
##print 'Four: ==================== Collaborative Filtering Cost Regularization...'
##J=CCF.cofiCostFunc(params,Y,R,num_users,num_movies,num_features,1.5)
##print 'Cost at loaded parameters (lambda=1.5): %f ' % J[0]
##print '(This value should be about 31.34)'

#Part Five: Collaborative Filtering Gradient Regularization
##print 'Five: ================== Checking Gradient (with regularization)...'
##checkCF.checkCostFunction(1.5)


#Part Six: Entering Ratings for a New User
print 'Six: ============== Entering Ratings for a New User...'
movieList=[]
fid=open('movie_ids.txt','r')
data=fid.readlines()
pos=data[0].find(' ')
for line in data:
    pos=line.find(' ')
    movieList.append(line[pos:-1])
fid.close()

my_ratings=np.zeros((1682,1))

my_ratings[0]=4
my_ratings[97]=2
my_ratings[6]=3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print 'New user ratings: '
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
        print 'Rated %d for %s' % (my_ratings[i],movieList[i])


#Part Seven: Learning Movie Ratings
print 'Seven: ============= Learning Movie Ratings...'
print 'Training collaborative filtering'
data=sio.loadmat('ex8_movies')
Y=data['Y']
R=data['R']

Y=np.hstack((my_ratings,Y))
R=np.hstack((np.int32(my_ratings !=0),R))
result=NR.normalizeRatings(Y,R)
Ynorm=result[0]
Ymean=result[1]
num_users=Y.shape[1]
num_movies=Y.shape[0]
num_features=10


X=np.random.randn(num_movies,num_features)
Theta=np.random.randn(num_users,num_features)

initial_params=np.hstack((X.flatten(),Theta.flatten()))
Lambda=10
func=lambda x:CCF.cofiCostFunc(x,Y,R,num_users,num_movies,num_features,Lambda)

theta=sop.minimize(func,initial_params,method='Newton-CG',jac=True)

theta=theta['x']
X=theta[0:num_movies*num_features].reshape(num_movies,num_features)
Theta=theta[num_movies*num_features:].reshape(num_users,num_features)
print 'Recommender system learning completed'

#Part Eight: Recommendation for you
print 'Eight: ======= Recommendation for You...'
p=np.dot(X,Theta.transpose())
my_prediction=p[:,0]+Ymean.flatten()

pos=my_prediction.size
zips=zip(my_prediction,range(pos))

rates=np.array(zips,dtype=[('x',float),('y',int)])
rates.sort(order='x')
rates=rates[::-1]
ix=rates['y']
for i in range(10):
    j=ix[i]
    print 'Predicting rating %.1f for movie %s' % (my_prediction[j],movieList[j])
    
print 'Original Ratings provided:'
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print 'Rated %d for %s ' % (my_ratings[i],movieList[i])












