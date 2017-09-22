# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:32:31 2017

@author: priya
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset
dataset = pd.read_csv('C:\\Users\priya\Downloads\dataset_1.csv')
#indexing column X from dataset
datasetx=dataset.iloc[:,0].values
datasetx
# calculating by the formula np.var
var_x=np.var(datasetx)
var_y=np.var(dataset.iloc[:,0]) 
#varx_x=np.var(datasetx)
#vard=np.var(dataset.iloc[:,0])
 # 0.080

# by the formula
mean_val=dataset.iloc[:,0].mean()
sum((datasetx-mean_val)**2)/len(dataset)
datasetx.mean
ss=dataset.iloc[:,0]
#sum((dataset.iloc[:,0]-mean_val)**2)/len(dataset)
dataset.iloc[:,0]

# PCA

from sklearn.preprocessing import StandardScaler
dataset1=dataset
sc_X = StandardScaler()
dataset1= sc_X.fit_transform(dataset1)

#
dataset3=dataset
mean_x=dataset3.iloc[:,0].values.mean()
mean_y=dataset3.iloc[:,1].values.mean()
mean_z=dataset3.iloc[:,2].values.mean()



dataset3.iloc[:,0]=dataset3.iloc[:,0].values-mean_x
dataset3.iloc[:,1]=dataset3.iloc[:,1].values-mean_y
dataset3.iloc[:,2]=dataset3.iloc[:,2].values-mean_z



print('NumPy covariance matrix: \n%s' %np.cov(dataset3.T))
cov_matx=np.cov(dataset3.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matx)
eignvalues=eig_vals
eigen_vec=eig_vecs

#cv=np.cov(dataset3.iloc[:].values.T)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('Everything ok!')


# sort the eigen values in descending form: forming a list
epairs=[(np.abs(eignvalues[i]),eigen_vec[:,i]) for i in range(len(eignvalues))]
    
    
epairs.sort()
epairs.reverse()

for ep in epairs:
    print ep[0]
    
    # forming the tranformation matrix P which can diag A
 pc1=epairs[0] [1].reshape(3,1)
 pc2=epairs[1][1].reshape(3,1)
d=np.hstack((pc1,pc2))
d
d[7]
# Y=XP
Y=np.dot(dataset3.iloc[:].values,d)
Y
Y1=np.dot(dataset3,d)
Y1
y2=dataset3.iloc[:].values.dot(d)
