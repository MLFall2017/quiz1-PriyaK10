# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:46:40 2017

@author: priya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv('C:\\Users\priya\Downloads\dataset_q1.csv')

#converting the dataframe into numphy array
data2=data1.iloc[:].values

data2[:,0]
# calculating variance of each columns 
var_x=np.var(data2[:,0])
var_x # 0.080529305883999994

var_y=np.var(data2[:,1])
var_y # 2.096902591519

var_z=np.var(data2[:,2])
var_z  #0.080501954878999998

# by formula for column x
mean_valx=data2[:,0].mean()
varf_x=sum((data2[:,0]-mean_valx)**2)/len(data2)
## 0.080529305883999994

mean_valy=data2[:,1].mean()
varf_y=sum((data2[:,1]-mean_valy)**2)/len(data2)
# 2.096902591519

mean_valz=data2[:,2].mean()
varf_z=sum((data2[:,2]-mean_valz)**2)/len(data2)
#0.080501954878999998

# covariance between x y 
x=data2[:,0]
y=data2[:,1]

cov_xy=np.cov(x,y)
cov_xy   # 0.40242878

# covariance between y and z
z=data2[:,2]
cov_yz=np.cov(y,z)
cov_yz  # -0.01439466

#PCA

#Standardirising the data ; axis=0 along columns
X_stnd=data2-np.mean(data2, axis=0)
#Covariance Calculation
cov_mat=np.cov(X_stnd.T)

# eigen vector and eigen value
evals, evecs = np.linalg.eig(cov_mat)

# forming a list of eigen values and its corresponding eigen vectors
e_pairs=[(np.abs(evals[i]),evecs[:,i]) for i in range(len(evals))]

e_pairs.sort()
e_pairs.reverse()

# bar plot to check the explained variance

tot = sum(evals)
var_exp = [(i / tot)*100 for i in sorted(evals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar((1,2,3), var_exp)
# selcecting PC1 and PC2 as axes
 # forming the tranformation matrix P which can diag A
p1=e_pairs[0][1].reshape(3,1)
p2=e_pairs[1][1].reshape(3,1)
pmat=np.hstack((p1,p2))

#Projection Onto the New Feature Space
Y = X_stnd.dot(pmat)
Y
# plotting 
#scatter plot
plt.scatter(x=Y[:,0], y=Y[:,1], alpha=0.5, edgecolors='green')


# eign vector and eigen values

A=np.mat('0, -1; 2,3')
eignval,eignvector=np.linalg.eig(A)
eignval
eignvector


