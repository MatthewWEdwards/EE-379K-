from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy as sp
import numpy as np
import seaborn as sb
import xgboost as xgb

n = 20 # number of points
X1 = np.array(datasets.make_gaussian_quantiles([0],.5,n,1,1)[0])
X2 = np.array(datasets.make_gaussian_quantiles([0],.5,n,1,1)[0])
X3 = np.array(datasets.make_gaussian_quantiles([0],.7,n,1,1)[0])
label_1 = np.concatenate((X1, X2, X3),1) 
print "label_1 covariance matrix"
print np.cov(label_1, rowvar=False)

X1 = np.array(datasets.make_gaussian_quantiles([1],.5,n,1,1)[0])
X2 = np.array(datasets.make_gaussian_quantiles([1],.5,n,1,1)[0])
X3 = np.array(datasets.make_gaussian_quantiles([1],.01,n,1,1)[0])
label_2 = np.concatenate((X1, X2, X3),1) 
print "label_2 covariance matrix"
print np.cov(label_2, rowvar=False)

plot = plt.figure().gca(projection='3d')
plot.set_xlim([-2,2])
plot.set_ylim([-2,2])
plot.set_zlim([-2,2])
plot.scatter(label_1[:,0], label_1[:,1], label_1[:,2], c='r')
plot.scatter(label_2[:,0], label_2[:,1], label_2[:,2], c='b')
plt.legend(['Label 1', 'Label 2'])
plt.show()

# Concatenate data, find covariance matrix
cat_data = np.concatenate((label_1, label_2), 0)

# Find the mean vector
sum = np.array([0, 0, 0])
for i in range(0, len(cat_data)):
    sum = sum + (cat_data[i])
mean = np.divide(sum, np.array([2*n, 2*n, 2*n]))

# Find the covariance matrix
cov_mat = np.array([[0, 0, 0], [0, 0, 0], [0,0,0]])
for i in range(0, len(cat_data)):
    var_vect_line = np.array([cat_data[i][0] - mean[0], cat_data[i][1] - mean[1], cat_data[i][2] - mean[2]])
    var_vect_col = np.array([[cat_data[i][0] - mean[0]], [cat_data[i][1] - mean[1]], [cat_data[i][2] - mean[2]]], dtype=float).T
    cov_mat = cov_mat + np.outer(var_vect_line, var_vect_col)
cov_mat = cov_mat / len(cat_data)

print mean
print cov_mat

