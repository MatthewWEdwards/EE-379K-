#1.1

import numpy as np

#generating samples
mean1 = np.array([0,0,0])
cov1 = np.array([[.5,0,0],[0,.5,0],[0,0,.7]])
x1,y1,z1 = np.random.multivariate_normal(mean1, cov1, 20).T

mean2 = np.array([1,1,1])
cov2 = np.array([[.5,0,0],[0,.5,0],[0,0,.01]])
x2,y2,z2 = np.random.multivariate_normal(mean2, cov2, 20).T


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plot the samples with markers
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1,y1,z1,c='r',marker='o') #label data points in the first set with a circle
ax.scatter(x2,y2,z2,c='b',marker='^') #label data points in the second set with a triangle

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()



#1.2
#concatenate all samples
x = np.concatenate((x1, x2), axis = 0)
y = np.concatenate((y1, y2), axis = 0)
z = np.concatenate((z1, z2), axis = 0)

#get each entry of the covariance matrix using the formula
m11 = 0
m12 = 0
m13 = 0
m21 = 0
m22 = 0
m23 = 0
m31 = 0
m32 = 0
m33 = 0
for i in range(0, len(x)):
    m11 += float(x[i] - np.mean(x)) * (x[i] - np.mean(x))
    m12 += float(x[i] - np.mean(x)) * (y[i] - np.mean(y))
    m13 += float(x[i] - np.mean(x)) * (z[i] - np.mean(z))
    m21 += float(y[i] - np.mean(y)) * (x[i] - np.mean(x))
    m22 += float(y[i] - np.mean(y)) * (y[i] - np.mean(y))
    m23 += float(y[i] - np.mean(y)) * (z[i] - np.mean(z))
    m31 += float(z[i] - np.mean(z)) * (x[i] - np.mean(x))
    m32 += float(z[i] - np.mean(z)) * (y[i] - np.mean(y))
    m33 += float(z[i] - np.mean(z)) * (z[i] - np.mean(z))
    
cov_mat = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])/(len(x)-1)

print cov_mat


#get the 2 eigenvectors with the 2 greatest eigen values
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_vec1, eig_vec2 = eig_vecs[0], eig_vecs[1]
print eig_vec1, '\n', eig_vec2


#dot the two vectors with the samples to get a projection
vecs = np.array([eig_vec1, eig_vec2])
all_samples = np.array([x,y,z])
result = vecs.dot(all_samples)

#show a scatter plot
plt.scatter(result[0][:20],result[1][:20], c='red', label = 'data set 1')
plt.scatter(result[0][20:],result[1][20:], c='blue', label = 'data set 2')
plt.show()