import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate points
label_1 = np.random.multivariate_normal([0,0,0], [[1,.9,.9],[.9,1,.9],[.9,.9,1]], size=20)
label_2 = np.random.multivariate_normal([0,0,1], [[1,.8,.8],[.8,1,.8],[.8,.8,1]], size=20)

# plot random points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(label_1[:,0], label_1[:,1], label_1[:,2],c='r',marker='o') 
ax.scatter(label_2[:,0], label_2[:,1], label_2[:,2],c='b',marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Linear Discriminant  Analysis
# Transform the points
mean_1 = np.mean(label_1, axis=0)
mean_2 = np.mean(label_2, axis=0)

cov_1 = np.array([[0,0,0],[0,0,0],[0,0,0]])
for vec in label_1:
	cov_1 = cov_1 + np.outer(vec - mean_1, vec - mean_1)

cov_2 = np.array([[0,0,0],[0,0,0],[0,0,0]])
for vec in label_2:
	cov_2 = cov_2 + np.outer(vec - mean_2, vec - mean_2)
	
s_within = cov_1 + cov_2
s_between = np.outer(mean_1 - mean_2, mean_1 - mean_2)

w_max = np.linalg.inv(s_within)*(mean_1 - mean_2) / np.linalg.norm(np.linalg.inv(s_within)*(mean_1 - mean_2)) 
for vec in range(0, len(w_max[:])):
	w_max[:, vec] = w_max[:, vec]/np.linalg.norm(w_max[:, vec])

label_1_transformed = np.dot(label_1, w_max)
label_2_transformed = np.dot(label_2, w_max)

# plot the transformation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(label_1_transformed[:,0], label_1_transformed[:,1], label_1_transformed[:,2],c='r',marker='o') 
ax.scatter(label_2_transformed[:,0], label_2_transformed[:,1], label_2_transformed[:,2],c='b',marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Project the tranformed points
w_lda = np.dot(np.linalg.inv(s_within), mean_1 - mean_2)
w_lda = w_lda / np.linalg.norm(w_lda)

label_1_projected = np.dot(label_1_transformed, w_lda)
label_2_projected = np.dot(label_2_transformed, w_lda)

# plot the transformed points
plt.plot(label_1_projected, np.zeros_like(label_1_projected), 'o', c='r')
plt.plot(label_2_projected, np.zeros_like(label_2_projected), 'o',  c='b')
plt.show()

# LDA using sklearn
from sklearn.lda import LDA
points = np.concatenate((label_1, label_2), axis=0)
label = np.concatenate((np.zeros((20, 1)), np.zeros((20,1)) + 1))

lda = LDA()
skl_transform = lda.fit_transform(points, label)

f, ax = plt.subplots(1,2)

ax[0].plot(skl_transform[:20], np.zeros_like(label_1_projected), 'o', c='r')
ax[0].plot(skl_transform[20:], np.zeros_like(label_2_projected), 'o',  c='b')
ax[0].set_title('SKlearn Projection')
ax[1].plot(label_1_projected, np.zeros_like(label_1_projected), 'o', c='r')
ax[1].plot(label_2_projected, np.zeros_like(label_2_projected), 'o',  c='b')
ax[1].set_title('Manual Projection')
plt.show()



