import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

corr1 = pd.read_csv("./CorrMat1.csv", header=None).as_matrix()
corr3 = pd.read_csv("./CorrMat3.csv", header=None).as_matrix()

# CorrMat1's corrupt entries clearly have value 9999. Because CorrMat1 should 
# be symmetric, we can clean the matrix this way.

# Count corrupted entries
num_corrupt = 0
for i in range(0, len(corr1)):
    for j in range(0, len(corr1[0])):
        if corr1[i][j] == 9999:
            num_corrupt = num_corrupt + 1
print("")
print("Number originally corrupted: " + str(num_corrupt))

# Correct corruptions
for i in range(0, len(corr1)):
    for j in range(0, len(corr1[0])):
        if corr1[i][j] == 9999:
            if corr1[j][i] != 9999:
                corr1[i][j] = corr1[j][i]
                
# Count corrupted entries
num_corrupt = 0
for i in range(0, len(corr1)):
    for j in range(0, len(corr1[0])):
        if corr1[i][j] == 9999:
            num_corrupt = num_corrupt + 1
print("")
print("Number corrupted after correction: " + str(num_corrupt))

# Use sklearn to calculate the variance of the components
pca = PCA(n_components=5)
pca.fit(corr1)
print("")
print("CorrMat1 variance components (after leveraging symmetry):")
print(pca.explained_variance_ratio_)

# The variance of the first few components should be sufficient. I will use n = 5.
k = 5
corr1_u, corr1_s, corr1_v = np.linalg.svd(corr1)
corr1_s = np.concatenate([np.array(corr1_s[:k]), np.zeros(len(corr1_s[k:]))], axis=0)
corr1_low_rank = corr1_u.dot(np.diag(corr1_s)).dot(corr1_v)

# Count greatly changed entries
num_corrupt = 0
for i in range(0, len(corr1)):
    for j in range(0, len(corr1[0])):
        if np.sqrt((corr1[i][j] - corr1_low_rank[i][j])**2) > 100:
            num_corrupt = num_corrupt + 1
print("")
print("Number of entries changed in value by over 100: " + str(num_corrupt))

# Look at the components of corr3
pca = PCA(n_components=10)
pca.fit(corr3)
print("")
print("CorrMat3 variance components:")
print(pca.explained_variance_ratio_)

k = 5
corr3_u, corr3_s, corr3_v = np.linalg.svd(corr3)
corr3_s = np.concatenate([np.array(corr3_s[:k]), np.zeros(len(corr3_s[k:]))], axis=0)
corr3_low_rank = corr3_u.dot(np.diag(corr3_s)).dot(corr3_v)

# Count greatly changed entries
num_corrupt = 0
for i in range(0, len(corr1)):
    for j in range(0, len(corr1[0])):
        if np.sqrt((corr3[i][j] - corr3_low_rank[i][j])**2) > 100:
            num_corrupt = num_corrupt + 1
print("")
print("Number of entries changed in value by over 100: " + str(num_corrupt))



