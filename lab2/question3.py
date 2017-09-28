from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np
import seaborn as sb

n_vec = np.linspace(10, 10000, 1000) 
b_vec = np.zeros(n_vec.size, dtype=float)
sqrt_n_vec = np.zeros(n_vec.size, dtype=float)
n_count = 0

for n in n_vec:
    x = np.array(datasets.make_gaussian_quantiles([0],1,n,1,1)[0])
    ones = np.array(np.ones(x.shape))
    e = np.array(datasets.make_gaussian_quantiles([0],1,n,1,1)[0])
    b_o = -3
    b_1 = 0
    y = b_o + b_1*x + e
    
    x_aug = np.concatenate((ones, x), 1)
    inv_mat = np.linalg.inv(np.dot(np.transpose(x_aug), x_aug))
    b_hat = np.dot(np.dot(inv_mat, np.transpose(x_aug)), y)
    b_vec[n_count] = abs(b_hat[1]) 
    sqrt_n_vec[n_count] = 1/np.sqrt(n)
    n_count = n_count + 1

plt.plot(n_vec, b_vec, n_vec, sqrt_n_vec, 'r-', antialiased=True)
plt.title('Question 3: Error vs. number of samples')
plt.xlabel('n')
plt.ylabel('error')
plt.legend(['empirical error', 'estimated error'])
plt.show()

	





