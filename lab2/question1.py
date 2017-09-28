from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np
import seaborn as sb

df1 = pd.read_csv("./DF1", header=None)
fig,axes = plt.subplots(nrows=4, ncols=4)

### plot using pandas ###
for i in range(1,5):
    for j in range(1,i):
	plot = df1.plot.scatter(x=[], y=[], ax=axes[i-1,j-1])
	plot.axis('off')
    for j in range(i,5):
	title = "column " + str(i) + " vs " + str(j)
	plot = df1.plot.scatter(i, j, ax=axes[i-1,j-1], title=title, 
	    xticks=[], yticks=[])
plt.show()

### plot using seaborn ###
df1 = pd.read_csv("./DF1", header=None, usecols = range(1,5))
seaborn_plot = sb.PairGrid(df1)
seaborn_plot = seaborn_plot.map(plt.scatter)
plt.show()

cov_matrix = np.cov(df1, rowvar=False)
print "\nCovariance Matrix"
print cov_matrix


#Formula for three-dimensinoal gaussian:
#X1 = N(2, 4), X2 = N(-3, 9), X3 = X2 + N(1, 16)

### Calculate covariance matrix and plot error ###
exp_cov = np.array([[4,0,0],[0,9,-8.735],[0,-8.735,25]])
top_left_err = np.zeros([998, 1])        
for n in range(2,1000):
    X1 = np.array(datasets.make_gaussian_quantiles([2],4,n,1,1)[0])
    X2 = np.array(datasets.make_gaussian_quantiles([-3],9,n,1,1)[0])
    X3 = X2 + np.array(datasets.make_gaussian_quantiles([1],16,n,1,1)[0])
    dataset = np.concatenate((X1, X2, X3),1) 
    cov_matrix = np.cov(dataset, rowvar=False)
    top_left_err[n-2] = abs(float(4 - cov_matrix[0,0]))

plt.plot(range(0,998), top_left_err)
plt.xlabel('n')
plt.ylabel('real correlation - empirical correlation')
plt.title('C_11 Epirical Corrleation Error vs. Number of Samples (n)')
plt.show()









