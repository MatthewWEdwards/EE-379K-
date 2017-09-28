
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np


###problem 1a###

d1 = datasets.make_gaussian_quantiles([-10], 25, 1000, 1, 1)[0]  # generate the first dataset
d2 = datasets.make_gaussian_quantiles([10], 25, 1000, 1, 1)[0]  # generate the second dataset

d_sum = d1 + d2  # element wise add them together to get the sum
d_sum = d_sum[:, 0]

df = pd.DataFrame({'sum': d_sum})  # turn the sum into a dataframe
df.hist(layout=(1, 1))  ###plot it

plt.show()  # show it

####problem 1b####

average = sum(d_sum) / len(d_sum)  # sum of all values divided by number of data points
print average

var_sum = 0
# the following code is basically the formula for finding the variance
for i in range(0, len(d_sum)):
    var_sum = var_sum + (d_sum[i] - average) ** 2
var = var_sum / (1000)

print var

#####problem 2###

z = [0] * 1000  # Dummy empty array

# small size n = 5
zsmall = np.array(z, dtype=float)
for j in range(0, 1000):
    # generate 5 equally likely random values with value -1 or 1,
    arr = sp.random.random_integers(0, 1, 5)
    for i in range(0, len(arr)):
        if arr[i] == 0:
            arr[i] = -1

            # calculate Z
    zsmall[j] = sum(arr) / float(len(arr))

df = pd.DataFrame({'zsmall': zsmall})
df.hist(layout=(1, 1))
plt.show()

# medium size n = 30
zmed = np.array(z, dtype=float)
for j in range(0, 1000):
    # generate 30 equally likely random values with value -1 or 1,
    arr = sp.random.random_integers(0, 1, 30)
    for i in range(0, len(arr)):
        if arr[i] == 0:
            arr[i] = -1

            # calculate Z
    zmed[j] = sum(arr) / float(len(arr))

df = pd.DataFrame({'zmed': zmed})
df.hist(layout=(1, 1))
plt.show()

# large size n = 250
zlarge = np.array(z, dtype=float)
for j in range(0, 1000):
    # generate 250 equally likely random values with value -1 or 1,
    arr = sp.random.random_integers(0, 1, 250)
    for i in range(0, len(arr)):
        if arr[i] == 0:
            arr[i] = -1

            # calculate Z
    zlarge[j] = sum(arr) / float(len(arr))

df = pd.DataFrame({'zlarge': zlarge})
df.hist(layout=(1, 1))
plt.show()

###problem 3

# generating dataset as required
d = datasets.make_gaussian_quantiles([0], 25, 25000, 1, 1)[0]

# sum divided by number to get mean
sum = 0
for num in d:
    sum = sum + num[0]
mean = sum / (25000)
print mean

# using the the variance formula
var_sum = 0
for i in range(0, len(d)):
    var_sum = var_sum + (d[i] - mean) * (d[i] - mean)
var = var_sum / (25000)
print(math.sqrt(var))

###problem 4

samples = datasets.make_gaussian_quantiles([-5, 5], [[20, .8], [.8, 30]], 10000, 1, 1)[0]

# Find the mean vector
sum = np.array([0, 0])
for i in range(0, len(samples)):
    sum = sum + (samples[i])
mean = np.divide(sum, np.array([10000, 10000]))

# Find the covariance matrix
cov_mat = np.array([[0, 0], [0, 0]])
for i in range(0, len(samples)):
    var_vect_line = np.array([samples[i][0] - mean[0], samples[i][1] - mean[1]])
    var_vect_col = np.array([[samples[i][0] - mean[0]], [samples[i][1] - mean[1]]], dtype=float).T
    cov_mat = cov_mat + np.outer(var_vect_line, var_vect_col)
cov_mat = cov_mat / len(samples)

print mean
print cov_mat

###problem 5a

data = pd.read_csv("./PatientData.csv", header=None)
print data.shape[0], '\n', data.shape[1]  # get the shape (number of rows and columns) from the dataframe

###problem 5c

import numpy as np

df = pd.read_csv("./PatientData.csv", header=None).T

data = df.values

# go through every feature and see if there are missing values
for i in range(len(data)):
    feature = data[i]

    if isinstance(feature[0],
                  str):  # since read_csv() will turn all the value in the feature column to string type if one or more of them is "?",
        # so when we see a string type, we know there is a missing value in the current column
        # and we can change those cells with "?" as their value to the average value of the current feature
        sum = 0
        num_invalid = (feature == '?').sum()
        for num in feature:
            if num == '?':
                num_invalid = num_invalid + 1
            else:
                sum = sum + int(num)
        average = sum / (len(feature) - num_invalid)  # getting average value of the current feature

        for j in range(len(feature)):
            if feature[j] == '?':
                feature[j] = str(average)
        data[i] = feature.astype(int)

data = np.array(data)  # turn the result into a numpy array

df.T.to_csv("./PatientDataNew.csv", header=False, index=False)  # write the result back







####problem 5d#####
#created a class to make parsing and comparing easier
class feature(object):
    def __init__(self, num, correlation=0):
        self.num = num
        self.correlation = correlation
    def __gt__(self, other):
        return self.correlation > other.correlation
    def __lt__(self, other):
        return self.correlation < other.correlation
    def __eq__(self, other):
        return self.correlation == other.correlation



#read the patient data
df = pd.read_csv("./PatientDataNew.csv", header=None)

#create an array so that we can rank each feature's influence
rankings = []

#get the condition of each patient
conditions = df.T.values[len(df.T.values)-1]

#calculte eacch feature's correlation with the patient's condition
for f in df.iteritems():
    feature_num = f[0] + 1
    correlation = np.correlate(f[1].values, conditions)
    new_feature = feature(feature_num, correlation)
    rankings.append(new_feature)

#sort the features by their correlation with the patient's condition
rankings.sort(reverse=True)

#print the result
print rankings[0].num, rankings[1].num, rankings[2].num

