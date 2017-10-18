import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as skl

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

original_train_data = pd.read_csv("./train_final.csv")
original_test_data = pd.read_csv("./test_final.csv")

nan_vals = np.array([])
for feat in range(1, 28):
    nan_val = len(original_train_data[pd.isnull(original_train_data["F" + str(feat)])].index)
    nan_vals = np.append(nan_vals, (nan_val))
plt.plot(nan_vals)
plt.show()    



#%%
#Remove outliers

#%%
#Fill NaN
train_data = original_train_data.fillna(original_train_data.mean())
test_data = original_test_data.fillna(original_test_data.mean())

#%% Get Features
train_y = train_data["Y"]
train_features = train_data.drop(["Y"], axis=1)
test_features = test_data.copy()

#Find some relationships between data
cov =  np.corrcoef(train_features.loc[:, "F1":"F27"], rowvar=False)

#Plot correlated relationships
for row in range(0, 27):
    for col in range(row, 27):
        if (cov[row][col] > .3) & (row != col):
            feat1 = "F" + str(row+1)
            feat2 = "F" + str(col)
            plt.scatter(train_features[feat1],train_features[feat2])
            plt.xlabel(feat1)
            plt.ylabel(feat2)
            plt.title("corrcoef = " + str(cov[row][col]))
            plt.show()


#%% Get Outliers
f2_outliers = train_features[train_features["F2"] > 10]
f3_outliers = train_features[train_features["F3"] > 10]
f10_outliers = train_features[train_features["F10"] > 20]
f14_outliers = train_features[train_features["F14"] > 20]
f21_outliers = train_features[train_features["F21"] > 1000]
f25_outliers = train_features[train_features["F25"] > 20]

#%% Plot histograms for train and test data
for feat in range(1, 28):
    plt.hist(train_features["F" + str(feat)], log=True, bins=100)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Train F" + str(feat))
    plt.show()

    plt.hist(test_features["F" + str(feat)], log=True, bins=100)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Test F" + str(feat))
    plt.show()
    
#%% examine features with missing data vs. output
y_likely = np.array([])
for val in range(100, 180):
    y_likely = np.append(y_likely,train_y[train_features["F18"] == val].mean())
plt.scatter(range(100, 180), y_likely)
