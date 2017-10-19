import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as skl

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

original_train_data = pd.read_csv("./train_final.csv")
original_test_data = pd.read_csv("./test_final.csv")

train_nan_vals = np.array([])
test_nan_vals = np.array([])
for feat in range(1, 28):
    train_nan_val = len(original_train_data[pd.isnull(original_train_data["F" + str(feat)])].index)
    test_nan_val = len(original_test_data[pd.isnull(original_test_data["F" + str(feat)])].index)
    train_nan_vals = np.append(train_nan_vals, (train_nan_val))
    test_nan_vals = np.append(test_nan_vals, (test_nan_val))
plt.plot(train_nan_vals)
plt.title("NaN values in the training data by feature")
plt.xlabel("Feature")
plt.ylabel("NaN count")
plt.show()    
plt.plot(test_nan_vals)
plt.title("NaN values in the testing data by feature")
plt.xlabel("Feature")
plt.ylabel("NaN count")
plt.show()    




#%%
#Remove outliers

#%%
#Fill NaN
# There is about 1000 NaNs for F4 and 10000 for F18. The information excluded
# from F18 is likely excluded based on the value of F18
train_data = original_train_data.copy()
train_data = train_data.drop(["id"], axis=1)
train_data["F5"] = train_data["F5"].fillna(0) # 0 is the most frequent value
train_data["F19"] = train_data["F19"].fillna(np.exp(8))
# predict F18

test_data = original_test_data.fillna(original_test_data.mean())

#%% Get Features
train_y = train_data["Y"]
train_features = train_data.drop(["Y"], axis=1)
test_features = test_data.copy()

#Find some relationships between data
cov =  np.corrcoef(train_features.loc[:, "F1":"F27"], rowvar=False)

#%%Plot correlated relationships
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


#%% Scatter plots
feat1 = range(1, 10)
feat2 = range(10, 20)

for f1 in feat1:
    for f2 in feat2:
        plt.scatter(train_features["F" + str(f1)], train_features["F" + str(f2)])
        plt.xlabel("F" + str(f1))
        plt.ylabel("F" + str(f2))
        plt.show()

#%% Get Outliers
f2_outliers = train_features[train_features["F2"] > 10]
f3_outliers = train_features[train_features["F3"] > 10]
f10_outliers = train_features[train_features["F10"] > 20]
f14_outliers = train_features[train_features["F14"] > 20]
f21_outliers = train_features[train_features["F21"] > 1000]
f25_outliers = train_features[train_features["F25"] > 20]

# Engineer some features
train_features["F6"] = np.log(train_features["F6"]+1)
train_features["F16"] = np.log(train_features["F16"]+1)
train_features["F19"] = np.log(train_features["F19"]+1)
train_features["F21"] = np.log(train_features["F21"]+1)
train_features["F23"] = np.log(train_features["F23"]+1)


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
plt.show()

#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.datasets import make_friedman1
GBR = GradientBoostingRegressor(n_estimators=20).fit(train_features, train_y)
plot_partial_dependence(GBR, train_features, range(1, 10)) 
plt.show()
plot_partial_dependence(GBR, train_features, range(10, 19)) 
plt.show()
plot_partial_dependence(GBR, train_features, range(19, 28)) 
plt.show()