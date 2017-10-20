import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
#Fill NaN
# There is about 1000 NaNs for F4 and 10000 for F18. The information excluded
# from F18 is likely excluded based on the value of F18
train_features = original_train_data.copy()
test_features = original_test_data.copy()

train_features = train_features.drop(["id"], axis=1)
train_features["F5"] = train_features["F5"].fillna(0) # 0 is the most frequent value
train_features["F19"] = train_features["F19"].fillna(np.exp(8))
test_features = test_features.drop(["id"], axis=1)
test_features["F5"] = test_features["F5"].fillna(0) # 0 is the most frequent value
test_features["F19"] = test_features["F19"].fillna(np.exp(8))

#%%

# Engineer some features
train_features["F3"] = np.log(train_features["F3"]+1)
train_features["F6"] = np.log(train_features["F6"]+1)
train_features["F9"] = np.log(train_features["F9"]+1)
train_features["F16"] = np.log(train_features["F16"]+1)
train_features["F19"] = np.log(train_features["F19"]+1)
train_features["F21"] = np.log(train_features["F21"]+1)
train_features["F22"] = np.log(train_features["F22"]+1)
train_features["F23"] = np.log(train_features["F23"]+1)

train_features["F3"] = np.log(train_features["F3"]+1)
train_features["F6"] = np.log(train_features["F6"]+1)
train_features["F9"] = np.log(train_features["F9"]+1)
train_features["F16"] = np.log(train_features["F16"]+1)
train_features["F19"] = np.log(train_features["F19"]+1)
train_features["F21"] = np.log(train_features["F21"]+1)
train_features["F22"] = np.log(train_features["F22"]+1)
train_features["F23"] = np.log(train_features["F23"]+1)

#%% Get Outliers, Drop
f2_outliers = train_features[train_features["F2"] > 10]
train_features = train_features.drop(f2_outliers.index)
f3_outliers = train_features[train_features["F3"] > 2]
train_features = train_features.drop(f3_outliers.index)
f6_outliers = train_features[train_features["F6"] > 15]
train_features = train_features.drop(f6_outliers.index)
f9_outliers = train_features[train_features["F9"] > 9]
train_features = train_features.drop(f9_outliers.index)
f10_outliers = train_features[train_features["F10"] > 20]
train_features = train_features.drop(f10_outliers.index)
f14_outliers = train_features[train_features["F14"] > 20]
train_features = train_features.drop(f14_outliers.index)
f16_outliers = train_features[train_features["F16"] > 8]
train_features = train_features.drop(f16_outliers.index)
f19_outliers = train_features[train_features["F19"] > 13]
train_features = train_features.drop(f19_outliers.index)
f21_outliers = train_features[train_features["F21"] > 8]
train_features = train_features.drop(f21_outliers.index)
f23_outliers = train_features[train_features["F23"] > 2]
train_features = train_features.drop(f23_outliers.index)
f25_outliers = train_features[train_features["F25"] > 20]
train_features = train_features.drop(f25_outliers.index)

#%% normalize
train_features["F11"] = preprocessing.scale(train_features["F11"])
train_features["F18"] = preprocessing.scale(train_features["F18"])
train_features["F19"] = preprocessing.scale(train_features["F19"])
train_features["F22"] = preprocessing.scale(train_features["F22"])
train_features["F23"] = preprocessing.scale(train_features["F23"])
train_features["F26"] = preprocessing.scale(train_features["F26"])

# Engineer some features
test_features["F3"] = np.log(test_features["F3"]+1)
test_features["F6"] = np.log(test_features["F6"]+1)
test_features["F9"] = np.log(test_features["F9"]+1)
test_features["F16"] = np.log(test_features["F16"]+1)
test_features["F19"] = np.log(test_features["F19"]+1)
test_features["F21"] = np.log(test_features["F21"]+1)
test_features["F22"] = np.log(test_features["F22"]+1)
test_features["F23"] = np.log(test_features["F23"]+1)

test_features["F3"] = np.log(test_features["F3"]+1)
test_features["F6"] = np.log(test_features["F6"]+1)
test_features["F9"] = np.log(test_features["F9"]+1)
test_features["F16"] = np.log(test_features["F16"]+1)
test_features["F19"] = np.log(test_features["F19"]+1)
test_features["F21"] = np.log(test_features["F21"]+1)
test_features["F22"] = np.log(test_features["F22"]+1)
test_features["F23"] = np.log(test_features["F23"]+1)

#%% Get Outliers, don't drop for test data
#f2_outliers = test_features[test_features["F2"] > 10]
#test_features = test_features.drop(f2_outliers.index)
#f3_outliers = test_features[test_features["F3"] > 2]
#test_features = test_features.drop(f3_outliers.index)
#f6_outliers = test_features[test_features["F6"] > 15]
#test_features = test_features.drop(f6_outliers.index)
#f9_outliers = test_features[test_features["F9"] > 9]
#test_features = test_features.drop(f9_outliers.index)
#f10_outliers = test_features[test_features["F10"] > 20]
#test_features = test_features.drop(f10_outliers.index)
#f14_outliers = test_features[test_features["F14"] > 20]
#test_features = test_features.drop(f14_outliers.index)
#f16_outliers = test_features[test_features["F16"] > 8]
#test_features = test_features.drop(f16_outliers.index)
#f19_outliers = test_features[test_features["F19"] > 13]
#test_features = test_features.drop(f19_outliers.index)
#f21_outliers = test_features[test_features["F21"] > 8]
#test_features = test_features.drop(f21_outliers.index)
#f23_outliers = test_features[test_features["F23"] > 2]
#test_features = test_features.drop(f23_outliers.index)
#f25_outliers = test_features[test_features["F25"] > 20]
#test_features = test_features.drop(f25_outliers.index)

#%% normalize
test_features["F11"] = preprocessing.scale(test_features["F11"])
test_features["F18"] = preprocessing.scale(test_features["F18"])
test_features["F19"] = preprocessing.scale(test_features["F19"])
test_features["F22"] = preprocessing.scale(test_features["F22"])
test_features["F23"] = preprocessing.scale(test_features["F23"])
test_features["F26"] = preprocessing.scale(test_features["F26"])

#%% Plot histograms for train and test data
not_one_hots = [2,3,6,9,10,11,14,16,18,19,21,22,23,25,26,27]
for feat in not_one_hots:
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

#%% One-hot encoding
feat_nums =    [1, 4, 5, 7, 8, 12,13,15,17,20,24]
feat_outlier = [6, 5, 6, 7, 6,  6, 6, 6, 6, 6, 6]

one_hot_df = pd.DataFrame([])
for feat in range(0, len(feat_nums)):
    feat_label = "F" + str(feat_nums[feat])
    train_features.loc[train_features[feat_label] > 6] = np.nan
    new_df = pd.get_dummies(train_features[feat_label], prefix = feat_label)
    one_hot_df = pd.concat([one_hot_df, new_df], axis=1)
    
for feat in range(0, len(feat_nums)):
    feat_label = "F" + str(feat_nums[feat])
    train_features = train_features.drop(feat_label, axis=1)
    
#train_features = pd.concat([train_features, one_hot_df], axis=1)

one_hot_df = pd.DataFrame([])
for feat in range(0, len(feat_nums)):
    feat_label = "F" + str(feat_nums[feat])
    test_features.loc[test_features[feat_label] > 6] = np.nan
    new_df = pd.get_dummies(test_features[feat_label], prefix = feat_label)
    one_hot_df = pd.concat([one_hot_df, new_df], axis=1)
    
for feat in range(0, len(feat_nums)):
    feat_label = "F" + str(feat_nums[feat])
    test_features = test_features.drop(feat_label, axis=1)
    
#test_features = pd.concat([test_features, one_hot_df], axis=1)

#%%
train_y = train_features["Y"]
train_features = train_features.drop(["Y"], axis=1)

#%% Logistic Regression

def save_csv(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    preds = pd.DataFrame()
    preds["id"] = pd.Series(range(49999, 99999))
    preds["Y"] = model.predict_proba(x_test)[:,1]
    preds.to_csv(path_or_buf="./preds.csv", header=["id", "Y"], index=False)

lr = LogisticRegression()
lr.fit(train_features, train_y)
preds = lr.predict_proba(train_features)[:,1]
print roc_auc_score(train_y, preds)
save_csv(lr, train_features, train_y, test_features)

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