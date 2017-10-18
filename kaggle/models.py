# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 23:26:27 2017

@author: Matthew Edwards
"""
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as skl
import xgboost as xgb

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

original_train_data = pd.read_csv("./train_final.csv")
original_test_data = pd.read_csv("./test_final.csv")

train_data = original_train_data.fillna(original_train_data.mean())
test_data = original_test_data.fillna(original_test_data.mean())

train_y = train_data["Y"]
train_features = train_data.drop(["Y"], axis=1)
test_features = test_data.copy()

def roc_cv(model, x, y):
    roc = cross_val_score(model, x, y, scoring="roc_auc", cv = 5)
    return(roc)

from sklearn.metrics import confusion_matrix

def conf_mat(model, x, y):
    model.fit(x, y)
    preds = model.predict(x)
    conf_mat = confusion_matrix(y,preds)
    return conf_mat



def save_csv(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    preds = pd.DataFrame()
    preds["id"] = x_test["id"]
    preds["Y"] = XGB.predict(x_test)
    preds.to_csv(path_or_buf="./preds.csv", header=["id", "Y"], index=False)
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


#%% Basic testing
KNN = KNeighborsClassifier()
GPC = GaussianProcessClassifier() # Causes memory error
DTC = DecisionTreeClassifier()
LR = LogisticRegression()
XGB = xgb.XGBClassifier()

models = [KNN, DTC, LR, XGB]
for model in models:
    print roc_cv(model, train_features, train_y)
    print conf_mat(model,)

#####Optomize on hyper parameters#####

#%% KNN
knn_vals = range(100,101) # Diminishing returns after 100 or so, peaks at .55
cv_knn = [roc_cv(KNeighborsClassifier(n_neighbors=knn_val), train_features, train_y).mean() for knn_val in knn_vals]
min_k = knn_vals[np.argmin(cv_knn)]
cv_knn = pd.Series(cv_knn, index = cv_knn)
plt.plot(range(2, 100), cv_knn)
plt.title("KNN CV")
plt.xlabel("K")
plt.ylabel("ROC")
plt.show()

#%% DTC
depth_vals = range(2,10)
dtc_cv = [roc_cv(DecisionTreeClassifier(max_depth=depth_val), train_features, train_y).mean() for depth_val in depth_vals]
min_depth = depth_vals[np.argmin(dtc_cv)]
dtc_cv = pd.Series(dtc_cv, index = dtc_cv)
plt.plot(range(2,10), dtc_cv)
plt.title("DTC CV")
plt.xlabel("Depth")
plt.ylabel("ROC")
plt.show()

#%% xgb
XGB = xgb.XGBClassifier()
print roc_cv(XGB, train_features, train_y).mean()


#%% Tests
XGB.fit(train_features, train_y)
xgb_preds = XGB.predict(test_features)
DTC = DecisionTreeClassifier(max_depth = 5)
DTC.fit(train_features, train_y)
dtc_preds = DTC.predict(test_features)

preds = xgb_preds | dtc_preds

#%%
final_preds = pd.DataFrame()
final_preds["id"] = test_features["id"]
final_preds["Y"] = preds
final_preds.to_csv(path_or_buf="./preds.csv", header=["id", "Y"], index=False)