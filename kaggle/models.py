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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


#%% PCA etc.

#from sklearn.decomposition import PCA
#from sklearn.cross_decomposition import PLSRegression
#pca = PLSRegression(n_components = 10)
#pca.fit(train_data, train_y)
#train_data = pca.transform(train_data)
#test_data = pca.transform(test_data)



#%%


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
    

#%% XGB

XGB = xgb.XGBClassifier()

parameters = {
              'objective':['binary:logistic'],
              'gamma': [.5], # Tested: .1, .2, .5, .7, 1, 2 Best=.5
              'learning_rate': [.1], # Tested: 0.001, 0.002, 0.005, .1, .05, .1 Best=.1
              'max_depth': [5], # Tested: 3, 5, 7, 9, 10 Best=5
              'min_child_weight': [2], # Tested: 1,2,3,4,5 Best=2
              'silent': [1], # Tested: 1 Best=1
              'subsample': [.6], # Tested: .6, .7, .8 Best=.6
              'colsample_bytree': [.8], # Tested: .6,.7,.8,.9,.10 Best=.8
              'n_estimators': [25],  # Tested: 100 Best=100 (more should be better)
              'reg_alpha': [1], # Tested: 0, .1, 1, 2, 5, 10 Best=1
              'reg_lambda': [10] # Tested: 0, .1, 1, 10, 100 Best=10
              
              }


clf = GridSearchCV(XGB, parameters, n_jobs=1, cv=5, 
                   scoring='roc_auc', refit=True)

clf.fit(train_data, train_y)



#%% DTC
depth_vals = range(2,10)
dtc_cv = [roc_cv(DecisionTreeClassifier(max_depth=depth_val), train_data, train_y).mean() for depth_val in depth_vals]
max_depth = depth_vals[np.argmax(dtc_cv)]
dtc_cv = pd.Series(dtc_cv, index = dtc_cv)
plt.plot(range(2,10), dtc_cv)
plt.title("DTC CV")
plt.xlabel("Depth")
plt.ylabel("ROC")
plt.show()

#%% Random Classifier Forest
# Get a good number of estimators
est_vals = range(3, 20)
rfc_cv = [roc_cv(RandomForestClassifier(n_estimators=depth_val), train_data, train_y).mean() for est_val in est_vals]
max_n_est = est_vals[np.argmax(rfc_cv)]
rfc_cv = pd.Series(rfc_cv, index = rfc_cv)
plt.plot(range(3,20), rfc_cv)
plt.title("RFC CV")
plt.xlabel("Estimators Count")
plt.ylabel("ROC")
plt.show()
max_n_est = 3 # Found using the above code. not optimal but close to it

# Max Depth

max_depth_vals = range(3, 20)
rfc_cv = [roc_cv(RandomForestClassifier(n_estimators=3, max_depth=max_depth_val), train_data, train_y).mean() for max_depth_val in max_depth_vals]
max_max_depth = est_vals[np.argmax(rfc_cv)]
rfc_cv = pd.Series(rfc_cv, index = rfc_cv)
plt.plot(range(3,20), rfc_cv)
plt.title("RFC CV")
plt.xlabel("Max Depth Val")
plt.ylabel("ROC")
plt.show()
max_max_depth = 7 # Found using above code and optimal

# Min sample split
min_split_vals = [2, 5, 10, 20, 50, 100, 200, 500]
rfc_cv = [roc_cv(RandomForestClassifier(n_estimators=3, max_depth=7, min_samples_split=min_split_val), train_data, train_y).mean() for min_split_val in min_split_vals]
max_split_val = est_vals[np.argmax(rfc_cv)]
rfc_cv = pd.Series(rfc_cv, index = rfc_cv)
plt.plot([2, 5, 10, 20, 50, 100, 200, 500], rfc_cv)
plt.title("RFC CV")
plt.xlabel("Min Split Val")
plt.ylabel("ROC")
plt.show()



#%% Enesmble models


XGB = xgb.XGBClassifier(base_score=0.5, gamma=0.5, learning_rate=0.05,
                        max_depth=5, min_child_weight=2, n_estimators=100, 
                        reg_alpha = 1, reg_lambda=10, subsampled=0.6,
                        colsample_bytree = .8)
XGB.fit(train_data, train_y)
xgb_train_preds = XGB.predict(train_data)
xgb_test_preds = XGB.predict(test_data)


DTC = DecisionTreeClassifier(max_depth = 4, min_samples_split = 40, criterion="entropy")
DTC.fit(train_data, train_y)
dtc_train_preds = DTC.predict(train_data)
dtc_test_preds = DTC.predict(test_data)

RFC = RandomForestClassifier(n_estimators=3, max_depth=7, min_samples_split=20)
RFC.fit(train_data, train_y)
rfc_train_preds = RFC.predict(train_data)
rfc_test_preds = RFC.predict(test_data)

super_log_reg_train = pd.DataFrame()
super_log_reg_train["xgb"] = xgb_train_preds
super_log_reg_train["dtc"] = dtc_train_preds
super_log_reg_train["rfc"] = rfc_train_preds

super_log_reg_test = pd.DataFrame()
super_log_reg_test["xgb"] = xgb_test_preds
super_log_reg_test["dtc"] = dtc_test_preds
super_log_reg_test["rfc"] = rfc_test_preds

LR_final = LogisticRegression()
LR_final.fit(super_log_reg_train, train_y)
print roc_cv(LR_final, super_log_reg_train, train_y)
train_preds = LR_final.predict_proba(super_log_reg_train)[:,1]
print roc_auc_score(train_y, train_preds)
ensembled_preds = LR_final.predict_proba(super_log_reg_test)
#%%
final_preds = pd.DataFrame()
final_preds["id"] = original_test_data["id"]
final_preds["Y"] = pd.Series(ensembled_preds, index=original_test_data.index)
final_preds.to_csv(path_or_buf="./preds.csv", header=["id", "Y"], index=False)

