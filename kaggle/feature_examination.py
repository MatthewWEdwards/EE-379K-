import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

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

# Log transform some features
train_features["F3"] = np.log(train_features["F3"]+1)
train_features["F6"] = np.log(train_features["F6"]+1)
train_features["F9"] = np.log(train_features["F9"]+1)
train_features["F16"] = np.log(train_features["F16"]+1)
train_features["F19"] = np.log(train_features["F19"]+1)
train_features["F21"] = np.log(train_features["F21"]+1)
train_features["F23"] = np.log(train_features["F23"]+1)
train_features["F27"] = np.log(train_features["F27"]+1)

train_features["F3"] = np.log(train_features["F3"]+1)
train_features["F6"] = np.log(train_features["F6"]+1)
train_features["F9"] = np.log(train_features["F9"]+1)
train_features["F16"] = np.log(train_features["F16"]+1)
train_features["F19"] = np.log(train_features["F19"]+1)
train_features["F21"] = np.log(train_features["F21"]+1)
train_features["F22"] = np.log(train_features["F22"]+1)
train_features["F23"] = np.log(train_features["F23"]+1)
train_features["F27"] = np.log(train_features["F27"]+1)

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
train_features["F27"] = preprocessing.scale(train_features["F27"])

#%% Log transform some features
test_features["F3"] = np.log(test_features["F3"]+1)
test_features["F6"] = np.log(test_features["F6"]+1)
test_features["F9"] = np.log(test_features["F9"]+1)
test_features["F16"] = np.log(test_features["F16"]+1)
test_features["F19"] = np.log(test_features["F19"]+1)
test_features["F21"] = np.log(test_features["F21"]+1)
test_features["F23"] = np.log(test_features["F23"]+1)
test_features["F27"] = np.log(test_features["F27"]+1)

test_features["F3"] = np.log(test_features["F3"]+1)
test_features["F6"] = np.log(test_features["F6"]+1)
test_features["F9"] = np.log(test_features["F9"]+1)
test_features["F16"] = np.log(test_features["F16"]+1)
test_features["F19"] = np.log(test_features["F19"]+1)
test_features["F21"] = np.log(test_features["F21"]+1)
test_features["F22"] = np.log(test_features["F22"]+1)
test_features["F23"] = np.log(test_features["F23"]+1)
test_features["F27"] = np.log(test_features["F27"]+1)

#%% Fix oddities in data

# This improved performance
#train_features = train_features[train_features["F19"] != 0]

# TODO:
#test_features[test_features["F2"] > 20] = 0
#test_features[test_features["F14"] > 20] = 0


#%% normalize
test_features["F11"] = preprocessing.scale(test_features["F11"])
test_features["F18"] = preprocessing.scale(test_features["F18"])
test_features["F19"] = preprocessing.scale(test_features["F19"])
test_features["F22"] = preprocessing.scale(test_features["F22"])
test_features["F23"] = preprocessing.scale(test_features["F23"])
test_features["F26"] = preprocessing.scale(test_features["F26"])
test_features["F27"] = preprocessing.scale(test_features["F27"])

#%% Grab features by type for later processing
not_one_hots = [2,3,6,9,10,11,14,16,18,19,21,22,23,25,26,27]
train_cont_feats = pd.DataFrame([])
test_cont_feats = pd.DataFrame([])
for num in not_one_hots:
    label = "F" + str(num)
    train_cont_feats[label] = train_features[label]
    test_cont_feats[label] = test_features[label]
   
one_hots = [1,4,5,7,8,12,13,15,17,20,24]
train_cat_feats = pd.DataFrame([])
test_cat_feats = pd.DataFrame([])
for num in one_hots:
    label = "F" + str(num)
    train_cat_feats[label] = train_features[label]
    test_cat_feats[label] = train_features[label]
    
#%% Plot histograms for train and test data
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
    
#%% Plot histograms for train and test data
for feat in one_hots:
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
feat_outlier = [6, 5, 6, 6, 6,  6, 5, 6, 6, 6, 6]

train_one_hot_df = pd.DataFrame([])
for feat in range(0, len(feat_nums)):
    feat_label = "F" + str(feat_nums[feat])
    train_features[feat_label].loc[train_features[feat_label] > feat_outlier[feat]] = np.nan
    new_df = pd.get_dummies(train_features[feat_label], prefix = feat_label)
    train_one_hot_df = pd.concat([train_one_hot_df, new_df], axis=1)
    
#Drop one-hot'd features
for feat in range(0, len(feat_nums)):
    feat_label = "F" + str(feat_nums[feat])
    train_features = train_features.drop(feat_label, axis=1)
    
train_features = pd.concat([train_features, train_one_hot_df], axis=1)

test_one_hot_df = pd.DataFrame([])
for feat in range(0, len(feat_nums)):
    feat_label = "F" + str(feat_nums[feat])
    test_features[feat_label].loc[test_features[feat_label] > feat_outlier[feat]] = np.nan
    new_df = pd.get_dummies(test_features[feat_label], prefix = feat_label)
    test_one_hot_df = pd.concat([test_one_hot_df, new_df], axis=1)
    
# Drop one-hot'd features
for feat in range(0, len(feat_nums)):
    feat_label = "F" + str(feat_nums[feat])
    test_features = test_features.drop(feat_label, axis=1)
    
# Training data ends up having some weird residual features due to what I think
## is noise. Drop 'em.
#train_features = train_features.drop(["F1_0.0"], axis=1)
#train_features = train_features.drop(["F7_0.0"], axis=1)
#train_features = train_features.drop(["F8_0.0"], axis=1)
#train_features = train_features.drop(["F12_0.0"], axis=1)
#train_features = train_features.drop(["F13_0.0"], axis=1)
#train_features = train_features.drop(["F15_0.0"], axis=1)
#train_features = train_features.drop(["F17_0.0"], axis=1)
#train_features = train_features.drop(["F20_0.0"], axis=1)
#train_features = train_features.drop(["F24_0.0"], axis=1)

    
test_features = pd.concat([test_features, test_one_hot_df], axis=1)

#%% Drop Y
train_y = train_features["Y"]
train_features = train_features.drop(["Y"], axis=1)

#%% PCA, SVD, PLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.decomposition import FactorAnalysis # For categoricals

fa = FactorAnalysis(n_components = 25)
fa.fit(train_features, train_y)
train_feats_fa = fa.transform(train_features)
test_feats_fa = fa.transform(test_features)

pca = PCA()
pca.fit(train_cont_feats)

pls = PLSRegression(n_components=8)   # This works good for the log reg model
pls.fit(train_cont_feats, train_y)
train_cont_feats_pls = pls.transform(train_cont_feats)
test_cont_feats_pls = pls.transform(test_cont_feats)

#%% GMM
from sklearn.mixture import BayesianGaussianMixture
gmm = GaussianMixture(n_components=1)

gmm.fit(np.transpose(train_features["F27"]))
gmm_preds = gmm.predict_proba(train_features["F27"])





#%% Feature selection
feat_count = 0
train_features_sel = pd.DataFrame([])
test_features_sel = pd.DataFrame([])

for column in train_features:
    if abs(lr_coefs[0, feat_count]) > .011:
        train_features_sel[column] = train_features[column]
        test_features_sel[column] = test_features[column]
        feat_count = feat_count + 1
    print column

#%% Hyper perameter tuning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def save_csv(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    preds = pd.DataFrame([])
    preds["id"] = pd.Series(range(49999, 99999))
    preds["Y"] = model.predict_proba(x_test)[:,1]
    preds.to_csv(path_or_buf="./preds.csv", header=["id", "Y"], index=False)

def roc_cv(model, x, y):
    roc = cross_val_score(model, x, y, scoring="roc_auc", cv = 5)
    return(roc)

from sklearn.metrics import confusion_matrix

def conf_mat(model, x, y):
    model.fit(x, y)
    preds = model.predict(x)
    conf_mat = confusion_matrix(y,preds)
    return conf_mat

#%% Logistic Regression

lr = LogisticRegression()
lr.fit(train_features, train_y)
lr_preds = lr.predict_proba(train_features)[:,1]
print roc_auc_score(train_y, lr_preds)


#%% XGB

XGB = xgb.XGBClassifier()

parameters = {
              'objective':['binary:logistic'],
              'gamma': [0.5], # Tested: 0.05, 0.1, 0.2, 0.5 Best=.5
              'learning_rate': [.1], # Tested: 0.05, 0.1, 0.2 Best=.1
              'max_depth': [3], # Tested: 3, 5, 7, 9 Best=3
              'min_child_weight': [4], # Tested: 2,4,6,10 Best=4
              'silent': [1], # Tested: 1 Best=1
              'subsample': [.8], # Tested:.6, .7, .8, .9, 1.0  Best=.8
              'colsample_bytree': [1.0], # Tested: .6, .7, .8, .9, 1.0 Best=1.0
              'n_estimators': [80],  # Tested:  Best=100 (more should be better)
              'reg_alpha': [1], # Tested:0, 0.1, 0.5, 1  Best=0
              'reg_lambda': [1] # Tested: 0, 0.1, 0.5, 1 Best=10
              
              }


clf = GridSearchCV(XGB, parameters, n_jobs=1, cv=5, 
                   scoring='roc_auc', refit=True)

#cross_val = roc_cv(clf, train_features, train_y)

clf.fit(train_features, train_y)
xgb_train_preds = clf.predict_proba(train_features)[:,1]
xgb_test_preds = clf.predict_proba(test_features)[:,1]

print roc_auc_score(train_y, xgb_train_preds)

#%% RFC
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

parameters = {
              'n_estimators': [60],
              'max_depth': [7],
              'min_samples_split': [5],
              'min_samples_leaf': [3],   
              'max_features': [20]
              }


clf = GridSearchCV(rfc, parameters, n_jobs=1, cv=10, 
                   scoring='roc_auc', refit=True)

clf.fit(train_features, train_y)

cross_val = roc_cv(clf, train_features, train_y)

rfc_train_preds = clf.predict_proba(train_features)[:,1]
rfc_test_preds = clf.predict_proba(test_features)[:,1]
print roc_auc_score(train_y, rfc_train_preds)

#%% Stack
from sklearn.ensemble import AdaBoostClassifier


stack_train_features = [rfc_train_preds, xgb_train_preds]
stack_test_features = [rfc_test_preds, xgb_test_preds]

stack_train_frame = pd.DataFrame([])
feat_count = 0
for feat in stack_train_features:
    stack_train_frame[str(feat_count)] = pd.Series(feat)
    feat_count = feat_count + 1

stack_test_frame = pd.DataFrame([])
feat_count = 0
for feat in stack_test_features:
    stack_test_frame[str(feat_count)] = pd.Series(feat)
    feat_count = feat_count + 1


stack_model = AdaBoostClassifier()
stack_model.fit(stack_train_frame, train_y)
stack_train_preds = stack_model.predict_proba(stack_test_frame)[:,1]
print roc_auc_score(train_y, rfc_train_preds)

save_csv(stack_model, stack_train_frame, train_y, stack_test_frame)
#%%Notes
#Use a gaussian mixture model of KDE for bimodal features.
