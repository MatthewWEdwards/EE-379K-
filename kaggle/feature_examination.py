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
train_features["F22"] = np.log(train_features["F22"]+1)
train_features["F23"] = np.log(train_features["F23"]+1)
train_features["F27"] = np.log(train_features["F27"]+1)

#train_features["F3"] = np.log(train_features["F3"]+1)
#train_features["F6"] = np.log(train_features["F6"]+1)
#train_features["F9"] = np.log(train_features["F9"]+1)
#train_features["F16"] = np.log(train_features["F16"]+1)
#train_features["F19"] = np.log(train_features["F19"]+1)
#train_features["F21"] = np.log(train_features["F21"]+1)
#train_features["F22"] = np.log(train_features["F22"]+1)
#train_features["F23"] = np.log(train_features["F23"]+1)
#train_features["F27"] = np.log(train_features["F27"]+1)

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
test_features["F22"] = np.log(test_features["F22"]+1)
test_features["F23"] = np.log(test_features["F23"]+1)
test_features["F27"] = np.log(test_features["F27"]+1)
#
#test_features["F3"] = np.log(test_features["F3"]+1)
#test_features["F6"] = np.log(test_features["F6"]+1)
#test_features["F9"] = np.log(test_features["F9"]+1)
#test_features["F16"] = np.log(test_features["F16"]+1)
#test_features["F19"] = np.log(test_features["F19"]+1)
#test_features["F21"] = np.log(test_features["F21"]+1)
#test_features["F22"] = np.log(test_features["F22"]+1)
#test_features["F23"] = np.log(test_features["F23"]+1)
#test_features["F27"] = np.log(test_features["F27"]+1)





#%% normalize
test_features["F11"] = preprocessing.scale(test_features["F11"])
test_features["F18"] = preprocessing.scale(test_features["F18"])
test_features["F19"] = preprocessing.scale(test_features["F19"])
test_features["F22"] = preprocessing.scale(test_features["F22"])
test_features["F23"] = preprocessing.scale(test_features["F23"])
test_features["F26"] = preprocessing.scale(test_features["F26"])
test_features["F27"] = preprocessing.scale(test_features["F27"])


#%% Fix oddities in data

test_features.loc[test_features["F2"] > 20, "F2"] = 0
test_features.loc[test_features["F14"] > 20, "F14"] = 0
test_features.loc[test_features["F25"] > 20, "F25"] = 0


train_features.loc[train_features["F19"] > -6, "F19"] = 0
test_features.loc[test_features["F19"] > -6, "F19"] = 0




#%% Grab features by type for later processing
not_one_hots = [2,3,5,6,9,10,11,14,16,18,19,21,22,23,25,26,27]
train_cont_feats = pd.DataFrame([])
test_cont_feats = pd.DataFrame([])
for num in not_one_hots:
    label = "F" + str(num)
    train_cont_feats[label] = train_features[label]
    test_cont_feats[label] = test_features[label]
   
one_hots = [1,4,7,8,12,13,15,17,20,24]
train_cat_feats = pd.DataFrame([])
test_cat_feats = pd.DataFrame([])
for num in one_hots:
    label = "F" + str(num)
    train_cat_feats[label] = train_features[label]
    test_cat_feats[label] = train_features[label]
    
#%% Engineer some features
#for feat in not_one_hots:
#    feat_label = "F" + str(feat)
#    train_features[feat_label+"_sq"] = train_features[feat_label] * train_features[feat_label]
#    test_features[feat_label+"_sq"] = test_features[feat_label] * test_features[feat_label]

    
#%% find colinear features
cov_mat = np.cov(train_features, rowvar=False)

    
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
feat_nums =    [1, 4, 7, 8, 12,13,15,17,20,24]
feat_outlier = [6, 5, 6, 6, 6,  5, 5, 6, 6, 6, 6]

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

    
test_features = pd.concat([test_features, test_one_hot_df], axis=1)

#%% Drop Y
train_y = train_features["Y"]
train_features = train_features.drop(["Y"], axis=1)

#%% PCA, SVD, PLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD

pca = PCA(n_components = 8)
pca_feats = [3, 5, 10, 14, 18, 19, 22, 23, 25, 26, 27]

train_pca_df = pd.DataFrame([])
test_pca_df = pd.DataFrame([])
for feat in pca_feats:
    feat_label = "F" + str(feat)
    train_pca_df[feat_label] = train_features[feat_label]
    test_pca_df[feat_label] = test_features[feat_label]

pls = PLSRegression(n_components=8)   # This works good for the log reg model
pls.fit(train_pca_df, train_y)
train_feats_pls = pd.DataFrame(pls.transform(train_pca_df), index=train_features.index)
test_feats_pls = pd.DataFrame(pls.transform(test_pca_df), index=test_features.index)

#%% Replace pca feats with new feats
for feat in pca_feats:
    feat_label = "F" + str(feat)
    train_features = train_features.drop([feat_label], axis=1)
    test_features = test_features.drop([feat_label], axis=1)
train_features = pd.concat([train_features, train_feats_pls], axis=1)
test_features = pd.concat([test_features, test_feats_pls], axis=1)

#%% Logistic Regression on the initial features

lr = LogisticRegression()
lr.fit(train_features, train_y)
lr_preds = lr.predict_proba(train_features)[:,1]
print roc_auc_score(train_y, lr_preds)

cross_val = roc_cv(lr, train_features, train_y)


#%% Sample features for tuning
selection_array = np.random.randint(1, 15, (len(train_features))) == 1
sel_train_features = train_features[selection_array]
sel_train_y = train_y[selection_array]


#%% Feature selection

import sklearn.linear_model as lm
from sklearn import metrics, preprocessing


class greedyFeatureSelection(object):

    def __init__(self, data, labels, scale=1, verbose=0):
        if scale == 1:
            self._data = preprocessing.scale(np.array(data))
        else:
            self._data = np.array(data)
        self._labels = labels
        self._verbose = verbose

    def evaluateScore(self, X, y):
        model = lm.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def selectionLoop(self, X, y):
        score_history = []
        good_features = set([])
        num_features = X.shape[1]
        while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]:
            scores = []
            for feature in range(num_features):
                if feature not in good_features:
                    selected_features = list(good_features) + [feature]

                    Xts = np.column_stack(X[:, j] for j in selected_features)

                    score = self.evaluateScore(Xts, y)
                    scores.append((score, feature))

                    if self._verbose:
                        print "Current AUC : ", np.mean(score)

            good_features.add(sorted(scores)[-1][1])
            score_history.append(sorted(scores)[-1])
            if self._verbose:
                print "Current Features : ", sorted(list(good_features))

        # Remove last added feature
        good_features.remove(score_history[-1][1])
        good_features = sorted(list(good_features))
        if self._verbose:
            print "Selected Features : ", good_features

        return good_features

    def transform(self, X):
        X = self._data
        y = self._labels
        good_features = self.selectionLoop(X, y)
        return X[:, good_features]
    

gs = greedyFeatureSelection(sel_train_features, sel_train_y, verbose=True)

new_feats = gs.transform(sel_train_features)

selected_feats = [0, 1, 2, 4, 6, 8, 9, 13, 14, 15, 16, 18, 20, 21, 23, 24, 27, 30, 31, 34, 38, 39, 40, 43, 44, 45, 46, 47, 48, 54, 55, 56, 59, 60, 63, 64, 65, 68, 70, 71]
sel_train_features = pd.DataFrame([])
sel_test_features = pd.DataFrame([])
for i in selected_feats:
    sel_train_features = pd.concat([sel_train_features, train_features.iloc[:, i]], axis=1)
    sel_test_features = pd.concat([sel_test_features, test_features.iloc[:, i]], axis=1)
    
#%% again, sample
selection_array = np.random.randint(1, 10, (len(train_features))) == 1
all_train_features = train_features.copy()
all_test_features = test_features.copy()
all_train_y = train_y.copy()
train_features = sel_train_features[selection_array]
train_y = train_y[selection_array]
test_features = sel_test_features

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






#%% XGB grid search


XGB = xgb.XGBClassifier()

parameters = {
              'objective':['binary:logistic'],
              'gamma': [0.05], # Tested: [0.1, 0.2, 0.3, 0.4, 0.5 Best=.05
              'learning_rate': [.05], # Tested: 0.05, 0.1, 0.2 Best=.1
              'max_depth': [2], # Tested:2, 3, 4, 5, 7, 9, 11 Best=2
              'min_child_weight': [3], # Tested:3,4,5,7,9,11 Best=4
              'silent': [1], # Tested: 1 Best=1
              'subsample': [.6], # Tested:.6, .7, .8, .9, 1.0  Best=.6
              'colsample_bytree': [0.8], # Tested: .6, .7, .8, .9, 1.0 Best=0.8
              'n_estimators': [100],  # Tested:  Best=100 (more should be better)
              'reg_alpha': [10], # Best=10
              'reg_lambda': [10] ,# Tested: 0, 0.1, 0.5, 1 Best=10
              'scale_pos_weight':[1.1],
              'base_score': [0.5] # best = 0.5
              }


clf_xgb = GridSearchCV(XGB, parameters, n_jobs=1, cv=5, 
                   scoring='roc_auc', refit=True)

cross_val = roc_cv(clf_xgb, train_features, train_y)

clf_xgb.fit(train_features, train_y)
xgb_train_preds = clf_xgb.predict_proba(train_features)[:,1]
xgb_test_preds = clf_xgb.predict_proba(test_features)[:,1]

print roc_auc_score(train_y, xgb_train_preds)
print clf_xgb.best_estimator_

#%% RFC
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

parameters = {
              'n_estimators': [60],
              'max_depth': [7], # best = 7
              'min_samples_split': [5],
              'min_samples_leaf': [5],    # best = 5
              'max_features': [13] # best=13
              }


clf_rfc = GridSearchCV(rfc, parameters, n_jobs=1, cv=5, 
                   scoring='roc_auc', refit=True)

clf_rfc.fit(train_features, train_y)

cross_val = roc_cv(clf_rfc, train_features, train_y)

rfc_train_preds = clf_rfc.predict_proba(train_features)[:,1]
rfc_test_preds = clf_rfc.predict_proba(test_features)[:,1]
print roc_auc_score(train_y, rfc_train_preds)
print clf_rfc.best_estimator_



#%% KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

parameters = {
              'n_neighbors': [2],
              'weights': ['distance'],
              }


clf_knn = GridSearchCV(knn, parameters, n_jobs=1, cv=10, 
                   scoring='roc_auc', refit=True)

clf.fit(train_cont_feats, train_y)

cross_val = roc_cv(clf_knn, train_cont_feats, train_y)

knn_train_preds = clf_knn.predict_proba(train_cont_feats)[:,1]
knn_test_preds = clf_knn.predict_proba(test_features)[:,1]
print roc_auc_score(train_y, knn_train_preds)

#%% Naive Bayes
from sklearn.naive_bayes import BernoulliNB, GaussianNB
gnb = GaussianNB()

gnb.fit(train_features, train_y)
cross_val = roc_cv(gnb, train_features, train_y)
gnb_train_preds = gnb.predict_proba(train_features)[:,1]
gnb_test_preds = gnb.predict_proba(test_features)[:,1]

print roc_auc_score(train_y, gnb_train_preds)


#%% Stack
from sklearn.ensemble import AdaBoostClassifier

#%% XGB
xgboost = xgb.XGBClassifier(gamma=0.05, learning_rate=.05, max_depth=2, min_child_weight=3, silent=0,
        subsample=0.6, colsample_bytree = 0.8, n_estimators=100, reg_alpha=10, 
        reg_lambda = 10, scale_pos_weight=1.1)

xgboost.fit(all_train_features, all_train_y)
xgb_train_preds = xgboost.predict_proba(all_train_features)[:,1]
xgb_test_preds = xgboost.predict_proba(all_test_features)[:,1]

rfc = RandomForestClassifier(n_estimators=60, max_depth = 7, min_samples_split = 7, 
                             min_samples_leaf=5, max_features=13)
rfc.fit(all_train_features, all_train_y)
rfc_train_preds = rfc.predict_proba(all_train_features)[:,1]
rfc_test_preds = rfc.predict_proba(all_test_features)[:,1]

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


stack_model = LogisticRegression()
stack_model.fit(stack_train_frame, all_train_y)
stack_train_preds = stack_model.predict_proba(stack_train_frame)[:,1]
print roc_auc_score(all_train_y, stack_train_preds)

cross_val = roc_cv(stack_model, stack_train_frame, all_train_y)


save_csv(stack_model, stack_train_frame, all_train_y, stack_test_frame)


