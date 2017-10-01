import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import sklearn as skl

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

#%% Init data (copied from https://www.kaggle.com/apapiu/regularized-linear-models)
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
plt.show()
train["SalePrice"] = np.log1p(train["SalePrice"])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

#%% Init Cross validation tools (https://www.kaggle.com/apapiu/regularized-linear-models)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.cross_validation import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))
    return(rmse)

#%% Cross validate Ridge (https://www.kaggle.com/apapiu/regularized-linear-models)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Ridge Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.axis((0, 70, 0.1,0.2))
plt.show()
cv_ridge.min()

#%% Cross validate Lasso 
#lasso_alphas = np.concatenate((np.linspace(0.0001, 0.001, num=10),
#                               np.linspace(0.001, 0.01, num=10),
#                               np.linspace(0.01, .1, num=10),
#                               np.linspace(0.1, 1, num=10)))
#cv_lasso = np.array([])
#coef_nonzero = np.array([])
#for alpha in lasso_alphas:
#    model_lasso = LassoCV(alphas = [alpha]).fit(X_train, y)
#    cv_lasso = np.append(cv_lasso, rmse_cv(model_lasso).mean())
#    coef_nonzero = np.append(coef_nonzero, sum(model_lasso.coef_ != 0))
#    
#fig = plt.figure() 
#ax = fig.add_subplot(111)
#cv_lasso = pd.Series(cv_lasso, index = lasso_alphas)
#cv_lasso.plot(title = "Lasso Validation", logx=True, ax=ax, color='r', grid=False)
#
#ax2 = ax.twinx()
#coef_nonzero = pd.Series(coef_nonzero, index = lasso_alphas)
#coef_nonzero.plot(ax=ax2)
#
#plt.xlabel("alpha")
#ax.set_ylabel("Cross validation score (red)")
#ax2.set_ylabel("Nonzero coefficients (blue)")
#plt.axis((0, 1, 0, 225))
#plt.show()

#%% A good alpha choice for ridge is 10, for lasso .0007.
a_ridge = 10
a_lasso = .0007
stack_data = X_train
stack_data["SalePrice"] = pd.Series(y, index=stack_data.index)
stack_data["ID"] = pd.Series(range(0, X_train.shape[0]), index=stack_data.index)
stack_data["FoldID"] = pd.Series(np.random.randint(1, high=6, size=(X_train.shape[0])), index=stack_data.index)
stack_meta = X_test.copy()
stack_train = stack_data.copy()
stack_train["Ridge"] = pd.Series(np.nan, index=stack_train.index)
stack_train["Lasso"] = pd.Series(np.nan, index=stack_train.index)


#%% Create training folds and fill out model predictions in the training data
for fold in range(1, 6):
    # Organize folds
    folds_combined = stack_data[stack_data["FoldID"] != fold]
    folds_train = folds_combined.drop("SalePrice", 1)
    folds_test = stack_data[stack_data["FoldID"] == fold]
    folds_test = folds_test.reindex()
    y_train = folds_combined["SalePrice"]
    y_test = folds_test["SalePrice"]
    folds_test = folds_test.drop("SalePrice", 1)
    
    # Train models on training folds, predict test fold
    model_ridge = Ridge(alpha=a_ridge).fit(folds_train, y_train)
    model_lasso = LassoCV(alphas=[a_lasso]).fit(folds_train, y_train)    
    ridge_preds = pd.DataFrame({"preds":model_ridge.predict(folds_test)}, index=folds_test.index)
    lasso_preds = pd.DataFrame({"preds":model_lasso.predict(folds_test)}, index=folds_test.index)
    
    folds_test["Ridge"] = ridge_preds
    folds_test["Lasso"]= lasso_preds
    stack_train.update(folds_test)
    
#%% Using original training data, create predictions on test data
X_train_no_extras = X_train.drop(["ID", "FoldID", "SalePrice"], axis=1)
normal_model_ridge = Ridge(alpha=a_ridge).fit(X_train_no_extras, y)
normal_model_lasso = LassoCV(alphas=[a_lasso]).fit(X_train_no_extras, y)
normal_ridge_preds = pd.DataFrame({"preds":normal_model_ridge.predict(X_test)})
normal_lasso_preds = pd.DataFrame({"preds":normal_model_lasso.predict(X_test)})
stack_meta["Ridge"] = normal_ridge_preds
stack_meta["Lasso"] = normal_lasso_preds

#%% Stack the models
stack_train_final = stack_train.drop(["ID", "FoldID", "SalePrice"], axis=1)
final_model_ridge = Ridge(alpha=10).fit(stack_train_final, y)

final_preds = pd.DataFrame({"preds":final_model_ridge.predict(stack_meta)})
final_preds= np.expm1(final_preds)

#%% Engineer some features
new_data = all_data.copy()
def year_dif(x):
    return 2017 - x      

def quad(x):
    return x**2

# Change year feature to "years before 2017"
new_data["YearBuilt"] = all_data["YearBuilt"].apply(year_dif)
new_data["GarageYrBlt"] = all_data["GarageYrBlt"].apply(year_dif)
new_data["YearRemodAdd"] = all_data["YearRemodAdd"].apply(year_dif)
new_data["YrSold"] = all_data["YrSold"].apply(year_dif)

# Add some quadratic features
new_data["OverallQualQuad"] = all_data["OverallQual"].apply(year_dif)
# Find the quartic of bathrooms because bathrooms are important
first_bath_op = all_data["FullBath"].apply(year_dif)
new_data["FullBathQuart"] = first_bath_op.apply(year_dif)


#%% Run stacked models on the engineered features

a_ridge = 10
a_lasso = .0007
stack_data = all_data[:1460]
stack_data["SalePrice"] = pd.Series(y, index=stack_data.index)
stack_data["ID"] = pd.Series(range(0, X_train.shape[0]), index=stack_data.index)
stack_data["FoldID"] = pd.Series(np.random.randint(1, high=6, size=(X_train.shape[0])), index=stack_data.index)
stack_meta = X_test.copy()
stack_train = stack_data.copy()
stack_train["Ridge"] = pd.Series(np.nan, index=stack_train.index)
stack_train["Lasso"] = pd.Series(np.nan, index=stack_train.index)
for fold in range(1, 6):
    # Organize folds
    folds_combined = stack_data[stack_data["FoldID"] != fold]
    folds_train = folds_combined.drop("SalePrice", 1)
    folds_test = stack_data[stack_data["FoldID"] == fold]
    folds_test = folds_test.reindex()
    y_train = folds_combined["SalePrice"]
    y_test = folds_test["SalePrice"]
    folds_test = folds_test.drop("SalePrice", 1)
    
    # Train models on training folds, predict test fold
    model_ridge = Ridge(alpha=a_ridge).fit(folds_train, y_train)
    model_lasso = LassoCV(alphas=[a_lasso]).fit(folds_train, y_train)    
    ridge_preds = pd.DataFrame({"preds":model_ridge.predict(folds_test)}, index=folds_test.index)
    lasso_preds = pd.DataFrame({"preds":model_lasso.predict(folds_test)}, index=folds_test.index)
    
    folds_test["Ridge"] = ridge_preds
    folds_test["Lasso"]= lasso_preds
    stack_train.update(folds_test)
    
#%% Using original training data, create predictions on test data
normal_model_ridge = Ridge(alpha=a_ridge).fit(X_train_no_extras, y)
normal_model_lasso = LassoCV(alphas=[a_lasso]).fit(X_train_no_extras, y)
normal_ridge_preds = pd.DataFrame({"preds":normal_model_ridge.predict(X_test)})
normal_lasso_preds = pd.DataFrame({"preds":normal_model_lasso.predict(X_test)})
stack_meta["Ridge"] = normal_ridge_preds
stack_meta["Lasso"] = normal_lasso_preds

#%% Stack the models
stack_train_final = stack_train.drop(["ID", "FoldID", "SalePrice"], axis=1)
final_model_ridge = Ridge(alpha=10).fit(stack_train_final, y)

final_preds = pd.DataFrame({"preds":final_model_ridge.predict(stack_meta)})
final_preds= np.expm1(final_preds)

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train_no_extras, y)
xgb_preds = pd.DataFrame({"preds":model_xgb.predict(X_test)})
np.expm1(xgb_preds).to_csv("xgb_predictions.csv")


        
	
