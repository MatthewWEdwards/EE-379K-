import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
plt.show()

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
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

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.cross_validation import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))
    return(rmse)

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

lasso_alphas = np.concatenate((np.linspace(0.0001, 0.001, num=10), np.linspace(0.01, .1, num = 10)))
cv_lasso = np.array([])
for alpha in lasso_alphas:
	model_lasso = LassoCV(alphas = [alpha]).fit(X_train, y)
	cv_lasso = np.append(cv_lasso, rmse_cv(model_lasso).mean())
print (cv_lasso)

cv_lasso = pd.Series(cv_lasso, index = lasso_alphas)
cv_lasso.plot(title = "Lasso Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.axis((0, 1, 0.1, 0.3))
plt.show()
cv_lasso.min()

