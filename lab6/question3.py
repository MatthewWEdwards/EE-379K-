
###### Problem 3 ######
### Part a ###
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score


df = pd.read_csv('./Boston.csv')

y = df['crim']
X = df[['zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']]

print y.shape
print X.shape

def mse_cv(model,X,y):
    mse= -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf_10)
    return mse

# lasso
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

model_lasso = linear_model.LassoCV(alphas = [0.001,0.01,0.1,1,10,100]).fit(X, y)
mse_cv(model_lasso,X,y).mean()

# ridge
model_ridge = linear_model.RidgeCV(alphas = [0.001,0.01,0.1,1,10,100]).fit(X, y)
mse_cv(model_ridge,X,y).mean()

# pcr
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set_style('darkgrid')

X_reduced = pca.fit_transform(scale(X))
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

n = len(X_reduced)
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=2)

regr = LinearRegression()
mse = []

score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)), y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
mse.append(score) 

for i in np.arange(1,14):
    score = -1*cross_validation.cross_val_score(regr, X_reduced[:,:i], y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    print score
    mse.append(score)

fig, (ax1) = plt.subplots(1, figsize=(12,5))
ax1.plot(mse, '-v')

for ax in fig.axes:
    ax.set_xlabel('Number of principal components in regression')
    ax.set_ylabel('MSE')
    ax.set_xlim((-0.2,13.2))