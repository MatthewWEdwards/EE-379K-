import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#%% Pass 0 is we don't want to plot
show_plots_flag = 1
if(len(sys.argv) > 1):
	if sys.argv[1] == '0':
		show_plots_flag = 0
		
#%% collect some data, do some basic transforms
college = pd.read_csv("./college.csv")
college_names = college["Name"]
college = college.drop(["Name"], axis=1)

def if_private(private):
	return private == "Yes"
college["Private"] = college["Private"].apply(if_private)

#%% Randomly select testing and training data
college["FoldID"] = pd.Series(np.random.randint(1, high=7, size=(college.shape[0])), index=college.index)
college_train = college[college["FoldID"] != 1]
college_test = college[college["FoldID"] == 1]
college_train.drop(["FoldID"], axis=1)
college_test.drop(["FoldID"], axis=1)
college_train_x = college_train.drop(["Apps"], axis=1)
college_train_y = college_train["Apps"]
college_test_x = college_test.drop(["Apps"], axis=1)
college_test_y = college_test["Apps"]

#%% Cross validation and rmse functions
from sklearn.cross_validation import cross_val_score

def rmse_cv(model, x, y):
    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

def rmse(model,x ,y):
    rmse = np.sqrt(((model.predict(x) - y)**2).mean())
    return (rmse)
	
#%% Naive linear regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
LRM = LinearRegression()
print "Linear Regression RMSE from cross validation:"
print rmse_cv(LRM, college_train_x, college_train_y).mean()
LRM.fit(college_train_x, college_train_y)
print "\nLinear Regression test RMSE"
print rmse(LRM, college_test_x, college_test_y)

#%% Ridge Regression
ridge_alphas = np.concatenate((np.linspace(0.1, 1, num=10),
                               np.linspace(1, 10, num=10),
                               np.linspace(10, 100, num=10),
                               np.linspace(100, 1000, num=10),
                               np.linspace(1000, 10000, num=10),
                               np.linspace(10000, 100000, num=10),
                               np.linspace(100000, 1000000, num=10)))

cv_ridge = [rmse_cv(Ridge(alpha = alpha), college_train_x, college_train_y).mean() for alpha in ridge_alphas]
min_alpha = ridge_alphas[np.argmin(cv_ridge)]

cv_ridge = pd.Series(cv_ridge, index = ridge_alphas)
cv_ridge.plot(title = "Ridge Cross Validation", logx=True)
plt.xlabel("Alpha")
plt.ylabel("Root Mean Square Error")
if show_plots_flag:
	plt.show()

best_ridge = Ridge()
best_ridge.fit(college_train_x, college_train_y)
print "\nRidge Regression test RMSE (alpha = " + str(min_alpha) + ")"
print rmse(best_ridge, college_test_x, college_test_y)

#%% Lasso Regression
lasso_alphas = np.concatenate((np.linspace(0.00001, 0.0001, num=10),
                               np.linspace(0.0001, 0.001, num=10),
                               np.linspace(0.001, 0.01, num=10),
                               np.linspace(0.1, 1, num=10),
                               np.linspace(1, 10, num=10),
                               np.linspace(10, 100, num=10),
                               np.linspace(100, 1000, num=10)))
cv_lasso = [rmse_cv(Lasso(alpha = alpha), college_train_x, college_train_y).mean() for alpha in lasso_alphas]
min_alpha = lasso_alphas[np.argmin(cv_lasso)]
cv_lasso = pd.Series(cv_lasso, index = lasso_alphas)
cv_lasso.plot(title = "Lasso Cross Validation", logx=True)
plt.xlabel("Alpha")
plt.ylabel("Root Mean Square Error")
if show_plots_flag:
	plt.show()

best_lasso = Lasso()
best_lasso.fit(college_train_x, college_train_y)
print "\nLasso Regression test RMSE (alpha = " + str(min_alpha) + ")"
print rmse(best_lasso, college_test_x, college_test_y)
print "\nLasso L0 norm"
print np.linalg.norm(best_lasso.coef_, ord=0)

#%% PCA
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(college_train_x, college_train_y)

cv_pcr = np.array([])
num_comp_array = range(1, 18)
for n_comp in num_comp_array:
	pca = PCA(n_components = n_comp)
	reduced_college_train_x = pca.fit_transform(college_train_x, college_train_y)
	pca_this_rmse = rmse_cv(LinearRegression(), reduced_college_train_x, college_train_y).mean()
	cv_pcr = np.append(cv_pcr, pca_this_rmse)

plt.plot(num_comp_array, cv_pcr)
plt.title('PCR Cross Validation')
plt.xlabel("Number of Components (M)")
plt.ylabel("Root Mean Square Error")
if show_plots_flag:
	plt.show()

opt_m = num_comp_array[np.argmin(cv_pcr)]
pcr_opt = PCA(n_components = opt_m)
pcr_opt.fit(college_train_x, college_train_y)
reduced_college_test_x = pcr_opt.transform(college_test_x)
reduced_college_train_x = pcr_opt.transform(college_train_x)
lrm = LinearRegression()
lrm.fit(reduced_college_train_x, college_train_y)
print "\nPCR RMSE (M = " + str(opt_m) + ")"
print rmse(lrm, reduced_college_test_x, college_test_y)

#%% PLS
from sklearn.cross_decomposition import PLSRegression

pls_components = range(1,18)

cv_pls = np.array([])
for m in pls_components:
	pls = PLSRegression(n_components = m)
	foo = np.transpose(college_train_x.get_values())
	transformed_college_train_x = pls.fit_transform(college_train_x, college_train_y)[0]
	lrm = LinearRegression()	
	pls_this_rmse = rmse_cv(LinearRegression(), transformed_college_train_x, college_train_y).mean()
	cv_pls = np.append(cv_pls, pls_this_rmse)

min_m = pls_components[np.argmin(cv_pls)]
cv_pls = pd.Series(cv_pls, index = pls_components)
cv_pls.plot(title = "PLSRegression Cross Validation")
plt.xlabel("Number of Components (M)")
plt.ylabel("Root Mean Square Error")
if show_plots_flag:
	plt.show()

best_pls = PLSRegression(n_components = min_m)
transformed_college_train_x = best_pls.fit_transform(college_train_x, college_train_y)[0]
transformed_college_test_x = best_pls.transform(college_test_x)
lrm = LinearRegression()
lrm.fit(transformed_college_train_x, college_train_y)
print "\nPLSRegression Regression test RMSE (M = " + str(min_m) + ")"
print rmse(lrm, transformed_college_test_x, college_test_y)

