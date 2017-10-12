########## question 4

######## a
import numpy as np
import math
np.random.seed(1)
y = np.random.randn(100)
x = np.random.randn(100)
y = x - 2 * np.power(x,2)+ np.random.randn(100)

######## b
plt.scatter(x,y)
plt.show()


######## c
from sklearn.cross_validation import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1)
p_order = np.arange(1,5)

# LeaveOneOut CV
regr = skl_lm.LinearRegression()
loo = LeaveOneOut(100)
scores = list()

for i in p_order:
    poly = PolynomialFeatures(i)
    X_poly = poly.fit_transform(x.reshape(-1,1))
    score = cross_val_score(regr, X_poly, y, cv=loo, scoring='neg_mean_squared_error').mean()
    scores.append(-score)
print scores



######## d
np.random.seed(2)
p_order = np.arange(1,5)

# LeaveOneOut CV
regr = skl_lm.LinearRegression()
loo = LeaveOneOut(100)
scores = list()

for i in p_order:
    poly = PolynomialFeatures(i)
    X_poly = poly.fit_transform(x.reshape(-1,1))
    score = cross_val_score(regr, X_poly, y, cv=loo, scoring='neg_mean_squared_error').mean()
    scores.append(-score)
print scores

######## e
# They are exactly the same, since LOOCV evaluates n folds of a single observation.

######## f