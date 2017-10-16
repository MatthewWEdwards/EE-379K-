###### Problem 1 ######

### Part 1 ###
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

# this block creates the matrix X with shape 51:50
def get_X():
    return np.random.normal(0,1,(51,50))

# this block generates the beta (vector of all 1's)
def get_b():
    b = np.ndarray((50,1),float)
    b.fill(1.0)
    return b

# this block creates the noise vector e with shape 51:1
def get_e():
    return np.random.normal(0,0.25,51).reshape(-1,1)


lr = LinearRegression(fit_intercept=False)
b_hats = []
for i in range(100):
    X = get_X()
    b = get_b()
    e = get_e()
    y = np.dot(X,b) + e
    lr.fit(X,y)
    b_hats.append(lr.coef_)

b_hats = np.asarray(b_hats)
print b_hats.mean(axis=0)
print b_hats.var(axis=0).mean()


### Part 2 ###
for a in [0.01,0.1,1,10,100]:
    ridge = Ridge(alpha=a)
    X = get_X()
    b = get_b()
    e = get_e()
    y = np.dot(X,b) + e
    ridge.fit(X,y)
    
    b_hat = ridge.coef_
    print "l1 norm: " + str(np.linalg.norm(b_hat,ord=1))
    print "score:" + str(ridge.score(X,y))