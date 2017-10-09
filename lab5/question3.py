####### problem 3 ########

# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2.robjects as robjects
import numpy as np
import sklearn.linear_model as skl_lm
from rpy2.robjects import pandas2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split



######## a & b
#note: since part b essentially tells you specifically how to fit a LR model, I will combine part a and b

# loading data
with localconverter(default_converter + pandas2ri.converter) as cv:
    robjects.r['load']('Default.rda')
    df = robjects.r['Default']
print df.head(10)


#### i
# initialize a logistic regression model
lr_model = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')

#split data into sets
df_train, df_test = train_test_split(df, test_size=0.20)

# get training data
X_train = pd.DataFrame(df_train['income'])
y_train = pd.DataFrame(df_train['default'])
mapping = {'No' : 0, 'Yes' : 1}
y_train = np.ravel(y_train['default'].map(mapping))
# y_train = np.ravel(y_train['default'])

# get validation data
X_vali = pd.DataFrame(df_test['income'])
y_vali = np.ravel(pd.DataFrame(df_test['default'].map(mapping)))


##### ii
lr_model = lr_model.fit(X_train, y_train)

#### iii
predictions = lr_model.predict_proba(X_vali)[:,1]
print "prosterior probabilities:", predictions

# classify a row based on its prosterior probability
for i in range(len(predictions)):
    if predictions[i] > .5:
        predictions[i] = 1
    else:
        predictions[i] = 0
print "predictions:", predictions


#### iv
#get the accuracy and then the error
accuracy = accuracy_score(predictions, y_vali)
error = 1 - accuracy
print error






######## c

#### train:test = 6:4

# initialize a logistic regression model
lr_model = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')

#split data into sets
df_train, df_test = train_test_split(df, test_size=0.40)

# get training data
X_train = pd.DataFrame(df_train['income'])
y_train = pd.DataFrame(df_train['default'])
mapping = {'No' : 0, 'Yes' : 1}
y_train = np.ravel(y_train['default'].map(mapping))
# y_train = np.ravel(y_train['default'])

# get validation data
X_vali = pd.DataFrame(df_test['income'])
y_vali = np.ravel(pd.DataFrame(df_test['default'].map(mapping)))

lr_model = lr_model.fit(X_train, y_train)

predictions = lr_model.predict_proba(X_vali)[:,1]

# classify a row based on its prosterior probability
for i in range(len(predictions)):
    if predictions[i] > .5:
        predictions[i] = 1
    else:
        predictions[i] = 0
#get the accuracy and then the error
accuracy = accuracy_score(predictions, y_vali)
error = 1 - accuracy
print error



#### train:test = 4:6

# initialize a logistic regression model
lr_model = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')

#split data into sets
df_train, df_test = train_test_split(df, test_size=0.60)

# get training data
X_train = pd.DataFrame(df_train['income'])
y_train = pd.DataFrame(df_train['default'])
mapping = {'No' : 0, 'Yes' : 1}
y_train = np.ravel(y_train['default'].map(mapping))
# y_train = np.ravel(y_train['default'])

# get validation data
X_vali = pd.DataFrame(df_test['income'])
y_vali = np.ravel(pd.DataFrame(df_test['default'].map(mapping)))

lr_model = lr_model.fit(X_train, y_train)

predictions = lr_model.predict_proba(X_vali)[:,1]

# classify a row based on its prosterior probability
for i in range(len(predictions)):
    if predictions[i] > .5:
        predictions[i] = 1
    else:
        predictions[i] = 0
#get the accuracy and then the error
accuracy = accuracy_score(predictions, y_vali)
error = 1 - accuracy
print error


#### train:test = 2:8

# initialize a logistic regression model
lr_model = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')

#split data into sets
df_train, df_test = train_test_split(df, test_size=0.80)

# get training data
X_train = pd.DataFrame(df_train['income'])
y_train = pd.DataFrame(df_train['default'])
mapping = {'No' : 0, 'Yes' : 1}
y_train = np.ravel(y_train['default'].map(mapping))
# y_train = np.ravel(y_train['default'])

# get validation data
X_vali = pd.DataFrame(df_test['income'])
y_vali = np.ravel(pd.DataFrame(df_test['default'].map(mapping)))

lr_model = lr_model.fit(X_train, y_train)

predictions = lr_model.predict_proba(X_vali)[:,1]

# classify a row based on its prosterior probability
for i in range(len(predictions)):
    if predictions[i] > .5:
        predictions[i] = 1
    else:
        predictions[i] = 0
#get the accuracy and then the error
accuracy = accuracy_score(predictions, y_vali)
error = 1 - accuracy
print error








######## d

#### without using a student dummy variable
# initialize a logistic regression model
lr_model = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')

#split data into sets
df_train, df_test = train_test_split(df, test_size=0.80)

# get training data
X_train = pd.DataFrame(df_train[['income','balance']])
y_train = pd.DataFrame(df_train['default'])
mapping = {'No' : 0, 'Yes' : 1}
y_train = np.ravel(y_train['default'].map(mapping))
# y_train = np.ravel(y_train['default'])

# get validation data
X_vali = pd.DataFrame(df_test[['income','balance']])
y_vali = np.ravel(pd.DataFrame(df_test['default'].map(mapping)))

lr_model = lr_model.fit(X_train, y_train)

predictions = lr_model.predict_proba(X_vali)[:,1]

# classify a row based on its prosterior probability
for i in range(len(predictions)):
    if predictions[i] > .5:
        predictions[i] = 1
    else:
        predictions[i] = 0
#get the accuracy and then the error
accuracy = accuracy_score(predictions, y_vali)
error = 1 - accuracy
print error




#### using a student dummy variable
# initialize a logistic regression model
lr_model = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')

df_train['student'] = df_train['student'].map({'Yes': 1, 'No': 0})
df_train['default'] = df_train['default'].map({'Yes': 1, 'No': 0})
df_test['student'] = df_test['student'].map({'Yes': 1, 'No': 0})
df_test['default'] = df_test['default'].map({'Yes': 1, 'No': 0})

# get training data
X_train = pd.DataFrame(df_train[['income','balance','student']])
y_train = np.ravel(pd.DataFrame(df_train['default']))

# get validation data
X_vali = pd.DataFrame(df_test[['income','balance','student']])
y_vali = np.ravel(pd.DataFrame(df_test['default']))

lr_model = lr_model.fit(X_train, y_train)

predictions = lr_model.predict_proba(X_vali)[:,1]

# classify a row based on its prosterior probability
for i in range(len(predictions)):
    if predictions[i] > .5:
        predictions[i] = 1
    else:
        predictions[i] = 0
#get the accuracy and then the error
accuracy = accuracy_score(predictions, y_vali)
error = 1 - accuracy
print error



