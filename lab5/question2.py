import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
import sklearn as skl

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

weekly = pd.read_csv("./weekly.csv", usecols = range(1, 10))

# Binarize direction column
def cat_direction(direction):
	return direction == "Up"
weekly['Direction'] = weekly['Direction'].apply(cat_direction)

# Plot every feature vs every other feature
#seaborn_plot = sb.PairGrid(weekly) 
#seaborn_plot = seaborn_plot.map(plt.scatter)
#plt.show()

# Calculate percet volume change
weekly['PercentVolumeChange'] = (weekly['Volume'].shift(1) - weekly['Volume']) / weekly['Volume'].shift(1)
weekly['PercentVolumeChange'][0] =  weekly['PercentVolumeChange'][1]
plt.scatter(weekly['Today'], weekly['PercentVolumeChange'])
plt.title('Percent Change of Volume vs. Today') 
plt.xlabel('Today')
plt.ylabel('Percent Volume Change')
plt.show()

# Plot select feature comparisons
# Lags vs. Year
plt.scatter(weekly['Lag1'], weekly['Year'], c='r')
plt.scatter(weekly['Lag2'], weekly['Year'], c='y')
plt.scatter(weekly['Lag3'], weekly['Year'], c='g')
plt.scatter(weekly['Lag4'], weekly['Year'], c='b')
plt.scatter(weekly['Lag5'], weekly['Year'], c='k')
plt.title('Lags vs. Year Plotted')
plt.xlabel('Lag')
plt.ylabel('Year')
plt.show()

# Direction Vs year
up_count = np.array([])
for year in range(1990,2011):
	year_vals = weekly.loc[weekly['Year'] == year]
	up_count_val = 0
	for entry in year_vals.iterrows():
		if entry[1]['Direction'] == True:
			up_count_val = up_count_val + 1
	up_count = np.append(up_count, up_count_val)
plt.plot(range(1990, 2011), up_count)
plt.title('Number of Positive Directions Per Year')
plt.xlabel('Year')
plt.ylabel('Up Count')
plt.show()

# Volume over time
plt.plot(range(0, len(weekly['Volume'])), weekly['Volume'])
plt.title('Volume vs. Week (Startin in 1990)')
plt.xlabel('Week')
plt.ylabel('Volume')
plt.show()

# Covariance matrix
print np.cov(weekly, rowvar=False) 

# Logistic Regression (Parts b and c)
weekly["TrainID"] = pd.Series(np.random.randint(0, high=2, size=(len(weekly['Year']))), index=weekly.index)
train_weekly = weekly[weekly['TrainID'] == 1]
test_weekly = weekly[weekly['TrainID'] == 0]
train_weekly_y = train_weekly["Direction"]
train_weekly_x = train_weekly[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
test_weekly_y = test_weekly["Direction"]
test_weekly_x = test_weekly[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

log_reg = LogisticRegression()
log_reg.fit(train_weekly_x, train_weekly_y)
log_reg_weekly_y_preds = log_reg.predict(test_weekly_x)
score = log_reg.score(test_weekly_x, test_weekly_y)
conf_matrix = confusion_matrix(test_weekly_y, log_reg_weekly_y_preds)	
print "\nLogistic Regression Coefficients [Lag1, Lag2, Lag3, Lag4, Lag5, Volume]: " + str(log_reg.coef_)
print "Confusion Matrix:"
print conf_matrix
print "Fraction of Correct Predictions: " + str(score)

# Logistic Regression (Part d)
train_weekly = weekly[weekly["Year"] < 2009]
test_weekly = weekly[weekly["Year"] > 2008]
train_weekly_x = train_weekly[['Lag2']]
train_weekly_y = train_weekly[['Direction']]
test_weekly_x = test_weekly[['Lag2']]
test_weekly_y = test_weekly[['Direction']]
log_reg = LogisticRegression()
log_reg.fit(train_weekly_x, train_weekly_y)
log_reg_weekly_y_preds = log_reg.predict(test_weekly_x)
score = log_reg.score(test_weekly_x, test_weekly_y)
conf_matrix = confusion_matrix(test_weekly_y, log_reg_weekly_y_preds)	
print "\nLogistic Regression Coefficients [Lag2]: " + str(log_reg.coef_)
print "Confusion Matrix:"
print conf_matrix
print "Fraction of Correct Predictions: " + str(score)

# LDA using sklearn
from sklearn.lda import LDA
lda = LDA()
lda.fit(train_weekly_x, train_weekly_y)
lda_preds = lda.predict(test_weekly_x)
lda_score = lda.score(test_weekly_x, test_weekly_y)
conf_matrix = confusion_matrix(test_weekly_y, lda_preds)

print "\nLDA Results"
print "Confusion Matrix:"
print conf_matrix
print "Fraction of Correct Predictions: " + str(lda_score)

# QDA using sklearn
from sklearn.qda import QDA

qda = QDA()
qda.fit(train_weekly_x, train_weekly_y)
qda_preds = qda.predict(test_weekly_x)
qda_score = qda.score(test_weekly_x, test_weekly_y)
conf_matrix = confusion_matrix(test_weekly_y, qda_preds)

print "\nQDA Results"
print "Confusion Matrix:"
print conf_matrix
print "Fraction of Correct Predictions: " + str(qda_score)

# KNN using sklearn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_weekly_x, train_weekly_y)
knn_preds = knn.predict(test_weekly_x)
knn_score = knn.score(test_weekly_x, test_weekly_y)
conf_matrix = confusion_matrix(test_weekly_y, knn_preds)

print "\nKNN Results (k=1)"
print "Confusion Matrix:"
print conf_matrix
print "Fraction of Correct Predictions: " + str(knn_score)

# KNN with different k values
k_max = 1
k_max_score = .5
for k in range(2, 11):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(train_weekly_x, train_weekly_y)
	knn_preds = knn.predict(test_weekly_x)
	knn_score = knn.score(test_weekly_x, test_weekly_y)
	if knn_score > k_max_score:
		k_max_score = knn_score
		k_max = k

knn = KNeighborsClassifier(n_neighbors=k_max)
knn.fit(train_weekly_x, train_weekly_y)
knn_preds = knn.predict(test_weekly_x)
conf_matrix = confusion_matrix(test_weekly_y, knn_preds)

print "\nKNN Results (k=" + str(k_max) + ") The best k between 1 and 10"
print "Confusion Matrix:"
print conf_matrix
print "Fraction of Correct Predictions: " + str(k_max_score)
	
# Running logistic regression on the outputs of other methods

# Generate independent predictions for training data
models = [LogisticRegression(), LDA(), KNeighborsClassifier(n_neighbors=k_max)]

train_data = weekly[weekly['Year'] < 2009]
train_data = train_data[['Lag2','Direction']]
train_x = train_data.drop('Direction', 1)
train_y = train_data['Direction']
test_data = weekly[weekly['Year'] > 2008]
test_data = test_data[['Lag2','Direction']]
test_x = test_data.drop('Direction', 1)
test_y = test_data['Direction']

stack_data = train_data.copy()
stack_data["FoldID"] = pd.Series(np.random.randint(1, high=6, size=(train_data.shape[0])), index=stack_data.index)
stack_meta = test_x.copy()
stack_train = stack_data.copy()
for i in range(0, len(models)):
    stack_train["Model " + str(i)] = pd.Series(np.nan, index=stack_train.index)

#%% Create training folds and fill out model predictions in the training data
for fold in range(1, 6):
    # Organize folds
    folds_combined = stack_data[stack_data["FoldID"] != fold]
    folds_train = folds_combined.drop("Direction", 1)
    folds_test = stack_data[stack_data["FoldID"] == fold]
    folds_test = folds_test.reindex()
    y_train = folds_combined["Direction"]
    y_test = folds_test["Direction"]
    folds_test = folds_test.drop("Direction", 1)

    # Train models on training folds, predict test fold
    stack_preds = np.array([])
    for i in range(0, len(models)):
		model = models[i]
		model = model.fit(folds_train, y_train)
		stack_preds = np.append(stack_preds, pd.DataFrame({"preds":model.predict(folds_test)}, index=folds_test.index))
    for i in range(0, len(models)):
        folds_test["Model " + str(i)] = stack_preds[i]
    stack_train.update(folds_test)
    
#%% Using original training data, create predictions on test data

normal_preds = np.array([])
for i in range(0, len(models)):
	model = models[i]
	model = model.fit(train_x, train_y)
	stack_meta["Model " + str(i)] = pd.Series(model.predict(test_x), index=stack_meta.index)

#%% Stack the models
stack_train_final = stack_train.drop(["FoldID", "Direction"], axis=1)

log_reg = LogisticRegression()
log_reg.fit(stack_train_final, train_y)
final_preds = log_reg.predict(stack_meta)
score = log_reg.score(stack_meta, test_y)
conf_matrix = confusion_matrix(test_y, final_preds)	
print "\nStacked Logistic Regression Coefficients [Lag2, LDA, KNN]: " + str(log_reg.coef_)
print "Confusion Matrix:"
print conf_matrix
print "Fraction of Correct Predictions: " + str(score)
