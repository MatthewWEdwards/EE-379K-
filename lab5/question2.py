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
variances = np.linalg.inv((np.diag(np.var(weekly)))**.5)
corr_matrix = variances * np.cov(weekly, rowvar=False) *variances
print corr_matrix

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
confusion_matrix = confusion_matrix(test_weekly_y, log_reg_weekly_y_preds)	
print "Score: " + str(score)
print "Logistic Regression Coefficients [Lag1, Lag2, Lag3, Lag4, Lag5, Volume]: " + str(log_reg.coef_)
print "Confusion Matrix:"
print confusion_matrix

# Logistic Regression (Part d)
