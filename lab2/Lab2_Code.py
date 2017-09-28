from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np
import seaborn as sb

#%%
### Question 1 ###
df1 = pd.read_csv("./DF1", header=None)
fig,axes = plt.subplots(nrows=4, ncols=4)

#plot using pandas
for i in range(1,5):
    for j in range(1,i):
        t = df1.plot.scatter(x=[], y=[], ax=axes[i-1,j-1])
        t.axis('off')
    for j in range(i,5):
        title = "column " + str(i) + " vs " + str(j)
        t = df1.plot.scatter(i, j, ax=axes[i-1,j-1], title=title, 
        xticks=[], yticks=[])
plt.show()

#plot using seaborn
df1 = pd.read_csv("./DF1", header=None, usecols = range(1,5))
seaborn_plot = sb.PairGrid(df1)
seaborn_plot = seaborn_plot.map(plt.scatter)
plt.show()

cov_matrix = np.cov(df1, rowvar=False)
print "\nCovariance Matrix"
print cov_matrix

#Formula for three-dimensional gaussian:
#X1 = N(2, 4), X2 = N(-3, 9), X3 = X2 + N(1, 16)

#Calculate covariance matrix and plot error
exp_cov = np.array([[4,0,0],[0,9,8.735],[0,8.735,25]])
calc_cov_matrix_1_2 = np.zeros(998)        
for n in range(2,1000):
    X1 = np.array(datasets.make_gaussian_quantiles([2],4,n,1,1)[0])
    X2 = np.array(datasets.make_gaussian_quantiles([-3],9,n,1,1)[0])
    X3 = X2 + np.array(datasets.make_gaussian_quantiles([1],16,n,1,1)[0])
    dataset = np.concatenate((X1, X2, X3),1) 
    cov_matrix = np.cov(dataset, rowvar=False)
    calc_cov_matrix_1_2[n-2] = cov_matrix[1,2]
    
exp_cov_vec = np.ones(998) * 8.735
plt.plot(range(0,998), calc_cov_matrix_1_2, range(0,998), exp_cov_vec, 'r')
plt.xlabel('n')
plt.ylabel('Correlation')
plt.title('Empircal Correlation Converging to True Corrleation')
plt.legend(['empirical correlation', 'true correlation'])
plt.show()

#%%
### Question 2 ###
df = pd.read_csv("./DF2",index_col=0)
plt.scatter(df.T.values[0], df.T.values[1])
plt.show() #shows the original plot

#in order to normalize one dimensional data, we divide each datapoint by its standard deviation,
#with 2 dimensional data, what we can do is divide each datapoint by the square root of the covariance matrix
cov_mat = np.array(df.cov())
toBeDivided = np.linalg.inv(np.sqrt(cov_mat))
Z = toBeDivided.dot(df.T)
plt.scatter(Z[0], Z[1])
plt.show()

#with the plot from the normalized data, we can see that the datapoint at (-1, 1) is an outlier

#%%
### Question 3 ###
n_vec = np.linspace(30, 1000, 25, dtype=int)
b_std_dev_vec = np.zeros(n_vec.size, dtype=float)

sqrt_n_vec = np.zeros(n_vec.size, dtype=float)
n_count = 0     

for n in n_vec:
    b_vec = np.zeros(100)
    #Calculate empirical standard deviation
    for sample in range(0, 100): # Choice of 100 is arbitrary
        #Get data
        x = np.array(datasets.make_gaussian_quantiles([0],1,n,1,1)[0])
        ones = np.array(np.ones(x.shape))
        e = np.array(datasets.make_gaussian_quantiles([0],1,n,1,1)[0])
        b_o = -3
        b_1 = 0
        y = b_o + b_1*x + e

        #Calculate error        
        b_vec[sample] = np.sum(x*e)/np.sum(x*x)
        
    #Calculate standard deviation of error
    b_std_dev_vec[n_count] = np.sqrt(np.sum(b_vec * b_vec) / n)
    #Generate error trend line
    sqrt_n_vec[n_count] = 1/np.sqrt(n)
    n_count = n_count + 1

plt.plot(n_vec, b_std_dev_vec, n_vec, sqrt_n_vec, 'r-', antialiased=True)
plt.title('Question 3: Standard Deviation of Error vs. Number of Samples')
plt.xlabel('n')
plt.ylabel('error')
plt.legend(['empirical standard deviation error', 'error trend'])
plt.show()

#%%
### Question 4 ###

#part 1
def top_names_in_year(year, top_k):

    # read from the data file
    try:
        file_path = "./Names/yob" + str(year) + ".txt"
        file = open(file_path, "r")
    except (OSError, IOError) as e:
        print "no data for this year"
    file.close()

    #load csv data to a dataframe
    df = pd.read_csv(file_path, names=["name", "gender", "freq"])

    #first we group the same names together, because one name may be used by both men and women
    df = df.groupby('name')
    #sum the frequencies of the same name
    df = df.sum().reset_index()
    #sort the dataframe in descending order
    df = df.sort_values('freq', ascending=False)
    return df.head(top_k)['name']

top_names_in_year(2010, 10)


#part 2
def frequency(name):
    #initialize a result dataframe
    result = pd.DataFrame(columns=["name", "gender", "frequency", "year"],index=None)
    #for each year
    for year in range(1880, 2016):
        #read and load data
        file_path = "./Names/yob" + str(year) + ".txt"
        df = pd.read_csv(file_path, names=["name", "gender", "frequency"])
        #locate the row where the "name" field is the same as name
        toAppend = df.loc[df['name'] == name]
        #specify the year
        toAppend['year'] = year
        #append it to result
        result = result.append(toAppend,ignore_index=True)
    return result.loc[result['gender'] == "M"].sum()['frequency'], result.loc[result['gender'] == "F"].sum()['frequency']

frequency("Taylor")

#part 3
def relativeFrequency(name):
    #initialize result dataframe
    result = pd.DataFrame(columns=["name", "gender", "relative frequency", "year"], index=None)
    #for each year
    for year in range(1880, 2016):
        #read data
        file_path = "./Names/yob" + str(year) + ".txt"
        df = pd.read_csv(file_path, names=["name", "gender", "frequency"])

        #add a 2-element array specifying the total frequencies for

        total = {'M': sum(df[df['gender'] == 'M']['frequency'].values), 'F': sum(df[df['gender'] == 'F']['frequency'].values)}
        df['total'] = df['gender'].map(total)

        toAppend = df.loc[df['name'] == name]
        toAppend['year'] = year

        toAppend['relative frequency'] = df['frequency']/df['total']
        result = result.append(toAppend, ignore_index=True)
    return result.drop(['frequency', 'total'], axis=1)

relativeFrequency('Taylor')

#part 4
def names_with_changing_gender():
    #initialize the "final" dataframe
    dff = pd.DataFrame()

    #within this loop, we append the dataframe for each year to dff, also we add a column "realtive frequency" to dff
    for year in range(1880, 2016):
        file_path = "./Names/yob" + str(year) + ".txt"
        df = pd.read_csv(file_path, names=["name", "gender", "frequency"])

        #this block gets the relative frequency
        total = {'M': sum(df[df['gender'] == 'M']['frequency']), 'F': sum(df[df['gender'] == 'F']['frequency'])}
        df['total'] = df['gender'].map(total)
        multiply = df['gender'].map({'M': 1, 'F': -1})
        df['relative frequency'] = df['frequency'] / df['total']
        df['relative frequency'] = df['relative frequency'] * multiply


        df = df.drop('frequency', axis=1).drop('total', axis=1)#drop frequency as it's no longer needed
        groupped = df.groupby('name')['relative frequency'].agg(['min', 'max']).reset_index()

        # this line finds populates the "more popular" cell such that it's positive when it's more popular among males, and negative when it's more popular among females,
        #simply put, it finds out whether the name is more popular among men or women in this current year
        groupped['more popular'] = groupped['min'] + groupped['max']

        groupped = groupped.drop('min',axis=1).drop('max',axis=1) #drop max and min as they are no longer needed
        groupped['year'] = year

        dff = dff.append(groupped)#append to the large data frame

    dff = dff.groupby('name')['more popular'].agg(['min', 'max'])
    return dff[dff['min'] * dff['max'] < 0].reset_index() #this finds all names that were once male popular then became female popular, or the other way around

print names_with_changing_gender()

#%%
### Question 5 ###
def assign_state(row):
    # Notice how the states in the following arrays match index-wise. I use this to my advantage below.
    postal_codes = [ 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN',
		     'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
		     'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
		     'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    state_names =  [ 'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
		     'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
		     'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine',
		     'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri',
		     'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico',
		     'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
		     'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee',
		     'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin',
 		     'wyoming']

    potential_states = [] 
    location = row["user_location"]   
    if type(location) is not str:
        return 'NaN'
    location_words = location.split()
    
    for word in location_words:
	 for i in range(0, 50):
            if word.upper() == postal_codes[i]:
                potential_states.append(postal_codes[i])
            if word.lower() == state_names[i]:
                potential_states.append(postal_codes[i])
    if len(potential_states) == 0:
         return 'NaN'
    if len(potential_states) == 1:
	      return potential_states[0]
    else:
       #Handle multiple possible states
    	 return potential_states[0]

tweets = pd.read_csv("tweets.csv")
tweets["state"] = tweets.apply(assign_state, axis=1)
counts = tweets["state"].value_counts()
print counts




