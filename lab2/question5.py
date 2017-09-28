from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np
import seaborn as sb


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

    potential_states = [] # Indices of potential states
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
	# Handle multiple possible states
	return potential_states[0]

tweets = pd.read_csv("tweets.csv")
tweets["state"] = tweets.apply(assign_state, axis=1)
counts = tweets["state"].value_counts()
print counts


