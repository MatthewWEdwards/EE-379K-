import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import sklearn as skl

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

train_data = pd.read_csv("./train_final.csv")
test_data = pd.read_csv("./test_final.csv")

train_features = train_data.drop("Y")
test_features = test_data.copy()

#Find some relationships between data


