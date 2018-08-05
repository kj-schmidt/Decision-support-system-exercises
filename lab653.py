import pandas as pd
import numpy as np
import math
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import multiprocessing as mlt
import random

random.seed(1)
data = pd.read_csv("Hitters.csv")
data.head()

# Again we only work on the training set, so we remove the player
# with no salary information
data = data[np.isfinite(data['Salary'])]
# Create dummy variables
# We need to define dummy variables for League, Division og NewLeague
# because all of these are categorical variables
dumVar = pd.get_dummies(data[['League', 'Division', 'NewLeague']])
# Define predictors and drop salary and names
data_drop = data.drop(['Unnamed: 0', 'League', 'Division', 'NewLeague'], axis=1)
# Concat X_drop and dumVar. This is the new data set, with dummy
# variables and only training data
data = pd.concat([data_drop, dumVar[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

data_train=data.sample(frac=0.6,random_state=200)
data_valid=data.drop(data_train.index)

# TODO calculate the validation error

