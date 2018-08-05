# Karl-Johan Schmidt, 201270751

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# 3.6.3 Simple Linear Regression ###

# Load the csv file
rawdata = np.genfromtxt('boston.csv', dtype=float, delimiter=',', names=True)


data = pd.DataFrame({'y': rawdata['medv'],'x1': rawdata['crim'], 'x2': rawdata['zn'],'x3': rawdata['indus'],'x4': rawdata['chas'],'x5': rawdata['nox'],'x6': rawdata['rm'],'x7': rawdata['age'],'x8': rawdata['dis'],'x9': rawdata['rad'],'x10': rawdata['tax'], 'x11': rawdata['ptratio'],'x12': rawdata['black'],'x13': rawdata['lstat']})

model = ols("y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13",data).fit()

print(model.summary())

