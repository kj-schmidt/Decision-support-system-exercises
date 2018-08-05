
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats

import csv

### 3.6.2 Simple Linear Regression ###

# Load the csv file
rawdata = np.genfromtxt('boston.csv', dtype=float, delimiter=',', names=True)
# Column names: "crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat","medv"
# Get the column names: print(rawdata.dtype.names)
# print(rawdata['medv'])

x = rawdata['lstat']
y = rawdata['medv']
# Get visual overview of samples
plt.plot(x,y,'o')
fit = np.polyfit(x,y,1) # Gives [slope, intercept]
fit_func = np.poly1d(fit)
plt.plot(x, fit_func(x),'r')
plt.ylabel('Median house value')
plt.xlabel('Percent of households with low socioeconomic status')
#plt.show()

# Assessing the Accuracy of the Coefficient Estimates from slide 7
RSS = sum((y-fit_func(x))**2) # Residual sum of squares
RSE = np.sqrt(RSS/(len(y)-2))**2 # Residual square error
SE_slope = np.sqrt(RSE/sum((x-np.mean(x))**2)) # Beta_1 (slope)
SE_int = np.sqrt(RSE*(1/len(x)+np.mean(x)**2/(sum((x-np.mean(x))**2)))) # Beta_0 (intercept)
#print('SE_slope', SE_slope, 'SE_int', SE_int)

# Calculate the confidens intervals
conf_slope = [fit[0]-2*SE_slope, fit[0]+2*SE_slope]
conf_int = [fit[1]-2*SE_int, fit[1]+2*SE_int]

# Calculate test statistics
t_slope = (fit[0]-0)/SE_slope
t_int = (fit[1]-0)/SE_int

print("Slope: \t\t %.4f [%.4f, %.4f], T-value: %.4f" % (fit[0],conf_slope[0],conf_slope[1],t_slope))
print("Intercept: \t %.4f [%.4f, %.4f], T-value: %.4f" % (fit[1],conf_int[0],conf_int[1],t_int))

#print tabulate([['Intercept', fit[1], conf_int[0],conf_int[1]],['Slope', fit[0], conf_slope[0],conf_slope[1]]], headers=['Param','Value','Lower conf','upper conf'],tablefmt='orgtbl')

# These 3 lines calcs the parameters for the linear regression: 
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# print("Slope: %.4f, Intercept: %.4f, P-Value: %.6f" % (slope, intercept, p_value))
# print('Std. err.:', std_err)

