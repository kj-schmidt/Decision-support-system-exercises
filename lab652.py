import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import multiprocessing as mlt


data = pd.read_csv("Hitters.csv")
data.head()

#We remove players with no salary information
data = data[np.isfinite(data['Salary'])]

# We need to define dummy variables for League, Division og NewLeague
# because all of these are categorical variables
dumVar = pd.get_dummies(data[['League', 'Division', 'NewLeague']])
y = data.Salary
# Define predictors and drop salary and names
X_drop = data.drop(['Unnamed: 0', 'Salary', 'League', 'Division', 'NewLeague'], axis=1)
# Concat X_drop and dumVar
X = pd.concat([X_drop, dumVar[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)


def processSubset(input_set):
   """This methods fit the predictors and calculates the RSS. This
   will be called for every combination of predictors"""
   model = sm.OLS(y, X[list(input_set)])
   regr = model.fit()
   RSS = ((regr.predict(X[list(input_set)])-y) ** 2).sum()
   Rsquared = ((regr.predict(X[list(input_set)])-np.mean(y)) ** 2).sum()

   return {'RSQ':Rsquared, 'RSS': RSS, 'Predictors': list(input_set)}


def forwardSubsetSelection(k, fixedPredictors):
    start = time.time()
    minModel, modelCount = [], 0
    curMin = 10**10
    Xtemp = X.drop(fixedPredictors, axis=1)
    for combo in itertools.combinations(Xtemp.columns,
                                        k-len(fixedPredictors)):
        modelCount += 1
        curRSS = processSubset(fixedPredictors +
                               list(combo))['RSS']
        if curRSS < curMin:
            curMin = curRSS
            minModel = processSubset(fixedPredictors
                                     + list(combo))

    end = time.time()
    print('Forward subset selection with %d models on %d'
          ' predictors in %.5f seconds'
          % (modelCount, k, end-start))
    print minModel
    return minModel

# TODO implement backward selection
def backwardSubsetSelection(k, fixedPredictors):
    start = time.time()
    minModel, modelCount = [], 0
    curMin = 10 ** 10
    Xtemp = X.drop(fixedPredictors, axis=1)
    xList = X.ix[:,:k]
    for combo in itertools.combinations(Xtemp.columns, k-1):
        modelCount += 1
        curRSS = processSubset(fixedPredictors +
                               list(combo))['RSS']
        if curRSS < curMin:
            curMin = curRSS
            minModel = processSubset(fixedPredictors + list(combo))

    end = time.time()
    print('Backward subset selection with %d models on '
          '%d predictors in %.5f seconds'
          % (modelCount, k, end - start))
    print minModel
    return minModel


def plotRSSandRSQ(rss, rsq):
    """Plots the numbers of predictor as a function of RSS and 
    R-squared"""
    plt.close('all')
    plt.subplot(211)
    plt.plot(noOfPredictors, rss)
    plt.xlabel('Number of predictors')
    plt.ylabel('RSS')

    plt.subplot(212)
    plt.plot(noOfPredictors, rsq)
    plt.xlabel('Number of predictors')
    plt.ylabel('R-squared')
    plt.show()


models = pd.DataFrame(columns=['RSS', 'model'])
start = time.time()
k = 4
RSQ,RSS, subsetPredictors, fixedPredictors = [], [], [], []
for i in range(1,k):
#for i in range(k,1,-1):
    tempDict = forwardSubsetSelection(i, fixedPredictors)
    RSS.append(tempDict['RSS'])
    RSQ.append(tempDict['RSQ'])
    subsetPredictors.append(tempDict['Predictors'])
    fixedPredictors = tempDict['Predictors']
end = time.time()
print subsetPredictors
print('Total elapsed time %.4f seconds' % (end-start))

noOfPredictors = list(range(1,k))

plotRSSandRSQ(RSS, RSQ)