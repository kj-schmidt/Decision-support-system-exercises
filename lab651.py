import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt


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
   return {'model':regr, 'RSS': RSS}

def bestSubsetSelect(k):
    start = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))

    models = pd.DataFrame(results)
    best_subset = models.loc[models['RSS'].argmin()]

    end = time.time()

    print("Processed ", models.shape[0], "models on", k, "predictors in",(end - start), "seconds.")
    return best_subset

models = pd.DataFrame(columns=['RSS', 'model'])
start = time.time()
k = 8
for i in range(1,k):
    models.loc[i] = bestSubsetSelect(i)
end = time.time()

print('Total elapsed time %.4f seconds' % (end-start))
RSS = models['RSS'].tolist()
noOfPredictors = list(range(1,k))

plt.plot(noOfPredictors, RSS)
plt.xlabel('Number of predictors')
plt.ylabel('RSS')
plt.show()

