import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
import operator
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier

"""Forklaring fra "http://scikit-learn.org/stable/modules/neighbors.html":
The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance
to the new point, and predict the label from these. The number of samples can be a user-defined constant 
(k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). 
The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. 
Neighbors-based methods are known as non-generalizing machine learning methods, since they simply "remember" all of 
its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree.)"""



data = pd.read_csv("Smarket.csv")
data = data.drop('Unnamed: 0', 1) # removing index column
trainData = data[data['Year'] != 2005]
testData = data[data['Year'] == 2005]

# Sort the data
trainX = np.column_stack([trainData['Lag1'], trainData['Lag2']])  # Training data
trainY = trainData['Direction']
testX = np.column_stack([testData['Lag1'], testData['Lag2']])  # Test data
y_true = np.array(testData['Direction'])
""" Done using predefined library
# K nearest neighbor classification
knn = KNeighborsClassifier(n_neighbors=3)
# Train the model
knn.fit(trainX, trainY)
results = knn.predict(testX)

# Interpret results
ups_true = 0
ups_false = 0
downs_true = 0
downs_false = 0
for i in range(len(results)):
    if results[i] == 'Up' and testResult[i] == 'Up':
        ups_true +=1
    elif results[i] == 'Up' and testResult[i] == 'Down':
        ups_false +=1
    elif results[i] == 'Down' and testResult[i] == 'Down':
        downs_true +=1
    else:
        downs_false +=1

print('Confusion Matrix:')
print([downs_true, downs_false])
print([ups_false, ups_true])

mean_predict = float((downs_true+ups_true))/len(results)
#mean_predict = 141/252
print('Prediction mean', mean_predict)."""

k = 3
n_train = np.size(trainX,0)
n_test = np.size(testX,0)
prediction = []
for i in range(n_test):
    dists = []
    neighborIdx = []
    upScore, downScore = 0, 0
    for ii in range(n_train): # Compute the distance to each point with respect to xtest[i]
        dists.append(np.linalg.norm(testX[i]-trainX[ii]))
    dists = np.array(dists)
    neighborIdx = dists.argsort()[:k] # Contains the index of k neighbors

    for ii in range(len(neighborIdx)): # Determine which class the k'th neighbors in the training set that belows to
        if trainY[neighborIdx[ii]] == 'Up':
            upScore += 1
        else:
            downScore += 1

    if upScore > downScore: # Determine whether the current test observation should belong to "up" or "down"
        prediction.append('Up')
    else:
        prediction.append('Down')

ups_true = 0
ups_false = 0
downs_true = 0
downs_false = 0
for i in range(len(prediction)):
    if prediction[i] == 'Up' and y_true[i] == 'Up':
        ups_true +=1
    elif prediction[i] == 'Up' and y_true[i] == 'Down':
        ups_false +=1
    elif prediction[i] == 'Down' and y_true[i] == 'Down':
        downs_true +=1
    else:
        downs_false +=1

print('Confusion Matrix:')
print([downs_true, downs_false])
print([ups_false, ups_true])

mean_predict = float((downs_true+ups_true))/len(prediction)
print('Prediction mean', mean_predict)




