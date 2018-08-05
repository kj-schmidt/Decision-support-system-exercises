import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



data = pd.read_csv("Smarket.csv")
data = data.drop('Unnamed: 0', 1) # removing index column
trainData = data[data['Year'] != 2005]
testData = data[data['Year'] == 2005]

X = np.column_stack([trainData['Lag1'], trainData['Lag2']])
X_test = np.column_stack([testData['Lag1'], testData['Lag2']])
y = trainData['Direction']

y_true = np.array(testData['Direction'])

downs = 0
ups = 0
for i in y_true:
    if i == 'Up':
        ups += 1
    else:
        downs +=1
print(ups, downs)


clf = LinearDiscriminantAnalysis()
clf = clf.fit(X,y)

# Prior prob of groups
print('Prior', clf.priors_)

# Group means
print('Group means',clf.means_)

# LDA coef
print('Coefficients', clf.coef_)

# Interpret results
results = clf.predict(X_test)
ups_true = 0
ups_false = 0
downs_true = 0
downs_false = 0
for i in range(len(results)):
    if results[i] == 'Up' and y_true[i] == 'Up':
        ups_true +=1
    elif results[i] == 'Up' and y_true[i] == 'Down':
        ups_false +=1
    elif results[i] == 'Down' and y_true[i] == 'Down':
        downs_true +=1
    else:
        downs_false +=1

print('Confusion Matrix:')
print([downs_true, downs_false])
print([ups_false, ups_true])

mean_predict = float((downs_true+ups_true))/len(results)
#mean_predict = 141/252
print('Prediction mean', mean_predict)


#clf.fit(data.loc[:,'Lag1', 'Lag2'].values, data.Direction)

