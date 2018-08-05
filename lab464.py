import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib import colors
from scipy import linalg
import matplotlib as mpl



data = pd.read_csv("Smarket.csv")
data = data.drop('Unnamed: 0', 1) # removing index column
trainData = data[data['Year'] != 2005]
testData = data[data['Year'] == 2005]

X = np.column_stack([trainData['Lag1'], trainData['Lag2']])
X_test = np.column_stack([testData['Lag1'], testData['Lag2']])
y = trainData['Direction']
y_true = np.array(testData['Direction'])

clf = QuadraticDiscriminantAnalysis()
clf = clf.fit(X,y, store_covariances=True)

# Prior prob of groups
print('Prior', clf.priors_)

# Group means
print('Group means',clf.means_)
print('Covariances', clf.covariances_)

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
print('Prediction mean', mean_predict)




# Plot results
#plt.subplot(2,1,1)
x1up, x2up, x1down, x2down = [],[],[],[]
ld1up, ld2up, ld1down, ld2down = [],[],[],[]
for i in range(len(results)):
    if y_true[i] == 'Up':
        x1up.append(X_test[i, 0])
        x2up.append(X_test[i, 1])
    else:
        x1down.append(X_test[i, 0])
        x2down.append(X_test[i, 1])
# plt.plot(x1up, x2up , 'go', label="Up")
# plt.plot(x1down, x2down,'ro', label="Down")
# plt.xlabel('Lag1')
# plt.ylabel('Lag2')
# plt.legend(loc=1)

# changes up/down to binary values
results[results== 'Up'] = 1
results[results== 'Down'] = 0

# Calcultes the disciminant values
prostProb = clf.predict_proba(X_test)



plt.subplot(2,1,2)
for i in range(len(results)):
    if results[i] == 1:
        ld1up.append(prostProb[i,0])
        ld2up.append(prostProb[i,1])
    else:
        ld1down.append(prostProb[i,0])
        ld2down.append(prostProb[i,1])


plt.plot(ld1up, ld2up, 'go', label="Up")
plt.plot(ld1down, ld2down, 'ro', label="Down")
plt.xlabel('QD1')
plt.ylabel('QD2')
plt.legend(loc=1)




# The following plot is test points ontop of a grid marking the decision boundary
resolution = 60
points = np.zeros((resolution*resolution, 2))

#plt.figure(figsize=(10, 4))
plt.subplot(2,1,1)

for i in range(1, resolution+1):
    for n in range(1, resolution+1):

        point = [min(X[:,0])+(max(X[:,0])-min(X[:,0]))/resolution*n, min(X[:,1])+(max(X[:,1])-min(X[:,1]))/resolution*i]

        points[(n-1)+(resolution*(i-1)), :] = point


        if clf.predict(point) == 'Up':
            plt.scatter(points[(n-1)+(resolution*(i-1)), 0], points[(n-1)+(resolution*(i-1)), 1],s=1, color='red')
        else:
            plt.scatter(points[(n - 1) + (resolution * (i - 1)), 0], points[(n - 1) + (resolution * (i - 1)), 1],
                        s=1, color='blue')

x1up, x2up, x1down, x2down = [], [], [], []
ld1up, ld2up, ld1down, ld2down = [], [], [], []
for i in range(len(results)):
    if y_true[i] == 'Up':
        x1up.append(X_test[i, 0])
        x2up.append(X_test[i, 1])
    else:
        x1down.append(X_test[i, 0])
        x2down.append(X_test[i, 1])
plt.plot(x1up, x2up, 'go', label="Up")
plt.plot(x1down, x2down, 'ro', label="Down")
plt.xlabel('Lag1')
plt.ylabel('Lag2')
plt.legend(loc=1)
plt.axis('tight')



plt.show()