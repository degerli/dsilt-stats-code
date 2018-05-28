#Python code for chapter 5 of DSILT: Statistics

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

train = pd.read_csv("train_clean.csv", sep=",")
test = pd.read_csv("test_clean.csv", sep=",")
train['Set'] = 'train'
test['Set'] = 'test'
alldata = pd.concat([train.drop(['Survived'], axis=1), test], ignore_index=True)

#Covariance and correlation between 2 variables
print(alldata.cov()['Fare_Per_Person'])
print(alldata.corr(method='pearson')['Fare_Per_Person'])

plt.scatter(alldata['Fare_Per_Person'], alldata['Age'])
plt.show()

print(stats.pointbiserialr(np.asarray(alldata[~np.isnan(alldata['Age'])]['DeckA']), np.asarray(alldata[~np.isnan(alldata['Age'])]['Age'])))
