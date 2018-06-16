#Python code for chapter 22 DSILT: Statistics

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

d = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/17-20 and 22 Iris and Motor Trends Cars/iris.csv')
le = LabelEncoder()
d['Species'] = le.fit_transform(d['Species'])
d = d.values

n_iterations = 1000
n_size = int(len(d)*0.5)

'''
-------------------------------------------------------------------------------
-------------------------------Bootstrapping-----------------------------------
-------------------------------------------------------------------------------
'''

#Bootstrap estimate the classification accuracy of a decision tree
scores = list()
for i in range(n_iterations):
    train = resample(d, n_samples=n_size)
    x_train = train[:,0:3]
    y_train = train[:,4].reshape(-1,1)
    test = np.array([x for x in d if x.tolist() not in train.tolist()])
    x_test = test[:,0:3]
    y_test = test[:,4].reshape(-1,1)
    dtc_model = DecisionTreeClassifier()
    dtc_model.fit(x_train, y_train)
    predictions = dtc_model.predict(x_test)
    score = accuracy_score(y_test, predictions)
    scores.append(score)
plt.hist(scores)
plt.show()
alpha_level_of_sig = 0.95
p_val_lower = ((1.0-alpha_level_of_sig)/2.0)*100
lower_ci = max(0.0, np.percentile(scores, p_val_lower))
p_val_upper = (alpha_level_of_sig+((1.0-alpha_level_of_sig)/2.0))*100
upper_ci = min(1.0, np.percentile(scores, p_val_upper))
print('95% Confidence Interval:', lower_ci*100, upper_ci*100)

#Bootstrap estimate of regression coefficients
#Regress Spepal.Length on the other vars
coefs = list()
for i in range(n_iterations):
    train = resample(d, n_samples=n_size)
    x_train = train[:,1:4]
    y_train = train[:,0].reshape(-1,1)
    test = np.array([x for x in d if x.tolist() not in train.tolist()])
    x_test = test[:,1:4]
    y_test = test[:,0].reshape(-1,1)
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    coefs.append(lr_model.coef_)
coefsarr = np.array(coefs).reshape((1000, 3)).T
plt.hist(coefsarr[0])  #Plot of coefficients for first var only
plt.show()
alpha_level_of_sig = 0.95
p_val_lower = ((1.0-alpha_level_of_sig)/2.0)*100
lower_ci = max(0.0, np.percentile(coefsarr[0], p_val_lower))
p_val_upper = (alpha_level_of_sig+((1.0-alpha_level_of_sig)/2.0))*100
upper_ci = min(1.0, np.percentile(coefsarr[0], p_val_upper))
print('95% Confidence Interval:', lower_ci*100, upper_ci*100)

'''
-------------------------------------------------------------------------------
-------------------------K-Fold Cross Validation-------------------------------
-------------------------------------------------------------------------------
'''

from sklearn.model_selection import KFold, RepeatedKFold, train_test_split, GridSearchCV, cross_val_score

#K-fold cross val to get the average accuracy of a model over the folds
#This is useful for robust model assessment so different models can be compared

#Specify the model
class_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=20)
#Set up k-fold
kf = KFold(n_splits=10, shuffle=True, random_state=14)
#Fit the model to each fold and track accuracy
scores = list()
for train_index, test_index in kf.split(d):
    x_train, x_test = d[train_index, 0:3], d[test_index, 0:3]  #Columns 0-3 are predictors
    y_train, y_test = d[train_index, 4], d[test_index, 4]      #Column 4 is target
    class_tree.fit(x_train, y_train)
    predictions = class_tree.predict(x_test)
    score = accuracy_score(y_test, predictions)
    scores.append(score)
print('Mean accuracy score over 10 folds:', np.mean(scores))

#K-fold cross val to tune a model's hyperparameters

#Split dataset into training and test sets (CV will further split up the training set)
x_train, x_test, y_train, y_test = train_test_split(d[:, 0:3], d[:, 4], test_size=0.20, random_state=14)
#Specify the model
class_tree = DecisionTreeClassifier(random_state=14)
#List the hyperparameters to be tuned
param_grid = [{'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 5, 10, 20]}]
#Perform grid search 10-fold cross validation on the hyperparameters listed, using accuracy as the evaluation metric
gs = GridSearchCV(estimator=class_tree, param_grid=param_grid, scoring='accuracy', cv=10, return_train_score=False)
gs.fit(x_train, y_train)
print('Hyperparameters of best model (the model with the highest mean accuracy over 10 folds):', gs.best_params_)
print(pd.DataFrame.from_dict(gs.cv_results_))  #Shows the full results that can be inspected to verify the best_params_ and mean score
#Optional outer cross validation loop: this is nested cross validation, as the outer CV is run over the model tuned with inner CV
scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=10)
print('Mean accuracy of optimally tuned classification tree over 10 folds: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#Same thing using repeated 10-fold cross validation

#Split dataset into training and test sets (CV will further split up the training set)
x_train, x_test, y_train, y_test = train_test_split(d[:, 0:3], d[:, 4], test_size=0.20, random_state=14)
#Specify the model
class_tree = DecisionTreeClassifier(random_state=14)
#List the hyperparameters to be tuned
param_grid = [{'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 5, 10, 20]}]
#Set up k-fold
rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=14)
#Perform grid search 10-fold cross validation on the hyperparameters listed, using accuracy as the evaluation metric
gs = GridSearchCV(estimator=class_tree, param_grid=param_grid, scoring='accuracy', cv=rkf, return_train_score=False)
gs.fit(x_train, y_train)
print('Hyperparameters of best model (the model with the highest mean accuracy over 10 folds):', gs.best_params_)
print(pd.DataFrame.from_dict(gs.cv_results_))  #Shows the full results that can be inspected to verify the best_params_ and mean score
#Optional outer cross validation loop: this is nested cross validation, as the outer CV is run over the model tuned with inner CV
scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=10)
print('Mean accuracy of optimally tuned classification tree over 10 folds: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

