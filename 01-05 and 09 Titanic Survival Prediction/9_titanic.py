#Python code for chapter 9 of DSILT: Statistics

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

#Fixes a bug in printing output from IDLE, it may not be needed on all machines
import sys
sys.__stdout__ = sys.stdout

train = pd.read_csv("train_clean.csv", sep=",")
test = pd.read_csv("test_clean.csv", sep=",")
train['Set'] = 'train'
test['Set'] = 'test'
alldata = pd.concat([train.drop(['Survived'], axis=1), test], ignore_index=True)

#-------------------------------------------------------------------------------------------------#
#-----------------------------------Final Feature Engineering-------------------------------------#
#-------------------------------------------------------------------------------------------------#

print(alldata.info())
print(alldata.head())

#Add calculated family size (siblings + spourse + parents + children + 1 for self)
alldata['Family_Size'] = alldata.SibSp + alldata.Parch + 1

#Add simple binary variable to indicate whether a passenger had a family
alldata['Has_Family'] = alldata['Family_Size'].apply(lambda x: 1 if (x > 1) else 0)

#Isolate titles from names
alldata['Title'] = alldata['Name'].apply(lambda x: re.split(',|\.', x)[1].strip())

print(alldata['Title'].value_counts())

#Combine similar titles that mean the same thing
myfilter = alldata['Title'][alldata['Title'].apply(lambda x: x in ['Dona', 'Jonkheer', 'Lady', 'Mlle', 'Mme', 'the Countess'])]
alldata.loc[myfilter.index, 'Title'] = 'Mme'
myfilter = alldata['Title'][alldata['Title'].apply(lambda x: x in ['Capt', 'Col', 'Don', 'Sir', 'Major'])]
alldata.loc[myfilter.index, 'Title'] = 'Sir'
myfilter = alldata['Title'][alldata['Title'].apply(lambda x: x in ['Miss', 'Ms'])]
alldata.loc[myfilter.index, 'Title'] = 'Miss'

alldata['Missing_Age'] = alldata['Age'].apply(lambda x: 1 if np.isnan(x) else 0)

#Create distributions to sample from, and then perform piecewise single random imputation
age_dist_child = alldata['Age'][(alldata.Age < 16) & (~np.isnan(alldata.Age))]
age_dist_adult = alldata['Age'][(alldata.Age >= 16) & (~np.isnan(alldata.Age))]

alldata.hist(column='Age')
plt.suptitle('Histogram of Age Before Piecewise Single Random Imputation')
#plt.show()
myfilter = alldata['Age'][(alldata['Title'].apply(lambda x: x in ['Master', 'Miss'])) & (np.isnan(alldata['Age']))]
alldata.loc[myfilter.index, 'Age'] = alldata.loc[myfilter.index, 'Age'].apply(lambda x: random.sample(list(age_dist_child), 1)[0])
alldata.loc[pd.isnull(alldata.Age), 'Age'] = alldata['Age'][pd.isnull(alldata.Age)].apply(lambda x: random.sample(list(age_dist_adult), 1)[0])
alldata.hist(column='Age')
plt.suptitle('Histogram of Age After Piecewise Single Random Imputation')
#plt.show()

#Child indicator
alldata['Is_Child'] = alldata['Age'].apply(lambda x: 1 if x < 16 else 0)

#Dummy encode title - be sure to set dummy encode instead of one-hot
title_dummy = pd.get_dummies(alldata['Title']).drop(alldata.sort_values('Title')['Title'].unique()[0], axis=1).add_prefix('Title_')
alldata = pd.concat([alldata, title_dummy], axis=1)
del title_dummy

#Remove the title column and one column from each group of the one-hot encoded column groups
del alldata['Title'], alldata['Sex_female'], alldata['Embarked_C'], alldata['DeckA']

#Split back into training and test sets
train_clean_feats = alldata.copy().loc[alldata['Set']=='train']
del train_clean_feats['Set']
train_clean_feats['Survived'] = train['Survived']
test_clean_feats = alldata.copy().loc[alldata['Set']=='test']
del test_clean_feats['Set']
#train_clean_feats.to_csv("train_clean_feats.csv", index=False)
#test_clean_feats.to_csv("test_clean_feats.csv", index=False)

#-------------------------------------------------------------------------------------------------#
#--------------------------------------Logistic Regression----------------------------------------#
#-------------------------------------------------------------------------------------------------#

train = pd.read_csv("train_clean_feats.csv")
test = pd.read_csv("test_clean_feats.csv")
del train_clean_feats, test_clean_feats, age_dist_child, age_dist_adult

######
#Testing assumptions

#Are all predictors quantitative or categorical with no more than 2 categories?
print(train.info())  #Oops!  Forgot to get rid of name
del train['Name'], test['Name']
print(train.info())  #Yes - assumption met
print(train.info())  #Yes - assumption met

#Is there a linear relationship between the predictors and the logit?
#Cannot tell yet - need to build the model to get the observed vs predicted probabilities

#Is there multicollinearity among the predictors?
corm = train.corr()
plt.matshow(corm)
#plt.show()  #Possible problems with collinearity here

#Are the residuals homoskedastic?
#Cannot tell yet - need to build the model to get the residuals

#Are the residuals autocorrelated?
#Cannot tell yet - need to build the model to get the residuals

######
#Modeling

#Split training dataset into x and y numpy arrays
array = train.values
x = array[:, :train.shape[1]-1]
y = array[:, train.shape[1]-1]

#Build logit model
logistic_reg = LogisticRegression(fit_intercept=True, random_state=14).fit(x, y)
logistic_reg_preds = logistic_reg.predict(x)
logistic_reg_prob_preds = logistic_reg.predict_proba(x)
print(logistic_reg.get_params, '\n')
print('Intercept:', logistic_reg.intercept_)
for idx, c in enumerate(logistic_reg.coef_[0]):
    print('Coefficient for', train.columns[idx], c)
print('Classification Accuracy:', logistic_reg.score(x, y))

#Confusion matrix
cm = metrics.confusion_matrix(y, logistic_reg_preds)
sns.heatmap(cm, annot=True, fmt=".2f", square=True)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

#Log loss (a.k.a. negative log likelihood)
print('Log loss:', metrics.log_loss(y, logistic_reg_preds))

#Plot ROC curve
fpr, tpr, threshold = metrics.roc_curve(y, logistic_reg_preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'b--')  #Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()

#VIF test for multicollinearity
vif = pd.DataFrame()
vif['VIF_Factor'] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
vif['Feature'] = train.columns[:train.shape[1]-1]
print(vif)

#Get rid of the highly correlated variables and try again
train.drop(['Fare', 'Family_Size', 'DeckZ',
            'Title_Master', 'Title_Miss',
            'Title_Mme', 'Title_Mr', 'Title_Mrs',
            'Title_Rev', 'Title_Sir'], axis=1, inplace=True)
test.drop(['Fare', 'Family_Size', 'DeckZ',
            'Title_Master', 'Title_Miss',
            'Title_Mme', 'Title_Mr', 'Title_Mrs',
            'Title_Rev', 'Title_Sir'], axis=1, inplace=True)

array = train.values
x = array[:, :train.shape[1]-1]
y = array[:, train.shape[1]-1]

vif = pd.DataFrame()
vif['VIF_Factor'] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
vif['Feature'] = train.columns[:train.shape[1]-1]
print(vif)  #Looks better

logistic_reg = LogisticRegression(fit_intercept=True, random_state=14).fit(x, y)
logistic_reg_preds = logistic_reg.predict(x)
logistic_reg_prob_preds = logistic_reg.predict_proba(x)
print(logistic_reg.get_params, '\n')
print('Intercept:', logistic_reg.intercept_)
for idx, c in enumerate(logistic_reg.coef_[0]):
    print('Coefficient for', train.columns[idx], c)
print('Classification Accuracy:', logistic_reg.score(x, y))

#DW test for autocorrelation
residuals = y - logistic_reg_preds
print('DW Statistic:', durbin_watson(residuals))

######
#Evaluating the model on new data

x_test = test.values

logistic_reg_preds = logistic_reg.predict(x_test)
logistic_reg_prob_preds = logistic_reg.predict_proba(x_test)
