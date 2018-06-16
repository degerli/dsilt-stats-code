#Python code for chapter 10 of DSILT: Statistics

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm

#Fixes a bug in printing output from IDLE, it may not be needed on all machines
import sys
sys.__stdout__ = sys.stdout

train = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression/train.csv")
test = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression/test.csv")

print(train.info())
print(train.head())

print("Mean:", np.mean(train.Calories))
print("Variance:", np.var(train.Calories))

#Negative binomial regression
negb_reg = smf.glm("Calories ~ Steps + Floors_Climbed + Times_Awake + Day_Saturday + Day_Sunday",
                   data=train,
                   family=sm.families.NegativeBinomial(link=sm.genmod.families.links.identity)).fit()
print(negb_reg.summary())

#Refer to this question about errors when running negative binomial regression - values should be scaled to remove errors
#https://stackoverflow.com/questions/44398081/statsmodels-negative-binomial-doesnt-converge-while-glm-does-converge
#Refer to the documentation to see valid link functions
#http://www.statsmodels.org/dev/glm.html#links

train_std_feats = MinMaxScaler().fit_transform(train)
test_std_feats = MinMaxScaler().fit_transform(test)
#Convert arrays back to dataframes
train_std = pd.DataFrame(train_std_feats, index=train.index, columns=train.columns)
test_std = pd.DataFrame(test_std_feats, index=test.index, columns=test.columns)
#Re-run on scaled data and get the Newey-West Hac estimates of the standard errors
negb_reg = smf.glm("Calories ~ Steps + Floors_Climbed + Times_Awake + Day_Saturday + Day_Sunday",
                   data=train_std,
                   family=sm.families.NegativeBinomial(link=sm.genmod.families.links.identity)).fit(cov_type='HAC', cov_kwds={'maxlags':1})
print(negb_reg.summary())
