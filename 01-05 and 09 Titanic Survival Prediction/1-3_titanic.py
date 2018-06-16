#Python code for chapters 1-3 of DSILT: Statistics
#The code for each chapter is set up so that it can run independently of the other chapters

#-------------------------------------------------------------------------------------------------#
#---------------------------Chapter 1 - Experimental Design---------------------------------------#
#-------------------------------------------------------------------------------------------------#

import pandas as pd

train = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/01-05 and 09 Titanic Survival Prediction/train.csv", sep=",")
test = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/01-05 and 09 Titanic Survival Prediction/test.csv", sep=",")

print(train.head(n=6))

#Examine the details for one variable
print(train.PassengerId.dtype)
print(min(train.PassengerId))
print(max(train.PassengerId))

#Examine the details for all variables
print(train.dtypes)
print(train.info())

#-------------------------------------------------------------------------------------------------#
#---------------------------Chapter 2 - Descriptive Statistics------------------------------------#
#-------------------------------------------------------------------------------------------------#

import pandas as pd
import statistics as stats
from scipy import stats as scistats
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/01-05 and 09 Titanic Survival Prediction/train.csv", sep=",")
test = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/01-05 and 09 Titanic Survival Prediction/test.csv", sep=",")

len(train.Fare)             #Number of observations
stats.mean(train.Fare)      #Mean
stats.median(train.Fare)    #Median
stats.mode(train.Fare)      #Mode
stats.variance(train.Fare)  #Sample Variance
stats.stdev(train.Fare)     #Sample Standard Deviation
stats.pvariance(train.Fare) #Population Variance
stats.pstdev(train.Fare)    #Population Standard Deviation
scistats.sem(train.Fare)    #Standard Error of the Mean

def summaryStats(x):
    return print('\n',
                 'Observations: ' + str(len(x)) + '\n',
                 'Mean: ' + str(stats.mean(x)) + '\n',
                 'Median: ' + str(stats.median(x)) + '\n',
                 'Mode: ' + str(stats.mode(x)) + '\n',
                 'Variance: ' + str(stats.variance(x)) + '\n',
                 'Std Dev: ' + str(stats.stdev(x)) + '\n',
                 'Pop Variance: ' + str(stats.pvariance(x)) + '\n',
                 'Pop Std Dev: ' + str(stats.pstdev(x)) + '\n',
                 'SEM: ' + str(scistats.sem(x)) + '\n',
                 )

summaryStats(train.Fare)

#Show quartiles for every column
train.quantile(q=[0.25, 0.5, 0.75])

#Plot options
plt.figure(figsize=(12, 9))             #Specify plot dimensions
ax = plt.subplot(111)                   #Access a specific plot
ax.spines["top"].set_visible(False)     #Remove top frame
ax.spines["right"].set_visible(False)   #Remove right frame
ax.get_xaxis().tick_bottom()            #x-axis ticks on bottom
ax.get_yaxis().tick_left()              #y-axis ticks on left

plt.title("Histogram of Passenger Class")
plt.xlabel("Class")
plt.ylabel("Number of Passengers")

#Added tick range after initial plot drawing to make x-axis clearer
plt.xticks(np.arange(min(train.Pclass), max(train.Pclass)+1, 1.0))

b = len(np.unique(train.Pclass))

plt.hist(train.Pclass, color="#3F5D70", bins=b)
plt.show()

#Plot normal distribuion
x = np.arange(-4, 4, 0.01)
plt.title("Normal Distribution")
plt.xlabel("Standard Deviation")
plt.ylabel("Probability")
plt.plot(x, scistats.norm.pdf(x, 0, 1))
plt.show()

#Plot probability density of Fare
dens = scistats.gaussian_kde(train.Fare)
dist_range = np.arange(min(train.Fare), max(train.Fare), 1)
#dens.covariance_factor = lambda : 0.4
#dens._compute_covariance()
plt.title("Probability Density of Fare")
plt.plot(dist_range, dens(dist_range))
plt.show()

#Cross tabulate passengers by class
pd.crosstab(len(train), train.Pclass,
            rownames=["Number of Passengers"])
#Cross tabulate passengers by class and embarkation port
pd.crosstab(train.Pclass, train.Embarked,
            rownames=["Class"])
#Cross tabulate passengers by class and embarkation port with perc
ct = pd.crosstab(train.Pclass, train.Embarked,
            rownames=["Perc of Passengers by Class"])
print(ct.apply(lambda r: r/len(train), axis=1)) #Perc by cell
print(ct.apply(lambda r: r/sum(r), axis=1))     #Perc by row
print(ct.apply(lambda r: r/sum(r), axis=0))     #Perc by col

#-------------------------------------------------------------------------------------------------#
#-----------------------------Chapter 3 - Statistical Modeling------------------------------------#
#-------------------------------------------------------------------------------------------------#

import pandas as pd
import statistics as stats
import math
from scipy import stats as scistats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import qqplot
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/01-05 and 09 Titanic Survival Prediction/train.csv", sep=",")
test = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/01-05 and 09 Titanic Survival Prediction/test.csv", sep=",")

#Two-tailed t-test to compare mean passenger age to mean UK population age
#First combine the training and test sets, removing the survived column and NA values for age from the training set
t_all = pd.concat([train.ix[:, train.columns != 'Survived'], test])
t_all = t_all.dropna(subset=['Age'])
xbar = stats.mean(t_all['Age']) #Sample mean of passenger age
mu = 34                         #The hypothesized mean age
s = stats.stdev(t_all['Age'])   #Sample standard deviation of passenger age
n = t_all.shape[0]              #The sample size
t = (xbar-mu)/(s/math.sqrt(n))  #The t-test statistic

#Compare the t-test statistic to a confidence interval (CI between 2 critical values)
alpha = 0.05                                 #Level of significance
tdist_half_alpha = scistats.t.ppf(1-alpha/2, df=n-1)
print([-tdist_half_alpha, tdist_half_alpha]) #The confidence interval for alpha
print(t)

#Alternatively, calculate the t-test statistic's p-value and compare it to a level of significance
pval = 2*scistats.t.cdf(t, df=n-1)
print(pval)

#Function to perform z-tests
def ztest(data, mu, sig, alpha=0.05, tails='two'):
    #data is a numeric vector
    #mu is the estimated population value to compare against
    #sig is the population standard deviation
    #alpha is the significance level, defaults to 0.05
    #tails is the number of tails for the test, defaults to 'two', other options are 'upper' and 'lower'
    xbar = stats.mean(data)
    n = len(data)
    z = (xbar-mu)/(sig/math.sqrt(n))
    if (tails=='two'):
        zdist_half_alpha = scistats.norm.ppf(1-alpha/2)
        pval = 2*scistats.norm.cdf(z)
        print('Confidence Interval:', -zdist_half_alpha, ',', zdist_half_alpha)
        print('z-statistic:', z)
        print('z-statistic p-value:', pval)
    elif (tails=='lower'):
        zdist_alpha = scistats.norm.ppf(1-alpha)
        pval = scistats.norm.cdf(z)
        print('Critical Value:', -zdist_alpha)
        print('z-statistic:', z)
        print('z-statistic p-value:', pval)
    elif (tails=='upper'):
        zdist_alpha = scistats.norm.ppf(1-alpha)
        pval = scistats.norm.sf(z)
        print('Critical Value:', zdist_alpha)
        print('z-statistic:', z)
        print('z-statistic p-value:', pval)
    else:
        return print('Error: invalid tails argument')
def ztest_prop(data, criterion, p0, alpha=0.05, tails='two'):
    #data is a numeric vector
    #criterion is a numeric vector of the number of samples that meet some criterion (a subset of data)
    #p0 is the estimated proportion to compare against
    #alpha is the significance level, defaults to 0.05
    #tails is the number of tails for the test, defaults to 'two', other options are 'upper' and 'lower'
    pbar = len(criterion)/len(data)
    n = len(data)
    z = (pbar-p0)/math.sqrt(p0*(1-p0)/n)
    if (tails=='two'):
        zdist_half_alpha = scistats.norm.ppf(1-alpha/2)
        pval = 2*scistats.norm.sf(z)
        print('Confidence Interval:', -zdist_half_alpha, ',', zdist_half_alpha)
        print('z-statistic:', z)
        print('z-statistic p-value:', pval)
    elif (tails=='lower'):
        zdist_alpha = scistats.norm.ppf(1-alpha)
        pval = scistats.norm.cdf(z)
        print('Critical Value:', -zdist_alpha)
        print('z-statistic:', z)
        print('z-statistic p-value:', pval)
    elif (tails=='upper'):
        zdist_alpha = scistats.norm.ppf(1-alpha)
        pval = 2*scistats.norm.sf(z)
        print('Critical Value:', zdist_alpha)
        print('z-statistic:', z)
        print('z-statistic p-value:', pval)
    else:
        return print('Error: invalid tails argument')
#Function to perform t-tests
def ttest(data, mu, alpha=0.05, tails='two'):
    #data is a numeric vector
    #mu is the estimated population value to compare against
    #alpha is the significance level, defaults to 0.05
    #tails is the number of tails for the test, defaults to 'two', other options are 'upper' and 'lower'
    data = data.dropna()
    xbar = stats.mean(data)
    s = stats.stdev(data)
    n = len(data)
    t = (xbar-mu)/(s/math.sqrt(n))
    if (tails=='two'):
        tdist_half_alpha = scistats.t.ppf(1-alpha/2, df=n-1)
        pval = 2*scistats.t.cdf(t, df=n-1)
        print('Confidence Interval:', -tdist_half_alpha, ',', tdist_half_alpha)
        print('t-statistic:', t)
        print('t-statistic p-value:', pval)
    elif (tails=='lower'):
        tdist_alpha = scistats.t.ppf(1-alpha, df=n-1)
        pval = scistats.t.cdf(t, df=n-1)
        print('Critical Value:', -tdist_alpha)
        print('t-statistic:', t)
        print('t-statistic p-value:', pval)
    elif (tails=='upper'):
        tdist_alpha = scistats.t.ppf(1-alpha, df=n-1)
        pval = scistats.t.sf(t, df=n-1)
        print('Critical Value:', tdist_alpha)
        print('t-statistic:', t)
        print('t-statistic p-value:', pval)
    else:
        return print('Error: invalid tails argument')

#Run t-tests for example
ttest(t_all['Age'], 34)
ttest(t_all['Age'], 34, tails='lower')
ttest(t_all['Age'], 34, tails='upper')

#Clean data for sex
t_all = pd.concat([train.ix[:, train.columns != 'Survived'], test])
t_all = t_all.dropna(subset=['Sex'])

#Run proportional z-test for example
ztest_prop(t_all['Sex'], t_all[t_all['Sex']=='male'].filter(['Sex'], axis=1), p0=0.5)
#Inspect the sex ratio of Titanic passengers
print(pd.crosstab(t_all["Sex"], columns="Prop")/(pd.crosstab(t_all["Sex"], columns="Prop").sum()))

#Validate normality of age visually...
t_all = pd.concat([train.ix[:, train.columns != 'Survived'], test])
t_all = t_all.dropna(subset=['Age'])
plt.hist(t_all['Age'], bins='auto')  #Histogram
plt.show()
#plt.hist(t_all['Age'], bins=np.unique(t_all['Age']))
#plt.show()
sns.kdeplot(np.array(t_all['Age']))  #Smoothed density plot
plt.show()
#...through a q-q plot...
qqplot(t_all['Age'])
plt.show()
#...and through Shapiro-Wilk and Jarque-Bera
print(scistats.shapiro(t_all['Age']))
print(scistats.jarque_bera(t_all['Age']))

#Validate homogeneity of variance with Levene's test
le = LabelEncoder()
t_all['Sex_Encoded'] = le.fit_transform(np.array(t_all['Sex']))
def levenes_test(num_variable, *group_variables, center='median'):
    temp = list(num_variable.groupby(group_variables))
    temp = [temp[i][1] for i,v in enumerate(temp)]
    return scistats.levene(*temp, center=center)
print(levenes_test(t_all['Age'], t_all['Sex_Encoded']))
print(levenes_test(t_all['Age'], t_all['Pclass']))
print(levenes_test(t_all['Age'], t_all['Sex_Encoded'], t_all['Pclass']))

#Validate homogenetiy of variance with variance ratio (Hartley's F Max)
def hartleys_f_max(num_variable, group_variable):
    #num_variable is a numeric variable to compare variances
    #group_variable is the variable with the groups to compare the variance
    group_variances = num_variable.groupby(group_variable).apply(stats.variance)
    f = max(group_variances)/min(group_variances)
    print('F-max statistic:', f)
    print('Degrees of Freedom:', len(num_variable)-1, ', k:', len(np.unique(group_variable)))
    print('Compare the F-max statistic for n-1 DoF and k to the F-max table here: \n',
          'http://archive.bio.ed.ac.uk/jdeacon/statistics/table8.htm \n',
          'and if the F-max statistic is smaller than the critical value in the table, then the variance is homogenous.')
hartleys_f_max(t_all['Age'], t_all['Sex'])
hartleys_f_max(t_all['Age'], t_all['Pclass'])

#Look at data to verify last 2 standard assumptions
t_all.head()
