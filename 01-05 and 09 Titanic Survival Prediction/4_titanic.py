#Python code for chapter 4 of DSILT: Statistics

import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from outliers import smirnov_grubbs as grubbs
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv", sep=",")
test = pd.read_csv("test.csv", sep=",")

print(train.info())

#-------------------------------------------------------------------------------------------------#
#-------------------------------Dealing with Missing Data-----------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Define a function to count the nulls in every field
def naCol(df):
    y = dict.fromkeys(df.columns)
    for idx, key in enumerate(y.keys()):
        if df.dtypes[list(y.keys())[idx]] == 'object':
            y[key] = pd.isnull(df[list(y.keys())[idx]]).sum() + (df[list(y.keys())[idx]]=='').sum()
        else:
            y[key] = pd.isnull(df[list(y.keys())[idx]]).sum()
    print("Number of nulls by column")
    print(y)
    return y

naCol(train)
naCol(test)

#Define a function to count the nulls by row to see if any rows have too many missing values to be useful
def naRow(df, threshold=0.5):
    y = dict.fromkeys(df.index)
    for idx, key in enumerate(y.keys()):
        y[key] = sum(df.iloc[[idx]].isnull().sum())
    print("Rows with more than 50% null columns")
    print([r for r in y if y[r]/df.shape[1] > threshold])
    return y

naRow(train)
naRow(test)

#Extract the deck level
train['Deck'] = train['Cabin'].str[:1]
test['Deck'] = test['Cabin'].str[:1]

#Cross tabulate class and deck - only doing this for training set, but if class and deck are related, they will be for both sets
ct = pd.crosstab(train['Deck'], train['Pclass'], rownames=["Passengers on Each Deck by Class"])
print(ct)
print(ct.apply(lambda r: r/len(train), axis=1)) #Perc by cell
print(ct.apply(lambda r: r/sum(r), axis=1))     #Perc by row
print(ct.apply(lambda r: r/sum(r), axis=0))     #Perc by col

#Drop cabin
del train['Cabin'], test['Cabin']

#Encode the missing deck values as something else
print(train.Deck.unique())
print(test.Deck.unique())  #Make sure the encoded value is not an existing category
train.Deck.fillna('Z', inplace=True)
test.Deck.fillna('Z', inplace=True)

#Fill in the 1 missing fare from the test set with the mean fare price for that passenger class
missing_fare_pclass = int(test.loc[test['Fare'].isnull()]['Pclass'])
test.Fare.fillna(round(np.nanmean(pd.concat([train.loc[train['Pclass']==missing_fare_pclass]['Fare'], test.loc[test['Pclass']==missing_fare_pclass]['Fare']], ignore_index=True)), 4), inplace=True)

#Fill in the 2 missing embarkation ports from the training set with the mode
train.Embarked.fillna(stats.mode(pd.concat([train['Embarked'], test['Embarked']], ignore_index=True)), inplace=True)

#Check if age is missing at random
print(pd.crosstab(train.loc[train.Age.isnull()]['Survived'], train.loc[train.Age.isnull()]['Pclass'], rownames=["Rows Missing Age by Survived and Pclass"]))

#-------------------------------------------------------------------------------------------------#
#---------------------------------Dealing with Outliers-------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Boxplots for numeric variables
numeric_cols = [col for col in train.columns if train[col].dtype == 'float64']
for col in numeric_cols:
    sns.boxplot(y=train[~np.isnan(train[col])][col])
    plt.title("Box Plot for " + col)
    plt.show()

#Grubbs test (note - this is the generalized extreme studentized deviates test/iterative Grubbs)
print(train.loc[train.Fare == grubbs.max_test_outliers(train['Fare'], alpha=0.05)[0]])

#Compute fare per person, since some passengers bought group tickets producing fares that are sums of the individual ticket prices
train['Set'] = 'train'
test['Set'] = 'test'
alldata = pd.concat([train.drop(['Survived'], axis=1), test], ignore_index=True)
alldata['Group_Size'] = alldata.groupby(['Fare', 'Ticket'])['PassengerId'].transform("count")
alldata['Fare_Per_Person'] = alldata.Fare/alldata.Group_Size

#Plot fare by passenger class
sns.boxplot(y=alldata[alldata.Pclass==1]['Fare_Per_Person'].values)
plt.title("Box Plot for Fare Per Person - First Class")
plt.show()
sns.boxplot(y=alldata[alldata.Pclass==2]['Fare_Per_Person'].values)
plt.title("Box Plot for Fare Per Person - Second Class")
plt.show()
sns.boxplot(y=alldata[alldata.Pclass==3]['Fare_Per_Person'].values)
plt.title("Box Plot for Fare Per Person - Third Class")
plt.show()

#Inspect outliers by class
print(alldata[(alldata.Fare_Per_Person > 100) & (alldata.Pclass==1)].head())
print(alldata[(alldata.Fare_Per_Person > 15) & (alldata.Pclass==3)].head())

#Change fares that equal 0 to the mean fare for the class
firstclassmean = stats.mean(alldata[alldata.Pclass==1]['Fare'])
firstclassmeanpp = stats.mean(alldata[alldata.Pclass==1]['Fare_Per_Person'])
secclassmean = stats.mean(alldata[alldata.Pclass==2]['Fare'])
secclassmeanpp = stats.mean(alldata[alldata.Pclass==2]['Fare_Per_Person'])
thirdclassmean = stats.mean(alldata[alldata.Pclass==3]['Fare'])
thirdclassmeanpp = stats.mean(alldata[alldata.Pclass==3]['Fare_Per_Person'])
alldata.loc[(alldata.Fare==0) & (alldata.Pclass==1)]['Fare'].replace(0, firstclassmean)
alldata.loc[(alldata.Fare_Per_Person==0) & (alldata.Pclass==1)]['Fare_Per_Person'].replace(0, firstclassmeanpp)
alldata.loc[(alldata.Fare==0) & (alldata.Pclass==2)]['Fare'].replace(0, secclassmean)
alldata.loc[(alldata.Fare_Per_Person==0) & (alldata.Pclass==2)]['Fare_Per_Person'].replace(0, secclassmeanpp)
alldata.loc[(alldata.Fare==0) & (alldata.Pclass==3)]['Fare'].replace(0, firstclassmean)
alldata.loc[(alldata.Fare_Per_Person==0) & (alldata.Pclass==3)]['Fare_Per_Person'].replace(0, firstclassmeanpp)
del firstclassmean ,firstclassmeanpp, secclassmean, secclassmeanpp, thirdclassmean, thirdclassmeanpp

#Bar plots for categorical variables
categorical_cols = [col for col in alldata.columns if alldata[col].dtype == 'int64' or alldata[col].dtype == 'object']
for col in categorical_cols:
    sns.countplot(alldata[col])
    #plt.bar([indx for indx, _ in enumerate(alldata[col])], height=alldata[col])
    plt.title("Bar Plot for " + col)
    plt.show()


#Inspect the 11 passengers with the same ticket
print(alldata[alldata.Group_Size==11])

#Inspect passengers with same name
print(alldata.ix[alldata.groupby('Name')['Name'].transform('count')[alldata.groupby('Name')['Name'].transform('count')>1].index])

#-------------------------------------------------------------------------------------------------#
#---------------------------------Transforming the Data-------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Check to see how many levels the object variables have
object_cols = [col for col in alldata.columns if alldata[col].dtype == 'object']
for col in object_cols:
    print("Unique categories of " + col + ": " + str(len(alldata[col].unique())))

#One-hot encode sex, embarked, and deck
#The commented out portion shows how to dummy encode rather than one-hot encode
sex_dummy = pd.get_dummies(alldata['Sex']).add_prefix('Sex_')#.drop(alldata.sort_values('Sex')['Sex'].unique()[0], axis=1)
embarked_dummy = pd.get_dummies(alldata['Embarked']).add_prefix('Embarked_')#.drop(alldata.sort_values('Embarked')['Embarked'].unique()[0], axis=1)
deck_dummy = pd.get_dummies(alldata['Deck']).add_prefix('Deck')#.drop(alldata.sort_values('Deck')['Deck'].unique()[0], axis=1)
dummies = pd.concat([sex_dummy, embarked_dummy, deck_dummy], axis=1)
alldata = pd.concat([alldata, dummies], axis=1)
del sex_dummy, embarked_dummy, deck_dummy

#Label encode ticket
enc = LabelEncoder()
alldata['Ticket_Enc'] = enc.fit_transform(alldata['Ticket'])

#Remove unneeded columns
alldata.drop(['PassengerId', 'Sex', 'Embarked', 'Deck', 'Ticket'], axis=1, inplace=True)

#Split back to training and test sets and save as cleaned data
train_clean = alldata.loc[alldata.Set=='train']
train_clean.is_copy = False
train_clean.drop(['Set'], axis=1, inplace=True)
train_clean['Survived'] = train['Survived']
test_clean = alldata.loc[alldata.Set=='test']
test_clean.is_copy = False
test_clean.drop(['Set'], axis=1, inplace=True)
train_clean.to_csv('train_clean.csv', index=False)
test_clean.to_csv('test_clean.csv', index=False)













