#Python code for chapters 12-15 DSILT: Statistics

#-------------------------------------------------------------------------------------------------#
#----------------------------------------Chapter 12: GLMs-----------------------------------------#
#-------------------------------------------------------------------------------------------------#

import pandas as pd
from scipy import stats
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import statsmodels.formula.api as smf

#Fixes a bug in printing output from IDLE, it may not be needed on all machines
import sys
sys.__stdout__ = sys.stdout

train = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/12-16 Titanic and Adult Salaries/train_clean_feats.csv')
test = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/12-16 Titanic and Adult Salaries/test_clean_feats.csv')
train['Set'] = 'train'
test['Set'] = 'test'
alldata = train.drop('Survived', axis=1).append(test, ignore_index=True)
print(alldata.info())

######
#t-test

#Two sample independent t-test (compare group means for 2 groups)
#Compare the mean fare per person between males and females
t, p = stats.ttest_ind(alldata[alldata['Sex_male']==0]['Fare_Per_Person'], alldata[alldata['Sex_male']==1]['Fare_Per_Person'], equal_var=True)
print("ttest_ind: t = %g  p = %g" % (t, p))

#Compare the mean fare per person with the mean fare
t, p = stats.ttest_ind(alldata['Fare_Per_Person'], alldata['Fare'], equal_var=True)
print("ttest_ind: t = %g  p = %g" % (t, p))

######
#One-Way ANOVA

#Test for heteroskedasticity across groups
#Resurrect the function defined in chapter 3 to test Levene's test over several categories
def levenes_test(num_variable, *group_variables, center='median'):
    temp = list(num_variable.groupby(group_variables))
    temp = [temp[i][1] for i,v in enumerate(temp)]
    return stats.levene(*temp, center=center)
print(levenes_test(alldata['Fare_Per_Person'], alldata['Group_Size']))
def bartlett_test(num_variable, *group_variables):
    temp = list(num_variable.groupby(group_variables))
    temp = [temp[i][1] for i,v in enumerate(temp)]
    return stats.bartlett(*temp)
print(bartlett_test(alldata['Fare_Per_Person'], alldata['Group_Size']))

#Heteroskedasticity is present, so nonparametric test (Kruskal-Wallis) should be used - wait to do this until later chapter

#Check to make sure none of the groups have only one distinct value
print(alldata.groupby(['Group_Size'])['Fare_Per_Person'].nunique())

#Compare mean fare per person by group size, omitting group of size 11
def oneway_test(num_variable, *group_variables):
    temp = list(num_variable.groupby(group_variables))
    temp = [temp[i][1] for i,v in enumerate(temp)]
    return stats.f_oneway(*temp)
F, p = oneway_test(alldata[alldata['Group_Size']!=11]['Fare_Per_Person'], alldata[alldata['Group_Size']!=11]['Group_Size'])
print("one-way ANOVA: F = %g  p = %g" % (F, p))
#Note difference from R b/c scipy assumes = variance

#Add embarkation port c back into the data and recreate the original embarkation port variable
alldata['Embarked_C'] = np.where((alldata['Embarked_Q']+alldata['Embarked_S'])==0, 1, 0)
alldata['Embarked'] = 0
alldata['Embarked'] = np.where(alldata['Embarked_Q']==1, 'Q', alldata['Embarked'])
alldata['Embarked'] = np.where(alldata['Embarked_S']==1, 'S', alldata['Embarked'])
alldata['Embarked'] = np.where(alldata['Embarked_C']==1, 'C', alldata['Embarked'])

#Perform ANOVA to compare means by embarkation port
anova_reg = ols("Fare_Per_Person ~ Embarked", alldata).fit()
anova_results = anova_lm(anova_reg)
print('\nANOVA results\n', anova_results)

#Check for heteroskedasticity
sm.qqplot(anova_reg.resid, line='s')
plt.show()

######
#Post Hoc Tests for One-way ANOVA

#Tukey test - good when groups are the same size and have and homogeneous variance
postHoc = pairwise_tukeyhsd(alldata['Fare_Per_Person'], alldata['Embarked'], alpha=0.05)
print(postHoc)

#Pairwise comparison using Bonferroni correction of p-values
mc = MultiComparison(alldata['Fare_Per_Person'], alldata['Embarked'])
#print(mc.allpairtest(stats.ttest_rel, method='Holm')[0])  #For paired t-test
print(mc.allpairtest(stats.ttest_ind, method='b')[0])     #For independent t-test

######
#ANCOVA

#Look for heteroskedasticity
plt.plot(alldata[(alldata['Pclass']==2) & (alldata['Sex_male']==1)]['Fare_Per_Person'], alldata[(alldata['Pclass']==2) &(alldata['Sex_male']==1)]['Group_Size'], 'bo')
plt.show()
#Second class male passengers with a fare price > 0 seem OK
#There are a couple group sizes with only 1 observation with these criteria though, so make sure to filter them out too

#Test for heteroskedasticity
print(levenes_test(alldata[(alldata['Pclass']==2) & (alldata['Sex_male']==1) & (alldata['Fare']>0) & (alldata['Group_Size'].isin([1,2,3,4,8,9,10,11]))]['Fare_Per_Person'], alldata[(alldata['Pclass']==2) & (alldata['Sex_male']==1) & (alldata['Fare']>0) & (alldata['Group_Size'].isin([1,2,3,4,8,9,10,11]))]['Group_Size']))
print(bartlett_test(alldata[(alldata['Pclass']==2) & (alldata['Sex_male']==1) & (alldata['Fare']>0) & (alldata['Group_Size'].isin([1,2,3,4,8,9,10,11]))]['Fare_Per_Person'], alldata[(alldata['Pclass']==2) & (alldata['Sex_male']==1) & (alldata['Fare']>0) & (alldata['Group_Size'].isin([1,2,3,4,8,9,10,11]))]['Group_Size']))

sub = alldata[(alldata['Pclass']==2) & (alldata['Sex_male']==1) & (alldata['Fare']>0) & (alldata['Group_Size'].isin([1,2,3,4,8,9,10,11]))]
print(sub.head())

#Show ANOVA to see how ANCOVA is different
anova_reg = ols("Fare_Per_Person ~ Group_Size", data=sub).fit()
anova_results = anova_lm(anova_reg)
print('\nANOVA results\n', anova_results)
mc = MultiComparison(alldata['Fare_Per_Person'], alldata['Embarked'])
print(mc.allpairtest(stats.ttest_ind, method='b')[0])     #For independent t-test
#PostHocs show that fare per person for groups sizes of 1 and 2 are different from the rest

import seaborn as sns
sns.boxplot(sub['Group_Size'], sub['Fare_Per_Person'])
plt.show()
#Box plot confirms what the post hoc tests reported

#Create the ANCOVA regression
ancova_reg = smf.ols("Fare_Per_Person ~ Age + Group_Size", data=sub).fit()
#print(ancova_reg.summary())
#Print the model summary with type III sums of squares
ancova_results = anova_lm(ancova_reg, typ="III")
print('\nANCOVA results\n', ancova_results)
sm.qqplot(ancova_reg.resid, line='s')
plt.show()

######
#Post Hoc Tests for ANCOVA

postHoc = pairwise_tukeyhsd(sub['Fare_Per_Person'], sub['Embarked'], alpha=0.05)
print(postHoc)

#Add interaction term to ANCOVA model to look for homogeneity in the regression slopes
ancova_reg = smf.ols("Fare_Per_Person ~ Age * Group_Size", data=sub).fit()
#print(ancova_reg.summary())
#Print the model summary with type III sums of squares
ancova_results = anova_lm(ancova_reg, typ="III")

######
#Chi-Square Test

#Perform Chi-Square test to compare passenger class by embarkation port
freq_tbl = pd.crosstab(alldata['Pclass'], alldata['Embarked'])
print(freq_tbl)
freq_tbl = np.array(freq_tbl)
chi2, p, dof, expected = stats.chi2_contingency(freq_tbl)
print("Chi-square: chi2 = %g  p = %g  dof = %g" % (chi2, p, dof))
print("Expected frequencies:", expected)
print("Actual frequencies:", freq_tbl)

######
#Loglinear Analysis

#No loglinear models for Python

#-------------------------------------------------------------------------------------------------#
#---------------------------------Chapter 13: Factorial ANOVA-------------------------------------#
#-------------------------------------------------------------------------------------------------#

import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

#Fixes a bug in printing output from IDLE, it may not be needed on all machines
import sys
sys.__stdout__ = sys.stdout

#Load adult dataset
d = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/12-16 Titanic and Adult Salaries/adult.data',
                names=['age', 'workclass', 'fnlwgt', 'education', 'education_nbr', 'marital_status', 'occupation', 'relationship', 'race',
                       'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary_bin'])
print(d.info())
print(d.head())

#Test for heteroskedasticity across groups
def levenes_test(num_variable, *group_variables, center='median'):
    temp = list(num_variable.groupby(group_variables))
    temp = [temp[i][1] for i,v in enumerate(temp)]
    return stats.levene(*temp, center=center)
print(levenes_test(d['hours_per_week'], d['education']))
print(levenes_test(d['hours_per_week'], d['relationship']))
def bartlett_test(num_variable, *group_variables):
    temp = list(num_variable.groupby(group_variables))
    temp = [temp[i][1] for i,v in enumerate(temp)]
    return stats.bartlett(*temp)
print(bartlett_test(d['hours_per_week'], d['education']))
print(bartlett_test(d['hours_per_week'], d['relationship']))

sns.boxplot(d['education'], d['hours_per_week'])
plt.show()

#Ignore the heteroskedasticity for now, and proceed
#n-way Factorial ANOVA
#Note that C() forces a varaible to be treated as categorical
anova_reg = ols("hours_per_week ~ C(education) + C(relationship) + C(education):C(relationship)", data=d).fit()
#print(anova_reg.summary())
aov_table = anova_lm(anova_reg, typ="III")
print(aov_table)

#QQ Plot of residuals
sm.qqplot(anova_reg.resid, line='s')
plt.show()

#Post Hoc tests for education only
mc = MultiComparison(d['hours_per_week'], d['education'])
print(mc.allpairtest(stats.ttest_ind, method='b')[0])     #For independent t-test

#-------------------------------------------------------------------------------------------------#
#------------------------------Chapter 14: Nonparametric Tests------------------------------------#
#-------------------------------------------------------------------------------------------------#

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from sklearn.preprocessing import LabelEncoder

#Fixes a bug in printing output from IDLE, it may not be needed on all machines
import sys
sys.__stdout__ = sys.stdout

#Load adult dataset
d = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/12-16 Titanic and Adult Salaries/adult.data',
                names=['age', 'workclass', 'fnlwgt', 'education', 'education_nbr', 'marital_status', 'occupation', 'relationship', 'race',
                       'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary_bin'])
print(d.info())
print(d.head())

######
#Kruskal-Wallis test (nonparametric ANOVA)

def kruskal_test(num_variable, *group_variables):
    temp = list(num_variable.groupby(group_variables))
    temp = [temp[i][1] for i,v in enumerate(temp)]
    return stats.kruskal(*temp)
print(kruskal_test(d['hours_per_week'], d['education']))
#If p value for test stat (H or chi-squared) is < 0.05, then the independent var does significantly affect the outcome
#Post hoc tests are needed to see which groups were responsible for the diff

#Post Hoc tests
mc = MultiComparison(d['hours_per_week'], d['education'])
print(mc.allpairtest(stats.ttest_ind, method='b')[0])     #For independent t-test

######
#Wilcoxon signed-rank/rank-sum test (nonparametric t-test)

le = LabelEncoder()
d['sex'] = le.fit_transform(d['sex'])

t, p = stats.wilcoxon(d['hours_per_week'], d['sex'])
print("Wilcoxon: t = %g  p = %g" % (t, p))

#-------------------------------------------------------------------------------------------------#
#-------------------------Chapter 15: LDA and QDA for Classification------------------------------#
#-------------------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import metrics

#Set seed for repeatability
seed = 14
np.random.seed(seed)

#Load adult dataset
d = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/12-16 Titanic and Adult Salaries/adult.data',
                names=['age', 'workclass', 'fnlwgt', 'education', 'education_nbr', 'marital_status', 'occupation', 'relationship', 'race',
                       'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary_bin'])
print(d.info())
print(d.head())

#Fixes a bug in printing output from IDLE, it may not be needed on all machines
import sys
sys.__stdout__ = sys.stdout

######
#Data Cleaning

#Define a function to count the nulls in every field
def naCol(df):
    y = dict.fromkeys(df.columns)
    for idx, key in enumerate(y.keys()):
        if df.dtypes[list(y.keys())[idx]] == 'object':
            y[key] = pd.isnull(df[list(y.keys())[idx]]).sum() + (df[list(y.keys())[idx]]=='').sum() +(df[list(y.keys())[idx]]==' ?').sum()
        else:
            y[key] = pd.isnull(df[list(y.keys())[idx]]).sum()
    print("Number of nulls by column")
    print(y)
    return y

naCol(d)

#Since there are 32k rows and <2k rows with nulls, it is safe to discard them
d = d.dropna()
d = d[(d != '?').all(1)]
d = d[(d != ' ?').all(1)]

#Convert variables to desired data types
d[['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']].apply(pd.to_numeric)
encoder = LabelEncoder()
categorical_vars = ['workclass', 'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'native_country', 'salary_bin']
categorical_var_mapping = dict()
for cv in categorical_vars:
    d[cv] = d[cv].str[1:]                 #Gets rid of the leading white spaces in the text
    d[cv] = encoder.fit_transform(d[cv])  #Encodes as integer
    categorical_var_mapping[cv] = list(encoder.classes_)  #Saves integer to category mapping

#Boxplots for numeric variables to check for outliers
for col in ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']:
    sns.boxplot(d[col])
    plt.title('Box Plot for ' + col)
    plt.show()

#Look at correlation matrix
corm = d[['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']].corr()
plt.matshow(corm)
plt.show()

#Get rid of education because it is already encoded as integer in 'education_nbr'
d.drop(['education'], axis=1, inplace=True)

#Save the version of the dataset without dummies
d_no_dummies = d.copy()

print(d.columns)
#Dummy encode categorical variables except for the target (salary_bin)
#Note that pd.get_dummies automatically removes the originals after dummy encoding
d = pd.get_dummies(d, columns=categorical_vars[:6], drop_first=True)
print(d.columns)
print(d.info())

#Perform z-score standardization on the numeric features
#Standardization is only needed if the matrix of the data will be decomposed instead of the covariance matrix
d[['age', 'fnlwgt', 'education_nbr', 'capital_gain', 'capital_loss', 'hours_per_week']] = StandardScaler().fit_transform(d[['age', 'fnlwgt', 'education_nbr', 'capital_gain', 'capital_loss', 'hours_per_week']])
d_no_dummies[['age', 'fnlwgt', 'education_nbr', 'capital_gain', 'capital_loss', 'hours_per_week']] = StandardScaler().fit_transform(d_no_dummies[['age', 'fnlwgt', 'education_nbr', 'capital_gain', 'capital_loss', 'hours_per_week']])

#Convert the dataframe into numpy array and specify dependent variable
x = d.drop(['salary_bin'], axis=1).values.astype(float)
y = d[['salary_bin']].values
x_nd = d_no_dummies.drop(['salary_bin'], axis=1).values.astype(float)
x_numeric = d_no_dummies[['age', 'fnlwgt', 'education_nbr', 'capital_gain', 'capital_loss', 'hours_per_week']].values.astype(float)

#Split data into training and test sets - be sure to stratify since this is for classification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
x_nd_train, x_nd_test, y_nd_train, y_nd_test = train_test_split(x_nd, y, test_size=0.3, stratify=y, random_state=seed)
xn_train, xn_test, yn_train, yn_test = train_test_split(x_numeric, y, test_size=0.3, stratify=y, random_state=seed)

######
#LDA

#Run LDA for classification
#Note if n_components=None, then all of them are kept
lda = LinearDiscriminantAnalysis(n_components=None, solver='svd')
lda.fit(x_train, y_train)
print(lda.get_params())
print('Priors:', lda.priors_)       #Class prior probabilities
print('Classification Accuracy:', lda.score(x_train, y_train))

#Explore the percentage of between class variance explained by each linear discriminant
print('Explained variance:', lda.explained_variance_ratio_)

######
#Evaluating the model on new data

#Make income predictions for validation set
post_lda = lda.predict(x_test)
post_lda = post_lda.reshape(post_lda.shape[0], 1)
print('Classification Accuracy:', lda.score(x_test, y_test))

#Confusion matrix
cm = metrics.confusion_matrix(y_test, post_lda)
sns.heatmap(cm, annot=True, fmt=".2f", square=True)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

#Log loss (a.k.a. negative log likelihood)
print('Log loss:', metrics.log_loss(y_test, post_lda))

#Plot ROC curve
fpr, tpr, threshold = metrics.roc_curve(y_test, post_lda)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'b--')  #Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()

######
#QDA

#Run QDA for classification
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)
print(qda.get_params())
print('Priors:', qda.priors_)       #Class prior probabilities
print('Classification Accuracy:', qda.score(x_train, y_train))

#Look at complete correlation matrix
print(corm)

