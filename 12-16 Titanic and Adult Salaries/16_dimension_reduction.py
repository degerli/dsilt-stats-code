#Python code for chapter 16 DSILT: Statistics

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

#-------------------------------------------------------------------------------------------------#
#------------------------------------PCA of Numeric Data------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Apply PCA, keeping n components < the number of original features
#Note if n_components=None, then all of them are kept
#Note that sklearn uses SVD instead of eigendecomposition
pca_model = PCA(n_components=None, whiten=False, random_state=seed)
pca_dim = pca_model.fit_transform(xn_train)

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1 (explains most variance)')
plt.ylabel('Latent Variable 2 (explains 2nd most variance)')
plt.title('PCA 2-Dimension Plot with Observation Class')
plt.scatter(pca_dim[:, 0], pca_dim[:, 1], c=yn_train.ravel())
plt.colorbar()
plt.show()

#Get percentage of variance explained (eigenvalue) by each latent variable (component)
#Note - alternative equation can be found here: http://stackoverflow.com/questions/29611842/scikit-learn-kernel-pca-explained-variance
var_explained = pca_model.explained_variance_ratio_
#Calculate cumulative variance explained by each latent variable (component)
cum_var_explained = np.cumsum(np.round(var_explained, decimals=4)*100)
#Plot cumulative explained variance to see how many components should be extracted
plt.plot(cum_var_explained)
plt.xlabel('Latent Variable')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Latent Variables')
plt.xticks(np.arange(0, len(var_explained), 1), (np.arange(0, len(var_explained), 1)+1))
plt.show()

#Choose extracted components based on graph or based on some desired variance threshold in loop below (default 80%)
#Note that for datasets with many dimensions, the fewer components extracted the better, even if less variance is explained
var_explained_thresh = 80.0
for idx, cumvar in enumerate(cum_var_explained):
    if (cumvar >= var_explained_thresh):
        pca_extracted_components = idx+1
        break
#pca_extracted_components = 2
pca_features = pca_dim[:, 0:pca_extracted_components]
print('PCA Number of Extracted Features:', pca_features.shape[1])

#Use pca_model.transform(xn_test) to fit the PCA from the training set onto the test set
#Note that KernelPCA (nonlinear PCA) is also available in sklearn

#-------------------------------------------------------------------------------------------------#
#-------------------------------Factor Analysis of Numeric Data-----------------------------------#
#-------------------------------------------------------------------------------------------------#

#Use 'randomized' svd_method if lapack is too slow or dataset is large (randomized sacrifices a little accuracy)
fact_model = FactorAnalysis(n_components=None, svd_method='lapack', random_state=seed)
fact_dim = fact_model.fit_transform(xn_train)

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1 (explains most variance)')
plt.ylabel('Latent Variable 2 (explains 2nd most variance)')
plt.title('Factor Analysis 2-Dimension Plot with Observation Class')
plt.scatter(fact_dim[:, 0], fact_dim[:, 1], c=yn_train.ravel())
plt.colorbar()
plt.show()

#Get percentage of variance explained (eigenvalue) by each latent variable (component)
#Note - this equation can be found here: http://stackoverflow.com/questions/29611842/scikit-learn-kernel-pca-explained-variance
var_explained = np.var(fact_dim, axis=0)/np.sum(np.var(fact_dim, axis=0))
#Calculate cumulative variance explained by each latent variable (component)
cum_var_explained = np.cumsum(np.round(var_explained, decimals=4)*100)
#Plot cumulative explained variance to see how many components should be extracted
plt.plot(cum_var_explained)
plt.xlabel('Latent Variable')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Latent Variables')
plt.xticks(np.arange(0, len(var_explained), 1), (np.arange(0, len(var_explained), 1)+1))
plt.show()

#Choose extracted components based on graph or based on some desired variance threshold in loop below (default 80%)
#Note that for datasets with many dimensions, the fewer components extracted the better, even if less variance is explained
var_explained_thresh = 80.0
for idx, cumvar in enumerate(cum_var_explained):
    if (cumvar >= var_explained_thresh):
        fact_extracted_components = idx+1
        break
#fact_extracted_components = 2
fact_features = fact_dim[:, 0:fact_extracted_components]
print('Factor Analysis Number of Extracted Features:', fact_features.shape[1])

#Use fact_model.transform(xn_test) to fit the factor analysis from the training set onto the test set
#Note factor analysis of mixed data (FAMD) is not available in Python

#-------------------------------------------------------------------------------------------------#
#---------------------------------LDA for Dimension Reduction-------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Run LDA for dimension reduction, ignoring the multicollinearity and keeping n components < the number of original features
#Note if n_components=None, then all of them are kept
lda = LinearDiscriminantAnalysis(n_components=None, solver='svd')
lda_dim_reduced = lda.fit_transform(x_nd_train, y_nd_train.ravel())

#Explore LDA results from the model that was just built
print('Priors:', lda.priors_)       #Class prior probabilities
print('Class means:', lda.means_)   #Class specific means for each predictor
print('Loadings:', lda.coef_)       #The factor loadings of the predictors on the latent variables (linear discriminants)
print('Explained variance:', lda.explained_variance_ratio_)
print('Mean classification accuracy:', lda.score(x_nd_train, y_nd_train))
#Plot the features by their loadings on the linear discriminant
import seaborn as sns
sns.barplot(lda.coef_[0], d_no_dummies.drop(['salary_bin'], axis=1).columns)
plt.xlabel('Coefficient Loading')
plt.title('Feature Loadings on the First Linear Discriminant')
plt.show()
