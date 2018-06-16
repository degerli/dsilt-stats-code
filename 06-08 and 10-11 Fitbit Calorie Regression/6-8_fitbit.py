#Python code for chapters 6 and 8 of DSILT: Statistics

#-------------------------------------------------------------------------------------------------#
#---------------------------------Chapter 6: Linear Regression------------------------------------#
#-------------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf

alldata = pd.read_csv("/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression/fitbit.csv", sep=",")
alldata.columns = alldata.columns.to_series().str.replace('\s+', '_')

print(alldata.info())
print(alldata.head())

#Convert Date to a date
alldata['Date'] = pd.to_datetime(alldata['Date'])

#Convert integers to numeric
alldata['Steps'] = alldata['Steps'].astype(float)
alldata['Calories'] = alldata['Calories'].astype(float)
alldata['Active_Minutes'] = alldata['Active_Minutes'].astype(float)
alldata['Floors_Climbed'] = alldata['Floors_Climbed'].astype(float)
alldata['Times_Awake'] = alldata['Times_Awake'].astype(float)

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

#Define a function to count the nulls by row to see if any rows have too many missing values to be useful
def naRow(df, threshold=0.5):
    y = dict.fromkeys(df.index)
    for idx, key in enumerate(y.keys()):
        y[key] = sum(df.iloc[[idx]].isnull().sum())
    print("Rows with more than 50% null columns")
    print([r for r in y if y[r]/df.shape[1] > threshold])
    return y

naCol(alldata)
naRow(alldata)

'''
!!!!!!Attention: If you are going to skip data cleaning b/c of problems installing or running fancyimpute's
MICE function, then jump down to the next comment that looks like this and ignore all the code in between.
'''

#Use multiple imputation to fill in the missing values for hours slept and times awake
from fancyimpute import MICE
import random

def estimate_by_mice(df):
    df_estimated_variables = df.copy()
    random.seed(14)
    mice = MICE()  # model=RandomForestClassifier(n_estimators=100))
    result = mice.complete(np.asarray(df.values, dtype=float))
    df_estimated_variables.loc[:, df.columns] = result[:][:]
    return df_estimated_variables

impdata = estimate_by_mice(alldata.drop(['Date'], axis=1))
#alldata = pd.concat([alldata[['Date']], impdata], axis=1)

#Plot distributions of original data and imputed to see if imputation is on track
fig = plt.figure()
plta = fig.add_subplot(211)
alldata.hist(column='Hours_Slept', ax=plta)
plta.set_title('Histogram of Hours Slept Before Imputation')
pltb = fig.add_subplot(212)
impdata.hist(column='Hours_Slept', ax=pltb)
pltb.set_title('Histogram of Hours Slept After Imputation')
fig.tight_layout()
plt.show()
plt.close(fig)
fig = plt.figure()
plta = fig.add_subplot(211)
alldata.hist(column='Times_Awake', ax=plta)
plta.set_title('Histogram of Times Awake Before Imputation')
pltb = fig.add_subplot(212)
impdata.hist(column='Times_Awake', ax=pltb)
pltb.set_title('Histogram of Times Awake After Imputation')
fig.tight_layout()
plt.show()
plt.close(fig)

#Boxplots for numeric variables to check for outliers
numeric_cols = [col for col in alldata.columns if alldata[col].dtype == 'float64']
for col in numeric_cols:
    sns.boxplot(y=alldata[~np.isnan(alldata[col])][col])
    plt.title("Box Plot for " + col)
    plt.show()

#Inspect rows with potential outliers
print(alldata[alldata['Times_Awake']>30].head())
print(alldata[alldata['Hours_Slept']<6].head())
print(alldata[alldata['Floors_Climbed']>100].head())
print(alldata[alldata['Active_Minutes']>100].head())

#Drop date and look at correlation matrix
nodate = alldata.drop(['Date'], axis=1)
allcor = nodate.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)  #Sets up diverging palette for seaborn
sns.heatmap(allcor, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Numeric Variables')
plt.show()

#Drop distance
nodate.drop(['Distance'], axis=1, inplace=True)

######
#Testing assumptions

#Are the predictors independent?
print(nodate.drop(['Calories'], axis=1).columns)  #Yes - there's no reason to think any of these depend on the others

#Are all predictors quantitative or categorical with no more than 2 categories?
print(nodate.drop(['Calories'], axis=1).info())  #Yes - assumption met

#Do the predictors have non-zero variance?
for col in list(nodate.drop(['Calories'], axis=1).columns):
    print(stats.variance(nodate[col]))  #Yes - assumption met

#Is there multicollinearity among the predictors?
print(allcor)  #No - assumption met

#Might the predictors correlate with variables that are not in dataset?
#Possibly - date was dropped, but steps and activity might be correlated with season, since summer months are more conducive to activity

#Are the residuals homoskedastic?
#Cannot tell yet - need to build the model to get the residuals

#Are the residuals normally distributed?
#Cannot tell yet - need to build the model to get the residuals

#Are the residuals autocorrelated?
#Cannot tell yet - need to build the model to get the residuals

######
#Modeling

#Add features for the days of the week, using Monday as the baseline
alldata['Day'] = alldata['Date'].dt.dayofweek
dummies = pd.get_dummies(alldata['Day'], drop_first=True).add_prefix('Day_')
alldata = pd.concat([alldata, dummies], axis=1)
nodate = pd.concat([nodate, dummies], axis=1)
print(nodate.head())

#Split data into training and test sets
x = nodate.drop(['Calories'], axis=1).values
y = nodate['Calories'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=14)
train = np.concatenate((x_train, y_train.reshape(y_train.shape[0],1)), axis=1)
train = pd.DataFrame(data=train, columns=list(nodate.drop(['Calories'], axis=1).columns)+['Calories'])
#train.to_csv('train.csv', index=False)
test = np.concatenate((x_test, y_test.reshape(y_test.shape[0],1)), axis=1)
test = pd.DataFrame(data=test, columns=list(nodate.drop(['Calories'], axis=1).columns)+['Calories'])
#test.to_csv('test.csv', index=False)

'''
!!!!!!Attention: If you skipped data cleaning b/c of problems installing or running fancyimpute's
MICE function, then uncomment the 2 lines below to read in the data.  Note that R used Friday as the
baseline dummy.
'''
#train = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression/train.csv')
#test = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression/test.csv')

x_train = train.drop(['Calories'], axis=1).values
y_train = train['Calories'].values
x_test = test.drop(['Calories'], axis=1).values
y_test = test['Calories'].values
train.columns = ['Steps', 'Calories', 'Active_Minutes', 'Floors_Climbed', 'Hours_Slept',
       'Times_Awake', 'Day_0', 'Day_5', 'Day_6',
       'Day_3', 'Day_1', 'Day_2']
test.columns = ['Steps', 'Calories', 'Active_Minutes', 'Floors_Climbed', 'Hours_Slept',
       'Times_Awake', 'Day_0', 'Day_5', 'Day_6',
       'Day_3', 'Day_1', 'Day_2']

#Build linear model
linear_reg = LinearRegression().fit(x_train, y_train)
print("Regression Parameters",
      "\nIntercept:", linear_reg.intercept_)
for idx, col in enumerate(train.drop(['Calories'], axis=1).columns):
    print(col, "Coefficient:", linear_reg.coef_[idx])
print("R^2:", linear_reg.score(x_train, y_train))

#Get prettier output
def get_lr_output(lr, x, y):
    #Takes a fitted sklearn linear regression as input and outputs results
    #Thanks to stackoverflow: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    import numpy as np
    import scipy.stats as scistats
    parameters = np.append(lr.intercept_, lr.coef_)
    predictions = lr.predict(x)
    X = pd.DataFrame({"Constant":np.ones(len(x))}).join(pd.DataFrame(x))
    mse = (sum((y-predictions)**2))/(len(X)-len(X.columns))
    sd = np.sqrt(mse*(np.linalg.inv(np.dot(X.T, X)).diagonal()))
    ts = parameters/sd
    p_values = [2*(1-scistats.t.cdf(np.abs(i), (len(X)-1))) for i in ts]
    sd = np.round(sd, 4)
    ts = np.round(ts, 4)
    p_values = np.round(p_values, 4)
    parameters = np.round(parameters, 4)
    predictors = pd.Series('intercept')
    predictors = predictors.append(pd.Series(x.columns)).reset_index()[0]
    output_df = pd.DataFrame()
    output_df['predictor'] = predictors
    output_df['coefficient'] = parameters
    output_df['standard_error'] = sd
    output_df['t-statistic'] = ts
    output_df['p-value'] = p_values
    r_squared = round(lr.score(x, y), 4)
    adj_r_squared = round(1-(1-lr.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1), 4)
    print(output_df, '\n R-squared:', r_squared, '\n Adjusted R-squared:', adj_r_squared)
    return [output_df, r_squared, adj_r_squared]

linear_reg_output = get_lr_output(linear_reg, train.drop(['Calories'], axis=1), train['Calories'])

#Testing normally distributed residuals assumption
train_preds = linear_reg.predict(x_train)
residuals = y_train-train_preds
print(np.mean(residuals))  #This should be near zero if the assumption of residual normality holds
sns.distplot(residuals)
plt.show()

#Correlogram to look for significant lags in the residuals
plot_acf(residuals, lags=30)
plt.show()

#Test idea that weekends were more active and therefore could cause serial correlation
#View average steps by day in bar chart
alldata.groupby(['Day'])['Steps'].mean().plot(kind='bar', title="Average Steps by Day")
plt.xlabel('Day of Week (Monday=0)')
plt.ylabel('Average Number of Steps')
plt.show()
alldata.groupby(['Day'])['Active_Minutes'].mean().plot(kind='bar', title="Average Active Minutes by Day")
plt.xlabel('Day of Week (Monday=0)')
plt.ylabel('Average Active Minutes')
plt.show()

#No Newey-West HAC for sklearn

#Remove unnecessary variables from the model to see the result
x_train_red = train[['Steps', 'Floors_Climbed', 'Times_Awake', 'Day_5']].values
x_test_red = test[['Steps', 'Floors_Climbed', 'Times_Awake', 'Day_5']].values
reduced_linear_reg = LinearRegression().fit(x_train_red, y_train)
reduced_linear_reg_output = get_lr_output(reduced_linear_reg, train[['Steps', 'Floors_Climbed', 'Times_Awake', 'Day_5']], train['Calories'])

def AIC(y, y_pred, k):
    '''
    Takes residuals of a model and number of predictors k, and outputs AIC
    '''
    resid = y - y_pred.ravel()
    sse = sum(resid ** 2)
    return 2*k - 2*np.log(sse+1e-32)

print(AIC(y_train, linear_reg.predict(x_train), x_train.shape[1]))
print(AIC(y_train, reduced_linear_reg.predict(x_train_red), x_train_red.shape[1]))

#Predict the calories burnt in the test set, using the fitted model
test_preds = np.round(reduced_linear_reg.predict(x_test_red), 0)  #Rounding since target var is rounded to whole number
mae = np.mean(abs(y_test-test_preds))
rmse = np.sqrt(np.mean((y_test-test_preds)**2))
baseline_model = np.round(np.mean(y_train), 0)
mae_baseline = np.mean(abs(y_test-baseline_model))
rmse_baseline = np.sqrt(np.mean((y_test-baseline_model)**2))

#-------------------------------------------------------------------------------------------------#
#-----------------------------Chapter 8: Hyperparameter Optimization------------------------------#
#-------------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import SGDRegressor
from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK

#Ignore all deprecation warnings from hyperopt
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression/train.csv')
test = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression/test.csv')

train.columns = ['Steps', 'Calories', 'Active_Minutes', 'Floors_Climbed', 'Hours_Slept',
       'Times_Awake', 'Day_0', 'Day_5', 'Day_6',
       'Day_3', 'Day_1', 'Day_2']
test.columns = ['Steps', 'Calories', 'Active_Minutes', 'Floors_Climbed', 'Hours_Slept',
       'Times_Awake', 'Day_0', 'Day_5', 'Day_6',
       'Day_3', 'Day_1', 'Day_2']

#Convert to numpy arrays and standardize all values, otherwise SGD error will be large
x_train = StandardScaler().fit_transform(train.drop(['Calories'], axis=1).values)
y_train = StandardScaler().fit_transform(train['Calories'].values.reshape(-1,1))
x_test = StandardScaler().fit_transform(test.drop(['Calories'], axis=1).values)
y_test = StandardScaler().fit_transform(test['Calories'].values.reshape(-1,1))

#Set up k-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=14)

#Train SGD regression with default hyperparameters
sgd_reg = SGDRegressor()
sgd_results = cross_val_score(sgd_reg, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

#Train model with optimized hyperparameters
def sgd_model_opt(opt_type='bayesian'):
    
    #Define objective functionto optimize
    def obj_func(params):
        clf = SGDRegressor(**params)
        return {'loss': -cross_val_score(clf, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error').mean(), 'status': STATUS_OK}
    
    #Define search space of hyperparameters
    search_space = {
        'loss': hp.choice('loss', ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
        'penalty': hp.choice('penalty', [None, 'l2', 'l1', 'elasticnet']),
        'alpha': hp.uniform('alpha', 0.0001, 10.0),
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
        'epsilon': hp.uniform('epsilon', 1.00001, 20.0),
        'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling'])
    }
    
    #Minimize the objective functionover the search space using hyperopt
    #Then train the model with the optimized hyperparameters
    trials = Trials()
    if (opt_type=='bayesian'):
        best_params = fmin(obj_func, search_space, algo=tpe.suggest, max_evals=100, trials=trials)
        
        #If condition to account for each item in the list passed to loss
        if (best_params['loss'] == 0):
            best_params['loss'] = 'squared_loss'
        elif (best_params['loss'] == 1):
            best_params['loss'] = 'huber'
        elif (best_params['loss'] == 2):
            best_params['loss'] = 'epsilon_insensitive'
        else:
            best_params['loss'] = 'squared_epsilon_insensitive'
        #If condition to account for each item in the list passed to penalty
        if (best_params['penalty'] == 0):
            best_params['penalty'] = None
        elif (best_params['penalty'] == 1):
            best_params['penalty'] = 'l2'
        elif (best_params['penalty'] == 2):
            best_params['penalty'] = 'l1'
        else:
            best_params['penalty'] = 'elasticnet'
        #If condition to account for each item in the list passed to learning_rate
        if (best_params['learning_rate'] == 0):
            best_params['learning_rate'] = 'constant'
        elif (best_params['learning_rate'] == 1):
            best_params['learning_rate'] = 'optimal'
        else:
            best_params['learning_rate'] = 'invscaling'
        
        best_model = SGDRegressor(loss=best_params['loss'],
                                  penalty=best_params['penalty'],
                                  alpha=best_params['alpha'],
                                  l1_ratio=best_params['l1_ratio'],
                                  epsilon=best_params['epsilon'],
                                  learning_rate=best_params['learning_rate'])
        best_results = cross_val_score(best_model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        return best_model, best_results
    elif (opt_type=='random'):
        best_params = fmin(obj_func, search_space, algo=rand.suggest, max_evals=100, trials=trials)
        
        #If condition to account for each item in the list passed to loss
        if (best_params['loss'] == 0):
            best_params['loss'] = 'squared_loss'
        elif (best_params['loss'] == 1):
            best_params['loss'] = 'huber'
        elif (best_params['loss'] == 2):
            best_params['loss'] = 'epsilon_insensitive'
        else:
            best_params['loss'] = 'squared_epsilon_insensitive'
        #If condition to account for each item in the list passed to penalty
        if (best_params['penalty'] == 0):
            best_params['penalty'] = None
        elif (best_params['penalty'] == 1):
            best_params['penalty'] = 'l2'
        elif (best_params['penalty'] == 2):
            best_params['penalty'] = 'l1'
        else:
            best_params['penalty'] = 'elasticnet'
        #If condition to account for each item in the list passed to learning_rate
        if (best_params['learning_rate'] == 0):
            best_params['learning_rate'] = 'constant'
        elif (best_params['learning_rate'] == 1):
            best_params['learning_rate'] = 'optimal'
        else:
            best_params['learning_rate'] = 'invscaling'
        
        best_model = SGDRegressor(loss=best_params['loss'],
                                  penalty=best_params['penalty'],
                                  alpha=best_params['alpha'],
                                  l1_ratio=best_params['l1_ratio'],
                                  epsilon=best_params['epsilon'],
                                  learning_rate=best_params['learning_rate'])
        best_results = cross_val_score(best_model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        return best_model, best_results
    else:
        print('Invalid opt_type')
        return

sgd_best_model, sgd_best_results = sgd_model_opt(opt_type='bayesian')

print('Stochastic Gradient Descent Cross Validated MSE:', abs(sgd_results.mean()))
print('Optimized Stochastic Gradient Descent Cross Validated MSE:', abs(sgd_best_results.mean()))
