#Python code for chapter 19 DSILT: Statistics

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt

d = pd.read_csv('mtcars.csv')
d.info()

def get_lr_output(lr, x, y):
    #Takes a fitted sklearn linear regression as input and outputs results
    #Thanks to stackoverflow: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    parameters = np.append(lr.intercept_, lr.coef_)
    predictions = lr.predict(x)
    X = pd.DataFrame({"Constant":np.ones(len(x))}).join(pd.DataFrame(x))
    mse = (sum((y-predictions)**2))/(len(X)-len(X.columns))
    sd = np.sqrt(mse*(np.linalg.inv(np.dot(X.T, X)).diagonal()))
    ts = parameters/sd
    p_values = [2*(1-stats.t.cdf(np.abs(i), (len(X)-1))) for i in ts]
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

#Standard linear regression for comparison
linear_reg = LinearRegression()
linear_reg.fit(d.drop(['mpg'], axis=1), d['mpg'])
linear_reg_output = get_lr_output(linear_reg, d.drop(['mpg'], axis=1), d['mpg'])
linear_reg_output
linear_reg_predictions = linear_reg.predict(d.drop(['mpg'], axis=1))
order = d['disp'].sort_values().index.tolist()
plt.scatter(d['disp'], d['mpg'])
plt.plot(d['disp'][order], linear_reg_predictions[order], color='black')
plt.xlabel('disp')
plt.ylabel('mpg')
plt.title('Linear Regression Fit')
plt.show()

'''
-------------------------------------------------------------------------------
----------------------------------Ridge----------------------------------------
-------------------------------------------------------------------------------
'''

#Try a bunch of alpha (shrinkage) values between 0.01 and 1,000
alphas = [10**i for i in np.arange(3, -2, -0.1)]
alphas

#Ridge regression
#Run a ridge regression for every alpha
ridge_results = []
ridge_errors = []
for a in alphas:
    ridge_reg = Ridge(alpha=a).fit(d.drop(['mpg'], axis=1), d['mpg'])
    ridge_predictions = ridge_reg.predict(d.drop(['mpg'], axis=1))
    ridge_results.append(ridge_predictions)
    ridge_mse = np.mean((d['mpg']-ridge_predictions)**2)
    ridge_errors.append(ridge_mse)
print(ridge_errors)

#Plot the fit of a couple different models
order = d['disp'].sort_values().index.tolist()
plt.scatter(d['disp'], d['mpg'])
plt.plot(d['disp'][order], ridge_results[10][order], color='red')
plt.plot(d['disp'][order], ridge_results[25][order], color='blue')
plt.plot(d['disp'][order], ridge_results[45][order], color='green')
plt.xlabel('disp')
plt.ylabel('mpg')
plt.title('Ridge Regressions of Different Shrinkages')
plt.show()

#Run the ridge regression using default alpha and view results
ridge_reg = Ridge().fit(d.drop(['mpg'], axis=1), d['mpg'])
ridge_predictions = ridge_reg.predict(d.drop(['mpg'], axis=1))
ridge_reg_output = get_lr_output(ridge_reg, d.drop(['mpg'], axis=1), d['mpg'])
ridge_reg_output

'''
-------------------------------------------------------------------------------
-----------------------------------LASSO---------------------------------------
-------------------------------------------------------------------------------
'''

#Lasso regression
#Run a lasso regression for every alpha
lasso_results = []
lasso_errors = []
for a in alphas:
    lasso_reg = Lasso(alpha=a).fit(d.drop(['mpg'], axis=1), d['mpg'])
    lasso_predictions = lasso_reg.predict(d.drop(['mpg'], axis=1))
    lasso_results.append(lasso_predictions)
    lasso_mse = np.mean((d['mpg']-lasso_predictions)**2)
    lasso_errors.append(lasso_mse)
print(lasso_errors)

#Plot the fit of a couple different models
order = d['disp'].sort_values().index.tolist()
plt.scatter(d['disp'], d['mpg'])
plt.plot(d['disp'][order], lasso_results[10][order], color='red')
plt.plot(d['disp'][order], lasso_results[25][order], color='blue')
plt.plot(d['disp'][order], lasso_results[45][order], color='green')
plt.xlabel('disp')
plt.ylabel('mpg')
plt.title('Lasso Regressions of Different Shrinkages')
plt.show()

#Run the lasso regression using default alpha and view results
lasso_reg = Lasso().fit(d.drop(['mpg'], axis=1), d['mpg'])
lasso_predictions = lasso_reg.predict(d.drop(['mpg'], axis=1))
lasso_reg_output = get_lr_output(lasso_reg, d.drop(['mpg'], axis=1), d['mpg'])
lasso_reg_output

'''
-------------------------------------------------------------------------------
----------------------------------Elastic Net----------------------------------
-------------------------------------------------------------------------------
'''

#Try a bunch of l1 ratios (determines mix of L1 and L2 regularization) between 0 and 1
l1ratios = list(np.arange(0, 1, 0.01))
l1ratios

#Elastic Net
#Run an elastic net regression for every l1 ratio - ignore the convergence warnings
en_results = []
en_errors = []
for lr in l1ratios:
    en_reg = ElasticNet(l1_ratio=lr).fit(d.drop(['mpg'], axis=1), d['mpg'])
    en_predictions = en_reg.predict(d.drop(['mpg'], axis=1))
    en_results.append(en_predictions)
    en_mse = np.mean((d['mpg']-en_predictions)**2)
    en_errors.append(en_mse)
print(en_errors)

#Plot the fit of a couple different models
order = d['disp'].sort_values().index.tolist()
plt.scatter(d['disp'], d['mpg'])
plt.plot(d['disp'][order], en_results[20][order], color='red')
plt.plot(d['disp'][order], en_results[45][order], color='blue')
plt.plot(d['disp'][order], en_results[90][order], color='green')
plt.xlabel('disp')
plt.ylabel('mpg')
plt.title('Elastic Nets of Different L1 to L2 Ratios')
plt.show()

#Run the elastic net regression using default l1_ratio of 0.5 and view results
en_reg = ElasticNet().fit(d.drop(['mpg'], axis=1), d['mpg'])
en_predictions = en_reg.predict(d.drop(['mpg'], axis=1))
en_reg_output = get_lr_output(en_reg, d.drop(['mpg'], axis=1), d['mpg'])
en_reg_output
