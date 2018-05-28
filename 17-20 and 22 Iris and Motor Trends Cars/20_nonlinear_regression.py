#Python code for chapter 20 DSILT: Statistics

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

d = pd.read_csv('mtcars.csv')
d.info()

def get_lr_output(lr, x, y):
    #Takes a fitted sklearn linear regression as input and outputs results
    #Thanks to stackoverflow: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    import numpy as np
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

linear_reg = LinearRegression()
linear_reg.fit(d.drop(['mpg'], axis=1), d['mpg'])
linear_reg_output = get_lr_output(linear_reg, d.drop(['mpg'], axis=1), d['mpg'])

'''
-------------------------------------------------------------------------------
---------------------------Polynomial Regression-------------------------------
-------------------------------------------------------------------------------
'''

def polynomial_regression(x, y, deg=1, x_exclude_from_poly=[]):
    #Takes dataframe input and outputs a fitted polynomial regression of order deg
    #The list passed as x_exclude_from_poly has features that should not have polynomial terms
    #PolynomialFeatures transforms a formula like a+b+c to a^2+ab+ac+b^2+bc+c^2 for example of 2nd degreee
    #Example: polynomial_regression(df[['a', 'b', 'c']], df['d'], deg=2, x_exclude_from_poly=['c'])
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import numpy as np
    x_poly = x.drop(x_exclude_from_poly, axis=1)
    x_remainder = x[[col for col in x.columns if col not in x_poly.columns]]
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(np.asarray(x_poly).reshape(-1,len(x_poly.columns)))
    x_poly = pd.DataFrame(x_poly[:,1:])  #Gets rid of first column of 1's
    colnames = poly.get_feature_names()[1:]
    x_poly.columns = colnames
    x = pd.concat([x_poly, x_remainder], axis=1)
    linear_reg = LinearRegression().fit(x, y)
    return linear_reg, x

#Polynomial regression of orders 3, 5, 10 with 1 predictor
poly_reg_three, x_poly_three = polynomial_regression(d[['disp']], d['mpg'], deg=3)
poly_reg_three_output = get_lr_output(poly_reg_three, x_poly_three, d['mpg'])
poly_reg_five, x_poly_five = polynomial_regression(d[['disp']], d['mpg'], deg=5)
poly_reg_five_output = get_lr_output(poly_reg_five, x_poly_five, d['mpg'])
poly_reg_ten, x_poly_ten = polynomial_regression(d[['disp']], d['mpg'], deg=10)
poly_reg_ten_output = get_lr_output(poly_reg_ten, x_poly_ten, d['mpg'])

print(poly_reg_three_output)
print(poly_reg_five_output)
print(poly_reg_ten_output)

#Plot the fitted lines
order = d['disp'].sort_values().index.tolist()
plt.scatter(d['disp'], d['mpg'])
plt.plot(d['disp'][order], poly_reg_three.predict(x_poly_three)[order], color='orange')
plt.plot(d['disp'][order], poly_reg_five.predict(x_poly_five)[order], color='green')
plt.plot(d['disp'][order], poly_reg_ten.predict(x_poly_ten)[order], color='purple')
plt.xlabel('disp')
plt.ylabel('mpg')
plt.title('Polynomial Regression of Orders 3, 5, and 10')
plt.show()

#Polynomial regression of order 3 with many predictors
poly_reg_three_complex, x_poly = polynomial_regression(d[['disp', 'hp', 'drat', 'wt']], d['mpg'], deg=3, x_exclude_from_poly=['drat', 'wt'])
poly_reg_three_complex_output = get_lr_output(poly_reg_three_complex, x_poly, d['mpg'])
print(poly_reg_three_complex_output)

'''
-------------------------------------------------------------------------------
----------------------------Isotonic Regression--------------------------------
-------------------------------------------------------------------------------
'''

#Simulate a function of mpg that is increasing
dsim = d[['mpg', 'disp']].copy()
dsim['mpg'] = dsim['mpg']+np.random.normal()
dsim['disp'] = dsim['disp']*(-1)
dsim.head()
#Convert df to numpy array
dsim_x = np.asarray(dsim['disp'])
dsim_y = np.asarray(dsim['mpg'])

#View scatter plot of x and mpg
plt.scatter(dsim['disp'], dsim['mpg'])
plt.xlabel('disp')
plt.ylabel('mpg')
plt.title('Scatterplot of x and y')
plt.show()

#Fit isotonic regression
iso_reg = IsotonicRegression()
print(iso_reg.get_params())
iso_fitted_values = iso_reg.fit_transform(dsim_x, dsim_y)
iso_predictions = iso_reg.predict(dsim_x)
print('R squared:', iso_reg.score(dsim_x, dsim_y))

#Plot the fitted line
order = dsim['disp'].sort_values().index.tolist()
plt.scatter(dsim['disp'], dsim['mpg'])
plt.plot(dsim['disp'][order], iso_fitted_values[order], color='brown')
plt.xlabel('disp')
plt.ylabel('mpg')
plt.title('Isotonic Regression')
plt.show()

'''
-------------------------------------------------------------------------------
--------------------------------Smoothing--------------------------------------
-------------------------------------------------------------------------------
'''

#Smoothing with moving average of a 6 value window (the observation and 6 values around it)
rolling_window = pd.Series(dsim['mpg']).rolling(window=6, center=True)
smooth_ma = rolling_window.mean()
print(smooth_ma)
plt.scatter(list(dsim.index), dsim['mpg'])
plt.plot(list(dsim.index), smooth_ma, color='darkgreen')
plt.xlabel('Row Index')
plt.ylabel('mpg')
plt.title('Moving Average Smoothing')
plt.show()

#Smoothing with LOESS
import statsmodels.api as sm
loess = sm.nonparametric.lowess(dsim_y, dsim_x)
loess_x = list(zip(*loess))[0]
loess_y = list(zip(*loess))[1]
plt.scatter(dsim['disp'], dsim['mpg'])
plt.plot(loess_x, loess_y, color='blue')
plt.xlabel('-1*disp')
plt.ylabel('mpg')
plt.title('LOESS Smoothing')
plt.show()

'''
-------------------------------------------------------------------------------
------------------------Generalized Additive Models----------------------------
-------------------------------------------------------------------------------
'''

#GAMs
#https://github.com/dswah/pyGAM
#https://codeburst.io/pygam-getting-started-with-generalized-additive-models-in-python-457df5b4705f
from pygam import LinearGAM, LogisticGAM
gam_model = LinearGAM().fit(d[['disp', 'wt']], d['mpg'])
print(gam_model.summary())
gam_predictions = gam_model.predict(d[['disp', 'wt']])
gam_mse = np.mean((gam_predictions-d['mpg'])**2)
print('MSE:', gam_mse)

#Plot the predictions with confidence intervals
plt.plot(list(d.index), gam_predictions, 'r--')
plt.plot(list(d.index), gam_model.prediction_intervals(d[['disp', 'wt']], width=.95), color='b', ls='--')
plt.scatter(list(d.index), d['mpg'], facecolor='gray', edgecolors='none')
plt.xlabel('Row Index')
plt.ylabel('mpg')
plt.title('GAM Prediction with 95% Condidence Interval')
plt.show()

#Plot with simulated posterior
for response in gam_model.sample(d[['disp', 'wt']], d['mpg'], quantity='y', n_draws=50, sample_at_X=d[['disp', 'wt']]):
    plt.scatter(list(d.index), response, alpha=0.03, color='k')
plt.plot(list(d.index), gam_predictions, 'r--')
plt.plot(list(d.index), gam_model.prediction_intervals(d[['disp', 'wt']], width=.95), color='b', ls='--')
plt.xlabel('Row Index')
plt.ylabel('mpg')
plt.title('GAM Prediction with 95% Condidence Interval')
plt.show()

#Plot the partial dependecies of the predictors with confidence intervals
plt.rcParams['figure.figsize'] = (12, 8)
fig, axs = plt.subplots(1, len(list(d[['disp', 'wt']].columns)))
titles = list(d[['disp', 'wt']].columns)
for i, ax in enumerate(axs):
    partial_dep, confidence = gam_model.partial_dependence(d[['disp', 'wt']], feature=i+1, width=0.95)
    print(partial_dep)
    order = d[['disp', 'wt']][titles[i]].sort_values().index.tolist()
    ax.plot(d[['disp', 'wt']][titles[i]].values[order], partial_dep[order])
    ax.plot(d[['disp', 'wt']][titles[i]].values[order], confidence[0][:, 0][order], c='grey', ls='--')
    ax.plot(d[['disp', 'wt']][titles[i]].values[order], confidence[0][:, 1][order], c='grey', ls='--')
    ax.set_title(titles[i])
plt.show()
#The strength & direction of the relationship corresponds to the slope of the line
#Nonlinear lines should have smoothing applied (they are already smoothed in this example)

#Try different hyperparameters
spline_exp = [3, 3]     #Type of spline to fit to each variable
nbr_splines = [10, 20]  #This must be > spline order
gam_model = LinearGAM(spline_order=spline_exp, n_splines=nbr_splines).fit(d[['disp', 'wt']], d['mpg'])
print(gam_model.summary())
gam_predictions = gam_model.predict(d[['disp', 'wt']])
gam_mse = np.mean((gam_predictions-d['mpg'])**2)
print('MSE:', gam_mse)

#Add binary and categorical predictors to the GAM
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
d['h_bin'] = encoder.fit_transform(pd.cut(d['hp'], 5))
gam_model = LinearGAM().fit(d[['disp', 'wt', 'vs', 'h_bin']], d['mpg'])
print(gam_model.summary())
gam_predictions = gam_model.predict(d[['disp', 'wt', 'vs', 'h_bin']])
gam_mse = np.mean((gam_predictions-d['mpg'])**2)
print('MSE:', gam_mse)

#Performing classification (logistic regression) with the GAM
d['mpg_bin'] = encoder.fit_transform(pd.cut(d['mpg'], [0, 20, 100]))
gam_model = LogisticGAM().gridsearch(d[['disp', 'wt', 'vs', 'h_bin']], d['mpg_bin'])
print(gam_model.summary())
print('Classification Accuracy:', gam_model.accuracy(d[['disp', 'wt', 'vs', 'h_bin']], d['mpg_bin']))

#This models the conditional probabilities for mpg being < 20 and >=20
#Note the y-axis of these plots is the logit

'''
-------------------------------------------------------------------------------
-------------------Classification and Regression Trees-------------------------
-------------------------------------------------------------------------------
'''

#Regression tree
reg_tree = DecisionTreeRegressor(criterion='mse', min_samples_split=20).fit(d.drop(['mpg', 'mpg_bin'], axis=1), d['mpg'])
print(reg_tree.get_params)
for i, f in enumerate(d.drop(['mpg', 'mpg_bin'], axis=1).columns):
    print('Importance of', f, reg_tree.feature_importances_[i])
reg_tree_predictions = reg_tree.predict(d.drop(['mpg', 'mpg_bin'], axis=1))
print('MSE:', reg_tree.score(d.drop(['mpg', 'mpg_bin'], axis=1), d['mpg']))

iris = pd.read_csv('iris.csv')
iris.info()

#Classification tree
class_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=20).fit(iris.drop(['Species'], axis=1), iris['Species'])
print(class_tree.get_params)
class_tree_predictions = class_tree.predict(iris.drop(['Species'], axis=1))
class_tree_prob_predictions = class_tree.predict_proba(iris.drop(['Species'], axis=1))
print('Classification Accuracy:', class_tree.score(iris.drop(['Species'], axis=1), iris['Species']))

'''
-------------------------------------------------------------------------------
---------------------------K-Nearest Neighbors---------------------------------
-------------------------------------------------------------------------------
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import metrics

#KNN regression
x = d.drop(['mpg'], axis=1).values
y = d[['mpg']].values
normalizer = MinMaxScaler()
x = normalizer.fit_transform(x)
y = normalizer.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=14)

knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(x_train, y_train)
preds = knn_reg.predict(x_test)
plt.scatter(y_test, preds)
plt.xlabel('actual mpg')
plt.ylabel('predicted mpg')
plt.show()
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))

#KNN classification, k=3
encoder = LabelEncoder()
categorical_vars = ['Species']
categorical_var_mapping = dict()
for cv in categorical_vars:
    iris[cv] = encoder.fit_transform(iris[cv])  #Encodes as integer
    categorical_var_mapping[cv] = list(encoder.classes_)  #Saves integer to category mapping

x = iris.drop(['Species'], axis=1).values
y = iris[['Species']].values
normalizer = MinMaxScaler()
x = normalizer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=14)

knn_class = KNeighborsClassifier(n_neighbors=3)
knn_class.fit(x_train, y_train.ravel())
preds = knn_class.predict(x_test)
pred_probs = knn_class.predict_proba(x_test)
print(metrics.log_loss(y_test, pred_probs, labels=np.array([0,1,2])))
print(metrics.accuracy_score(y_test, preds))
#View the confusion matrix with predicted values on the left
print(metrics.confusion_matrix(y_test, preds))

