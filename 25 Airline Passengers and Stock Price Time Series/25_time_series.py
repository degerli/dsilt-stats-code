#Python code for chapter 25 DSILT: Statistics

'''
-------------------------------------------------------------------------------
---------------------------Time Series Decomposition---------------------------
-------------------------------------------------------------------------------
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

d = pd.read_csv('/home/dsilt/Desktop/dsilt-stats-code/25 Airline Passengers and Stock Price Time Series/AirPassengers.csv')

#Fix the dates
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
d['Month'] = months*12
d['time'] = d['time'].astype('str')
d['time'] = d['time'].apply(lambda x: x[0:4])
d['time'] = d['time']+'-'+d['Month']
d['time'] = d['time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%b'))
d.index = d.time
del d['Month'], d['Unnamed: 0'], d['time']
print(d.head())
print(d.dtypes)

#Convert to series to simplify future commands
dts = pd.Series(d['value'])
#print(dts['1949'])
plt.plot(dts)
plt.show()

#Look at ACF and PACF, and perform Augmented Dickey-Fuller test for stationarity
plot_acf(dts)
plt.show()
plot_acf(dts, lags=30)
plt.show()
plot_pacf(dts)
plt.show()
plot_pacf(dts, lags=30)
plt.show()
adftest = adfuller(dts, autolag='AIC')
print('Augmented Dickey-Fuller Test Results',
      '\nTest Statistic:', adftest[0],
      '\np-value:', adftest[1],
      '\nLags Used:', adftest[2],
      '\nNumber of Observations:', adftest[3])
#Cannot reject the null that series is non-stationary

#Note that as the time series increases in magnitude, seasonal variation increases too.  This is a multiplicative relationship (trend*season*random)
#If the above statement were not true, it would be an additive relationship (trend+season+random)

#Smooth by using a centered moving average (data is monthly, so 12 month trend seems appropriate)
trend = dts.rolling(12).mean()
plt.plot(dts)
plt.plot(trend)
plt.show()

#Remove the trend: remove multiplicative trends by dividing and additive trends by subtracting
dts_detrend = dts/trend
plt.plot(dts_detrend)
plt.show()

#Average the seasonality over the period of the MA trend (it was 12 for this dataset)
mseas = np.asarray(dts_detrend).reshape(12, 12).T  #Create a matrix of the time periods, where columns are the periods
season = np.nanmean(mseas, axis=1)                 #Column means are the averaged seasonality for each period
season = np.asarray(list(season)*12)
plt.plot(season)
plt.show()

#Remove the seasonality: remove multiplicative seasonality by dividing and additive seasonality by subtracting
dts_random = dts / (trend*season)
plt.plot(dts_random)
plt.show()

#Do the same thing using the seasonal_decompose function (requires a pandas object) - this is classical decompostion with MAs
help(seasonal_decompose)
dts_decomp = seasonal_decompose(dts, model='multiplicative')
dts_decomp.plot()
plt.show()

#Do the same thing using LOESS decomposition
#This function was taken from https://github.com/jrmontag/STLDecompose
def stl_decompose(df, period=365, lo_frac=0.6, lo_delta=0.01, allow_multiplicative_trend=False):
    '''Create a seasonal-trend (with Loess, aka "STL") decomposition of observed time series data.
    This implementation is modeled after the ``statsmodels.tsa.seasonal_decompose`` method 
    but substitutes a Lowess regression for a convolution in its trend estimation.
    Defaults to an additive model, Y[t] = T[t] + S[t] + e[t]
    For more details on lo_frac and lo_delta, see: 
    `statsmodels.nonparametric.smoothers_lowess.lowess()`
    Args:
        df (pandas.Dataframe): Time series of observed counts. This DataFrame must be continuous (no 
            gaps or missing data), and include a ``pandas.DatetimeIndex``.  
        period (int, optional): Most significant periodicity in the observed time series, in units of
            1 observation. Ex: to accomodate strong annual periodicity within years of daily 
            observations, ``period=365``. 
        lo_frac (float, optional): Fraction of data to use in fitting Lowess regression. 
        lo_delta (float, optional): Fractional distance within which to use linear-interpolation 
            instead of weighted regression. Using non-zero ``lo_delta`` significantly decreases 
            computation time.
    Returns:
        `statsmodels.tsa.seasonal.DecomposeResult`: An object with DataFrame attributes for the 
            seasonal, trend, and residual components, as well as the average seasonal cycle. 
    '''
    import numpy as np
    import pandas as pd
    from pandas.core.nanops import nanmean as pd_nanmean
    from statsmodels.tsa.seasonal import DecomposeResult
    from statsmodels.tsa.filters._utils import _maybe_get_pandas_wrapper_freq
    import statsmodels.api as sm

    # use some existing pieces of statsmodels    
    lowess = sm.nonparametric.lowess
    _pandas_wrapper, _ = _maybe_get_pandas_wrapper_freq(df)

    # get plain np array
    observed = np.asanyarray(df).squeeze()

    # calc trend, remove from observation
    trend = lowess(observed, [x for x in range(len(observed))], 
                   frac=lo_frac, 
                   delta=lo_delta * len(observed),
                   return_sorted=False)
    detrended = observed - trend

    # calc one-period seasonality, remove tiled array from detrended
    period_averages = np.array([pd_nanmean(detrended[i::period]) for i in range(period)])
    # 0-center the period avgs
    period_averages -= np.mean(period_averages)
    seasonal = np.tile(period_averages, len(observed) // period + 1)[:len(observed)]    
    resid = detrended - seasonal

    # convert the arrays back to appropriate dataframes, stuff them back into 
    #  the statsmodel object
    results = list(map(_pandas_wrapper, [seasonal, trend, resid, observed]))    
    dr = DecomposeResult(seasonal=results[0],
                         trend=results[1],
                         resid=results[2], 
                         observed=results[3],
                         period_averages=period_averages)
    return dr

stl = stl_decompose(dts, period=12, allow_multiplicative_trend=False)
stl.plot()
plt.show()

#Test for stationarity after seasonality and trend have been removed
adftest = adfuller(stl.resid, autolag='AIC')
print('Augmented Dickey-Fuller Test Results',
      '\nTest Statistic:', adftest[0],
      '\np-value:', adftest[1],
      '\nLags Used:', adftest[2],
      '\nNumber of Observations:', adftest[3])
#Cannot reject the null that series is non-stationary

#Define differencing function, this was stolen from: https://machinelearningmastery.com/difference-time-series-dataset-python/
def diff(dataset, differences=1):
	diff = list()
	for i in range(differences, len(dataset)):
		value = dataset[i] - dataset[i - differences]
		diff.append(value)
	return pd.Series(diff)

#Perform first differencing to see if that makes series stationary
adftest = adfuller(diff(stl.resid, differences=1), autolag='AIC')
print('Augmented Dickey-Fuller Test Results',
      '\nTest Statistic:', adftest[0],
      '\np-value:', adftest[1],
      '\nLags Used:', adftest[2],
      '\nNumber of Observations:', adftest[3])
#Can reject the null that series is non-stationary, the series is difference stationary

'''
-------------------------------------------------------------------------------
------------------------------------ARIMA--------------------------------------
-------------------------------------------------------------------------------
'''

#Replicate the auto.arima function from R - it uses grid search to opmtimize AIC or BIC
def auto_arima(ts, max_p=5, max_d=5, max_q=5, ic='bic'):
    from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  #Ignores all convergence warnings
        best_ic = 1e24
        best_params = (0,0,0)
        for p in range(max_p):
            for d in range(max_d):
                for q in range(max_q):
                    try:
                        model = ARIMA(ts.astype('float32'), order=(p+1, d+1, q+1))
                        fitted = model.fit()
                        if ic=='aic' and fitted.aic < best_ic:
                            best_ic = fitted.aic
                            best_params = (p+1, d+1, q+1)
                            best_model = fitted
                        elif ic=='bic' and fitted.bic < best_ic:
                            best_ic = fitted.bic
                            best_params = (p+1, d+1, q+1)
                            best_model = fitted
                    except:
                        continue
    return best_model, best_params, best_ic

arima_model, arima_params, arima_ic = auto_arima(dts, max_p=7, max_d=2, max_q=5, ic='bic')
plt.plot(arima_model.fittedvalues)
plt.title('ARIMA Fitted Values')
plt.show()
plt.plot(arima_model.resid)
plt.title('ARIMA Redisuals')
plt.show()
print('Best ARIMA model had params', arima_params, 'and BIC', arima_ic)
#p-values are cumulative for this test, so restrict lags or get spurious significance in distant lags
lag_test_stats, lag_p_vals = acorr_ljungbox(arima_model.resid, lags=10, boxpierce=False)  

#Make future predictions for next 24 months
preds = arima_model.forecast(steps=24) #Returns tuple of preds, stderr, conf_int
plt.plot(np.array(dts))
plt.plot(np.array(range(144, 144+24)), preds[0])  #Shift preds x-axis bc there are 144 obs in original data
plt.title('ARIMA Predictions for Next 24 Months')
plt.show()

'''
-------------------------------------------------------------------------------
----------------------------------ARIMA+GARCH----------------------------------
-------------------------------------------------------------------------------
'''

import datetime as dt
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like  #Fixes import error for datareader
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox


start_dt = dt.datetime(2012, 1, 1)
end_dt = dt.datetime(2016, 1, 1)

d = web.DataReader('SPY', 'morningstar', start_dt, end_dt)
dts = pd.Series(d['Close'])
dts.index= d.index.levels[1]
log_returns = pd.Series(np.log(dts/dts.shift(1)).dropna())
#Note that log returns = percent change
#print(d.pct_change().dropna())
log_returns.plot()
plt.show()
dts.plot()
plt.show()


def auto_arima(ts, max_p=5, max_d=5, max_q=5, ic='bic'):
    from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  #Ignores all convergence warnings
        best_ic = 1e24
        best_params = (0,0,0)
        for p in range(max_p):
            for d in range(max_d):
                for q in range(max_q):
                    try:
                        model = ARIMA(ts.astype('float32'), order=(p+1, d+1, q+1))
                        fitted = model.fit()
                        if ic=='aic' and fitted.aic < best_ic:
                            best_ic = fitted.aic
                            best_params = (p+1, d+1, q+1)
                            best_model = fitted
                        elif ic=='bic' and fitted.bic < best_ic:
                            best_ic = fitted.bic
                            best_params = (p+1, d+1, q+1)
                            best_model = fitted
                    except:
                        continue
    return best_model, best_params, best_ic


arima_model, arima_params, arima_ic = auto_arima(log_returns, max_p=7, max_d=2, max_q=5, ic='bic')
plt.plot(arima_model.fittedvalues)
plt.title('ARIMA Fitted Values')
plt.show()
plt.plot(arima_model.resid)
plt.title('ARIMA Redisuals')
plt.show()
print('Best ARIMA model had params', arima_params, 'and BIC', arima_ic)
#p-values are cumulative for this test, so restrict lags or get spurious significance in distant lags
lag_test_stats, lag_p_vals = acorr_ljungbox(arima_model.resid, lags=10, boxpierce=False)  

#Make future predictions for next 60 days
preds = arima_model.forecast(steps=60) #Returns tuple of preds, stderr, conf_int
plt.plot(np.array(log_returns))
plt.plot(np.array(range(len(dts), len(dts)+60)), preds[0])  #Shift preds x-axis
plt.title('ARIMA Predictions for Next 60 Days')
plt.show()

#Now fit a GARCH model, using the parameters of the best fitting ARIMA model
garch_model = arch_model(log_returns, p=arima_params[0], o=arima_params[1], q=arima_params[2], dist='StudentsT')
garch_fitted = garch_model.fit()
print(garch_fitted.summary())
plt.plot(garch_fitted.resid)
plt.title('GARCH Redisuals')
plt.show()

plot_acf(garch_fitted.resid, lags=30)
plt.show()
plot_pacf(garch_fitted.resid, lags=30)
plt.show()

#View predicted volatility
garch_fitted.hedgehog_plot(horizon=60)
plt.show()

'''
-------------------------------------------------------------------------------
------------------Pairs Trading Using Cointegration----------------------------
-------------------------------------------------------------------------------
'''

'''
Pairs trading uses linear regression to find cointegrated, stationary pairs
Regression finds the mean hedge ratio over a time period.  If the hedge ratio 
changes, then it must be smoothed.  The Kalman filter is one way to smooth it:
https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter
Pairs trading hedges two stocks and relies on the ADF test to look for 
stationarity.  Hedging a basket of stocks relies on the Johansson test for 
stationarity:
https://quant.stackexchange.com/questions/14327/need-help-on-cointegration
For basket trading, linear regression can again be used to find the hedge 
ratios, and the Kalman filter can again be used to smooth dynamic hedge ratios.
Hedge ratios can be assumed to be drawn from normal distributions, so z-scores
or confidence intervals can bound the hedge ratio.  The hedge ratio is mean 
reverting by definition (stationary time series are mean reverting), so any 
time it diverges from the mean, trading can be done.
'''

#Let's cheat and pick 2 that are known to be cointegrated to see what happens
#We'll also cherry pick a time frame during which these stocks were stationary
start_dt = dt.datetime(2008, 2, 1)
end_dt = dt.datetime(2012, 2, 1)

ewap = web.DataReader('EWA', 'morningstar', start_dt, end_dt)
ewcp = web.DataReader('EWC', 'morningstar', start_dt, end_dt)

ewapts = pd.Series(ewap['Close'])
ewcpts = pd.Series(ewcp['Close'])
ewapts.index = ewap.index.levels[1]
ewcpts.index = ewcp.index.levels[1]

plt.plot(ewapts)
plt.plot(ewcpts)
plt.show()

#Define function to check for cointegration using linear regression
def coint_test(stockA, stockB):
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.stattools import adfuller
    stockA = stockA.values.reshape(-1, 1)
    stockB = stockB.values.reshape(-1, 1)
    stockAm = LinearRegression().fit(stockB, stockA)
    stockBm = LinearRegression().fit(stockA, stockB)
    stockAm_resid = stockA-stockAm.predict(stockB)
    stockBm_resid = stockB-stockBm.predict(stockA)
    adfA = adfuller(stockAm_resid.reshape(stockAm_resid.shape[0],), autolag='AIC')
    adfB = adfuller(stockBm_resid.reshape(stockBm_resid.shape[0],), autolag='AIC')
    if (adfA[1] >= 0.05 and adfB[1] >= 0.05):
        print('Non-stationary, cannot do cointegration')
    elif adfA[0] < adfB[0]:
        print('Use the first input as the dependent variable in the linear combination.')
    else:
        print('Use the second input as the dependent variable in the linear combination.')

coint_test(ewapts, ewcpts)
from sklearn.linear_model import LinearRegression
ewa_hedge_model = LinearRegression().fit(ewcpts.values.reshape(-1, 1), ewapts.values.reshape(-1, 1))
print('Hedge ratio:', ewa_hedge_model.coef_)

#Use a Kalman filter to estimate the dynamic hedge ratio
from pykalman import KalmanFilter
def dynamic_regression_w_kalman(x, y):
    #w_t = covariance matrix of the noise term - decrease it to produce smoother result (the smaller, the less sensitive KF is to true values)
    w_t = 1e-5
    #A_t or state transition matrix is the matrix of beta coefficients for the regression - this will start off as a matrix of diagonal noise terms to simulate random walk
    state_transition_matrix = w_t/(1-w_t)*np.eye(2)
    #x_t or the observed matrix is the matrix of values of the independent variable on the left and the dependent variable on the right: [x, y] - initialize dependent vars as ones
    observation_matrix = np.vstack([x, np.ones(y.shape)]).T[:, np.newaxis]
    
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                      initial_state_mean=np.zeros(2), 
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2), 
                      transition_covariance=state_transition_matrix,
                      observation_matrices=observation_matrix, 
                      observation_covariance=1.0)
    
    return kf.filter(y.values)

state_means, state_covariances = dynamic_regression_w_kalman(ewcpts, ewapts)
dynamic_reg_params = pd.DataFrame({
        'slope': state_means[:, 0],
        'intercept': state_means[:, 1]
        }, index=ewapts.index)
dynamic_reg_params.plot()
plt.title('Dynamic Hedge Ratio of EWA to EWC')
plt.show()

'''
-------------------------------------------------------------------------------
--------------------------Change Point Detection-------------------------------
-------------------------------------------------------------------------------
'''

import datetime as dt
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like  #Fixes import error for datareader
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import ruptures as rpt

start_dt = dt.datetime(2008, 2, 1)
end_dt = dt.datetime(2017, 2, 1)

aapl = web.DataReader('AAPL', 'morningstar', start_dt, end_dt)

aaplpts = pd.Series(aapl['Close'])
aaplpts.index = aapl.index.levels[1]

plt.plot(aaplpts)
plt.show()

cp_model = rpt.Pelt(model='rbf', min_size=2).fit(aaplpts.values)
cp_preds = cp_model.predict(pen=3)  #Smaller penalty -> more change points
cp_preds = cp_preds[:-1]  #Drop the final change point, as it is at the end of the series

plt.plot(aaplpts)
plt.title('Change Points in AAPL Price')
for cp in cp_preds:
    plt.axvline(x=aaplpts.index[cp], color='purple', linestyle='--')
plt.show()

print('Change points occurred at:')
for cp in cp_preds:
    print(aaplpts.index[cp])

