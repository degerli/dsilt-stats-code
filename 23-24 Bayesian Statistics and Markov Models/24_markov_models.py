#Python code for chapter 24 DSILT: Statistics

'''
-------------------------------------------------------------------------------
------------------------------Hidden Markov Models-----------------------------
-------------------------------------------------------------------------------
'''

'''
Evaluation Problem
'''

import numpy as np
from hmmlearn import hmm
import math

states = ('healthy', 'sick')
observations = ('no-symptoms', 'cold', 'dizzy')

start_probability = {'healthy': 0.8, 'sick': 0.2}
startm = np.array([0.8, 0.2])
transition_probability = {
   'healthy' : {'healthy': 0.8, 'sick': 0.2},
   'sick' : {'healthy': 0.4, 'sick': 0.6},
   }
transm = np.array([[0.8, 0.2], 
                   [0.4, 0.6]])
emission_probability = {
   'healthy' : {'no-symptoms': 0.6, 'cold': 0.3, 'dizzy': 0.1},
   'sick' : {'no-symptoms': 0.1, 'cold': 0.3, 'dizzy': 0.6},
   }
emism = np.array([[0.6, 0.3, 0.1], 
                  [0.1, 0.3, 0.6]])

hmm_model = hmm.MultinomialHMM(n_components=len(states), algorithm='viterbi')
hmm_model.startprob_ = startm
hmm_model.transmat_ = transm
hmm_model.emissionprob_ = emism

#Evaluation: given a model, what is the probability of sequence y?
#Note that the score method produces the log likelihood, so to get prob, exponentiate
y = np.array([[0]])
print('Probability of first observation in a sequence being', 
      observations[0], 'regardless of state is', math.exp(hmm_model.score(y)))
y = np.array([[1]])
print('Probability of first observation in a sequence being', 
      observations[1], 'regardless of state is', math.exp(hmm_model.score(y)))
y = np.array([[2]])
print('Probability of first observation in a sequence being', 
      observations[2], 'regardless of state is', math.exp(hmm_model.score(y)))
y = np.array([[1, 1, 0]])
print('Probability of', observations[1], observations[1], observations[0], 
      'regardless of state is', math.exp(hmm_model.score(y)))
y = np.array([[0, 0, 2]])
print('Probability of', observations[0], observations[0], observations[2], 
      'regardless of state is', math.exp(hmm_model.score(y)))

'''
Decoding Problem
'''

#Decoding: given a model and a sequence of observations, what is the most likely sequence of states?
#Note that the score method produces the log likelihood, so to get prob, exponentiate
y = np.array([[2]]).T
print('Given observation', observations[2],
      'the most likely hidden state is', states[hmm_model.decode(y)[1][0]],
      'with probability', math.exp(hmm_model.decode(y)[0]))
y = np.array([[0, 1, 0]]).T
print('Given observations', observations[0], observations[1], observations[0],
      'the most likely sequence of hidden states is', states[hmm_model.decode(y)[1][0]],
      states[hmm_model.decode(y)[1][1]], states[hmm_model.decode(y)[1][2]], 
      'with probability', math.exp(hmm_model.decode(y)[0]))
y = np.array([[2, 1, 2]]).T
print('Given observations', observations[2], observations[1], observations[2],
      'the most likely sequence of hidden states is', states[hmm_model.decode(y)[1][0]],
      states[hmm_model.decode(y)[1][1]], states[hmm_model.decode(y)[1][2]], 
      'with probability', math.exp(hmm_model.decode(y)[0]))

'''
Learning problem
'''

import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import warnings
from hmmlearn.hmm import GaussianHMM

start_dt = dt.datetime(1993, 2, 1)
end_dt = dt.datetime(2016, 1, 1)

#d = web.DataReader('SPY', 'morningstar', start_dt, end_dt)
#d.to_csv('spy.csv')  #Write to csv so future runs don't need to call the API
d = pd.read_csv('spy.csv')
d = d[['Date', 'Close']]
date_index = pd.to_datetime(d['Date'])
dts = pd.Series(d['Close'])
dts.index = date_index
log_returns = pd.Series(np.log(dts/dts.shift(1)).dropna())
#Note that log returns = percent change
#print(d.pct_change().dropna())
log_returns.plot()
plt.title('Log Returns for SPY 1993-2016')
plt.show()
dts.plot()
plt.title('Closing Price of SPY 1993-2016')
plt.show()

#Convert log returns to a numpy array
log_returns_arr = np.column_stack([log_returns]).reshape(-1, 1)

#Ignore the litany of deprecation warnings
warnings.filterwarnings("ignore")

#Create a HMM with Gaussian emissions and assume there are 2 hidden states (bear and bull markets)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(log_returns_arr)
hidden_states = hmm_model.predict(log_returns_arr)

#Stack plots for each of the hidden states
#This plotting code comes from hmmlearn's documentation:
#http://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_hmm_stock_analysis.html
fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True)
colors = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))
for i, (ax, color) in enumerate(zip(axs, colors)):
    mask = hidden_states == i
    ax.plot_date(dts[1:].index[mask], d['Close'][1:][mask], ".", linestyle='none', c=color)
    ax.set_title("Hidden State #%s" % i)
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.grid(True)
plt.show()

'''
-------------------------------------------------------------------------------
--------------------------------Kalman Filter----------------------------------
-------------------------------------------------------------------------------
'''


'''
Kalman Filter on a 2D random walk
'''

import numpy as np
import random
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

random.seed(14)

def randomWalk2d(n_steps=1000, walkType="financial"):
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    if walkType == "random":
        for s in range(1, n_steps):
            val = random.randint(1,4)
            if val == 1:
                x[s] = x[s - 1] + 1
                y[s] = y[s - 1]
            elif val == 2:
                x[s] = x[s - 1] - 1
                y[s] = y[s - 1]
            elif val == 3:
                x[s] = x[s - 1]
                y[s] = y[s - 1] + 1
            else:
                x[s] = x[s - 1]
                y[s] = y[s - 1] - 1
    else:
        for s in range(1, n_steps):
            val = random.randint(1,3)
            if val == 1:
                x[s] = x[s - 1] + 1
                y[s] = y[s - 1]
            elif val == 2:
                x[s] = x[s - 1]
                y[s] = y[s - 1] + 1
            else:
                x[s] = x[s - 1]
                y[s] = y[s - 1] - 1
    return x,y

x, y = randomWalk2d()

#Plot random walk
plt.title("Random Walk ($n = " + str(1000) + "$ steps)")
plt.plot(x, y)
plt.show()

robotpath = np.array((x,y)).T

'''
Every state in this system consists of a coordinate pair (x,y) and nothing
else.  So the initial state is a matrix x equal to (x y) <- imagine those 
vertically stacked.

X
The initial state is assumed to be drawn from a normal distribution.  We can
set its mean equal to zero, so initial state is a 1*2m matrix, where m is the 
number of rows in the initial state matrix.  Since the initial state matrix
has 2 rows, the initial state mean matrix is a 1*2m matrix of where the 
diagonal contains the values of the x and y coordinates (x in first row, 
change in x in second row, y in third row, change in y in fourth row)

P
Since we have no idea what the covariance of each state is, it is safe to set
them all equal to 1.  The covariance matrix is an m*m matrix, where m is the 
number of rows in the initial state matrix.  Since the initial state matrix 
has 2 rows, the initial coveriance matrix is a 2*2 matrix of diagonal ones.  

H
The observation matrix maps each state to the next one.  Since this example is 
simply a random walk, the transition is the last state plus noise, so the 
observation matrix is filled with ones where the value of the coordinate 
changes.  So first row would be [1, 0, 0, 0] because it represents 
[x_coord, change in x_coord, y_coord, change in y_coord], and the second row
would be [0, 0, 1, 0]

Q
The process errror (measurement error) is essentially 0 since there are no 
measurements taken because we know the data directly.  So the observation 
covariance matrix is an m*m matrix of zeros. 

A
Each row and column of the transition matrix corresponds to a variable to be 
updated.  So if there were 2 variables: position and velocity, then H would 
be a 4*4 matrix.  The first row would be [1, 1, 0, 0] because it represents 
[x_coord, change in x_coord, y_coord, change in y_coord] and 
next_position = position + velocity*delta_time.  The second row would be 
[0, 1, 0, 0] because next_velocity = velocity. 

R
The measurement error covariance can be set to some small value.  This can be 
set to any value - it should really be played with to see how it affects the 
model. So the transition covariance is a m*m matrix of diagonal small values,
such as 0.2.
'''

#Parameter set up
init_state_mean = [robotpath[0, 0],
                   0,
                   robotpath[0, 1],
                   0]
init_state_cov = np.ones((4, 4))
obs_mat= [[1, 0, 0, 0],
          [0, 0, 1, 0]]
obs_cov = np.eye(2)
trans_mat = [[1, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 1],
             [0, 0, 0, 1]]
trans_cov = np.eye(4)*1e-6

kf = KalmanFilter(n_dim_obs=2, n_dim_state=4,
                  initial_state_mean=init_state_mean,
                  initial_state_covariance=init_state_cov,
                  observation_matrices=obs_mat,
                  observation_covariance=obs_cov,
                  transition_matrices=trans_mat,
                  transition_covariance=trans_cov)

state_means, state_covs = kf.filter(robotpath)

#Smoothing with expectation maximization
kf1 = kf.em(robotpath, n_iter=5)
smoothed_state_means, smoothed_state_covariances = kf1.smooth(robotpath)

#Plot the fitted Kalman filter
plt.plot(#x, y, 'bo',
         #state_means[:, 0], state_means[:, 2], 'ro',
         x, y, 'b--',
         state_means[:, 0], state_means[:, 2], 'r--',)
plt.title('Kalman Filter Fitted to a 2D Random Walk')
plt.show()
plt.plot(#x, y, 'bo',
         #state_means[:, 0], state_means[:, 2], 'ro',
         x, y, 'b--',
         smoothed_state_means[:, 0], smoothed_state_means[:, 2], 'r--',)
plt.title('Kalman Filter Fitted to a 2D Random Walk and Smoothed with Expectation Maximization')
plt.show()

#Plot the change in each variable over time
times = range(robotpath.shape[0])
plt.plot(#times, robotpath[:, 0], 'bo',
         #times, robotpath[:, 1], 'ro',
         times, state_means[:, 0], 'b--',
         times, state_means[:, 2], 'r--',)
plt.title('Split Plot of the Change in x and y throughout the Random Walk')
plt.show()
plt.plot(#times, robotpath[:, 0], 'bo',
         #times, robotpath[:, 1], 'ro',
         times, smoothed_state_means[:, 0], 'b--',
         times, smoothed_state_means[:, 2], 'r--',)
plt.title('Split Plot of the Smoothed Change in x and y throughout the Random Walk')
plt.show()

'''
Now try deleting some points from the random walk to see how the Kalman
filter adapts to missing states
'''

#Remove observations 400-500
x_h = x.copy()
y_h = y.copy()
x_h[range(400, 500)] = np.nan
y_h[range(400, 500)] = np.nan
robotpath_obscured = np.array((x_h, y_h)).T

#Parameter set up
init_state_mean = [robotpath_obscured[0, 0],
                   0,
                   robotpath_obscured[0, 1],
                   0]
init_state_cov = np.ones((4, 4))
obs_mat= [[1, 0, 0, 0],
          [0, 0, 1, 0]]
obs_cov = np.eye(2)
delta_x = x_h[1] - x_h[0]  #If x were time, this would be more meaningful, but here it is just 1.0
#Note that this trans_mat is the same as before, but we're using delta_x to show how it would change if dt were not 1.0
trans_mat = [[1, delta_x, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, delta_x],
             [0, 0, 0, 1]]
trans_cov = np.eye(4)*1e-6

kf = KalmanFilter(n_dim_obs=2, n_dim_state=4,
                  initial_state_mean=init_state_mean,
                  initial_state_covariance=init_state_cov,
                  observation_matrices=obs_mat,
                  observation_covariance=obs_cov,
                  transition_matrices=trans_mat,
                  transition_covariance=trans_cov)

#This is how the KF would be applied if all points were known - if we try it, it will fail when the values disappear
#state_means_h, state_covs_h = kf.filter(robotpath_obscured)
'''
#Plot the fitted Kalman filter - this shows how the previous approach fails when points are obscured
plt.plot(#x_h, y_h, 'bo',
         #state_means_h[:, 0], state_means_h[:, 2], 'ro',
         x_h, y_h, 'b--',
         state_means_h[:, 0], state_means_h[:, 2], 'r--',)
plt.show()
'''

#Make sure the tuples have the same number of dimensions as the initial state mean and initial state cov matrix (4, and (4,4) in this case)
state_means_h = np.zeros((x_h.shape[0], 4))
state_covs_h = np.zeros((x_h.shape[0], 4, 4))

#Apply the KF iteratively to iterpolate missing states
#Note that both x and y are missing, but we are assuming x always increments by 1 (delta_x). so we really only need to interpolate y
for t in range(robotpath_obscured[:,0].shape[0]):
    if np.isnan(robotpath_obscured[t,0]):
        x_h[t] = x_h[t-1]+1  #Increments x[t] by x[t-1]+1 when x[t] is missing
        state_means_h[t], state_covs_h[t] = (
            kf.filter_update(state_means_h[t-1],
                             state_covs_h[t-1],
                             observation=x_h[t])  #Only x is available
            )
    else:
        state_means_h[t], state_covs_h[t] = (
            kf.filter_update(state_means_h[t-1],
                             state_covs_h[t-1],
                             observation=robotpath_obscured[t])  #Both x and y are available
            )

#Plot the fitted Kalman filter
plt.plot(x, y, 'b--', label="Random Walk")
plt.plot(state_means[:, 0], state_means[:, 2], 'g--', label="KF with No Missing Values")
plt.plot(state_means_h[:, 0], state_means_h[:, 2], 'r--', label="KF with Missing Values")
plt.grid()
plt.legend(loc="upper left")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random Walk with Hidden States')
plt.show()

'''
Kalman Filter on 1D data

Example of Kalman filter with 1 variable: dealing with noisy readings from a 
voltmeter and estimating the true voltage.  This can be applied to any sensor, 
but note that a Kalman filter may not always be the best way to determine the 
true signal with noise.  Other filters could be more effective.

The class to simulate a voltmeter was taken from:
    http://greg.czerniak.info/guides/kalman1/
'''

class Voltmeter:
    def __init__(self, _truevoltage, _noiselevel):
        self.truevoltage = _truevoltage
        self.noiselevel = _noiselevel
    def GetVoltage(self):
        return self.truevoltage
    def GetVoltageWithNoise(self):
        return random.gauss(self.GetVoltage(), self.noiselevel)

voltmeter = Voltmeter(1.25, 0.25)

random.seed(14)

#Simulate voltmeter readings for 100 time steps
measuredvoltage = []
truevoltage = []
for i in range(100):
    measured = voltmeter.GetVoltageWithNoise()
    measuredvoltage.append(measured)
    truevoltage.append(voltmeter.GetVoltage())

#Parameter set up
init_state_mean = np.array((measuredvoltage[0]))
init_state_cov = np.ones((1, 1))
obs_mat= np.matrix([1])
obs_cov = np.eye(1)
trans_mat = np.matrix([1])
trans_cov = np.eye(1)*1e-6

kfv = KalmanFilter(n_dim_obs=1, n_dim_state=1,
                   initial_state_mean=init_state_mean,
                   initial_state_covariance=init_state_cov,
                   observation_matrices=obs_mat,
                   observation_covariance=obs_cov,
                   transition_matrices=trans_mat,
                   transition_covariance=trans_cov)

observations = np.array((measuredvoltage)).reshape(100,1)
state_means_v, state_covs_v = kfv.filter(observations)

#Plot the change in each variable over time
times = range(len(measuredvoltage))
plt.plot(times, measuredvoltage, 'b--', label='Measured Voltage')
plt.plot(times, truevoltage, 'g--', label='True Voltage')
plt.plot(times, state_means_v, 'r--', label='Kalman Filter')
plt.title('Noisy Voltage Measurements Smoothed by Kalman Filter')
plt.legend(loc="upper left")
plt.show()
