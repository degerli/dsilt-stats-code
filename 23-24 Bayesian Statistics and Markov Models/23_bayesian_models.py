#Python code for chapter 23 DSILT: Statistics

'''
-------------------------------------------------------------------------------
-----------------------Combinations and Permutations---------------------------
-------------------------------------------------------------------------------
'''

import math

def permutation(n, k):
    return math.factorial(n) / math.factorial(k)

def combination(n, k):
    return permutation(n, k) / math.factorial(k)

'''
-------------------------------------------------------------------------------
--------------------Bayes' Theorem and Bayesian Inference----------------------
-------------------------------------------------------------------------------
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#Simulate 1000 coin flips (Bernoulli trials) from a binomial dist
bernoulli_trials = np.random.binomial(n=1, p=0.5, size=1000)
print(np.mean(bernoulli_trials))

#PMF of Bernoulli dist gives probability of a positive hit for a single Bernoulli trial
print(stats.bernoulli.pmf(1, 0.5))

#Generate 1000 random binary numbers to simulate coin flipping data
d = np.random.randint(2, size=1000)

#For indpendent events, the probability of observing the data is the product of the PMF
print(np.product(stats.bernoulli.pmf(d, 0.5)))

#Look at the probability of the data on an x scale of 0-1
x = np.linspace(0, 1, 100)
prob = [np.product(stats.bernoulli.pmf(d, p)) for p in x]
print(prob)

#Plot distribution of a fair coin (prior)
fair_flips = bernoulli_flips = np.random.binomial(n=1, p=.5, size=1000)
p_fair = np.array([np.product(stats.bernoulli.pmf(fair_flips, p)) for p in x])
p_fair = p_fair / np.sum(p_fair)
plt.plot(x, p_fair)
plt.title('Prior Probability')
plt.show()

#Look at the probability of an unfair coin (sample)
d = np.random.binomial(n=1, p=0.8, size=1000)
prob = np.array([np.product(stats.bernoulli.pmf(d, p)) for p in x])
print(prob)

#Plot the posterior distribution after applying Bayes' Theorem
#This function stolen from http://dataconomy.com/2015/02/introduction-to-bayes-theorem-with-python/
def bayes(n_sample, n_prior=100, observed_p=0.8, prior_p=0.5):
    x = np.linspace(0, 1, 100)
    sample = np.random.binomial(n=1, p=observed_p, size=n_sample)
    observed_dist = np.array([np.product(stats.bernoulli.pmf(sample, p)) for p in x])
    prior_sample = np.random.binomial(n=1, p=prior_p, size=n_prior)
    prior = np.array([np.product(stats.bernoulli.pmf(prior_sample, p)) for p in x])
    prior = prior / np.sum(prior)
    posterior = [prior[i] * observed_dist[i] for i in range(prior.shape[0])]
    posterior = posterior / np.sum(posterior)
     
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,8))
    axes[0].plot(x, observed_dist)
    axes[0].set_title("Sampling Distribution")
    axes[1].plot(x, prior)
    axes[1].set_title("Prior Distribution")
    axes[2].plot(x, posterior)
    axes[2].set_title("Posterior Distribution")
    plt.tight_layout()
    plt.show()
     
    return posterior

bayes(100)
bayes(1000)

'''
-------------------------------------------------------------------------------
--------------------------------Markov Chains----------------------------------
-------------------------------------------------------------------------------
'''

'''
Example 1: Trump Speech Generator
'''

import numpy as np

#Trump speech generator
d = open('trump_speeches.txt', encoding='utf8').read()
corpus = d.split()
#Function to generate all pairs of words
#Keys are all unique words and all of the words that appear after the key are stored as a list of values
def word_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])
pairs = word_pairs(corpus)
words = {}
for w1, w2 in pairs:
    if w1 in words.keys():
        words[w1].append(w2)
    else:
        words[w1] = [w2]

#Choose a word to initialize the Markov chain and the length of the chain in number of words
first_word = np.random.choice(corpus)
nbr_words = 40
def markov_chain_text(initial_state, word_pair_dict, stop_time):
    state = initial_state
    chain = [state]
    for w in range(stop_time):
        state = np.random.choice(word_pair_dict[chain[-1]])
        chain.append(state)
    return ' '.join(chain)

print(markov_chain_text(first_word, words, nbr_words))

'''
Example 2: Markov Chain for voting in presidential elections
'''

#Model voting democrat vs republican in consecutive presidential elections
#Assume the following split of voting population for this elect (time t)
rep_1 = 0.55
dem_1 = 0.45
#Assume the following transition probabilities
rep_to_rep = 0.8
rep_to_dem = 0.2
dem_to_dem = 0.7
dem_to_rep = 0.3
#Calculate split of voting population at next election (t+1)
rep_2 = rep_1*rep_to_rep + dem_1*dem_to_rep
dem_2 = dem_1*dem_to_dem + rep_1*rep_to_dem
print('Republican vs Democrat Split for Next Election:', rep_2, 'vs', dem_2)
#Define initial state and transition matrix to make calculations simpler in the future
state_1 = np.array([rep_1, dem_1])
transm = np.array([[rep_to_rep, rep_to_dem], [dem_to_rep, dem_to_dem]])
#Define function to generalize the calculations above to any future time
def markov_chain(initial_state, transition_matrix, stop_time):
    state = initial_state
    chain = [state]
    for t in range(stop_time):
        state = np.round(np.dot(state, transm), 4)
        chain.append(state)
    return chain
print(markov_chain(state_1, transm, 10))

#Notice the proportion appears to converge around a rep to dem ratio of 60:40
#Steady state is when initial_state * transition_matrix = initial_state, meaning there are no more changes to state
#Adjust markov_chain to stop if a steady state has been reached
def markov_chain(initial_state, transition_matrix, stop_time):
    state = initial_state
    chain = [state]
    for t in range(stop_time):
        if (np.round(np.dot(state, transm), 4) == state).all():
            print('Steady state reached at time t =', t)
            break
        else:
            state = np.round(np.dot(state, transm), 4)
        chain.append(state)
    return chain
print(markov_chain(state_1, transm, 20))

'''
-------------------------------------------------------------------------------
-----------------------------Bayesian Regression-------------------------------
-------------------------------------------------------------------------------
'''

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def glm_mcmc(df, formula, nbr_iters, likelihood=None, mcmc_sampler=None, target_accept=0.8, tune=500):
    '''
    This function is a wrapper for GLM Bayesian regression, using MCMC.  The 
    default MCMC method is NUTS (No U-Turn Sampler), but others are available:
        Binary variables --> BinaryMetropolis
        Discrete variables --> Metropolis
        Continuous variables --> NUTS
    
    Parameters:
        df = pandas dataframe
        formula = regression formula as specified in R or Patsy
        nbr_iters = number of iterations to sample over in MCMC
        likelihood = the family of likelihood functions for the data (e.g. pm.glm.families.Normal())
        mcmc_sampler = the MCMC sampling method (e.g. pm.NUTS, pm.Metropolis), default NUTS
        target_accept = adapts the step size to make acceptance probability closer to target, higher target --> smaller steps, range(0,1)
        tune = number of tuning samples to use in each iteration, higher tune --> more probability space explored
    
    If acceptance probability is higher than the target, increase the tune 
    parameters.  For more info about this, see:
        https://discourse.pymc.io/t/warning-when-nuts-probability-is-greater-than-acceptance-level/594
    
    If there are divergences after tuning, increase the target_accept 
    parameter.  For more info about this, see:
        https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html
        
    A great intro to GLM Bayesian regression with PYMC3 can be found here:
        https://docs.pymc.io/notebooks/GLM-linear.html
    '''
    bayesian_model = pm.Model()  #Initializes a Bayesian model container
    with bayesian_model:
        if likelihood is None:
            #Priors and observed sampling dist (likelihood) are automatically set
            pm.glm.GLM.from_formula(formula, df)
        else:
            pm.glm.GLM.from_formula(formula, df, family=likelihood)
        if mcmc_sampler is None:
            mcmc_sampler = pm.NUTS(target_accept=target_accept)
        else:
            mcmc_sampler = mcmc_sampler(target_accept=target_accept)
        #Perform inference to estimate the posterior, using nbr_iters MCMC sample posteriors
        trace = pm.sample(draws=nbr_iters, step=mcmc_sampler, 
                          init='auto', n_init=200000,
                          cores=None, tune=tune, progressbar=True,
                          random_seed=14)
    return bayesian_model, trace

def patsy_formula(df, dependent_var, *excluded_cols):
    '''
    This function generates the R style formula for Patsy, omitting the 
    dependent variable and any other specified variables from the right 
    side of the equation.  This is useful because Patsy does not have 
    a '.' operator.
    '''
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)

#GLM of mpg on disp
d = pd.read_csv('mtcars.csv')
formula = patsy_formula(d, 'mpg')

#Bayesian regression
bayesian_reg, trace = glm_mcmc(d, formula, nbr_iters=2000)
print(pm.summary(trace))  #Regression results
#SD is the error term
pm.autocorrplot(trace)
plt.show()

#Plot the last half of the posterior estimates, as the first few traces are likely poor estimates of the parameters
#First few estimates are called "burn-in"
pm.traceplot(trace[1000:], lines={k: v['mean'] for k,v in pm.summary(trace[1000:]).iterrows()})
plt.show()
#The traces, produced to the right, should be stationary if the model converged
#The traces will show autocorrelation if there is any
#The distributions, produced to the left, can be used to get the mean of the parameter estimate, credible regions, and look at skew
#Distributions should look normal - if they do, and performance is not satisfactory, specifying normal priors might improve it

#Model PPC and RMSE for evaluation
ppc = pm.sample_ppc(trace[1000:], samples=500, model=bayesian_reg)
rmse = np.sqrt(np.sum((ppc['y'].mean(0).mean(0).T - d['mpg'])**2) / d.shape[0])

def get_r2(df, ppc, target):
    sse_model = np.sum((ppc['y'].mean(0).mean(0).T - df[[target]])**2)[0]
    sse_mean = np.sum((df[[target]] - df[target].mean())**2)[0]
    return 1-(sse_model / sse_mean)
print('R Squared:', get_r2(d, ppc, 'mpg'))

#Get the posterior predictions
traces = pm.trace_to_dataframe(trace)[['Intercept'] + list(d.drop(['mpg'], axis=1).columns)]
x = d.drop(['mpg'], axis=1)
x.insert(0, 'intercept', 1)
likelihoods = np.dot(x, traces.T)
likelihoods_sd = np.tile(pm.trace_to_dataframe(trace)[['sd']].T, (d.shape[0], 1))
likelihood = np.random.normal(likelihoods, likelihoods_sd)
print(likelihood.shape)  #Should have d.shape[0] rows and as many columns as MCMC samples

#Get credible intervals for the target variable
dfp = pd.DataFrame(np.percentile(likelihood,[2.5, 25, 50, 75, 97.5], axis=1).T,
                   columns=['0_25','25','50','75','97_5'])

#Plot target variable over index and the credible intervals from Bayesian regression
pal = sns.color_palette('Purples')
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.fill_between(d.index, dfp['0_25'], dfp['97_5'], alpha=0.7, color=pal[1], label='CR 95%')
ax.fill_between(d.index, dfp['25'], dfp['75'], alpha=0.5, color=pal[4], label='CR 50%')
ax.plot(d.index, dfp['50'], alpha=0.5, color=pal[5], label='Median')
plt.plot(d.index, d['mpg'], 'bo')
plt.legend()
plt.title('Bayesian Regression Predicted Credible Intervals vs Actual')
plt.show()

#Bayesian LASSO regression - this can be done by using a Laplacian prior
#First, look at Laplacian dist - the prior will force dist near 0 and only variables significantly different from 0 will escape
#Thus, by using Lapacian prior, only the variables with strong coefficients will stand out
from scipy.stats import norm, laplace

def plot_laplace_vs_normal(norm_sd=1., b=1.):

    dstrb = pd.DataFrame(index=np.linspace(-10, 10, 1000))
    dstrb['normal'] = norm.pdf(dstrb.index.values, loc=0, scale=norm_sd)
    b0 = max(b*.5,0)
    b2 = min(b*2,10)
    dstrb['laplace b={}'.format(b0)] = laplace.pdf(dstrb.index.values, loc=0, scale=b0)
    dstrb['laplace b={}'.format(b)] = laplace.pdf(dstrb.index.values, loc=0, scale=b)    
    dstrb['laplace b={}'.format(b2)] = laplace.pdf(dstrb.index.values, loc=0, scale=b2)
    dstrb.plot(style=['--','-','-','-'], figsize=(12,4))
    plt.show()
    
plot_laplace_vs_normal()

def lasso_mcmc(df, formula, nbr_iters, likelihood=None, mcmc_sampler=None, target_accept=0.8, tune=500):
    '''
    This function is a wrapper for LASSO Bayesian regression, using MCMC.  The 
    default MCMC method is Metropolis, but others are available:
        Binary variables --> BinaryMetropolis
        Discrete variables --> Metropolis
        Continuous variables --> NUTS (No U-Turn Sampler)
    
    Parameters:
        df = pandas dataframe
        formula = regression formula as specified in R or Patsy
        nbr_iters = number of iterations to sample over in MCMC
        likelihood = the family of likelihood functions for the data (e.g. pm.glm.families.Normal())
        mcmc_sampler = the MCMC sampling method (e.g. pm.NUTS, pm.Metropolis), default NUTS
        target_accept = adapts the step size to make acceptance probability closer to target, higher target --> smaller steps, range(0,1)
        tune = number of tuning samples to use in each iteration, higher tune --> more probability space explored
    
    If acceptance probability is higher than the target, increase the tune 
    parameters.  For more info about this, see:
        https://discourse.pymc.io/t/warning-when-nuts-probability-is-greater-than-acceptance-level/594
    
    If there are divergences after tuning, increase the target_accept 
    parameter.  For more info about this, see:
        https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html
        
    A great intro to GLM Bayesian regression with PYMC3 can be found here:
        https://docs.pymc.io/notebooks/GLM-linear.html
    '''
    bayesian_model = pm.Model()  #Initializes a Bayesian model container
    with bayesian_model:
        
        priors = {"Intercept": pm.Laplace.dist(mu=0, b=0.1),
                  "Regressor": pm.Laplace.dist(mu=0, b=0.1)
                  }
        
        if likelihood is None:
            #Priors and observed sampling dist (likelihood) are automatically set
            pm.glm.GLM.from_formula(formula, df, priors=priors, 
                                    family=pm.glm.families.Normal())
        else:
            pm.glm.GLM.from_formula(formula, df, priors=priors,
                                    family=likelihood)
        if mcmc_sampler is None:
            mcmc_sampler = pm.Metropolis(target_accept=target_accept)
        else:
            mcmc_sampler = mcmc_sampler(target_accept=target_accept)
        #Perform inference to estimate the posterior, using nbr_iters MCMC sample posteriors
        trace = pm.sample(draws=nbr_iters, step=mcmc_sampler, 
                          init='auto', n_init=200000,
                          cores=None, tune=tune, progressbar=True,
                          random_seed=14)
    return bayesian_model, trace

lasso_reg, trace = lasso_mcmc(d, formula, nbr_iters=2000)
print(pm.summary(trace))
pm.traceplot(trace[1000:], lines={k: v['mean'] for k,v in pm.summary(trace[1000:]).iterrows()})
plt.show()
#So using Laplacian priors was a horrible idea

#Model PPC and RMSE for evaluation
ppc = pm.sample_ppc(trace[1000:], samples=500, model=lasso_reg)
rmse = np.sqrt(np.sum((ppc['y'].mean(0).mean(0).T - d['mpg'])**2) / d.shape[0])

print('R Squared:', get_r2(d, ppc, 'mpg'))  #R^2 shows the model does not fit the data at all

#Get the posterior predictions
traces = pm.trace_to_dataframe(trace)[['Intercept'] + list(d.drop(['mpg'], axis=1).columns)]
x = d.drop(['mpg'], axis=1)
x.insert(0, 'intercept', 1)
likelihoods = np.dot(x, traces.T)
likelihoods_sd = np.tile(pm.trace_to_dataframe(trace)[['sd']].T, (d.shape[0], 1))
likelihood = np.random.normal(likelihoods, likelihoods_sd)
print(likelihood.shape)  #Should have d.shape[0] rows and as many columns as MCMC samples

#Get credible intervals for the target variable
dfp = pd.DataFrame(np.percentile(likelihood,[2.5, 25, 50, 75, 97.5], axis=1).T,
                   columns=['0_25','25','50','75','97_5'])

#Plot target variable over index and the credible intervals from Bayesian regression
pal = sns.color_palette('Purples')
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.fill_between(d.index, dfp['0_25'], dfp['97_5'], alpha=0.7, color=pal[1], label='CR 95%')
ax.fill_between(d.index, dfp['25'], dfp['75'], alpha=0.5, color=pal[4], label='CR 50%')
ax.plot(d.index, dfp['50'], alpha=0.5, color=pal[5], label='Median')
plt.plot(d.index, d['mpg'], 'bo')
plt.legend()
plt.title('Bayesian Regression Predicted Credible Intervals vs Actual')
plt.show()

'''
-------------------------------------------------------------------------------
---------------------------------Naive Bayes-----------------------------------
-------------------------------------------------------------------------------
'''

from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import confusion_matrix

d = pd.read_csv('iris.csv')

#Train model with default hyperparameters
nb_model = GaussianNB()
nb_model.fit(d.drop(['Species'], axis=1), d['Species'])
preds = nb_model.predict(d.drop(['Species'], axis=1))
pred_probs = nb_model.predict_proba(d.drop(['Species'], axis=1))
print(confusion_matrix(d['Species'], preds))
