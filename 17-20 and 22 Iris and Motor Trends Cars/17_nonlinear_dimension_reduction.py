#Python code for chapter 17 DSILT: Statistics

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
import matplotlib.pyplot as plt

#Set seed for repeatability
seed = 14
np.random.seed(seed)

#Load Iris data and specify dependent variable
alldata = pd.read_csv('C:/Users/Nick/Documents/Word Documents/Data Science Books/DSILT Stats Code/17-20 and 22 Iris and Motor Trends Cars/iris.csv')
print(alldata.info())
print(alldata.head())

#Split data into x and y, and format as numpy arrays
array = alldata.values
x = array[:, 0:4].astype(float)
y = array[:, 4]
#Convert the classes (species) into integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

#Scale every feature using min-max normalization and z-score standardization
#Note that y is not scaled because it is categorical, but if it weren't, it would have to be scaled too
x_norm = MinMaxScaler().fit_transform(x)
x_std = StandardScaler().fit_transform(x)


'''
-------------------------------------------------------------------------------
-------------------------Multi-Dimensional Scaling-----------------------------
-------------------------------------------------------------------------------
'''

#Apply metric MDS, keeping n components < the number of original features
#kernel choices: linear (default), poly (polynomial of degree=degree), rbf, sigmoid, cosine
mds_model = MDS(n_components=2, metric=True, random_state=seed)
mds_model.fit_transform(x_std)
print(mds_model.get_params())
mds_dim = mds_model.embedding_
print(mds_dim.shape)  #There should be 2 latent variables represented
print('Stress:', mds_model.stress_)

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.title('Metric MDS 2-Dimension Plot with Observation Class')
plt.scatter(mds_dim[:, 0], mds_dim[:, 1], c=y)
plt.colorbar()
plt.show()

#Apply non-metric MDS, keeping n components < the number of original features
#kernel choices: linear (default), poly (polynomial of degree=degree), rbf, sigmoid, cosine
mds_model = MDS(n_components=2, metric=False, random_state=seed)
mds_model.fit_transform(x_std)
print(mds_model.get_params())
mds_dim = mds_model.embedding_
print(mds_dim.shape)  #There should be 2 latent variables represented
print('Stress:', mds_model.stress_)

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.title('Nonmetric MDS 2-Dimension Plot with Observation Class')
plt.scatter(mds_dim[:, 0], mds_dim[:, 1], c=y)
plt.colorbar()
plt.show()

#Apply metric MDS for many different choices of dimensions
#Limitation is that nbr dimensions must be < the number of original features
nbr_dim = range(3)
mds_dim_nbr = []
mds_stress_results = []
for nd in nbr_dim:
    mds_model = MDS(n_components=nd+1, metric=True, random_state=seed)
    mds_model.fit_transform(x_std)
    mds_dim_nbr.append(nd+1)
    mds_stress_results.append(mds_model.stress_)
mds_results = {'nbr_dim': mds_dim_nbr, 'stress': mds_stress_results}
mds_results = pd.DataFrame.from_dict(mds_results)
#See which number of dimensions has the lowest stress
plt.plot(mds_results['nbr_dim'], mds_results['stress'])
plt.xlabel('Number of Latent Dimensions')
plt.ylabel('Stress')
plt.title('Scree Plot of Stress by Number of Latent Variables')
plt.show()

#Build MDS model with one less than the number of dimensions at the kneee of the scree plot
#In this case, that would be 1, so the index is substituted for the second latent variable
mds_model = MDS(n_components=1, metric=True, random_state=seed)
mds_model.fit_transform(x_std)
mds_dim = mds_model.embedding_
plt.figure(figsize=(10, 5))
plt.xlabel('Index')
plt.ylabel('Latent Variable 1')
plt.title('Metric MDS 1-Dimension Plot with Observation Class')
plt.scatter(mds_dim[:, 0], list(alldata.index), c=y)
plt.colorbar()
plt.show()

'''
-------------------------------------------------------------------------------
-----------------------------------Isomap--------------------------------------
-------------------------------------------------------------------------------
'''

#Apply isomap embedding, keeping n components < the number of original features
iso_model = Isomap(n_neighbors=5, n_components=2)
iso_model.fit_transform(x_std)
print(iso_model.get_params())
iso_dim = iso_model.embedding_
print(iso_dim.shape)  #There should be 2 latent variables represented

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1 (explains most variance)')
plt.ylabel('Latent Variable 2 (explains second most variance)')
plt.title('Isomap 2-Dimension Plot with Observation Class')
plt.scatter(iso_dim[:, 0], iso_dim[:, 1], c=y)
plt.colorbar()
plt.show()

#Apply isomap for many different choices of dimensions
#Limitation is that nbr dimensions must be < the number of original features
nbr_dim = range(3)
iso_dim_nbr = []
iso_reconstruction_errors = []
for nd in nbr_dim:
    iso_model = Isomap(n_neighbors=5, n_components=nd+1)
    iso_model.fit_transform(x_std)
    iso_dim_nbr.append(nd+1)
    iso_reconstruction_errors.append(iso_model.reconstruction_error())
iso_results = {'nbr_dim': iso_dim_nbr, 'error': iso_reconstruction_errors}
iso_results = pd.DataFrame.from_dict(iso_results)
#See which number of dimensions has the lowest reconstruction error
plt.plot(iso_results['nbr_dim'], iso_results['error'])
plt.xlabel('Number of Latent Dimensions')
plt.ylabel('Reconstruction Error')
plt.title('Plot of Error by Number of Latent Variables')
plt.show()

#Use iso_model.transform(x_test) to fit the isomap from the training set onto the test set

'''
-------------------------------------------------------------------------------
-------------------------------Modified LLE------------------------------------
-------------------------------------------------------------------------------
'''

#Apply modified LLE, keeping n components < the number of original features
#method = 'standard' for LLE, 'hessian' for HELLE, or 'modified' for modified LLE
mlle_model = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='modified', random_state=seed)
mlle_model.fit_transform(x_std)
print(mlle_model.get_params())
mlle_dim = mlle_model.embedding_
print(mlle_dim.shape)  #There should be 2 latent variables represented

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1 (explains most variance)')
plt.ylabel('Latent Variable 2 (explains second most variance)')
plt.title('Modified LLE 2-Dimension Plot with Observation Class, 5 neighbors')
plt.scatter(mlle_dim[:, 0], mlle_dim[:, 1], c=y)
plt.colorbar()
plt.show()

#Try a different number of neighbors
mlle_model = LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='modified', random_state=seed)
mlle_model.fit_transform(x_std)
mlle_dim = mlle_model.embedding_
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1 (explains most variance)')
plt.ylabel('Latent Variable 2 (explains second most variance)')
plt.title('Modified LLE 2-Dimension Plot with Observation Class, 15 neighbors')
plt.scatter(mlle_dim[:, 0], mlle_dim[:, 1], c=y)
plt.colorbar()
plt.show()

#Use mlle_model.transform(x_test) to fit the modified LLE from the training set onto the test set

'''
-------------------------------------------------------------------------------
-------------------------------------t-SNE-------------------------------------
-------------------------------------------------------------------------------
'''

#Build TSNE model, learning rate defaults to 1000 but usually best around 200
#Perplexity balances local and global aspects of neighbors, usually best between 5 and 50
tsne_model = TSNE(n_components=2, perplexity=30.0, learning_rate=100.0, 
                  n_iter=2000, n_iter_without_progress=30, 
                  random_state=seed, method='barnes_hut')
tsne_model.fit_transform(x_std)
print(tsne_model.get_params())
tsne_dim = tsne_model.embedding_
print(tsne_dim.shape)  #There should be 2 latent variables represented
#print('Kullback-Leibler divergence:', tsne_model.kl_divergence_)

#Plot first 2 extracted features and the observation class
plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.title('t-SNE 2-Dimension Plot with Observation Class \nperplexity=30, learning_rate=100')
plt.scatter(tsne_dim[:, 0], tsne_dim[:, 1], c=y)
plt.colorbar()
plt.show()
