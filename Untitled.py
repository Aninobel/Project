#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import h5py
import numpy as np
import input_output as io
import seaborn as sns
import codes.analysis.rnn as rnn
import codes.util as util
import codes.analysis.machine_learning as ml
import codes.analysis as an
import codes.processing as p
import codes.viz as viz
import codes.util.input_output as io_transfer
plt.style.use('seaborn-white')
viz.format.custom_plt_format
from IPython.display import IFrame, HTML
from graphviz import Graph


# In[2]:


download_data = True

url = 'https://zenodo.org/record/3407773/files/Data_Zip.zip?download=1'
filename = 'Data_Zip.zip'
save_path = './'

io.download_and_unzip(filename, url, save_path, download_data)


# In[6]:


data = h5py.File('20_80_PZT/20_80_SHO fits.mat', 'r')
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print(key)
data.visititems(print_attrs)
data['Data']['outoffield_2']['A'].shape


# In[18]:


data = h5py.File('20_80_PZT/20_80_loop_fits.mat', 'r')
voltage = data.get(data['alldata2'][0][8]).value
loop_data_2080 = data.get(data['unfolddata'][1][0]).value
plt.plot(voltage[0],loop_data_2080.squeeze()[:,0,12])
loop_data_2080 = np.reshape(loop_data_2080,(2500,64),order='C')
print(loop_data_2080)
loop_data_2080 = np.rollaxis(loop_data_2080,1,0)
#  Removing all NaN values in the data
loop_data_2080 = np.nan_to_num(loop_data_2080)


# In[19]:


#  Standardizing the dataset by StandardScaler Method
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
loop_data_2080_std = scaler.fit_transform(loop_data_2080)
#  Applying PCA to evaluate Eigenvalues and eigenvectors
mean_vec = np.mean(loop_data_2080_std,axis=0)
covariance_matrix = (loop_data_2080_std - mean_vec).T.dot((loop_data_2080_std - mean_vec)) / (loop_data_2080_std.shape[0] - 1)
print("Covariance matrix is : " , covariance_matrix)
#  Performing Eigendecomposition on the Covariance matrix
eigenvalues,eigenvectors = np.linalg.eig(covariance_matrix)
print("Eigenvalues are :", eigenvalues)
print("Eigenvectors are :", eigenvectors)
#  Singular Value Decomposition applied on Data
u,s,v = np.linalg.svd(loop_data_2080_std.T)
print(u)
#  Selecting Principle Components
for e in eigenvectors.T:
    np.testing.assert_array_almost_equal(1.0,np.linalg.norm(e))
#  Making lists of Eigenvalue-Eigenvector tuples
pairs = [(np.abs(eigenvalues[i]),eigenvectors[:,i]) for i in range(len(eigenvalues))]
pairs.sort(key=lambda a : a[0], reverse=True)
print("Eigenvalues in descending order :")
for w in pairs:
    print(w[0])
#  Explained Variance
total = sum(eigenvalues)
explained_variance = [(i/total)*100 for i in sorted(eigenvalues,reverse=True)]
cumulative_explained_variance = np.cumsum(explained_variance)
#  Projection Matrix
projection_matrix = np.hstack((pairs[0][1].reshape(2500),pairs[1][1].reshape(2500)))
print("Projection matrix : ", projection_matrix)


# In[20]:


#Applying Pairwise Distance Calculation on the Data
from sklearn.metrics import pairwise_distances
D = pairwise_distances(loop_data_2080)
D.shape
plt.imshow(D,zorder=2,cmap='viridis',interpolation='nearest')
plt.colorbar()


# In[21]:


from sklearn.manifold import MDS
model = MDS(n_components=2,dissimilarity='precomputed',random_state=1)
out = model.fit_transform(D)
fig = plt.figure(figsize=(12,12))
plt.scatter(out[:,0],out[:,1])
plt.axis('equal')


# In[22]:


from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=63,n_components=2,method='modified',eigen_solver='dense')
out = model.fit_transform(loop_data_2080)
fig,ax = plt.subplots(figsize=(12,12))
ax.scatter(out[:,0],out[:,1])
ax.set_ylim(-0.15,0.15)


# In[30]:


# Applying NMF on the data
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
nmf = NMF(n_components=20,solver='mu',init='random',random_state=0)
X = np.array(loop_data_2080)
for i in range(0,64):
    for j in range(0,2500):
        if X[i][j] < 0:
           loop_data_2080[i][j] = 0
for i in range(0,64):
    for j in range(0,2500):
        if loop_data_2080[i][j] < 0:
           print(loop_data_2080[i][j])
W = nmf.fit_transform(loop_data_2080)
H = nmf.components_
print(W)
print(H)


# In[26]:


# Applying K-Means Clustering on the Data
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
Y = kmeans.fit_predict(loop_data_2080)
kmeans.cluster_centers_
kmeans.labels_
plt.scatter(loop_data_2080[:,2000],loop_data_2080[:,2000],c= Y)
sse = []
list_k = list(range(1,10))
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(loop_data_2080)
    sse.append(km.inertia_)
plt.figure(figsize=(6,6))   
plt.plot(list_k,sse)
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared distance")


# In[31]:


from sklearn.preprocessing import scale
A = scale(loop_data_2080)
cov = PCA(n_components=30)
cov.fit(A)
variance = cov.explained_variance_ratio_
var = np.cumsum(np.round(cov.explained_variance_ratio_,decimals=3)*100)
plt.plot(var)
plt.xlabel('Number of Features')
plt.ylabel('% Variance Explained')
plt.title('PCA Analysis')


# In[32]:


ls


# In[ ]:


# DEEP LEARNING MODELS APPLIED ON THE DATA
from fastai.vision import *
from fastai.metrics import error_rate


# In[49]:


arch = models.resnet50


# In[50]:


import numpy as np


# In[ ]:





# In[ ]:




