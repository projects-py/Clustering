#!/usr/bin/env python
# coding: utf-8

# # Clustering Algorithm
# #Hierarchical Clustering

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("Documents/customers data.csv")


# In[4]:


df.head()


# In[5]:


#We need to make clusters, but first we Normalize the data

from sklearn.preprocessing import normalize

data_scaled = pd.DataFrame(normalize(df), columns = df.columns)
data_scaled.head()


# In[6]:


#Forming clusters

import scipy.cluster.hierarchy as shc

plt.figure(figsize = (10,8))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method = 'ward'))


# In[8]:


#as verticle line with the maximum distance is the blue line and hence we can decide a threshold at 6 and cut the dendrogram

plt.figure(figsize = (10,8))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method = 'ward'))
plt.axhline(y=6, color = 'r', linestyle = '--')


# In[9]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(data_scaled)


# In[30]:


plt.figure(figsize = (10,8))
plt.scatter(data_scaled['Grocery'], data_scaled['Milk'], c= cluster.labels_)


# In[12]:


#So we have clearly 2 clusters


# In[16]:


# Using K-means clustering

from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df


# In[22]:


#finding the number of clusters we want using elbow method

# Arbitrarily selecting a range of values for K
K = range(1,10)
sum_of_squared_distances = []
# Using Scikit Learn’s KMeans Algorithm to find sum of squared distances
for k in K:
    model = KMeans(n_clusters=k).fit(data_scaled)
    sum_of_squared_distances.append(model.inertia_)
plt.plot(K, sum_of_squared_distances, "bx-")
plt.xlabel('K values')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method')
plt.show()


# In[23]:


#From here we see that 2 clusters must be made

data_kmeans = KMeans(n_clusters=2)
data_kmeans.fit(data_scaled)
labels = data_kmeans.predict(data_scaled)
print(labels)


# In[24]:


centroids = data_kmeans.cluster_centers_


# In[33]:


s = metrics.silhouette_score(data_scaled, labels, metric = 'euclidean')
print(f"Silhoutte Coefficient for the Dataset Clusters: {s: .2f}")
plt.figure(figsize = (10,8))

plt.scatter(data_scaled['Grocery'],data_scaled['Milk'], c = labels, cmap = 'rainbow')
plt.show()


# In[ ]:




