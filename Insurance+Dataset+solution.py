
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
get_ipython().magic('matplotlib inline')


# In[2]:


insur = pd.read_csv('insurance.csv')


# In[3]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[4]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(insur.iloc[:,:])


# In[5]:


df_norm


# In[6]:


###### screw plot or elbow curve ############
k = list(range(2,7))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[7]:


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[8]:


# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
insur['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()


# In[9]:


insur.head(100)

