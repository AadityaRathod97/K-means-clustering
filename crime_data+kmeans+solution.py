
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 

get_ipython().magic('matplotlib inline')


# In[2]:


crime = pd.read_csv('crime_data.csv')


# In[3]:


crime.head(10)


# In[4]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[5]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])


# In[6]:


df_norm


# In[7]:


###### screw plot or elbow curve ############
k = list(range(2,9))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[8]:


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[9]:


# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()


# In[10]:


crime = crime.iloc[:,1:]
crime.iloc[:,1:].groupby(crime.clust).mean()
crime.to_csv("crime_data.csv")


# In[11]:


crime.head(50)

