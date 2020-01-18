#********************************************
#*** Project 4: K Means Clustering
#*** Shweta Kharat
#***********************************
# coding: utf-8

# In[1]:


# Load dataset and plot it

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt

df = pd.read_csv('simple_iris_dataset.csv')

df1 = df['sepal_length'].values
df2 = df['sepal_width'].values
X = np.array(list(zip(df1, df2)))   # 100 x 2
plt.scatter(df1, df2, c='black', s=7)
print(X.shape)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# In[2]:


# Initialize - take 2 random samples from data set
k=2
N = len(X)
ctr1 = X[(np.random.randint(0,N)),:]
ctr2 = X[(np.random.randint(0,N)),:]
C = np.array(list(zip(ctr1, ctr2)), dtype=np.float32)  # 2x2
print(C.shape)

# Stores the value of old centroids
C_old = np.zeros(C.shape)

# Cluster indices
clusters = np.zeros(len(X))

# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)

MAX_ITR = 100
# Loop till convergence, i.e., centroids do not change anymore
# or until raching MAX_ITR

for itr in range(0,MAX_ITR):
    #Assign each data point to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    
    # Store old centroid values
    # Example: C_old = deepcopy(C)
    C_old = deepcopy(C)

    # Update the location of centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    
    # Algorithm converges if distance between new and old centroids is 0
    error = dist(C, C_old, None)
    if error == 0:
        break

print('Algorithm converges after {} iterations'. format(itr))


# In[3]:


# Plot the results
def plot_clusters(X, clusters, centroids):
    """
      X : 100x2 data matrix
      clusters: 100x1 vector indicating the cluster assignment of each data point
      centroids: 2x2 matrix, the row contains the coordinate of the centroid 
    """
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    k,_ = np.shape(C)

    fig, ax = plt.subplots()
    for i in range(k):
        print(colors[i])
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        #ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

    plt.scatter(C[:,0], C[:,1], marker='*', s=150, c='y')
    
plot_clusters(X, clusters, C)

