###########################################
# Project 3 : DISCRIMINANT ANALYSIS
# Name : Shweta Kharat
###########################################
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load training data - 135 observations, 4 features, 3 classes,
df = pd.read_csv("iris_corrupted_training_data.csv")
print(df.head())
df = df.values
tr_data = df
# Load validation data - 15 samples
df = pd.read_csv("iris_validation_data.csv")
print(df.head())
df = df.values
val_data = df


# In[2]:


# Covariance MLE function
def covmle(data, mean):

    #data is a matrix size Nx4
    #mean is a vector size N x 1

    X = np.array(data[:,0:4])
    m = np.array(mean)
    XM = X - m
    #print(np.shape(mean))
    return np.dot(XM.transpose(), XM) / len(X)


# In[3]:


# Compute various components of the disriminant function
N = len(tr_data)

X1 = np.ones((tr_data.shape[0], 5))
X2 = np.ones((tr_data.shape[0], 5))
X3 = np.ones((tr_data.shape[0], 5))
# Create data matrices for each class
x1_count = 0
x2_count = 0
x3_count = 0

for i in range (0, len(tr_data)) :
    if tr_data[i][4] == 1:
        X1[x1_count,:] = tr_data[i,:]
        x1_count = x1_count + 1
    elif tr_data[i][4] == 2:
        X2[x2_count,:] = tr_data[i,:]
        x2_count = x2_count + 1
    elif tr_data[i][4] == 3:
        X3[x3_count,:] = tr_data[i,:]
        x3_count = x3_count + 1    

NSubclass = N/3       #Count of sub class

# Calculate mean
X11 = np.array(X1[0:45,0:4])
meanX1 = (np.sum(X11,axis =0))/NSubclass
print(meanX1, "Mean for Class 1")
X22 = np.array(X2[0:45,0:4])
meanX2 = (np.sum(X22,axis =0))/NSubclass
print(meanX2, "Mean for Class 2")
X33 = np.array(X3[0:45,0:4])
meanX3 = (np.sum(X33,axis =0))/NSubclass
print(meanX3, "Mean for Class 3")

# Call covariance function
covX1 = covmle(X11, meanX1)
covX2 = covmle(X22, meanX2)
covX3 = covmle(X33, meanX3)

# X - mu
xm1 = X11 - meanX1
xm2 = X22 - meanX2
xm3 = X33 - meanX3

# X'
xmT1 = xm1.transpose()
xmT2 = xm2.transpose()
xmT3 = xm3.transpose()

# Covariance inverse
covInv1 = np.linalg.inv(covX1)
covInv2 = np.linalg.inv(covX2)
covInv3 = np.linalg.inv(covX3)

# Covariance determinants
covDet1 = np.linalg.det(covX1)
covDet2 = np.linalg.det(covX2)
covDet3 = np.linalg.det(covX2)


# In[4]:


correct_class = 0; # number of correctly predicted label
g1 = []
g2 = []
g3 = []
# Compute the discriminant function (g1, g2, g3) with prior = 1/3
for i in range(0, len(val_data)):
    testg1 = ((-0.5) * np.matmul((np.matmul((val_data[i,0:4]-meanX1), covInv1)),(val_data[i,0:4]-meanX1))) 
    - ((0.5) * np.log(covDet1)) + (1/3)
    g1.append(testg1)
    testg2 = ((-0.5) * np.matmul((np.matmul((val_data[i,0:4]-meanX2), covInv2)),(val_data[i,0:4]-meanX2))) 
    - ((0.5) * np.log(covDet2)) + (1/3)
    g2.append(testg2)
    testg3 = ((-0.5) * np.matmul((np.matmul((val_data[i,0:4]-meanX3), covInv3)),(val_data[i,0:4]-meanX3))) 
    - ((0.5) * np.log(covDet3)) + (1/3)
    g3.append(testg3)

# Count the number of correctly predicted labels
for i in range(0, len(val_data)):
    if g1[i]>g2[i] and g1[i]>g3[i]:
        if val_data[i,4] == 1:
            correct_class = correct_class+1
    if g2[i]>g1[i] and g2[i]>g3[i]:
        if val_data[i,4] == 2:
            correct_class = correct_class+1
    if g3[i]>g1[i] and g3[i]>g2[i]:
        if val_data[i,4] == 3:
            correct_class = correct_class+1
            
print('Classification accuracy = ', '{0:.4f}'. format(correct_class/15))


