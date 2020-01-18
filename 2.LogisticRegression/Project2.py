#####################################
# Project 2 : Logistic Regression
# Name : Shweta Kharat
#####################################

# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
import math

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

# read dataset
df = pd.read_csv("shuffled_2class_iris_dataset.csv")
print(df.head())

df = df.values
X = df[:,0:4]   # 100x4
y = df[:,4]     # 100x1

# Zero out the mean
diff = X.max(0) - X.min(0)
X = X - diff

# Create 90/10 trainig/test sets
tr_data = X[0:90,0:4]; tr_label = y[0:90]
test_data = X[90:100,0:4]; test_label = y[90:100]
np.random.seed(1)


# *************** Training **************
# Initialization
nu = 0.01
MAX_ITR = 1500
J = []   # J is the cost function
J = np.ones(MAX_ITR)
J = J.reshape(MAX_ITR,1)
N = len(tr_data)
X0 = np.ones((tr_data.shape[0], 1))
tr_data = np.hstack((X0, tr_data))
w = np.zeros(tr_data.shape[1])

# In[2]:


# Loop for 1500 iterations
for itr in range (0, MAX_ITR):
    #Apply sigmoid
    scores = np.dot(tr_data, w)
    predictions = sigmoid(scores)

    # Update weights with gradient
    output_error_signal = tr_label - predictions
    gradient = np.dot(tr_data.T, output_error_signal)
    w += nu/N * gradient
    
    # cost function
    scores1 = np.dot(tr_data, w)
    pred = sigmoid(scores1)
    cost = np.sum( tr_label*scores1 - np.log(1 + np.exp(scores1)) )
    J[itr,0] = -cost
    
# Plot cost function vs iterations
#print (np.shape(J))   # 1500x1
plt.plot(J)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()


# In[3]:


# **************** Testing *****************
correct_class_cnt = 0
N1 = len(test_data)
X0 = np.ones((test_data.shape[0], 1))
test_data = np.hstack((X0, test_data))
for i in range (0, len(test_data)):
    
    # sigmoid function
    final_scores = np.dot(test_data, w)
    preds = np.round(sigmoid(final_scores))
    
    
    #predicted label based on threshold = 0.5
    if preds[i] > 0.5:
        y_hat = 1
    else:
        y_hat = 0
       
    # accuracy
    print('y={}, y_hat={}'.format(test_label[i], y_hat))
    if (y_hat == test_label[i]):
        correct_class_cnt += 1

print('Average accuracy = {0:2f}'.format(correct_class_cnt/10))

