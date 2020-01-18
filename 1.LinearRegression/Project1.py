
###############################################################
# Project 1 : Linear Regression with Batch Gradient Descent
# Name : Shweta Kharat
###############################################################

# coding: utf-8

# In[2]:


# Linear Regression with Batch Gradient Descent
import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv("housing_price_data.csv")

df.head()
# Data points
X, y = (df["Size"].values,df["Price"].values)
X = X.reshape(len(X),1)
y = y.reshape(len(y),1)

# Plot the data points
plt.plot(X, y, 'ro')
plt.xlabel("# Square footage")
plt.ylabel("Price")
plt.show()

# Normalize the house size, e.g., divide by (mx-min) to ensure the scale is compatible with the bias, which is 1
normalize = X.max(0) - X.min(0)
X =  X / normalize

# Add bias
z=np.ones((len(X),1)) 
X = X.reshape(len(X),1)
X = np.concatenate((z,X), axis=1)                
y = y.reshape(len(X),1) 

# sort for testing/plotting
Xsort = np.sort(X, axis=0)

print(X.shape)
print(y.shape)


# Perform gradient descent
# Initialize vector w 
w = np.random.rand(2,1)

# Learning rate
nu = 0.1

# Number of iterations
MAX_ITR = 2000

mse = {}                      

# Loop for 2000 iterations
for itr in range (0, MAX_ITR):
    
    m = np.matmul(X,w)      #781x1
    h = np.subtract(m,y)    #781x1
    H = h.transpose()      #1x781
    # Updating the parameters
    
    w[0] = w[0] - nu * (1/MAX_ITR) * np.matmul(H , X[:, 0]);
    w[1] = w[1] - nu * (1/MAX_ITR) * np.matmul(H , X[:, 1]);
    m1 = np.matmul(H , h)
    m2 = m1 / (2 * MAX_ITR) 
    mse = np.vstack([mse,m2])
    mse[itr] = m1 / (2 * MAX_ITR) 
    
# Plot MSE vs iterations
print (m1.shape)
print (m2.shape)
print (mse.shape)
plt.plot(mse)  #
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()


# Plot the fitted curve

yhat = np.dot(np.sort(X, axis=0),w)

plt.plot(X[:,1] * normalize, y, 'ro')
plt.plot(Xsort[:,1] *normalize , yhat, 'b', label="Gradient descent")
plt.legend()
plt.xlabel("# Square footage")
plt.ylabel("Price")
plt.show()

