#!/usr/bin/env python
# coding: utf-8

# Double_layer_neural_network
# 
# content :
# 1) Importing library
# 
# 2) Unziping file and making it in a readable format
# 
# 3) Visualising data
# 
# 4) Implmenting double layer neural network
# 
# 5) Ploting error v/s iteration curve
# 
# 6) Calculating accuracy on training and testing data
#     training-data= 93.33905586240776
#     testing_data(activation=sigmoid)= 93.57689362930182
#     testing_data(activation=softmax)= 78.5
# 
#     

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gzip
import struct


# In[2]:


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte' %kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte' %kind)
    
    with open(labels_path,'rb') as lbpath:
        magic, n = struct.unpack(">II",lbpath.read(8))
        labels=np.fromfile(lbpath,dtype=np.uint8)
    
    with open(images_path,'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images=np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
    return images, labels


# In[3]:


X_train, Y_train = load_mnist('/home/sachin/Desktop/assignment_4/', kind='train')
X_train.shape
Y_train.shape


# In[4]:


X_test, Y_test = load_mnist('/home/sachin/Desktop/assignment_4/', kind='t10k')
m_train = X_train.shape[0]
m_test = X_test.shape[0]
#print(m_train)
print(m_test)


# In[5]:


print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("y_test shape: " + str(Y_test.shape))
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))


# In[6]:


np.random.seed(0);
indices = list(np.random.randint(m_train,size=9))
print(indices)
for i in range(len(indices)):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[indices[i]].reshape(28,28), cmap='gray',interpolation='none' )
    plt.title("Index {} Class {}".format(indices[i], Y_train[indices[i]]))
    plt.tight_layout()


# In[7]:


m_train = 59000
m_validation = 1000

mask = list(range(m_train, m_train + m_validation))
X_val = X_train[mask]
y_val = Y_train[mask]

mask = list(range(m_train))
X_train = X_train[mask]
y_train = Y_train[mask]

mask = list(range(m_test))
X_test = X_test[mask]
y_test = Y_test[mask]


# In[8]:


X_val.reshape(1000,-1)


# In[9]:


print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))
print("X_val shape: " + str(X_val.shape))
print("y_val shape: " + str(y_val.shape))
print("X_test shape: " + str(X_test.shape))
print("y_test shape: " + str(y_test.shape))
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of validation examples: m_validation = " + str(m_validation))
print ("Number of testing examples: m_test = " + str(m_test))


# In[10]:


input_layer_size= 784
hiden_layer_size= 50
output_layer_size=10


# In[11]:


def softmax(inputs):
    
    return np.exp(inputs) / (np.sum(np.exp(inputs),axis=0))

def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def cost_function(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L

 


# In[12]:



w1= np.random.randn(hiden_layer_size,input_layer_size+1)
w1.shape
w2=np.random.randn(output_layer_size,hiden_layer_size+1)
x_new = np.concatenate((np.ones((len(X_train),1)),X_train),axis=1)
x_new.shape

#print(len(z3))
delta_output=np.zeros((10,51))
delta_hidden=np.zeros((51,785))


# # one_hot_encoding

# In[16]:


x_one_hot = np.eye(10)[y_train]
x_one_hot


# In[18]:


iters=100
alpha=0.1
errors=[]
cost_array=[]
y_hot   =np.eye(10)[y_test]


for i in range (iters):
    # input layer with concatenating one(for bias term)
    a1 = x_new #(59000,785)
    z2 = x_new.dot(w1.T)#(59000,50)
    # hidden layer(values)
    a2 = sigmoid(z2)

    # concatenating one for biasing
    a2_new = np.concatenate((np.ones((len(a2),1)),a2),axis=1)#(59000,51)

    # outpout layer
    z3  = a2_new.dot(w2.T)#(59000,10)
    a3  = sigmoid(z3)#(59000,10)
    
    cost=cost_function(x_one_hot,a3)
    cost_array.append(cost)
    
        
    # error in output layer    
    error_output = a3 -x_one_hot  #(59000,10)

    # error in hidden layer
    err = error_output*sigmoid_derivative(a3) 
    error_layer_1 = np.dot(err,w2)#(59000,51)

    # delta value for output layer
    delta_output += (error_output.T).dot(a2_new)#(10,51)
    #print(delta_output.shape)

    # delta value for hinden layer
    delta_hidden += (error_layer_1.T).dot(a1)#(51,785)
    del_hidden = delta_hidden[1:,:]#(51,785)

    # gradient descendent for input and output value
    dw2 = delta_output/a1.shape[0]
    dw1 = del_hidden/a1.shape[0]

    # updating weight
    w2  = w2-alpha*dw2#(10,51)
    #print(w2.shape)
    w1  = w1-alpha*dw1#(50,785)
    
    error = np.mean(np.abs(error_output))
    errors.append(error)
    accuracy = (1-error)*100
    
    
    
print(a3)   


# In[19]:



print(accuracy)


# # cost v/s iteration curve

# In[23]:


plt.plot(np.arange(100),cost_array)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.title("cost v/s iteration curve")


# # error v/s iteration curve 

# In[24]:


plt.plot(np.arange(100),errors)
plt.xlabel("iteration")
plt.ylabel("errors")
plt.title("errors v/s iterations")


# # Predicting on test data

# In[25]:




x_test = np.concatenate((np.ones((len(X_test),1)),X_test),axis=1)
z2_test = x_test.dot(w1.T) 
a2_test = sigmoid(z2_test)

a2_new = np.concatenate((np.ones((len(a2_test),1)),a2_test),axis=1)
a2_new.shape
z3_test = a2_new.dot(w2.T)

a3_test = sigmoid(z3_test)
#test=softmax(z3_test)
print(a3_test)

a3_test.shape
    
print(a3_test.shape)

    


# In[527]:


y_hot   =np.eye(10)[y_test]
error_test = np.mean(np.abs(y_hot-a3_test))
accuracy = (1-error_test)*100
print(accuracy)


# In[29]:


def softmax(inputs):
    
    return np.exp(inputs) / (np.sum(np.exp(inputs),axis=0))
lst1=[]
lst2=[]
for i in range(len(a3_test)):
    prob = softmax(z3_test[i])

    lst1.append(np.sum(prob))
    lst2.append(np.argmax(prob))


# In[30]:


lst2


# In[31]:


sum=0

for i in range(10000):
    if lst2[i]==y_test[i]:
        sum=sum+1
print((sum/10000)*100)        


# In[ ]:


for i in range (iters):

