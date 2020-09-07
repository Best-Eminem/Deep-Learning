#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
print(tf.__version__)


# In[2]:


def corr2d(X, K):
    h, w = K.shape
    if len(X.shape) <= 1:
        X = tf.reshape(X, (X.shape[0],1))
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j].assign(tf.cast(tf.reduce_sum(X[i:i+h, j:j+w] * K), dtype=tf.float32))
    return Y


# In[12]:


def corr2d_multi_in(X, K):
    return tf.reduce_sum([corr2d(X[i], K[i]) for i in range(X.shape[0])],axis=0)

X = tf.constant([[[0,1,2],[3,4,5],[6,7,8]],
                 [[1,2,3],[4,5,6],[7,8,9]]])
K = tf.constant([[[0,1],[2,3]],
                 [[1,2],[3,4]]])

corr2d_multi_in(X, K)


# In[17]:


def corr2d_multi_in_out(X, K):
    return tf.stack([corr2d_multi_in(X, k) for k in K],axis=0)


# In[13]:


K = tf.stack([K, K+1, K+2],axis=0)
K.shape


# In[18]:


corr2d_multi_in_out(X, K)


# In[28]:


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = tf.reshape(X,(c_i, h * w))# 3*9
    K = tf.reshape(K,(c_o, c_i)) #2*3
    Y = tf.matmul(K, X)
    return tf.reshape(Y, (c_o, h, w))

X = tf.random.uniform((3,3,3))
K = tf.random.uniform((2,3,1,1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print(Y1,Y2)
tf.norm(Y1-Y2) < 1e-6

