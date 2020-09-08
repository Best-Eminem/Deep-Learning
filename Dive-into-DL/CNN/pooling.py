#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
print(tf.__version__)


# In[2]:


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1))
    Y = tf.Variable(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j].assign(tf.reduce_max(X[i:i+p_h, j:j+p_w]))
            elif mode =='avg':
                Y[i,j].assign(tf.reduce_mean(X[i:i+p_h, j:j+p_w]))
    return Y


# In[3]:


X = tf.constant([[0,1,2],[3,4,5],[6,7,8]],dtype=tf.float32)
pool2d(X, (2,2))


# In[4]:


pool2d(X, (2,2), 'avg')


# In[20]:


#tensorflow default data_format == 'channels_last'
#so (1,4,4,1) instead of (1,1,4,4)
X = tf.reshape(tf.constant(range(16)), (1,4,4,1))
X


# In[ ]:





# In[6]:


pool2d = tf.keras.layers.MaxPool2D(pool_size=[3,3])
pool2d(X)


# In[19]:


#I guess no custom padding settings in keras.layers?
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3,3],padding='valid',strides=2)
pool2d(X)


# In[ ]:





# In[17]:


X = tf.stack([X, X+1], axis=3)
X = tf.reshape(X, (2,4,4,1))
X.shape


# In[18]:


pool2d = tf.keras.layers.MaxPool2D(3, padding='same', strides=2)
pool2d(X)

