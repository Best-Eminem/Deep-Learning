#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
print(tf.__version__)
from matplotlib import pyplot as plt
import random


# In[5]:


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal((num_examples, num_inputs),stddev = 1)
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += tf.random.normal(labels.shape,stddev=0.01)


# In[ ]:





# In[6]:


print(features[0], labels[0])


# In[ ]:





# In[7]:


def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1], labels, 1)


# In[ ]:





# In[1]:


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i: min(i+batch_size, num_examples)]
        yield tf.gather(features, axis=0, indices=j), tf.gather(labels, axis=0, indices=j)


# In[10]:


batch_size=10
for X,y in data_iter(batch_size,features,labels):
    print(X,y)
    break


# In[11]:


w = tf.Variable(tf.random.normal((num_inputs,1),stddev=0.01))
b = tf.Variable(tf.zeros((1,)))


# In[12]:


def linreg(X, w, b):
    return tf.matmul(X, w) + b


# In[17]:


def squared_loss(y_hat,y):
    return (y_hat - tf.reshape(y,y_hat.shape))** 2 /2


# In[ ]:





# In[14]:


def sgd(params,lr,batch_size,grads):
    """Mini-batch stochastic gradient descent."""
    for i, param in enumerate(params):
        """enumerate用于获取index和value"""
        param.assign_sub(lr * grads[i] / batch_size)


# In[18]:


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as t:
            t.watch([w,b])
            l = tf.reduce_sum(loss(net(X, w, b), y))
        grads = t.gradient(l, [w, b])
        sgd([w, b], lr, batch_size, grads)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))


# In[1]:


print(true_w, w)
print(true_b, b)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




