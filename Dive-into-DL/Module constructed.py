#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
print(tf.__version__)


# In[7]:


net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(256))
net.add(tf.keras.layers.Dense(10))

X = tf.random.uniform((2,20))
Y = net(X)
Y


# In[8]:


net.weights[0], type(net.weights[0])


# In[21]:


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(
            units=10,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01),
            bias_initializer=tf.zeros_initializer()
        )
        self.d2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.ones_initializer(),
            bias_initializer=tf.ones_initializer()
        )

    def call(self, input):
        output = self.d1(input)
        output = self.d2(output)
        return output

net = Linear()
net(X)
net.get_weights()


# In[28]:


def my_init():
    return tf.keras.initializers.Ones()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, kernel_initializer=my_init(),
                               bias_initializer=tf.zeros_initializer()))

Y = model(X)
model.weights[1]


# In[30]:


X = tf.random.uniform((2,20))
x


# In[31]:


X


# In[32]:


class CenteredLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)


# In[33]:


layer = CenteredLayer()
layer(np.array([1,2,3,4,5]))


# In[34]:


net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(20))
net.add(CenteredLayer())

Y = net(X)
Y


# In[35]:


tf.reduce_mean(Y)


# In[43]:


class myDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_weight(name='w',
            shape=[input_shape[-1], self.units], initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b',
            shape=[self.units], initializer=tf.zeros_initializer())
        activation=tf.nn.relu

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


# In[37]:


dense = myDense(3)
dense(X)
dense.get_weights()


# In[44]:


net = tf.keras.models.Sequential()
net.add(myDense(8))
net.add(myDense(1))
net(X)


# In[ ]:




