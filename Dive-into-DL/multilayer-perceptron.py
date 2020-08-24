#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_tensorflow
import softmax_implement as d2l
print(tf.__version__)

from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
batch_size = 256
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
x_train = x_train/255.0
x_test = x_test/255.0
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens),mean=0, stddev=0.01, dtype=tf.float32))
b1 = tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32))
W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs),mean=0, stddev=0.01, dtype=tf.float32))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=0.1))

def relu(x):
    return tf.math.maximum(x,0)

def net(X):
    X = tf.reshape(X, shape=[-1, num_inputs])
    h = relu(tf.matmul(X, W1) + b1)
    return tf.math.softmax(tf.matmul(h, W2) + b2)

def loss(y_hat,y_true):
    return tf.losses.sparse_categorical_crossentropy(y_true,y_hat)

num_epochs, lr = 5, 0.5
params = [W1, b1, W2, b2]
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

