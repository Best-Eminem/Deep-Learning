#!/usr/bin/env python
# coding: utf-8

# In[11]:


import tensorflow as tf
from tensorflow import keras


# In[12]:


fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[13]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[14]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[15]:


loss = 'sparse_categorical_crossentropy'


# In[16]:


optimizer = tf.keras.optimizers.SGD(0.1)


# In[18]:


model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=256)


# In[19]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Acc:',test_acc)

