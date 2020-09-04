#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
mint=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)=mint.load_data()


# In[2]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[3]:


plt.imshow(x_train[0],cmap="gray")


# In[18]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu',),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.5),
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20,
              batch_size=256,
              validation_data=(x_test, y_test),
              validation_freq=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




