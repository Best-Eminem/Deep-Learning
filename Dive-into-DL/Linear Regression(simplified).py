#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal(shape=(num_examples,num_inputs),stddev=1)
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels += tf.random.normal(labels.shape,stddev = 0.01)


# In[2]:


from tensorflow import data as tfdata

batch_size = 10
# 将训练数据的特征和标签组合
dataset = tfdata.Dataset.from_tensor_slices((features, labels))
# 随机读取小批量 
# shuffle 的 buffer_size 参数应大于等于样本数，batch 可以指定 batch_size 的分割大小。
dataset = dataset.shuffle(buffer_size=num_examples) 
dataset = dataset.batch(batch_size)
#使用iter(dataset)的方式，只能遍历数据集一次，是一种比较 tricky 的写法
data_iter = iter(dataset)


# In[3]:


for X, y in data_iter:
    print(X, y)
    break


# In[4]:


for (batch, (X, y)) in enumerate(dataset):
    print(X, y)
    break


# In[6]:


# 使用Keras定义网络，定义模型和初始化参数
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init
model = keras.Sequential()
model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01)))


# In[7]:


# 定义损失函数
from tensorflow import losses
loss = losses.MeanSquaredError()


# In[8]:


# 定义优化算法
from tensorflow.keras import optimizers
trainer = optimizers.SGD(learning_rate=0.03)


# In[9]:


# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for (batch, (X, y)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            l = loss(model(X, training=True), y)

        grads = tape.gradient(l, model.trainable_variables)
        trainer.apply_gradients(zip(grads, model.trainable_variables))

    l = loss(model(features), labels)
    print('epoch %d, loss: %f' % (epoch, l))


# In[10]:


true_w, model.get_weights()[0]


# In[11]:


true_b, model.get_weights()[1]


# In[ ]:




