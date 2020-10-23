#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hernandezsm
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#for reproducibility between runs
np.random.seed(123)

# Load mnist and transform
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train = x_train.astype(dtype='float32')
x_test = x_test.astype(dtype='float32')
y_train = y_train.astype(dtype='float32')
y_test = y_test.astype(dtype='float32')

# Select only classes `5` and `9`.
x_train = x_train[(y_train == 5) | (y_train == 9)]
y_train = y_train[(y_train == 5) | (y_train == 9)]
x_test = x_test[(y_test == 5) | (y_test == 9)]
y_test = y_test[(y_test == 5) | (y_test == 9)]
y_train[y_train == 5] = -1
y_train[y_train == 9] = 1
y_test[y_test == 5] = -1
y_test[y_test == 9] = 1

# Reshape
x_train = x_train.reshape((x_train.shape[0], 28*28))
y_train = y_train.reshape((y_train.shape[0], 1))
x_test = x_test.reshape((x_test.shape[0], 28*28))
y_test = y_test.reshape((y_test.shape[0], 1))

# Intialize variables and placeholders
initialW = (np.random.rand(28*28,1) * 1e-6).astype(dtype='float32')
initialB=0.0
w = tf.Variable(initialW,name="w")
b = tf.Variable(initialB,name="b")
x = tf.placeholder(dtype=tf.float32,name='x')
y = tf.placeholder(dtype=tf.float32,name='y')

# 
# Parameters
# 
n_epochs = 100
n_train = x_train.shape[0]
L = 1e20
C = 100
batch_size = 2048

predictions = tf.matmul(x,w)+b
loss = tf.math.log(1+tf.math.exp(tf.multiply(-y,predictions)))
l1_penalty = (1/C) * tf.reduce_sum(tf.abs(w))
risk = tf.reduce_mean(loss) + l1_penalty

optimizer = tf.train.GradientDescentOptimizer(1e-6)
train = optimizer.minimize(risk)

# create a tensorflow session and initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

errors = []

for i in range(0,n_epochs):
    for j in range(0,n_train,batch_size):
        jS=j;jE=min(n_train,j+batch_size)
        x_batch=x_train[jS:jE,:]
        y_batch=y_train[jS:jE,:]
        _,curr_batch_risk,predBatchY,w_value=sess.run([train,risk,predictions,w],feed_dict={x: x_batch, y: y_batch});
        new_w_value = np.zeros(w_value.shape)
        if len(new_w_value[w_value >= 1/L]) > 0:
            new_w_value[w_value >= 1/L] = w_value[w_value >= 1/L] - (1/L)
        if len(new_w_value[w_value <= -1/L]) > 0:
            new_w_value[w_value <= -1/L] = w_value[w_value <= -1/L] + (1/L)
        new_w_assign = tf.assign(w,new_w_value)
        sess.run(new_w_assign)

    y_pred,curr_w,curr_b=sess.run([predictions,w,b],feed_dict={x: x_train, y: y_train})
    err=np.sum(np.abs(np.sign(y_pred) - y_train))
    errors.append(err)
    print(i, "Total Error count:",err)

plt.plot(errors)
plt.show()
