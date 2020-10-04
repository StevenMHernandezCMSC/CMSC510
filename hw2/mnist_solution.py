# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# @author: hernandezsm
# """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import torch as torch
import torchvision.datasets as datasets

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_trainset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

LABEL_1 = 9
LABEL_2 = 5

X_train = mnist_trainset.data
X_train = torch.reshape(X_train, (len(X_train), 28*28))
y_train = mnist_trainset.targets

idx = torch.logical_or(y_train == LABEL_1, y_train == LABEL_2)
print("idx", idx)
y_train = torch.reshape(y_train, (len(y_train), 1))

X_train = X_train[idx, :]
y_train = y_train[idx]

y_train[y_train == LABEL_2] = -1
y_train[y_train == LABEL_1] = 1

def f(w, b, X, y):
  import time
  out = None
  for m in range(len(y)):
    last_time = time.time()
    wT_x = None
    for j in range(len(w)):
      if wT_x == None:
        wT_x = torch.mul(w[j], X[m,j])
      else: 
        wT_x += torch.mul(w[j], X[m,j])
    if out == None:
      out = torch.log(1 + torch.exp(-y[m] * wT_x))
    else:
      out += torch.log(1 + torch.exp(-y[m] * wT_x))
    print("done", m, len(y), time.time() - last_time)
  return out

  # print(y.shape, w.transpose(0,1).shape, X.transpose(0,1).shape)
  # print("a+b", torch.mul(y, w.transpose(0,1)).shape)
  # print("b+c", torch.mul(w.transpose(0,1), X.transpose(0,1)).shape)
  # print("a+b+c", torch.mul(torch.mul(y, w.transpose(0,1)), X).shape)
  # print(torch.matmul(X_train, w))
  # return sum((((torch.matmul(X_train.float(), w.float())) + b.float()) - y_train)**2)
  # pred = (torch.matmul(X_train.float(), w.float())) + b.float()
  # return -sum(torch.mul(-y, torch.log(pred)) - torch.mul((1 - y), torch.log(1 - pred))) / len(y)
  # return sum(torch.log(1 + torch.exp(torch.mul(torch.mul(-y, w.transpose(0,1)),X_train)))) / len(y)
  # w = w.reshape(28*28)
  # # print("ywX", y[i].shape, w.shape, X[i,:].shape)
  # # print("ywX", y[i].shape, w.transpose(0,1).shape, X[i,:].shape)
  # # print("===1", torch.mul(w, X[i,:].reshape(28*28,1).transpose(0,1)).shape)
  # # print("===2", torch.mul(w.transpose(0,1), X[i,:].reshape(28*28,1)).shape)
  # # print("===1", torch.mul(w, X[i,:].reshape(28*28,1)).shape)
  # # print("===1", torch.mul(w.transpose(0,1), X[i,:].reshape(28*28,1).transpose(0,1)).shape)
  # print("???")
  # print(sum([w[j] * X[i,j] for j in range(len(w))]))
  # print(torch.log(1 + (-y[i] * sum([w[j] * X[i,j] for j in range(len(w))]))))
  # print(len(X))
  # print(([torch.log(1 + (-y[i] * sum([w[j] * X[i,j] for j in range(len(w))]))) for i in range(len(X))]))

  # # return sum([torch.log(1 + torch.exp(torch.mul(-y[i], torch.mul(w.transpose(0,1), X[i,:])))) for i in range(len(X))]) / len(y)
  # return sum([torch.log(1 + (-y[i] * sum([w[j] * X[i,j] for j in range(len(w))]))) for i in range(len(X))]) / len(y)

initialW = np.random.rand(28*28,1) * 1e-8
initialB = np.random.rand(1)
w = torch.tensor(initialW,requires_grad=True)
b = torch.tensor(initialB,requires_grad=True)
# b=1

# this will do gradient descent (fancy, adaptive learning rate version called Adam) for us
optimizer = torch.optim.Adam([w],lr=1e-4)

tprs = []
fprs = []
tnrs = []
fnrs = []

loss = []

num_pos = sum(y_train == 1)
num_neg = sum(y_train == -1)

for i in range(10000):
    # clear previous gradient calculations
    optimizer.zero_grad()
    # calculate f based on current value
    z=f(w,b,X_train,y_train)
    print(i)
    if (i % 100 == 0):
      pred_ind = ((torch.matmul(X_train.float(), w.float())) + b.float())
      tprs.append(np.count_nonzero(y_train[torch.logical_and(pred_ind >= 0, y_train ==  1)]))
      fprs.append(np.count_nonzero(y_train[torch.logical_and(pred_ind >= 0, y_train == -1)]))
      tnrs.append(np.count_nonzero(y_train[torch.logical_and(pred_ind <  0, y_train == -1)]))
      fnrs.append(np.count_nonzero(y_train[torch.logical_and(pred_ind <  0, y_train ==  1)]))
      loss.append(f(w,b, X_train, y_train))
    if (i % 100 == 0 ):
        print("Iter: ",i," w: ",w.data.cpu()[0:5],"... f(w): ",z.item())
        print("True positives",  tprs[-1])
        print("False positives", fprs[-1])
        print("True negatives",  tnrs[-1])
        print("False negatives", fnrs[-1])
        # pred = sum(((torch.matmul(X_train.float(), w.float())) + b) > 0)
        # print("pred", pred)
    # calculate gradient of f w.r.t. w
    z.backward();
    # use the gradient to change w
    optimizer.step();


# # print("True minimum: "+str(minimum_w.data.cpu()));
# print("Found minimum:"+str(w.data.cpu()));

# pred = sum(((torch.matmul(X_train.float(), w.float())) + b) > 0)
# print(pred)

# matplotlib.use('Qt5Agg') 
# plt.plot([1,2,3,4])
plt.subplot(2,1,1)
plt.plot(tprs)
plt.plot(fprs)
plt.plot(tnrs)
plt.plot(fnrs)
plt.legend([
  "True positives",
  "False positives",
  "True negatives",
  "False negatives",
])
plt.xlabel("Iteration")
plt.ylabel("Percentage")
plt.subplot(2,1,2)
plt.plot(loss)
plt.savefig("output/mnist.png")
plt.clf()
