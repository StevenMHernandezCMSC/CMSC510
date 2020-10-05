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


# 
# Load MNIST Dataset
# 
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

def transform_dataset(mnist_dataset):
  # Load data and labels
  X = mnist_dataset.data
  y = mnist_dataset.targets

  # Select samples which belong to the class: `9` or `5`
  LABEL_1 = 9
  LABEL_2 = 5
  idx = torch.logical_or(y == LABEL_1, y == LABEL_2)

  # Reshape X and y for training
  X = torch.reshape(X, (len(X), 28*28))
  y = torch.reshape(y, (len(y), 1))

  # Filter samples outside of the selected classes
  X = X[idx, :]
  y = y[idx]

  # Encode classes as bipolar values [-1,+1]
  y[y == LABEL_2] = -1
  y[y == LABEL_1] = 1

  return X, y

X_train, y_train = transform_dataset(mnist_trainset)
X_test, y_test = transform_dataset(mnist_testset)

# 
# Define loss function
# 
def f(w, b, X, y):
  l = len(y)

  def calculate_loss(i):
    linear_model = torch.mm(w.transpose(0,1).float(),X[i,:,None].float()) + b
    logistic_loss = torch.log(1 + torch.exp(torch.mul(-y[i], linear_model)))
    return logistic_loss

  return sum([calculate_loss(i) for i in range(l)]) / l

# 
# Setup linear model variables and tensors
# 
# initialW = ((np.random.rand(28*28,1) * 2) - 1) * 1e-8
initialW = np.random.rand(28*28,1) * 1e-8
initialB = np.random.rand(1)
w = torch.tensor(initialW,requires_grad=True)
b = torch.tensor(initialB,requires_grad=True)

# Set optimizer to be the "Adam" optimizer
optimizer = torch.optim.Adam([w],lr=1e-4)

train_loss = []
test_loss = []
train_error = []
test_error = []

num_pos = sum(y_train == 1)
num_neg = sum(y_train == -1)
num_total = len(y_train)

for i in range(100):
    print("Iteration:", i)
    # clear previous gradient calculations
    optimizer.zero_grad()

    # calculate f based on current value
    z=f(w,b,X_train,y_train)

    # calculate gradient of f w.r.t. w
    z.backward()

    # use the gradient to change w
    optimizer.step()

    # calculate loss
    train_loss.append(f(w,b, X_train, y_train))
    test_loss.append(f(w,b, X_test, y_test))

    # calculate error
    def calculate_error(X, y):
      pred_ind = ((torch.matmul(X.float(), w.float())) + b.float())
      return np.count_nonzero(y[(pred_ind >= 0) != (y ==  1)]) / num_total

    train_error.append(calculate_error(X_train,y_train))
    test_error.append(calculate_error(X_test,y_test))

plt.plot(train_loss)
plt.plot(test_loss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(["Training", "Testing"])
plt.savefig("output/mnist_loss.eps")
plt.clf()

plt.plot(train_error)
plt.plot(test_error)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.legend(["Training", "Testing"])
plt.savefig("output/mnist_error.eps")
plt.clf()
