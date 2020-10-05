#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hernandezsm
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import torch as torch


x=np.array([-1.67245526, -2.36540279, -2.14724263,  1.40539096,  1.24297767,
        -1.71043904,  2.31579097,  2.40479939, -2.22112823])
    
y=np.array([-18.56122168, -24.99658931, -24.41907817,  -2.688209  ,
        -1.54725306,  -19.18190097,   1.74117419,
         3.97703338, -24.80977847])
    
plt.scatter(x,y)

# create a tensor variable, this is a constant parameter, we do not need gradient w.r.t. to it
x=torch.tensor(x,requires_grad=False)
y=torch.tensor(y,requires_grad=False)

# define some function using pytorch operations (note torch. instead of np.) 
# this function is f(w)=||w-minimum||^2, and so has minimum at minimum_w, i.e. at vector [1.0,3.0]
# it is a convex function so has one minimum, no other local minima
def f(w):
    pred = sum([torch.mul(torch.pow(x, i), w[i]) for i in range(len(w))])
    return sum(torch.pow(pred - y,2)) / len(y)

# 
# Run training for different N
# i.e. wx^0 + wx^1 + ... + wx^N+1
# 
for N in range(2,11):
  #define starting value of W for gradient descent
  initialW=np.random.rand(N)

  #create a PyTorch tensor variable for w. 
  # we will be optimizing over w, finding its best value using gradient descent (df / dw) so we need gradient enabled
  w = torch.tensor(initialW,requires_grad=True);

  # this will do gradient descent (fancy, adaptive learning rate version called Adam) for us
  optimizer = torch.optim.Adam([w],lr=0.001)

  for i in range(10000):
      # clear previous gradient calculations
      optimizer.zero_grad();

      # calculate f based on current value
      z=f(w);
      if (i % 100 == 0 ):
          print("Iter: ",i," w: ",w.data.cpu()," f(w): ",z.item())

      # calculate gradient of f w.r.t. w
      z.backward()

      # use the gradient to change w
      optimizer.step()

  print("Found minimum:"+str(w.data.cpu()));

  """
  Calculate f(x) for all x in X given the weights $w$.
  """
  def calculate_f_x(X, w):
    output = []
    for x in X:
      out = 0
      for i in range(len(w)):
        out += (x**i)*w[i]
        
      output.append(out)
    return output


  """
  Plot samples, predicted values for $n$. Save to File
  """
  x_ = np.arange(-3,3,0.1)
  y_ = np.array(calculate_f_x(x_,w.data.cpu()))
  plt.plot(x_, y_, ':k')
  plt.plot(x, calculate_f_x(x,w.data.cpu()), '.r')
  plt.plot(x, y, '.b')
  plt.legend(["Prediction Function", "Prediction Per Sample", "Sample"])
  plt.savefig(f"output/n.{N}.eps")
  plt.clf()