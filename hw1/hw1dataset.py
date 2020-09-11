#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:30:10 2019

@author: tarodz
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.array([-1.67245526, -2.36540279, -2.14724263,  1.40539096,  1.24297767,
        -1.71043904,  2.31579097,  2.40479939, -2.22112823])
    
y=np.array([-18.56122168, -24.99658931, -24.41907817,  -2.688209  ,
        -1.54725306,  -19.18190097,   1.74117419,
         3.97703338, -24.80977847])
    
plt.scatter(x,y)
plt.show()
