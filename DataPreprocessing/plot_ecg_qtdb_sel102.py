#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 00:10:17 2018

@author: jayavardhanreddy
"""
import numpy as np
import matplotlib.pyplot as plt
'''
Plot output
'''
input_ = np.loadtxt('sel_102.txt',skiprows=2)
print(input_.shape)


mlii=input_[:,1]
v5=input_[:,2]
print(mlii.shape)
print(v5.shape)


plt.figure()
plt.plot(mlii[:1000])
plt.show()
    
plt.figure()
plt.plot(v5[:1000])
plt.show()