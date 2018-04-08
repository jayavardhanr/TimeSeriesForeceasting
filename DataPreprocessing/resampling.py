#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:13:19 2018

@author: jayavardhanreddy
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
'''
Plot output
'''
input_ = np.loadtxt('sel_102.txt',skiprows=2)
print(input_.shape)

times=input_[:,0]
mlii=input_[:,1]
v5=input_[:,2]
count=1.0
count_=0
previous=0

l=[]

print(len(times))
for time in times:
    if time>count:
        l.append(count_-previous-1)
        previous=count_-1
        count+=1.0
        count_+=1
    else:
        count_+=1
    
print('Number of points per seconds:',set(l))
print('total number of seconds',len(l))

mlii_1s_resample_trace=[]
start=0
for index in l:
    resample_1s_trace = signal.resample(np.asarray(mlii[start:start+index]),50)
    mlii_1s_resample_trace.append(resample_1s_trace)
    start=start+index
    
v5_1s_resample_trace=[]
start=0
for index in l:
    resample_1s_trace = signal.resample(np.asarray(v5[start:start+index]),20)
    v5_1s_resample_trace.append(resample_1s_trace)
    start=start+index
    
    
plt.figure()
plt.title("mlii signal original")
plt.plot(mlii[:5000])
plt.show()

plt.figure()
plt.title("mlii signal resample 50")
#plt.plot(mlii_1s_resample_trace[0]+mlii_1s_resample_trace[1]+mlii_1s_resample_trace[2])
plt.plot(mlii_1s_resample_trace[0])
plt.show()
    
plt.figure()
plt.title("v5 signal original 50")
plt.plot(v5[:750])
plt.show()

plt.figure()
plt.title("v5 signal resample 50")
#plt.plot(v5_1s_resample_trace[0]+v5_1s_resample_trace[1]+v5_1s_resample_trace[2])
plt.plot(v5_1s_resample_trace[0])
plt.show()
    

    
