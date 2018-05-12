#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:55:50 2018

@author: Gordon

Ref link: 
    https://keystrokecountdown.com/articles/poisson/index.html
"""
from __future__ import division
import random

import numpy as np
from numpy import array, zeros, argmin, inf
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances


def distance_cost_plot(distances):
    im = plt.imshow(distances, interpolation='nearest', cmap='Reds') 
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.colorbar();


def softmin(a, b, c, gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma
    
    max_val = max(max(a, b), c)
    tmp = 0
    tmp += np.exp(a - max_val)
    tmp += np.exp(b - max_val)
    tmp += np.exp(c - max_val)
    
    return -gamma * (np.log(tmp) + max_val)

#poisson distribution to sample time for events

#lambda=14/14400

event_time_sequence= []

np.random.seed(15)
lmbda=1/100

limit = 24 * 60 
times=[]
for iter in range(100000):
    singleday=[]
    timestamp = 0.0
    while True:
        timestamp += random.expovariate(lmbda)
        if timestamp >= limit:
            break
        singleday.append(timestamp)
    times.append(singleday)

#Event sampling
typeOfEvents = ['0', '1']
#events=[]
for singleDay in times:
    singleDayEvents=[]
    for i,time in enumerate(singleDay):
        singleDayEvents.append((time,random.choice(typeOfEvents)))
    event_time_sequence.append(singleDayEvents)
    
    

rand_nums_test = [x for x in range(100000)]
random.shuffle(rand_nums_test)

x1 = np.asarray(event_time_sequence[rand_nums_test[101]])
y1 = np.asarray(event_time_sequence[rand_nums_test[102]])
for i in range(x1.shape[0]):
    if x1[i][1] == '0':
        x1[i][1] = 0;

    elif x1[i][1] == '1':
        x1[i][1] = 1;

for j in range(y1.shape[0]):
    if y1[j][1] == '0':
        y1[j][1] = 0;

    elif y1[j][1] == '1':
        y1[j][1] = 1;


assert x1.shape[1] == y1.shape[1]
x1_ = x1.astype(np.float64)
y1_ = y1.astype(np.float64)
#D = SquaredEuclidean(x1_, y1_)

#x1_ = x1_[:,0]
#y1_ = y1_[:,0]

D = euclidean_distances(x1_, y1_, squared=True)
#sdtw_ = SoftDTW(D, gamma=self.gamma)
#loss = sdtw_.compute()

gamma=1.0
D_ = D.astype(np.float64)
m, n = D_.shape
R = np.zeros((m+2, n+2), dtype=np.float64)

R[0, 0] = 0
for i in range(1,m+1):
    for j in range(1,n+1):
        R[i, j] = D[i-1, j-1] + softmin(R[i-1, j],R[i-1,j-1],R[i,j-1],gamma)
        
        
#distance_cost_plot(R)
#m+1 = len(y)
#n+1 = len(x)



path = [[n, m]]

i = m
j = n
while i>0 and j>0:
    if i==0:
        j = j - 1
    elif j==0:
        i = i - 1
    else:
        if R[i-1, j] == min(R[i-1, j-1], R[i-1, j], R[i, j-1]):
            i = i - 1
        elif R[i, j-1] == min(R[i-1, j-1], R[i-1, j], R[i, j-1]):
            j = j - 1
        else:
            i = i - 1
            j = j - 1
    path.append([j, i])
path.append([0,0])


cost = 0
count_path = 0
for z in range(len(path)):
    [x,y] = path[z]
    if x == 0 or y == 0:
        break
    else:
        count_path += 1
        print(D[y-1,x-1])
        cost = cost + D[y-1,x-1]

avg_cost = cost/count_path
       
path_x = [point[0] for point in path]
path_y = [point[1] for point in path]

distance_cost_plot(R)
plt.plot(path_x, path_y);

# The above plot shows the optimum warping path which minimizes the sum of distance (DTW distance) along the path. 
# Let us wrap up the function by also incorporating the DTW distance between the two signals as well.
    
#plt.figure(2)
#plt.subplot(211)
#plt.plot(x1_[:,0])
#plt.subplot(212)
#plt.plot(y1_[:,0])

plt.figure(3)
plt.plot(x1_[:,0], 'bo-' ,label='x')
plt.plot(y1_[:,0], 'g^-', label = 'y')
#plt.legend();
#paths = path_cost(x, y, accumulated_cost, distances)[0]

for [map_y, map_x] in path:
    if map_y == 0 or map_x == 0:
        break
    plt.plot([map_x-1, map_y-1], [x1_[map_x-1][0], y1_[map_y-1][0]], 'r')

    
