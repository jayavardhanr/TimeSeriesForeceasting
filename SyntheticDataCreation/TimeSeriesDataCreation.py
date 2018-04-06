#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:55:50 2018

@author: jayavardhanreddy

Ref link: 
    https://keystrokecountdown.com/articles/poisson/index.html
"""
from __future__ import division
import random




#poisson distribution to sample time for events

#lambda=14/14400
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
typeOfEvents = ['A', 'B']
events=[]
for singleDay in times:
    singleDayEvents=[]
    for _ in range(len(singleDay)):
        singleDayEvents.append(random.choice(typeOfEvents))
    events.append(singleDayEvents)










    
