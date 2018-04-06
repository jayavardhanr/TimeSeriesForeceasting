#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 23:56:47 2018

@author: jayavardhanreddy
"""

from numpy import array, zeros, argmin, inf
import random
import operator
from matplotlib import pyplot as plt
plt.figure(figsize=(12,12))

def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def distanceFunction(value1,value2):
    """
    Function to Calculate distance between a tuple of
    (time1,event1) and (time2,event2)
    args:
        inputs:
            value1:tuple(time1,event1)
            value2:tuple(time2,event2)
        outputs:
            int : distance
    """
    time1=value1[0]
    time2=value2[0]
    event1=value1[1]
    event2=value2[1]
    
    #difference for events is as good as 60 mins time difference
    distance=0
    if event1!=event2:
        distance+=60
    
    #difference for times is absolute difference between times
    distance+=abs(time1-time2)
    return distance

class EventTimeSequences(object):
    def __init__(self):
        self.event_time_sequence= []
        
    def generateSequenceOfEventsAndTimeStamps(self,number_of_days):
        '''
        #Sampling Time Sequence for each day
        We use poisson distribution to sample time for events
        To get 14 events for 1 day(14400 mins), lambda=14/1440 ~= 1/100
        limit for each day is the total number of minutes in a day
        
        #Event sampling
        Random Sampling for a list of events
        '''
        lmbda=1.0/100
        limit = 24 * 60 
        times=[]
        for iter in range(number_of_days):
            singleday=[]
            timestamp = 0.0
            while True:
                timestamp += random.expovariate(lmbda)
                if timestamp >= limit:
                    break
                singleday.append(timestamp)
            times.append(singleday)
        
        typeOfEvents = ['A', 'B']
        for singleDay in times:
            singleDayEvents=[]
            for i,time in enumerate(singleDay):
                singleDayEvents.append((time,random.choice(typeOfEvents)))
            self.event_time_sequence.append(singleDayEvents)

    def getTwoDays(self,same_day,day_number=0):
        """
        Gets time series sequences for two days  
        This function is used for testing DTW
        """
        if same_day:   
            return self.event_time_sequence[day_number],self.event_time_sequence[day_number]
        return self.event_time_sequence[day_number],self.event_time_sequence[day_number+1]
    
    def getOneDay(self,day_number=0):
        """
        Gets time series sequences for One day 
        """
        return self.event_time_sequence[day_number]
    
    def testDTW(self,same_day=True,visualize=False):
        '''
        If same_day=True, the getValues function returns same sequences for x,y
        the distance printed out should be 0
        '''
        x,y=self.getTwoDays(same_day)
        print("day one:\n",x)
        print("day two:\n",y)
        dist_fun = distanceFunction
        dist, cost, acc, path = dtw(x, y, dist_fun)
        print("calculated distance: ",dist)
        
        if visualize:
            plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
            plt.plot(path[0], path[1], '-o') # relation
            plt.xticks(range(len(x)), x,rotation='vertical')
            plt.yticks(range(len(y)), y)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('tight')
            plt.title('Minimum distance: {}'.format(dist))
            plt.show()
            
        return dist
    
    def getDistance(self,one_day,second_day):
        x,y=one_day,second_day
        dist_fun = distanceFunction
        dist, cost, acc, path = dtw(x, y, dist_fun)
        return dist
            
    def generateDifferenceMatrixPlot(self,numberOfDays):
        
        dist_fun = distanceFunction
        Sequences=[]
        for i in range(numberOfDays):
            Sequences.append(self.getOneDay(day_number=i))
            
        matrix=zeros((numberOfDays,numberOfDays))
        for i,seq1 in enumerate(Sequences):
            for j,seq2 in enumerate(Sequences):
                dist, _, _, _ = dtw(seq1, seq2, dist_fun)
                matrix[i,j]=dist

        plt.title("Matrix plot of distance between different days")
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.show()
        
    def ClosestSequences(self,inputSequence, k):
        distances = []
        
        for x in range(len(self.event_time_sequence)):
            dist = self.getDistance(inputSequence, self.event_time_sequence[x])
            distances.append((self.event_time_sequence[x], dist))
        distances.sort(key=operator.itemgetter(1))
        
        closest_sequences = []
        for x in range(k):
            closest_sequences.append(distances[x])
        return closest_sequences

if __name__ == '__main__':
    print("eventTimeSequences - start\n")
    eventTimeSequences=EventTimeSequences()
    print("eventTimeSequences - done\n")
    
    print("generated eventTimeSequences - start\n")
    eventTimeSequences.generateSequenceOfEventsAndTimeStamps(10000)
    print("generated eventTimeSequences - done\n")
    
    print("testDTW - start\n")
    eventTimeSequences.testDTW(same_day=True)
    print("testDTW - done\n")
    
    print("matrix plot - start\n")
    eventTimeSequences.generateDifferenceMatrixPlot(10)
    print("matrix plot - done\n")
    
    print("find closest sequences-start\n")
    inputSequence=eventTimeSequences.getOneDay()
    numberOfSequences=10
    closest_sequences=eventTimeSequences.ClosestSequences(inputSequence,numberOfSequences)
    print(inputSequence)
    print('\n')
    for i in range(numberOfSequences):
        print(str(i+1)+' closest sequence\n')
        print(closest_sequences[i])
        print('\n')
    
    
    
    
    

    
    
    
        
