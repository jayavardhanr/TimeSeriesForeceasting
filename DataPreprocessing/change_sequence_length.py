#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:14:40 2018

@author: jayavardhanreddy
"""
import numpy as np
import matplotlib.pyplot as plt

trainDataArray=np.array(np.load('sel102_trainData.npy'))
testDataArray=np.array(np.load('sel102_testData.npy'))


# =============================================================================
# ########Sequenth Length =416
# trainDataArray=trainDataArray[:-1,:,:]
# x_dimension=trainDataArray.shape[0]/2
# newSequenceLength=trainDataArray.shape[1]*2
# reshaped_trainDataArray=trainDataArray.reshape((x_dimension,newSequenceLength,1))
# 
# 
# testDataArray=testDataArray[:-1,:,:]
# x_dimension=testDataArray.shape[0]/2
# newSequenceLength=testDataArray.shape[1]*2
# reshaped_testDataArray=testDataArray.reshape((x_dimension,newSequenceLength,1))
# 
# for trainingData in reshaped_trainDataArray[:5]:
#     plt.figure()
#     plt.title("Training Data 416")
#     plt.plot(list(trainingData))
#     plt.show()
#     
# for testData in reshaped_testDataArray[:5]:
#     plt.figure()
#     plt.title("Test Data 416")
#     plt.plot(list(testData))
#     plt.show()
#   
# 
# plt.figure()
# plt.title("Test Data Anamoly 416")
# plt.plot(list(reshaped_testDataArray[25]))
# plt.show()
# 
# np.save('sel102_trainData_416.npy',reshaped_trainDataArray)
# np.save('sel102_testData_416.npy',reshaped_testDataArray)
# =============================================================================

# =============================================================================
# #Sequenth Length =312
# trainDataArray=np.array(np.load('sel102_trainData.npy'))
# testDataArray=np.array(np.load('sel102_testData.npy'))
# 
# trainDataArray=trainDataArray[:-1,:,:]
# x_dimension=(trainDataArray.shape[0]*2)/3
# newSequenceLength=(trainDataArray.shape[1]*3)/2
# reshaped_trainDataArray=trainDataArray.reshape((x_dimension,newSequenceLength,1))
# 
# 
# testDataArray=testDataArray[:-2,:,:]
# x_dimension=(testDataArray.shape[0]*2)/3
# newSequenceLength=(testDataArray.shape[1]*3)/2
# reshaped_testDataArray=testDataArray.reshape((x_dimension,newSequenceLength,1))
# 
# for trainingData in reshaped_trainDataArray[:5]:
#     plt.figure()
#     plt.title("Training Data 312")
#     plt.plot(list(trainingData))
#     plt.show()
#     
# for testData in reshaped_testDataArray[:5]:
#     plt.figure()
#     plt.title("Test Data 312")
#     plt.plot(list(testData))
#     plt.show()
#   
# 
# plt.figure()
# plt.title("Test Data Anamoly 312")
# plt.plot(list(reshaped_testDataArray[33]))
# plt.show()
# 
# 
# np.save('sel102_trainData_312.npy',reshaped_trainDataArray)
# np.save('sel102_testData_312.npy',reshaped_testDataArray)
# =============================================================================


########Sequenth Length =520
trainDataArray=np.array(np.load('sel102_trainData.npy'))
testDataArray=np.array(np.load('sel102_testData.npy'))

trainDataArray=trainDataArray[:-4,:,:]
x_dimension=(trainDataArray.shape[0]*2)/5
newSequenceLength=(trainDataArray.shape[1]*5)/2
reshaped_trainDataArray=trainDataArray.reshape((x_dimension,newSequenceLength,1))


testDataArray=testDataArray[:-1,:,:]
x_dimension=(testDataArray.shape[0]*2)/5
newSequenceLength=(testDataArray.shape[1]*5)/2
reshaped_testDataArray=testDataArray.reshape((x_dimension,newSequenceLength,1))

for trainingData in reshaped_trainDataArray[:5]:
    plt.figure()
    plt.title("Training Data 520")
    plt.plot(list(trainingData))
    plt.show()
    
for testData in reshaped_testDataArray[:5]:
    plt.figure()
    plt.title("Test Data 520")
    plt.plot(list(testData))
    plt.show()
  

plt.figure()
plt.title("Test Data Anamoly 520")
plt.plot(list(reshaped_testDataArray[20]))
plt.show()

np.save('sel102_trainData_520.npy',reshaped_trainDataArray)
np.save('sel102_testData_520.npy',reshaped_testDataArray)

#######Sequenth Length =624
trainDataArray=np.array(np.load('sel102_trainData.npy'))
testDataArray=np.array(np.load('sel102_testData.npy'))

trainDataArray=trainDataArray[:-1,:,:]
x_dimension=trainDataArray.shape[0]/3
newSequenceLength=trainDataArray.shape[1]*3
reshaped_trainDataArray=trainDataArray.reshape((x_dimension,newSequenceLength,1))

testDataArray=testDataArray[:-2,:,:]
x_dimension=testDataArray.shape[0]/3
newSequenceLength=testDataArray.shape[1]*3
reshaped_testDataArray=testDataArray.reshape((x_dimension,newSequenceLength,1))

for trainingData in reshaped_trainDataArray[:5]:
    plt.figure()
    plt.title("Training Data 624")
    plt.plot(list(trainingData))
    plt.show()
    
for testData in reshaped_testDataArray[:5]:
    plt.figure()
    plt.title("Test Data 624")
    plt.plot(list(testData))
    plt.show()
  

plt.figure()
plt.title("Test Data Anamoly 624")
plt.plot(list(reshaped_testDataArray[16]))
plt.show()

np.save('sel102_trainData_624.npy',reshaped_trainDataArray)
np.save('sel102_testData_624.npy',reshaped_testDataArray)



