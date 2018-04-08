#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:17:48 2018

@author: jayavardhanreddy
"""

import numpy as np
import matplotlib.pyplot as plt

label=[]
sample_number=[]
i=0
with open('sel_102_ann.txt', 'rU') as f:
    for line in f:
        if i==0:
            print line
            i+=1
        else:
            splitLine=line.split()
            sample_number.append(int(splitLine[1]))
            label.append(splitLine[2])
        

anomolySampleNumberminus1=sample_number[label.index('V')-1]       
anomolySampleNumber=sample_number[label.index('V')]
anomolySampleNumberPlus1=sample_number[label.index('V')+1]
anomolySampleNumberPlus2=sample_number[label.index('V')+2]
print("Anomoly Sample Number ",anomolySampleNumber,' to ',anomolySampleNumberPlus1)
print("Anomoly Occurs at ",anomolySampleNumber,' to ',anomolySampleNumber+208)

ecg_input = np.loadtxt('sel_102.txt',skiprows=2)
mlii=list(ecg_input[:,1])
v5=list(ecg_input[:,2])

print("Dispaly Anomoly along with signal surrounding it")

'''
plt.figure()
plt.plot(mlii[max(anomolySampleNumber-2080,0):anomolySampleNumber+208+2080])
plt.show()
'''

plt.figure()
plt.title('Proper Signal')
plt.plot(v5[anomolySampleNumberminus1:anomolySampleNumber])
plt.show()
   
plt.figure()
plt.title('Exact Anamoly')
plt.plot(v5[anomolySampleNumber:anomolySampleNumberPlus1])
plt.show()

'''
plt.figure()
plt.plot(v5[anomolySampleNumberPlus1:anomolySampleNumberPlus2])
plt.show()
'''
  
plt.figure()
plt.title("Dispaly Anomoly along with signal surrounding it")
plt.plot(v5[max(anomolySampleNumber-2080,0):anomolySampleNumber+208+2080])
plt.show()


#Preparing Test and Train Data Sequences
#Creating Sequences of Length =208
#Reference: Anomoly Detection paper
test_start=anomolySampleNumber-(208*50)
test_end=anomolySampleNumber+(208*51)
testData=v5[test_start:test_end]

'''
plt.figure()
exactAnamoloy=v5[test_start+(208*50):test_start+(208*51)]
plt.plot(list(exactAnamoloy))
plt.title('Exact Anamoly')
plt.show()
'''

trainingData=v5[test_end:]

#Train Data Sequence Creation
trainDataSequences=[]
number_of_train_batches=len(trainingData)/208
print("number_of_train_batches: ",number_of_train_batches)

for batchnumber in range(number_of_train_batches):
    trainDataSequences.append(trainingData[50+batchnumber*208:50+(batchnumber+1)*208])

#Test Data Sequence Creation
testDataSequences=[]
number_of_test_batches=len(testData)/208
print("number_of_test_batches: ",number_of_test_batches)

for batchnumber in range(number_of_test_batches):
    testDataSequences.append(testData[batchnumber*208:(batchnumber+1)*208])

trainDataArray=np.array(trainDataSequences)  
testDataArray=np.array(testDataSequences)

trainDataArray=trainDataArray.reshape((trainDataArray.shape[0],trainDataArray.shape[1],1))
testDataArray=testDataArray.reshape((testDataArray.shape[0],testDataArray.shape[1],1))

print(trainDataArray.shape)
print(testDataArray.shape)

for trainingData in trainDataSequences[:10]:
    plt.figure()
    plt.title("Training Data")
    plt.plot(list(trainingData))
    plt.show()
    
for testData in testDataSequences[:10]:
    plt.figure()
    plt.title("Test Data")
    plt.plot(list(testData))
    plt.show()
    

plt.figure()
plt.title("Test Data Anamoly")
plt.plot(list(testDataSequences[50]))
plt.show()


#np.save('sel102_trainData.npy',trainDataArray)
#np.save('sel102_testData.npy',testDataArray)



  