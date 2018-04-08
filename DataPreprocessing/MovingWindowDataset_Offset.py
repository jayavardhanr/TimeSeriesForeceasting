#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 00:38:17 2018

@author: jayavardhanreddy
"""

import numpy as np
import matplotlib.pyplot as plt

ecg_input = np.loadtxt('sel_102.txt',skiprows=2)
v5=list(ecg_input[:,2])
#Anamoly 54265 to 54475
#Total Length of Signal=225000

trainDataSequences_Input=[]
trainDataSequences_Output=[]
testDataSequences_Input=[]
testDataSequences_Output=[]

length=1040

offsets=[5*i for i in range(length/5)]
i=0
while i+length+length<=60000:
    for offset in offsets:
        j=i
        if j+offset+length+length<=60000:
            testDataSequences_Input.append(v5[j+offset:j+offset+length])
            j+=length
            testDataSequences_Output.append(v5[j+offset:j+offset+length])
        else:
            break
    i+=length
 
print("Test final",i)

i=60000
while i+length+length<=225000:
    for offset in offsets:
        j=i
        if j+offset+length+length<=225000:
            trainDataSequences_Input.append(v5[j+offset:j+offset+length])
            j+=length
            trainDataSequences_Output.append(v5[j+offset:j+offset+length])
        else:
            break
    i+=length

print("Train final",i)

trainDataArray_Input=np.array(trainDataSequences_Input)
trainDataArray_Output=np.array(trainDataSequences_Output)
testDataArray_Input=np.array(testDataSequences_Input)
testDataArray_Output=np.array(testDataSequences_Output)

trainDataArray_Input=trainDataArray_Input.reshape((trainDataArray_Input.shape[0],trainDataArray_Input.shape[1],1))
trainDataArray_Output=trainDataArray_Output.reshape((trainDataArray_Output.shape[0],trainDataArray_Output.shape[1],1))
testDataArray_Input=testDataArray_Input.reshape((testDataArray_Input.shape[0],testDataArray_Input.shape[1],1))
testDataArray_Output=testDataArray_Output.reshape((testDataArray_Output.shape[0],testDataArray_Output.shape[1],1))

print(trainDataArray_Input.shape)
print(trainDataArray_Output.shape)
print(testDataArray_Input.shape)
print(testDataArray_Output.shape)

for trainingData_Input,trainingData_Output in zip(trainDataSequences_Input[:5],trainDataSequences_Output[:5]):
    plt.figure()
    plt.title("Training Data Input")
    plt.plot(list(trainingData_Input))
    plt.show()
    plt.figure()
    plt.title("Training Data Output")
    plt.plot(list(trainingData_Output))
    plt.show()

    
for testingData_Input,testingData_Output in zip(testDataSequences_Input[:5],testDataSequences_Output[:5]):
    plt.figure()
    plt.title("Testing Data Input")
    plt.plot(list(testingData_Input))
    plt.show()
    plt.figure()
    plt.title("Testing Data Output")
    plt.plot(list(testingData_Output))
    plt.show()




np.save('../seq2seq_ecg_codes/PredictionTask/inputData/sel102_trainDataInput_'+str(length)+'_offset.npy',trainDataArray_Input)
np.save('../seq2seq_ecg_codes/PredictionTask/inputData/sel102_trainDataOutput_'+str(length)+'_offset.npy',trainDataArray_Output)
np.save('../seq2seq_ecg_codes/PredictionTask/inputData/sel102_testDataInput_'+str(length)+'_offset.npy',testDataArray_Input)
np.save('../seq2seq_ecg_codes/PredictionTask/inputData/sel102_testDataOutput_'+str(length)+'_offset.npy',testDataArray_Output)