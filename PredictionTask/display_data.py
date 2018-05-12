#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 02:00:38 2018

@author: jayavardhanreddy
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ###################    Loading Train Data   ############### 
# =============================================================================

#Train Data Output - Sequence Length 208, hidden = 20
InputData=np.load('outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20.npy')
print(InputData.shape)

trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20.npy')
print(trueOutput.shape)

predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20.npy')
print(predictedOutput.shape)

# =============================================================================
# ##########    Printing True Signal and Predicted Signal in Train Set ##################
# =============================================================================
for i in range(5):
    input_data=InputData[:,i]
    print(input_data.shape)
    
    true=trueOutput[:,i]
    print(true.shape)
   
    pred = predictedOutput[:,i]
    print(pred.shape)
    
    plt.figure()
    plt.title("TrainData - Input Signal")
    plt.plot(list(input_data))
    plt.show()
    
    plt.figure()
    plt.title("TrainData - True Output")
    plt.plot(list(true))
    plt.show()
    
    
    plt.figure()
    plt.title("TrainData - Predicted Signal")
    plt.plot(list(pred))
    plt.show()
    
    
# =============================================================================
# ###################    Loading Test Data   ############### 
# =============================================================================


#Train Data Output - Sequence Length 208, hidden = 20
InputData=np.load('outputs/Test_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20.npy')
print(InputData.shape)

trueOutput=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20.npy')
print(trueOutput.shape)

predictedOutput=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20.npy')
print(predictedOutput.shape)



# =============================================================================
# ##########    Printing True Signal and Predicted Signal in Test Set ##################
# =============================================================================

for i in range(5):
    input_data=InputData[:,i]
    print(input_data.shape)
    
    true=trueOutput[:,i]
    print(true.shape)
   
    pred = predictedOutput[:,i]
    print(pred.shape)
    
    plt.figure()
    plt.title("Test Data - Input Signal")
    plt.plot(list(input_data))
    plt.show()
    
    plt.figure()
    plt.title("Test Data - True Signal Output")
    plt.plot(list(true))
    plt.show()
    
    
    plt.figure()
    plt.title("Test Data - Predicted Signal Output")
    plt.plot(list(pred))
    plt.show()
    

# =============================================================================
# ####Checking Anamoly
# ##Anamoly is at Sample 50 in Test Set for Sequence Length 208
# pred = predictedOutputTest[:,50]
# print(pred.shape)
# 
# true=trueOutputTest[:,50]
# print(true.shape)
# 
# plt.figure()
# plt.title("TestData Checking Anamoly - True Signal")
# plt.plot(list(true))
# plt.show()
# 
# 
# plt.figure()
# plt.title("TestData Checking Anamoly- Predicted Signal")
# plt.plot(list(pred))
# plt.show()
# =============================================================================


