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

# =============================================================================
# #Train Data Output - Sequence Length 208, hidden = 10
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_208_hidden_10.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_208_hidden_10.npy')
# print(trueOutput.shape)
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 208, hidden = 15
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_208_hidden_15.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_208_hidden_15.npy')
# print(trueOutput.shape)
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 208, hidden = 20
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_208_hidden_20.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_208_hidden_20.npy')
# print(trueOutput.shape)
# =============================================================================


# =============================================================================
# #Train Data Output - Sequence Length 312, hidden = 20
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_312_hidden_20.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_312_hidden_20.npy')
# print(trueOutput.shape)
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 416, hidden = 15
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_416_hidden_15.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_416_hidden_15.npy')
# print(trueOutput.shape)
# 
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 416, hidden = 20
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_416_hidden_20.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_416_hidden_20.npy')
# print(trueOutput.shape)
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 416, hidden = 25
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_416_hidden_25.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_416_hidden_25.npy')
# print(trueOutput.shape)
# =============================================================================



# =============================================================================
# #Train Data Output - Sequence Length 520, hidden = 20
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_520_hidden_20.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_520_hidden_20.npy')
# print(trueOutput.shape)
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 520, hidden = 25
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_520_hidden_25.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_520_hidden_25.npy')
# print(trueOutput.shape)
# =============================================================================

#Train Data Output - Sequence Length 520, hidden = 30
predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_520_hidden_30.npy')
print(predictedOutput.shape)

trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_520_hidden_30.npy')
print(trueOutput.shape)

# =============================================================================
# #Train Data Output - Sequence Length 624, hidden = 20
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_624_hidden_20.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_624_hidden_20.npy')
# print(trueOutput.shape)
# =============================================================================


# =============================================================================
# #Train Data Output - Sequence Length 624, hidden = 30
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_624_hidden_30.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_624_hidden_30.npy')
# print(trueOutput.shape)
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 624, hidden = 35
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_SeqLength_624_hidden_35.npy')
# print(predictedOutput.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_SeqLength_624_hidden_35.npy')
# print(trueOutput.shape)
# =============================================================================


# =============================================================================
# ##########    Printing True Signal and Predicted Signal in Train Set ##################
# =============================================================================
for i in range(5):
    pred = predictedOutput[:,i]
    print(pred.shape)
    
    true=trueOutput[:,i]
    print(true.shape)
    
    plt.figure()
    plt.title("TrainData - True Signal")
    plt.plot(list(true))
    plt.show()
    
    
    plt.figure()
    plt.title("TrainData - Predicted Signal")
    plt.plot(list(pred))
    plt.show()
    
    
# =============================================================================
# ###################    Loading Test Data   ############### 
# =============================================================================

# =============================================================================
# #Test Data Output - Sequence Length 208, hidden = 10
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_208_hidden_10.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_208_hidden_10.npy')
# print(trueOutputTest.shape)
# =============================================================================
    
# =============================================================================
# #Test Data Output - Sequence Length 208, hidden = 15
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_208_hidden_15.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_208_hidden_15.npy')
# print(trueOutputTest.shape)
# =============================================================================
    
# =============================================================================
# #Test Data Output - Sequence Length 208, hidden = 20
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_208_hidden_20.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_208_hidden_20.npy')
# print(trueOutputTest.shape)
# =============================================================================
    
# =============================================================================
# #Test Data Output - Sequence Length 312, hidden = 20
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_312_hidden_20.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_312_hidden_20.npy')
# print(trueOutputTest.shape)
# =============================================================================

# =============================================================================
# #Test Data Output - Sequence Length 416, hidden = 15
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_416_hidden_15.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_416_hidden_15.npy')
# print(trueOutputTest.shape)
# =============================================================================
    
# =============================================================================
# #Test Data Output - Sequence Length 416, hidden = 20
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_416_hidden_20.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_416_hidden_20.npy')
# print(trueOutputTest.shape)
# =============================================================================

# =============================================================================
# #Test Data Output - Sequence Length 416, hidden = 25
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_416_hidden_25.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_416_hidden_25.npy')
# print(trueOutputTest.shape)
# =============================================================================

# =============================================================================
# #Test Data Output - Sequence Length 520, hidden = 20
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_520_hidden_20.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_520_hidden_20.npy')
# print(trueOutputTest.shape)
# =============================================================================
    
# =============================================================================
# #Test Data Output - Sequence Length 520, hidden = 25
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_520_hidden_25.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_520_hidden_25.npy')
# print(trueOutputTest.shape)
# =============================================================================

#Test Data Output - Sequence Length 520, hidden = 30
predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_520_hidden_30.npy')
print(predictedOutputTest.shape)

trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_520_hidden_30.npy')
print(trueOutputTest.shape)

# =============================================================================
# #Test Data Output - Sequence Length 624, hidden = 20
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_624_hidden_20.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_624_hidden_20.npy')
# print(trueOutputTest.shape)
# =============================================================================

# =============================================================================
# #Test Data Output - Sequence Length 624, hidden = 30
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_624_hidden_30.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_624_hidden_30.npy')
# print(trueOutputTest.shape)
# =============================================================================

# =============================================================================
# #Test Data Output - Sequence Length 624, hidden = 35
# predictedOutputTest=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_SeqLength_624_hidden_35.npy')
# print(predictedOutputTest.shape)
# 
# trueOutputTest=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_SeqLength_624_hidden_35.npy')
# print(trueOutputTest.shape)
# =============================================================================




# =============================================================================
# ##########    Printing True Signal and Predicted Signal in Test Set ##################
# =============================================================================

for i in range(2):
    pred = predictedOutputTest[:,i]
    print(pred.shape)
    
    true=trueOutputTest[:,i]
    print(true.shape)
    
    plt.figure()
    plt.title("TestData - True Signal")
    plt.plot(list(true))
    plt.show()
    
    
    plt.figure()
    plt.title("TestData - Predicted Signal")
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

# =============================================================================
# #Checking Anamoly
# #Anamoly is at Sample 33 in Test Set for Sequence Length 312
# pred = predictedOutputTest[:,33]
# print(pred.shape)
# 
# true=trueOutputTest[:,33]
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


# =============================================================================
# # Checking Anamoly
# # Anamoly is at Sample 25 in Test Set for Sequence Length 416
# pred = predictedOutputTest[:,25]
# print(pred.shape)
# 
# true=trueOutputTest[:,25]
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


####Checking Anamoly
##Anamoly is at Sample 20 in Test Set for Sequence Length 520
pred = predictedOutputTest[:,20]
print(pred.shape)

true=trueOutputTest[:,20]
print(true.shape)

plt.figure()
plt.title("TestData Checking Anamoly - True Signal")
plt.plot(list(true))
plt.show()


plt.figure()
plt.title("TestData Checking Anamoly- Predicted Signal")
plt.plot(list(pred))
plt.show()

# =============================================================================
# ####Checking Anamoly
# ##Anamoly is at Sample 16 in Test Set for Sequence Length 624
# pred = predictedOutputTest[:,16]
# print(pred.shape)
# 
# true=trueOutputTest[:,16]
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


