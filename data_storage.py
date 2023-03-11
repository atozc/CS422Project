"""
Title: Project 1
Author: Crystal Atoz
"""

import numpy as np

def build_nparray(data):
    dataArray1 = np.array(data) 
    dataArray2 = np.array(dataArray1[1:]) 
    trainingFeatures = np.array([x[0:len(x) - 1] for x in dataArray2]).astype(float) 
    trainingLabels = np.array([x[len(x) - 1] for x in dataArray2]).astype(int) 
    
    return trainingFeatures, trainingLabels


def build_list(data):
    myData = data[1:].tolist() 
    trainingLabels = [x[len(x) - 1] for x in myData] 
    
    intLabels = []
    for item in trainingLabels:
        intLabels.append(int(item))

    trainingFeatures = [x[0:len(x) - 1] for x in myData]
    featuresData = []
    for item in trainingFeatures:
        features = []
        for x in item:
            features.append(float(x))
        featuresData.append(features)
    
    return featuresData, intLabels


def build_dict(data):
    myData = data.tolist() 
 
    dictList = []
    keys = myData[0][:-1]
    for item in range(1, len(myData)):
        dictFeature = {}
        count1 = 0
        for key in keys:
            dictFeature[key] = float(myData[item][count1])
            count1 += 1
        dictList.append(dictFeature)

    dictTrainingLabels = {}
    count2 = 0
    for item in range(1, len(myData)):
        dictTrainingLabels[count2] = int(myData[item][-1])
        count2 += 1
    
    return dictList, dictTrainingLabels