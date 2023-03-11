"""
Title: Project 1
Author: Crystal Atoz
"""

import numpy as np
import math
from typing import List


def Entropy(probabilities: List[float]):
    entropyReturn = 0
    for probability_i in probabilities:
        if probability_i > 0:
            entropyReturn -= probability_i*math.log2(probability_i)
    return entropyReturn


def IG(yTotal: List[int],
       yWhenXisYes: List[int],
       yWhenXisNo: List[int]):

    yTotal = np.array(yTotal)
    yWhenXisYes = np.array(yWhenXisYes)
    yWhenXisNo = np.array(yWhenXisNo)

    nTotal1 = np.count_nonzero(yTotal == 1)
    nTotal0 = np.count_nonzero(yTotal == 0)
    nTotal = len(yTotal)
    nXEqualsYes1 = np.count_nonzero(yWhenXisYes == 1)
    nXEqualsYes0 = np.count_nonzero(yWhenXisYes == 0)
    nXEqualsYes = len(yWhenXisYes)
    nXEqualsNo1 = np.count_nonzero(yWhenXisNo == 1)
    nXEqualsNo0 = np.count_nonzero(yWhenXisNo == 0)
    nXEqualsNo = len(yWhenXisNo)

    if (nTotal == 0 or nTotal1 == 0 or nTotal0 == 0):
        return 0

    entropyTotal = Entropy([nTotal1/nTotal, nTotal0/nTotal])
    entropyXEqualsYes = 0 if (nXEqualsYes1 == 0 or nXEqualsYes0 == 0) else Entropy(
        [nXEqualsYes1/nXEqualsYes, nXEqualsYes0/nXEqualsYes])
    entropyXEqualsNo = 0 if (nXEqualsNo0 == 0 or nXEqualsNo0 == 0) else Entropy(
        [nXEqualsNo0/nXEqualsNo, nXEqualsNo0/nXEqualsNo])

    IG = entropyTotal - \
        (nXEqualsYes/nTotal*entropyXEqualsYes) - \
        (nXEqualsNo/nTotal*entropyXEqualsNo)
    return IG


def igXY(single_X, Y):
    xEqualsYes = np.array(Y[single_X == 1]).flatten()
    xEqualsNo = np.array(Y[single_X == 0]).flatten()
    return IG(Y, xEqualsYes, xEqualsNo)


def igAllProvided(X_in_order, Y):
    igAllProvidedGiven = []
    for x in X_in_order:
        ig = igXY(x, Y)
        igAllProvidedGiven.append(ig)
    return igAllProvidedGiven


def xToBranch(igAllProvidedGiven, remaining_x):
    maxIGIndex = np.argmax(igAllProvidedGiven)
    return remaining_x[maxIGIndex]


def maxFrequencyNP(np_array):
    values, counts = np.unique(np_array, return_counts=True)
    return (values[np.argmax(counts)])


def conditionToBranch(X, Y, x_remaining, no_of_features, max_depth):
    if max_depth <= no_of_features-len(x_remaining) or len(x_remaining) <= 0:
        return False
    igAll = igAllProvided(X, Y)
    if max(igAll) == 0:
        return False
    return igAll


def removeFeatureToBranch(X, Y, x_remaining):
    igAll = igAllProvided(X, Y)
    indexToBranch = np.argmax(igAll)
    featureToBranch = x_remaining[indexToBranch]
    x_remaining = x_remaining[x_remaining != featureToBranch]

    y1 = Y[X[indexToBranch] > 0]
    y0 = Y[X[indexToBranch] <= 0]
    x0 = []
    x1 = []
    for xes_in_given_X in range(len(X)):
        if xes_in_given_X != indexToBranch:
            x0.append(X[xes_in_given_X][X[indexToBranch] <= 0])
            x1.append(X[xes_in_given_X][X[indexToBranch] > 0])
    return [{"X": x0, "Y": y0, "x_remaining": np.array(x_remaining).flatten()},
            {"X": x1, "Y": y1, "x_remaining": np.array(x_remaining).flatten()}]


def branchCreate(X, Y, x_remaining, no_of_features, max_depth):
    condition = conditionToBranch(
        X, Y, x_remaining, no_of_features, max_depth)
    if condition:
        maxIGIndex = np.argmax(condition)
        featureToBranch = x_remaining[maxIGIndex]
        branching_step = removeFeatureToBranch(
            X, Y, x_remaining)
        return {"X_po": featureToBranch,
                "div": [branchCreate(branching_step[0]["X"], branching_step[0]["Y"], branching_step[0]["x_remaining"], no_of_features, max_depth), branchCreate(branching_step[1]["X"], branching_step[1]["Y"], branching_step[1]["x_remaining"], no_of_features, max_depth)]}
    else:
        return {"X_po": None, "div": maxFrequencyNP(Y)}


def DT_train_binary(X, Y, max_depth):
    no_of_features = len(X[0])
    max_depth = min(no_of_features, no_of_features if (
        max_depth == -1) else max_depth)

    initial_remaining_x = np.arange(no_of_features)
    X = np.array(X).T
    Y = np.array(Y)
    DT = branchCreate(X, Y, initial_remaining_x, no_of_features, max_depth)
    return DT


def DT_test_binary(X, Y, DT):
    predictionY = np.array(
        [DT_make_prediction(x_instance, DT) for x_instance in X])
    predictionCorrect = np.sum(Y == predictionY)
    testDataCountTotal = len(Y)
    accuracy = (predictionCorrect/testDataCountTotal)
    return accuracy


def DT_make_prediction(x, DT):
    X_po = DT["X_po"]
    div = DT["div"]
    while X_po != None:
        if x[X_po] <= 0:
            X_po = div[0]["X_po"]
            div = div[0]["div"]
        else:
            X_po = div[1]["X_po"]
            div = div[1]["div"]
    return div


"""
def RF_build_random_forest(X, Y, max_depth, num_ofTrees):
    random_forest_sub_tree = []

    for i in range(num_ofTrees):
        sample = data(frac=1, replace=True)

        X = removeFeatureToBranch(sample)[0]
        Y = removeFeatureToBranch(sample)[1]

        random_forest_sub_tree.append(conditionToBranch(X, Y,
                                      X.drop(labels=['target'], axis=1).columns))

    return random_forest_sub_tree


def RF_test_random_forest(X, Y, RF):
    data['predictions'] = None
    for i in range(len(data)):
        query = data.iloc[i, :].drop('target').to_dict()
        data.loc[i, 'predictions'] = RandomForest_Predict(
            query, random_forest, default='p')
    accuracy = sum(data['predictions'] == data['target'])/len(data)*100

    return accuracy
"""
