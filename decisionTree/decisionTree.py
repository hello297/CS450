from sklearn.cross_validation import train_test_split as tsp
from scipy import stats

import pandas
import numpy as np

def loadFile(filename):
    df = pandas.read_csv(filename)

    data = df.ix[:, df.columns != "target"]
    target = df.ix[:, df.columns == "target"]

    labels = df.columns
    nlabels = labels[:-1]

    return data.values, target.values, nlabels



def makeTree(data, classes, colNames):
    tempLabel = len(colNames)
    length = len(data)

    if len(classes[0]) == length:
        return classes[0]

    else:
        infoGain = np.zeros(tempLabel)

        for col in range(tempLabel):
            infoGain[col] = weightedAverage(data, classes, col)


        best = np.argmin(infoGain)

        val = getValues(data, best)

        tree = {colNames[best]: {}}

        tempData = []
        tempClass = []

        for tempVal in val:
            index = 0
            for dPoints in data:
                if dPoints[best] == tempVal:
                    if best == 0:
                        dPoint = dPoints[1:]
                        new = colNames[1:]
                    elif best == tempLabel:
                        dPoint = dPoints[:-1]
                        new = colNames[:-1]
                    else:
                        dPoint = dPoints[:best]
                        if isinstance(dPoint, np.ndarray):
                            dPoint = dPoint.tolist()
                        dPoint.extend(dPoints[best + 1:])

                        new = colNames[:best]
                        new.append(colNames[best + 1:])

                    tempData.append(dPoint)
                    tempClass.append(classes[index])

                index += 1
            subtree = makeTree(tempData, tempClass, new)

            tree[colNames[best]][tempVal] = subtree
        return tree

def getValues(info, col):
    values = []
    for point in info:
        if point[col] not in values:
            values.append(point[col])

    return values



def findEntropy(data):
    if data != 0:
        return -data * np.log2(data)
    else:
        return 0
    
def weightedAverage(data, classTarget, feature):
    counted = len(data)

    featureVal = getValues(data, feature)

    valCount = np.zeros(len(featureVal))
    entropy = np.zeros(len(featureVal))

    i = 0

    for v in featureVal:
        dIndex = 0
        newClasses = []

        for point in data:
            if point[feature] == v:
                valCount[i] += 1
                newClasses.append(classTarget[dIndex])

            dIndex += 1

        cVal = []
        for nClass in newClasses:

            if cVal.count(nClass) == 0:
                cVal.append(nClass)

        tempVal = np.zeros(len(cVal))

        cIndex = 0

        for temp in cVal:
            for nClass in newClasses:
                if nClass == temp:
                    tempVal[cIndex] += 1
            cIndex += 1

        for j in range(len(cVal)):
            entropy[i] += findEntropy(float(tempVal[j]) / sum(tempVal))

        weight = valCount[i] / counted
        entropy[i] = (entropy[i] * weight)
        i += 1

        return sum(entropy)

class Node:
    def __init__(self, name="", childNode={}):
        self.label = name
        self.childNode = childNode


class ID3:

    #def predict(self, traitempData, train_target, test_data):


    def train(selfs, data_set, train_target, colNames):
        a = makeTree(data_set, train_target, colNames)
        print(a)