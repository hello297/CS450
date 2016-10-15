from sklearn import datasets
import random
import heapq
from collections import Counter
import csv
import numpy as np

iris = datasets.load_iris()


class neuralNetworkOneLayer:
    def __init__(self, targets):
        results = []
        self.targetCount = len(set(targets))

    def train(self, inputs, targets):
        self.Nodes = []
        for i in inputs:
            for x in range(self.targetCount):
                self.Nodes.append(Node(i))

    def predict(self):
        output = []
        temp = []
        for inc, current in enumerate(self.Nodes):
            total = 0
            for weightI, inputI in enumerate(current.inputs):
                total += inputI * current.weights[weightI]
            if total >= 0:
                total = 1
            else:
                total = 0
            temp.append(total)
            if inc % 3 == 2:
                output.append(temp)
                temp = []
        return output


class Node:
    bias = 1
    def __init__(self, inputs):
        self.weights = []
        self.inputs = []
        for x in inputs:
            self.inputs.append(x)
        self.inputs.append(1)
        for x in self.inputs:
            self.weights.append(round(random.uniform(-0.5, 0.5), 2))




irisNeural = neuralNetworkOneLayer(iris.target)
irisNeural.train(iris.data, iris.target)
print(irisNeural.predict())



file = open("C:/Users/Takeshi Yangita/Desktop/2016 Fall/CS 450/neuralNetwork/pimaDiabetes.csv", "r")
reader = csv.reader(file, skipinitialspace=False)
pimaData = []
for train in reader:
    pimaData.append(np.array([float(i) for i in train]))

file = open("C:/Users/Takeshi Yangita/Desktop/2016 Fall/CS 450/neuralNetwork/pimaDiabetesTarget.csv", "r")
reader = csv.reader(file, skipinitialspace=False)
pimaTargets = []
for train in reader:
    pimaTargets.append(np.array([int(i) for i in train]))

pimaNeural = neuralNetworkOneLayer(pimaTargets)
pimaNeural.train(pimaData, pimaTargets)
print(pimaNeural.predict())