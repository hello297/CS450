from sklearn import datasets
import random
import math
from sklearn import preprocessing
import heapq
from collections import Counter
import csv
import numpy as np

iris = datasets.load_iris()

numLayers = int(input("Enter the number of hidden layers (Keep it relatively small): "))

numNodes = []
for i in range(numLayers):
    question = "                 Enter the number of nodes in hidden layer {}: ".format(i)
    numNodes.append(int(input(question)))

numOutputs = int(input("                            Enter the number of output nodes: "))


class NeuralNetworkSystem:
    def __init__(self, numLayers, numNodes, numInputs, numOutputs):
        self.layers = []
        previousNodes = numInputs
        for i in range(numLayers):
            self.layers.append(Layer(numNodes[i], previousNodes))
            previousNodes = numNodes[i]
        self.layers.append(Layer(numOutputs, previousNodes))

    def predict(self, inputs):
        tempInputs = inputs
        for current in self.layers:
            outputs = []
            current.setInputs(tempInputs)
            for i, currentNode in enumerate(current.nodes[1:]):
                currentNode.calculate(current.inputs, current.weights[i])
                outputs.append(currentNode.output)
            tempInputs = outputs
        return outputs



class Layer:
    def __init__(self, numNodes, previousNodes):
        self.inputs  = []
        self.outputs = []
        self.nodes   = []
        self.weights = []

        self.nodes.append(Node())
        for x in range(numNodes):
            self.nodes.append(Node())

        for _ in range(numNodes):
            temp = []
            for _ in range(previousNodes + 1):
                temp.append(round(random.uniform(-.5, .5), 3))
            self.weights.append(temp)

    def setInputs(self, inputs):
        self.inputs = []
        self.inputs.append(1)
        for x in inputs:
            self.inputs.append(x)

class Node:
    def calculate(self, inputs, weights):
        print(inputs)
        print(weights)
        a = 0
        for i, w in zip(inputs, weights):
            print(i, w)
            a = a + (i * w)
        print("       a:      ", a)
        self.output = 1 / (1 + math.exp(-a))
        print("output: ", self.output)
        return self.output

irisData = preprocessing.normalize(iris.data)
irisMultiLayer = NeuralNetworkSystem(numLayers, numNodes, len(irisData[0]), numOutputs)

irisResults = []

for i in irisData:
    irisResults.append(irisMultiLayer.predict(i))

for results in irisResults:
    print(results)


pimaData = preprocessing.normalize(np.array(list(csv.reader(open("C:/Users/Takeshi Yangita/Desktop/2016 Fall/CS 450/neuralNetwork/pimaDiabetes.csv", "r"), delimiter=','))).astype('float'))
pimaTargets = np.array(list(csv.reader(open("C:/Users/Takeshi Yangita/Desktop/2016 Fall/CS 450/neuralNetwork/pimaDiabetesTarget.csv", "r"), delimiter=','))).astype('int')
pimaTargets = pimaTargets.flatten()


pimaNeural = NeuralNetworkSystem(numLayers, numNodes, len(pimaData[0]), numOutputs)

pimaResults = []
#for p in pimaData:
#   pimaResults.append(pimaNeural.predict(p))

#for p in pimaResults:
#    print(p)