from sklearn import datasets
import random
import math
from sklearn import preprocessing
import csv

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

    def train(self, inputs, expected):
        expResult = []

        compare = lambda a, b: len(a) == len(b) and len(a) == sum([1 for i, j in zip(a, b) if i == j])

        if expected == 0:
            expResult = [1, 0, 0]
        elif expected == 1:
            expResult = [0, 1, 0]
        else:
            expResult = [0, 0, 1]


        while True:
            tempInputs = inputs
            for current in self.layers:
                outputs = []
                current.setInputs(tempInputs)
                for i, currentNode in enumerate(current.nodes[1:]):
                    currentNode.calculate(current.inputs, current.weights[i])
                    outputs.append(currentNode.output)
                tempInputs = outputs

            for index, x in enumerate(outputs):
                if x < 0.5:
                    outputs[index] = 0
                if x > 0.5:
                    outputs[index] = round(x)

            if compare(outputs, expResult):
                print(outputs)
                return outputs

            for i, current in enumerate(reversed(self.layers)):
                for j, currentNode in enumerate(current.nodes[1:]):
                    if i == 0:
                        currentNode.calculateOutputError(expResult[j])
                    else:
                        propWeights = []
                        propErrors = []
                        for x in self.layers[len(self.layers) - i].nodes[1:]:
                            propErrors.append(x.error)
                        for x in self.layers[len(self.layers) - i].weights[1:][j]:
                            propWeights.append(x)
                        currentNode.calculateHiddenError(propWeights, propErrors)

            weightInputs = [1]
            for p in inputs:
                weightInputs.append(p)
            for layer in self.layers:
                weightOutputs = []
                for i, node in enumerate(layer.nodes[1:]):
                    for j, w in enumerate(layer.weights[i]):
                        layer.weights[i][j] = w - (0.1) * node.error * weightInputs[j]
                        weightOutputs.append(node.output)
                weightInputs = weightOutputs


    def predict(self, inputs):
        tempInputs = inputs
        for current in self.layers:
            outputs = []
            current.setInputs(tempInputs)
            for i, currentNode in enumerate(current.nodes[1:]):
                currentNode.calculate(current.inputs, current.weights[i])
                outputs.append(currentNode.output)
            tempInputs = outputs

        for index, i in enumerate(outputs):
            outputs[index] = round(i)
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
    def __init__(self):
        self.error = 0.0
        self.output = 0.0

    def calculateOutputError(self, expVal):
        self.error = self.output * (1 - self.output) * (self.output - expVal)

    def calculateHiddenError(self, weights, errors):
        totalError = 0.0
        for w, d in zip (weights, errors):
            totalError = totalError + (w * d)
        self.error = self.output * (1 - self.output) * totalError

    def calculate(self, inputs, weights):
        a = 0.0
        for i, w in zip(inputs, weights):
            a = a + (i * w)
        self.output = 1 / (1 + math.exp(-a))
        return self.output


data = (iris.data)
target = iris.target
combined = list(zip(data, target))
random.shuffle(combined)


trainData, trainTarget = zip(*combined[0:100])
testData, testTarget = zip(*combined[100:len(data)+1])

irisMultiLayer = NeuralNetworkSystem(numLayers, numNodes, len(data[0]), numOutputs)

outputs = []
irisResults = []

for z in range(5):
    for index, i in enumerate(trainData):
        print(index, " ", end="", flush=True)
        outputs.append(irisMultiLayer.train(i, trainTarget[index]))

for i in data:
    irisResults.append(irisMultiLayer.predict(i))

print("\n")
for results, target in zip(irisResults, testTarget):
    print(results, target)

"""
numLayers = int(input("Enter the number of hidden layers (Keep it relatively small): "))

numNodes = []
for i in range(numLayers):
    question = "                 Enter the number of nodes in hidden layer {}: ".format(i)
    numNodes.append(int(input(question)))

numOutputs = int(input("                            Enter the number of output nodes: "))

pimaData = preprocessing.normalize(np.array(list(csv.reader(open("C:/Users/Takeshi Yangita/Desktop/2016 Fall/CS 450/neuralNetwork/pimaDiabetes.csv", "r"), delimiter=','))).astype('float'))
pimaTargets = np.array(list(csv.reader(open("C:/Users/Takeshi Yangita/Desktop/2016 Fall/CS 450/neuralNetwork/pimaDiabetesTarget.csv", "r"), delimiter=','))).astype('int')
pimaTargets = pimaTargets.flatten()


pimaNeural = NeuralNetworkSystem(numLayers, numNodes, len(pimaData[0]), numOutputs)

pimaResults = []
for p in pimaData:
   pimaResults.append(pimaNeural.predict(p))

for p in pimaResults:
    print(p)"""