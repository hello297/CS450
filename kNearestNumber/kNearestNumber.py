from sklearn import datasets
import random
import heapq
from collections import Counter
import csv
import numpy as np
iris = datasets.load_iris()


class kNearestNumber:

    def train(self, data, target, split):
        #Combine the data set and target and shuffle
        self.combined = list(zip(data, target))
        random.shuffle(self.combined)

        #Split into train and test sets
        self.trainData, self.trainTarget = zip(*self.combined[0:split])
        self.testData, self.testTarget = zip(*self.combined[split:len(data)+1])

    def predict(self, k):
        rotated = []
        for j in range(0, len(self.trainData[0])):
            column = [row[j] for row in self.trainData]
            rotated.append(column)

        prediction = []
        for tempPredict in self.testData:
            z_scores = []
            distances = []
            for x, mean in enumerate(tempPredict):
                std_dev = (1 / len(rotated[x]) * sum([(x_i - mean) ** 2 for x_i in rotated[x]])) ** 0.5
                z = [abs((x_i - mean) / std_dev) for x_i in rotated[x]]
                z_scores.append(z)
            for idx in range(0, len(z_scores[0])):
                distance = (0 - z_scores[0][idx])**2 + (0 - z_scores[1][idx])**2 + (0 - z_scores[2][idx])**2 + (0 - z_scores[3][idx])**2
                distances.append(distance)
            kSmallest = heapq.nsmallest(k, distances)
            kIndices = []
            for kSmall in kSmallest:
                tempIndex = distances.index(kSmall)
                kIndices.append(tempIndex)
            majority = []
            for idx in kIndices:
                    majority.append(self.trainTarget[idx])
            data = Counter(majority)
            prediction.append(data.most_common(1)[0][0])
        return prediction

    def predictCar(self, k):
        rotated = []
        for j in range(0, len(self.trainData[0])):
            column = [row[j] for row in self.trainData]
            rotated.append(column)

        prediction = []
        for tempPredict in self.testData:
            z_scores = []
            distances = []
            for x, mean in enumerate(tempPredict):
                std_dev = (1 / len(rotated[x]) * sum([(x_i - mean) ** 2 for x_i in rotated[x]])) ** 0.5
                z = [abs((x_i - mean) / std_dev) for x_i in rotated[x]]
                z_scores.append(z)
            for idx in range(0, len(z_scores[0])):
                distance = (0 - z_scores[0][idx])**2 + (0 - z_scores[1][idx])**2 + (0 - z_scores[2][idx])**2 + (0 - z_scores[3][idx])**2
                distances.append(distance)
            kSmallest = heapq.nsmallest(k, distances)
            kIndices = []
            for kSmall in kSmallest:
                tempIndex = distances.index(kSmall)
                kIndices.append(tempIndex)
            majority = []
            for idx in kIndices:
                if (isinstance(self.trainData[idx], np.ndarray)):
                    majority.append(self.trainData[idx][0])
                else:
                    majority.append(self.trainTarget[idx])
                # print(self.trainTarget[idx])
                # print(type(self.trainTarget[idx]))
            data = Counter(majority)
            prediction.append(data.most_common(1)[0][0])
        return prediction

    def compare(self, predictions):
        match = 0

        for p, a in zip(predictions, self.testTarget):
            if (p == a):
                match += 1
        return match / len(self.testTarget) * 100


knn = kNearestNumber()
knn.train(iris.data, iris.target, 100)
prediction = knn.predict(5)
percentage = knn.compare(prediction)
print("Accuracy for iris: ", percentage, "%")

######################　
file = open("C:/Users/Takeshi Yangita/Desktop/2016 Fall/CS 450/kNearestNumber/car.csv", "r")
reader = csv.reader(file, skipinitialspace=False)
carData = []
for train in reader:
    carData.append(np.array([float(i) for i in train]))

file = open("C:/Users/Takeshi Yangita/Desktop/2016 Fall/CS 450/kNearestNumber/carTargets.csv", "r")
reader = csv.reader(file, skipinitialspace=False)
carTargets = []
for train in reader:
    carTargets.append(np.array([int(i) for i in train]))

knnCars = kNearestNumber()
knnCars.train(carData, carTargets, 1152)
carPrediction = knnCars.predictCar(5)
carPercentage = knnCars.compare(carPrediction)
print("Accuracy cars: ", round(carPercentage, 2), "%")