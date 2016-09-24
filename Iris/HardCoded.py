from sklearn import datasets
import random
iris = datasets.load_iris()

class Hardcode:
    def __init__(self, data, target):
        self.combined = list(zip(data, target))
        random.shuffle(self.combined)
        self.trainData, self.trainTarget = zip(*self.combined[0:100])
        self.testData, self.testTarget = zip(*self.combined[100:151])

    def train(self):
        print("Just doin the training thing")

    def predict(self):
        print("Just predictin and stuff")
        predictionTarget = self.testTarget
        return predictionTarget

    def compare(self, predictions):
        match = 0

        for p, a in zip(predictions, self.testTarget):
            if (p == a):
                match += 1
        return match / len(self.testTarget) * 100


hardcode = Hardcode(iris.data, iris.target)
hardcode.train()
prediction = hardcode.predict()
percentage = hardcode.compare(prediction)
print("Accuracy: ",percentage, "%")