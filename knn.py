import csv
import random
import math
import operator
from sklearn.preprocessing import MinMaxScaler

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset2 = dataset
        scaler = MinMaxScaler()
        scaler.fit(dataset)
        MinMaxScaler(copy=True, feature_range=(0, 1))
        dataset = scaler.transform(dataset)
        for a in range(len(dataset)):
            dataset[a][10] = dataset2[a][10]
        for x in range(0,len(dataset)):
            for y in range(9):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length, chrom2):
    distance = 0
    for x in range(1,length):
        if chrom2[x]==1:
            distance += pow((float(instance1[x])-float(instance2[x])), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k, chrom2):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length, chrom2)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def fitnessValue(chrom):
    random.seed(2)
    trainingSet=[]
    testSet=[]
    split = 0.8
    loadDataset('glass.data', split, trainingSet, testSet)
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k, chrom)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testSet, predictions)
    return repr(accuracy)