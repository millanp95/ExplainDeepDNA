from keras.models import model_from_json
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from helpers import genome2one_hot,getStats
from itertools import product
from sklearn.preprocessing import normalize

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain

from keras.models import  Model
from keras import backend as K

batchSize = 5
#load the keras model
dataset = sys.argv[1]
keras_model_weights =  'data/'+ dataset +'/best_weights5_one_hot.h5'
keras_model_json =  'data/'+ dataset +'/architecture_one_hot.json'

model = model_from_json(open(keras_model_json).read())
model.load_weights(keras_model_weights)

#load training data
dataset = sys.argv[1]

TrainDataFile = 'data/' + dataset +'/train.p'
train = pickle.load(open(TrainDataFile, "rb"))
max_ = len(max(train, key = lambda x: len(x[1]))[1])
print ("max genome length:", max_)  # max sequence length

unique_labels = sorted(set(map(lambda x: x[0], train)))
numClasses = len(unique_labels)

# get some stats about training and testing dataset
diTrain = getStats(train)

# Create Labels for Classes
diLabels = {}
classId = 0;
numClasses = len(diTrain)

for item in diTrain:
    classId += 1
    diLabels[item] = classId;

## Let's delete all the missrepresented classes.

# Select the minimun amount of elements per class
minimum = 20

# Create Labels for Classes
diLabels = {}
classId = 0
numClasses = len(diTrain)

i = 0
while i < len(train):
    if diTrain[train[i][0]] < minimum:
        train.pop(i)
    else:
        i += 1

# get some stats about the training dataset
diTrain = getStats(train)
unique_labels = sorted(set(map(lambda x: x[0], train)))
numClasses = len(unique_labels)

TestDataFile = 'data/' + dataset +'/test.p'

# read and prepare testing data
test = pickle.load(open(TestDataFile, "rb"))
n_test = len(test)

test_genomes = []
test_labels = []

for i in range(len(test)):
    test_genomes.append(genome2one_hot(test[i][1], max_))
    test_labels.append(test[i][0])

print('Shape of data tensor:', np.asarray(test_genomes).shape)
print('Shape of label tensor:', np.asarray(test_labels).shape)

x_test = np.reshape(np.asarray(test_genomes), (n_test, max_, 4)).astype('float16')
y_test = np.array(test_labels)

y_pred = model.predict(x_test, batch_size=batchSize, verbose=2)

print("predicted images size :")
print(y_pred.shape)
y_pred = y_pred.astype('float16')

y_pred = np.argmax(y_pred, axis=1)
y_pred = list(map(lambda x: unique_labels[x], y_pred))

p = list(unique_labels)

confusionMatrix = np.zeros((len(p), len(p)), dtype='int')

unknown = 0


for y1, y2 in zip(y_test, y_pred):
    if y1 in p and y2 in p:
        i = p.index(y1)
        j = p.index(y2)
        confusionMatrix[i, j] += 1
    else:
        unknown += 1

print("Overall Accuracy: ", sum(np.diag(confusionMatrix)) / np.sum(confusionMatrix))
print("Number of Unknown samples in the test set: ", unknown)
print("")

print("------Accuracy subdivided by classes:------ ")

for i in range(len(p)):
    print(p[i], "Number of samples:", sum(confusionMatrix[i, :]), "Accuracy: ",
          confusionMatrix[i][i] / sum(confusionMatrix[i, :]))
