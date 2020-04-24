import numpy as np
import sys
import pickle
from helpers import getStats, kmer_count
from sklearn.preprocessing import normalize

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
from joblib import dump, load

dataset = sys.argv[1]
TrainDataFile = '../data/'+ dataset +'/train.p'

train = pickle.load(open(TrainDataFile, "rb"))
unique_labels = list(set(map(lambda x: x[0], train)))

# get some stats about training and testing dataset
diTrain = getStats(train)

# Create Labels for Classes
diLabels = {}
classId = 0
numClasses = len(diTrain)

for item in diTrain:
    classId += 1
    diLabels[item] = classId

## Let's delete all the missrepresented classes.

# Select the minimun amount of elements per class
minimum = 20

# Create Labels for Classes
diLabels = {}
classId = 0
numClasses = len(diTrain)

i=0
while i < len(train):
    if diTrain[train[i][0]] < minimum:
        train.pop(i)
    else:
        i+=1

#get some stats about the training dataset
diTrain = getStats(train)

n_train = len(train)
k = int(sys.argv[2])
train_features = []
train_labels = []
a_size = int(np.sqrt(4 ** k))

unique_labels = list(set(map(lambda x: x[0], train)))
numClasses = len(unique_labels)

for i in range(len(train)):
    t = kmer_count(train[i][1], k)
    t = np.array(t)
    t = normalize(t.reshape(1, -1)).reshape(-1)  # Normalizado Maricas !!!!!
    train_features.append(t)
    train_labels.append(train[i][0])

x_train = np.asarray(train_features).astype('float32')
y_train = np.asarray(train_labels)

TestDataFile = '../data/'+ dataset +'/test.p'
test = pickle.load(open(TestDataFile, "rb"))

test_features = []
test_labels = []

for i in range(len(test)):
    t = kmer_count(test[i][1], k)
    t = np.array(t)
    t = normalize(t.reshape(1, -1)).reshape(-1)  # Normalizado Maricas !!!!!
    test_features.append(t)
    test_labels.append(test[i][0])

x_test = np.asarray(test_features).astype('float32')
y_test = np.asarray(test_labels)

# This is the machine Learning Pipeline, taken from Kameris
#-----------------------------------------------
def build_pipeline(num_features):

    normalize_features = True
    dim_reduce_fraction = 0.1

    # setup normalizers if needed
    normalizers = []

    normalizers.append(('scaler', StandardScaler(with_mean=False)))

    # reduce dimensionality to some fraction of its original
    # normalizers.append(('dim_reducer',TruncatedSVD(n_components=int(
    #                     np.ceil(num_features * dim_reduce_fraction)))))

    # Classifier
    normalizers.append(('classifier',LinearSVC(penalty='l2')))

    return Pipeline(normalizers)

# Build, train and test the classifier.
print(x_train.shape,y_train.shape)
pipeline = build_pipeline(1024)
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

print(y_test)
print(y_pred)

print("predicted images size :")
print(y_pred.shape)

print(accuracy_score(y_test, y_pred))

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

model_name = 'saved_models/'+ dataset + '_' + str(k) + '.joblib'
dump(pipeline,model_name)
