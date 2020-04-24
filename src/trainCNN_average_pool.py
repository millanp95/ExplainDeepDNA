#This code is based on the work by A. Fabijanska and S. Grabowsk
# that can be found on: https://github.com/afabijanska/VGDC.

from keras.layers import Conv1D, MaxPooling1D, Dropout, Input, AveragePooling1D
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.models import Model
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import pickle
import configparser
import numpy as np

from helpers import genome2one_hot,getStats
import sys

# define CNN architecture

def getNetwork(maxLen, numClasses, maskSize):

    inputs = Input(shape=(maxLen,4))

    conv1 = Conv1D(filters=8, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(inputs)
    pool1 = AveragePooling1D(pool_size=(2), strides=(2), padding='valid')(conv1)
    norm1 = BatchNormalization()(pool1)

    conv2 = Conv1D(filters=16, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm1)
    pool2 = AveragePooling1D(pool_size=(2), strides=(2), padding='valid')(conv2)
    norm2 = BatchNormalization()(pool2)

    conv3 = Conv1D(filters=32, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm2)
    pool3 = AveragePooling1D(pool_size=(2), strides=(2), padding='valid')(conv3)
    norm3 = BatchNormalization()(pool3)

    conv4 = Conv1D(filters=64, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm3)
    pool4 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv4)
    norm4 = BatchNormalization()(pool4)

    # conv5 = Conv1D(filters=128, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm4)
    # pool5 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv5)
    # norm5 = BatchNormalization()(pool5)

    flat6 = Flatten()(norm4)
    dens6 = Dense(256, activation='relu')(flat6)
    drop6 = Dropout(0.4)(dens6)
    norm6 = BatchNormalization()(drop6)

    dens7 = Dense(128, activation='relu')(norm6)
    drop7 = Dropout(0.4)(dens7)
    norm7 = BatchNormalization()(drop7)

    dens8 = Dense(64, activation='relu')(norm7)
    drop8 = Dropout(0.4)(dens8)
    norm8 = BatchNormalization()(drop8)

    dens9 = Dense(numClasses, activation='softmax')(norm8)

    model = Model(inputs=inputs, outputs=dens9)

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])

    return model

#load training data
dataset = sys.argv[1]

TrainDataFile = 'data/' + dataset +'/train.p'
train = pickle.load(open(TrainDataFile, "rb"))
max_ = len(max(train, key = lambda x: len(x[1]))[1])
print ("max genome length:", max_)  # max sequence length

# get some stats about training and testing dataset
diTrain = getStats(train)

# Create Labels for Classes
diLabels = {}
classId = 0;
numClasses = len(diTrain)

for item in diTrain:
    classId += 1
    diLabels[item] = classId;

print(diLabels)

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

#prepare training data for feeding it to CNN
n_train = len(train)

train_genomes = []
train_labels = []

for i in range(len(train)):
    train_genomes.append(genome2one_hot(train[i][1], max_))
    labels = np.zeros(numClasses,dtype='float16')
    labels[unique_labels.index(train[i][0])] = 1
    train_labels.append(labels)

print('Shape of data tensor:', np.asarray(train_genomes).shape)
print('Shape of label tensor:', np.asarray(train_labels).shape)

x = np.reshape(np.asarray(train_genomes), (n_train, max_, 4)).astype('float16')
y = np.reshape(np.asarray(train_labels), (n_train, numClasses))

#Training params
maskSize = 7
poolStrides = 2
batchSize = 50
numEpochs = 100
valSplit = 0.1
modelJsonFile = 'data/'+ dataset +'/architecture_avg_pool.json'
modelVisFile =  'data/'+ dataset +'/model_avg_pool.png'
bestWeightsFile =  'data/'+ dataset + '/best_weights5_avg_pool.h5'
lastWeightsFile =  'data/'+ dataset +'/last_weights5_avg_pool.h5'

#Build the Network
model = getNetwork(max_, numClasses, maskSize)

plot_model(model, to_file=modelVisFile, show_shapes='True')
json_string = model.to_json()
open(modelJsonFile, 'w').write(json_string)
checkpointer = ModelCheckpoint(bestWeightsFile, verbose=0, monitor='val_loss', mode='auto', save_best_only=True)

model.fit(x, y, epochs = numEpochs, batch_size = batchSize, verbose=2, shuffle=True, validation_split=valSplit, callbacks=[checkpointer])
model.save_weights(lastWeightsFile, overwrite=True)
print(unique_labels)

