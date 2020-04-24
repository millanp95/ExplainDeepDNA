from keras.models import model_from_json
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from helpers import genome2tabInt, getStats, genome2one_hot
from itertools import product
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import random
from joblib import dump, load
from scipy.spatial import distance

from keras.models import  Model
from keras import backend as K


# Import DeepExplain
from deepexplain.tensorflow import DeepExplain

from keras.models import  Model
from keras import backend as K

# COnditional Kamer Count.
def build_kmer_explain(explanation,seq,thr,k):
    kmerDict = {}

    for k_mer in product('ACGT',repeat=k):
        kmer = ''.join(k_mer)
        kmerDict[kmer]=0

    idx=0

    while idx < len(seq)-k:
        try:
            x = explanation[idx:idx+k]
            if sum(x > thr) > 2:
              kmerDict[seq[idx:idx+k]]+=1
        except KeyError:
            pass
        idx +=1
    return list(kmerDict.values())

# Load Model for computing the explanations.
def load_trained_model(dataset, model_type):
    if model_type == 'ascii':
        keras_model_weights = 'data/' + dataset + '/best_weights5.h5'
        keras_model_json = 'data/' + dataset + '/architecture.json'

    elif model_type == 'one_hot':
        keras_model_weights = 'data/' + dataset + '/best_weights5_one_hot.h5'
        keras_model_json = 'data/' + dataset + '/architecture_one_hot.json'

    elif model_type == 'avg_pool':
        keras_model_weights = 'data/' + dataset + '/best_weights5_avg_pool.h5'
        keras_model_json = 'data/' + dataset + '/architecture_avg_pool.json'

    else:
        pass

    model = model_from_json(open(keras_model_json).read())
    model.load_weights(keras_model_weights)

    return model

# Load data to compute the explanations.
def load_data(dataset, model_type):

    TrainDataFile = 'data/' + dataset + '/train.p'
    train = pickle.load(open(TrainDataFile, "rb"))

    max_ = len(max(train, key=lambda x: len(x[1]))[1])
    print("max genome length:", max_)  # max sequence length

    # get some stats about training and testing dataset
    diTrain = getStats(train)

    # Create Labels for Classes
    diLabels = {}
    classId = 0

    for item in diTrain:
        classId += 1
        diLabels[item] = classId

    ## Let's delete all the missrepresented classes.

    # Select the minimum amount of elements per class
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
    unique_labels = sorted(list(diTrain.keys()))
    numClasses = len(diTrain)
    print(unique_labels)

    # prepare training data for feeding it to CNN
    n_train = len(train)

    genomes = []
    labels = []
    sequences = []

    # get 1000 random samples from the training dataset.
    idx = random.sample(range(n_train), 500)
    n_train = 500

    for i in idx:
        if model_type == 'ascii':
            genomes.append(genome2tabInt(train[i][1], max_))
            width_ = 1

        elif model_type == 'one_hot':
            genomes.append(genome2one_hot(train[i][1], max_))
            width_ = 4
        elif model_type == 'avg_pool':
            genomes.append(genome2one_hot(train[i][1], max_))
            width_ = 4
        else:
            pass

        labels.append(train[i][0])
        sequences.append(train[i][1])

    print('Shape of data tensor:', np.asarray(genomes).shape)
    print('Shape of label tensor:', np.asarray(labels).shape)

    x = np.reshape(np.asarray(genomes), (n_train, max_, width_)).astype('float16')

    return x, labels, sequences, unique_labels


def explain(model, x):

    with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context

        input_tensor = model.input

        fModel = Model(inputs=input_tensor, outputs=model.layers[-1].output)
        target_tensor = fModel(input_tensor)
        print(target_tensor)

        xs = x
        ys = model.predict(xs)

        gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
        ig    = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
        dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
        elrp = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)

        explanations = [gradin,ig,dl,elrp]



    # Compare Gradient * Input with approximate Shapley Values
    # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
    # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
    # attributions_sv     = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=100)

    return explanations

# idx = {0:gradin, 1:ig, 2:dl, 3:elrp}
def compute_importance(sequences, labels, explanations, idx, unique_labels, interest_label, k):

    kmer_importance = []

    for i, explanation in enumerate(explanations[idx]):
      explanation = normalize(abs(explanation[:,0]).reshape(1,-1),norm='l2').reshape(-1)
      kmer_importance.append(build_kmer_explain(explanation,sequences[i],1e-4, k))

    kmer_importance = np.array(kmer_importance)
    print("shape of the explanation array")
    print(kmer_importance.shape)

    # We separate the explanations per class to compare them with the SVM.
    importance = [[] for x in range(len(unique_labels))]

    for i, sample in enumerate(kmer_importance):
        w = np.array(sample)
        w = normalize(w.reshape(1, -1), norm='l2').reshape(-1)
        importance[unique_labels.index(labels[i])].append(w)

    for i, label_explanation in enumerate(importance):
        label_explanation = np.array(label_explanation)
        importance[i] = np.mean(label_explanation, axis=0)

    print("Labels class:", interest_label)

    return importance[unique_labels.index(interest_label)]

###############################---------------------------------------------
# This is the part for building the Kameris Data.

def compute_kameris_explanations(dataset,k, label):

    model_name = 'Kameris Models/saved_models/'+ dataset + '_' + str(k) + '.joblib'
    pipeline = load(model_name)

    classifier = pipeline.named_steps['classifier']
    scaler = pipeline.named_steps['scaler']
    u = scaler.mean_
    s = scaler.var_
    weights = classifier.coef_

    l = list(classifier.classes_)
    idx = l.index(label)

    w = abs(weights[idx] / s)
    w = normalize(w.reshape(1, -1)).reshape(-1)

    return w

def main():
    dataset = sys.argv[1]
    k = sys.argv[2]   #[3,4,5]
    model_type = sys.argv[3]  #[ascii, one_hot, average]
    interest_label = sys.argv[4] #be careful

    model = load_trained_model(dataset, model_type)
    features, labels, sequences, unique_labels = load_data(dataset, model_type)
    explanations = explain(model,features)
    w_kmer = compute_kameris_explanations(dataset, k, interest_label)

    w_deep = []
    for i in range(4):
        dict = {0:'gradin', 1:'ig', 2:'dl', 3:'elrp'}
        w_deep.append(compute_importance(sequences, labels, explanations , i, unique_labels, interest_label, int(k)))  # idx = {0:gradin, 1:ig, 2:dl, 3:elrp}
        print("cosine :", distance.cosine(w_kmer,w_deep[i]))
        print("Manhattan :", distance.cityblock(w_kmer,w_deep[i]))
        print("Euclidean :", distance.euclidean(w_kmer,w_deep[i]))


    fig, ax = plt.subplots(1, 5)
    ax[0].set_title("Attributions linear model")
    ax[0].stem(w_kmer)
    ax[0].set_xlabel("Index in %s-mer dict" %k)

    ax[1].set_title("Attributions Grad*Int")
    ax[1].stem(w_deep[0])
    ax[1].set_xlabel("Index in %s-mer dict" %k)

    ax[2].set_title("Attributions Integrated Grads")
    ax[2].stem(w_deep[1])
    ax[2].set_xlabel("Index in %s-mer dict" %k)

    ax[3].set_title("Attributions DeepLift")
    ax[3].stem(w_deep[2])
    ax[3].set_xlabel("Index in %s-mer dict" %k)

    ax[4].set_title("Attributions eLRP")
    ax[4].stem(w_deep[3])
    ax[4].set_xlabel("Index in %s-mer dict" %k)

    plt.show()

if __name__ == '__main__':
    main()







