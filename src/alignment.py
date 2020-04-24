# from Bio import pairwise2
# import numpy as np
# import pickle
# import sys
#
# #import format alignment method
# from Bio.pairwise2 import format_alignment
#
# #load training and testing data
# dataset = sys.argv[1]
#
# TrainDataFile = 'data/' + dataset +'/train.p'
# train = pickle.load(open(TrainDataFile, "rb"))
#
# TestDataFile = 'data/' + dataset +'/test.p'
# test = pickle.load(open(TestDataFile, "rb"))
#
# def top_k(sample,k):
#     scores = []
#     for i, X in enumerate(train):
#         alignments = pairwise2.align.globalms(sample[1], X[1], 2, -3, 0, 0)
#         score = alignments[0][2]
#         scores.append(score)
#
#     scores = np.array(scores)
#     top = np.argsort(scores)
#     top = top[::-1]
#
#     return top[:k]
#
# print(top_k(test[3],train))

import sys
import pickle
import os
import shutil

dataset = sys.argv[1]

TrainDataFile = 'data/' + dataset +'/train.p'
train = pickle.load(open(TrainDataFile, "rb"))
seq_dir = os.path.join('data/', dataset, 'data_seq')
train_dir = os.path.join('data/', dataset, 'train')
test_dir = os.path.join('data/', dataset, 'test')

for sample in train:
    accesion_number = sample[2]
    file = os.path.join(seq_dir,accesion_number)
    src = file + '.fasta'

    file = os.path.join(train_dir,accesion_number)
    dst = file + '.fasta'

    shutil.move(src,dst)

TestDataFile = 'data/' + dataset +'/test.p'
test = pickle.load(open(TestDataFile, "rb"))

for sample in test:
    accesion_number = sample[2]
    file = os.path.join(seq_dir, accesion_number)
    src = file + '.fasta'

    file = os.path.join(test_dir, accesion_number)
    dst = file + '.fasta'

    shutil.move(src, dst)

#blastn -query example.fasta -db dengue_train -out results.txt -outfmt "7 qacc sacc bitscore qstart qend sstart send" -max_target_seqs 5