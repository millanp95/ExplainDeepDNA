import json
import sys
import os
from Bio import SeqIO
import pickle
import random

dataset = sys.argv[1]
seq_dir = os.path.join('data/', dataset, 'data_seq')
metadata_file = 'data/' + dataset + '/' + dataset + '.json'

with open(metadata_file, 'r') as f:
    dict = json.load(f)

Sequences = []
Targets = []
IDs = []

for filename in os.listdir(seq_dir):

    # Get the accession number.
    accession_number = filename.split('.')[0]
    filename = os.path.join(seq_dir,filename)

    # Read fasta file.
    files = SeqIO.parse(filename, "fasta")

    for file in files:
        seq = str(file.seq)
        seq = seq.replace('-','').upper()
        # Get label.
        for sample in dict:
            if accession_number == sample['id']:
                label = sample['subtype']

        #Getting the sub-sequences.
        # i=0
        # while i < len(seq):
        #     sub = seq[i:i+600]
        #     i+=600
        #
        #     # Add tuple to list.
        #     IDs.append(accession_number)
        #     Targets.append(label)
        #     Sequences.append(sub)

        IDs.append(accession_number)
        Targets.append(label)
        Sequences.append(seq)

#Now we divide the data into training and testing.

n = len(IDs)

train_idx = random.sample(range(n), int(n * 0.70))
test_idx = set(range(n)).difference(set(train_idx))

train = []
unlabeled = []

for i in train_idx:
    if Targets[i] == '-':  # Check Only labeled Sequences
        unlabeled.append((Targets[i], str(Sequences[i])))
    else:
        train.append((Targets[i], str(Sequences[i]),IDs[i]))

pathTrain = os.path.join('data/',dataset,'train.p')
pickle.dump(train, open(pathTrain, "wb"))

test = []

for i in test_idx:

    if Targets[i] == '-':
        unlabeled.append((Targets[i], str(Sequences[i])))
    else:
        test.append((Targets[i], str(Sequences[i]),IDs[i]))  # Check Only Labeled Sequences

pathTest = os.path.join('data/',dataset,'test.p')
pickle.dump(test,open(pathTest,"wb"))

max_ = len(max(train, key = lambda x: len(x[1]))[1])
print ("max genome length:", max_)  # max sequence length
