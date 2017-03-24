import numpy as np
from Bio import SeqIO
import os
import copy
import pickle

cwd = os.getcwd()

# needs to be able to return a class that behaves in the same way as the tensorflow mnist class
# input sequences must be encoded as sequences of one hot embeddings, output classes must be again encoded as one hot


def read_data(max_length):
    # Define file names
    train_dir = cwd + '\\data\\train\\'
    test_dir = cwd + '\\data\\test\\'
    cyto = train_dir + 'cyto.fasta'
    mito = train_dir + 'mito.fasta'
    nucleus = train_dir + 'nucleus.fasta'
    secreted = train_dir + 'secreted.fasta'
    test = test_dir + 'blind.fasta'

    # Load cyto
    cyto_seqs, cyto_labels, cyto_seq_lengths = load_fasta(cyto, 0)

    # Load mito
    mito_seqs, mito_labels, mito_seq_lengths = load_fasta(mito, 1)

    # Load nucleus
    nucleus_seqs, nucleus_labels, nucleus_seq_lengths = load_fasta(nucleus, 2)

    # Load secreted
    secreted_seqs, secreted_labels, secreted_seq_lengths = load_fasta(secreted, 3)

    # Load test
    test_seqs, _, test_seq_lengths, test_identifiers = load_fasta(test, None, identifier=True)

    # Combine
    train_seqs = cyto_seqs + mito_seqs + nucleus_seqs + secreted_seqs
    train_labels = combine_tensors([cyto_labels, mito_labels, nucleus_labels, secreted_labels], dim=0)
    train_seq_lengths = combine_tensors([cyto_seq_lengths, mito_seq_lengths, nucleus_seq_lengths, secreted_seq_lengths],
                                        dim=0)

    # Perform further processing
    # Process to shorten sequence lengths
    train_seq_lengths_copy = copy.copy(train_seq_lengths)
    test_seq_lengths_copy = copy.copy(test_seq_lengths)
    shortened_train_seqs, shortened_train_seq_lengths = shorten_seqs(train_seqs, train_seq_lengths_copy, max_length, mode='symmetric')
    shortened_test_seqs, shortened_test_seq_lengths = shorten_seqs(test_seqs, test_seq_lengths_copy, max_length, mode='symmetric')

    # Convert sequence strings to numpy array of indexes
    train_inputs = str2index(shortened_train_seqs)
    test_inputs = str2index(shortened_test_seqs)

    # Compute molecular weights
    train_molecular_weights = calc_molecular_weight(train_seqs)
    test_molecular_weights = calc_molecular_weight(test_seqs)

    # Create hydropathy sequences
    train_hydropathy_seqs = create_hydropathy_seq(shortened_train_seqs, max_length)
    test_hydropathy_seqs = create_hydropathy_seq(shortened_test_seqs, max_length)

    # Create isoelectric point sequences
    train_isoelectric_point_seqs = create_isoelectric_point_seq(shortened_train_seqs, max_length)
    test_isoelectric_point_seqs = create_isoelectric_point_seq(shortened_test_seqs, max_length)

    # Create pk1 sequences
    train_pk1_seqs = create_pk1_seq(shortened_train_seqs, max_length)
    test_pk1_seqs = create_pk1_seq(shortened_test_seqs, max_length)

    # Create pk2 sequences
    train_pk2_seqs = create_pk2_seq(shortened_train_seqs, max_length)
    test_pk2_seqs = create_pk2_seq(shortened_test_seqs, max_length)

    # Shuffle
    shuffle_ids = np.random.permutation(len(train_inputs))
    train_inputs = train_inputs[shuffle_ids]
    train_labels = train_labels[shuffle_ids]
    shortened_train_seq_lengths = shortened_train_seq_lengths[shuffle_ids]
    train_seq_lengths = train_seq_lengths[shuffle_ids]
    train_molecular_weights = train_molecular_weights[shuffle_ids]
    train_hydropathy_seqs = train_hydropathy_seqs[shuffle_ids]
    train_isoelectric_point_seqs = train_isoelectric_point_seqs[shuffle_ids]
    train_pk1_seqs = train_pk1_seqs[shuffle_ids]
    train_pk2_seqs = train_pk2_seqs[shuffle_ids]

    # Create data object
    data = DataContainer([train_inputs, train_labels, shortened_train_seq_lengths,
                          train_seq_lengths, train_molecular_weights, train_hydropathy_seqs,
                          train_isoelectric_point_seqs, train_pk1_seqs, train_pk2_seqs],
                         [test_inputs, shortened_test_seq_lengths, test_seq_lengths,
                          test_molecular_weights, test_hydropathy_seqs, test_isoelectric_point_seqs,
                          test_pk1_seqs, test_pk2_seqs, test_identifiers])

    return data


def shorten_seqs(seqs, seq_lengths, max_length, mode='symmetric'):

    # Go through sequences
    for i, seq in enumerate(seqs):
        if len(seq) > max_length:
            # Shorten sequence to max length using mode
            if mode is 'symmetric':
                seqs[i] = seq[:max_length//2] + seq[-max_length//2:]
            elif mode is 'left':
                seqs[i] = seq[:max_length]
            elif mode is 'right':
                seqs[i] = seq[-max_length:]
            else:
                print('mode must be symmetric, left or right')

            # Update sequence length
            seq_lengths[i] = max_length

    return seqs, seq_lengths


def calc_molecular_weight(seqs):
    # Define amino acid weight dict
    amino_acid_weight = {'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165, 'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131, 'M': 149,
                         'N': 132, 'P': 115, 'Q': 146, 'R': 174, 'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181, 'X': 0,
                         'U': 168, 'B': 133}

    # Define numpy array of zeros to fill
    molecular_weights = np.zeros([len(seqs)])

    for i, seq in enumerate(seqs):
        for amino_acid in seq:
            molecular_weights[i] += amino_acid_weight[amino_acid]

    return molecular_weights


def create_hydropathy_seq(shortened_seqs, max_seq_length):
    # Define hydropathy dict
    amino_acid_hydropathy = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9,
                             'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3, 'X': 0,
                             'U': 0, 'B': 0}

    # Define numpy array of zeros to be filled
    hydropathy_seq = np.zeros([len(shortened_seqs), max_seq_length])

    for i, seq in enumerate(shortened_seqs):
        for j, amino_acid in enumerate(seq):
            hydropathy_seq[i, j] = amino_acid_hydropathy[amino_acid]

    return hydropathy_seq


def create_isoelectric_point_seq(shortened_seqs, max_seq_length):
    # Define hydropathy dict
    amino_acid_isoelectric_point = {'A': 6.01, 'C': 5.05, 'D': 2.85, 'E': 3.15, 'F': 5.49, 'G': 2.35, 'H': 7.6, 'I': 6.05, 'K': 9.6, 'L': 6.01, 'M': 5.74,
                                    'N': 5.41, 'P': 6.3, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.6, 'V': 6.0, 'W': 5.89, 'Y': 5.64, 'X': 7.0,
                                    'U': 5.47, 'B': 7.0}

    # Define numpy array of zeros to be filled
    isoelectric_point_seq = np.zeros([len(shortened_seqs), max_seq_length])

    for i, seq in enumerate(shortened_seqs):
        for j, amino_acid in enumerate(seq):
            isoelectric_point_seq[i, j] = amino_acid_isoelectric_point[amino_acid]

    return isoelectric_point_seq


def create_pk1_seq(shortened_seqs, max_seq_length):
    # Define hydropathy dict
    amino_acid_pk1 = {'A': 2.35, 'C': 1.92, 'D': 1.99, 'E': 2.1, 'F': 2.2, 'G': 2.35, 'H': 1.8, 'I': 2.32, 'K': 2.16, 'L': 2.33, 'M': 2.13,
                      'N': 2.14, 'P': 1.95, 'Q': 2.17, 'R': 1.82, 'S': 2.19, 'T': 2.09, 'V': 2.39, 'W': 2.46, 'Y': 2.2, 'X': 2.14,
                      'U': 1.91, 'B': 2.14}

    # Define numpy array of zeros to be filled
    pk1_seq = np.zeros([len(shortened_seqs), max_seq_length])

    for i, seq in enumerate(shortened_seqs):
        for j, amino_acid in enumerate(seq):
            pk1_seq[i, j] = amino_acid_pk1[amino_acid]

    return pk1_seq


def create_pk2_seq(shortened_seqs, max_seq_length):
    # Define hydropathy dict
    amino_acid_pk2 = {'A': 9.87, 'C': 10.7, 'D': 9.9, 'E': 9.47, 'F': 9.31, 'G': 9.78, 'H': 9.33, 'I': 9.76, 'K': 9.06, 'L': 9.74, 'M': 9.28,
                      'N': 8.72, 'P': 10.64, 'Q': 9.13, 'R': 8.99, 'S': 9.21, 'T': 9.1, 'V': 9.74, 'W': 9.41, 'Y': 9.21, 'X': 9.54,
                      'U': 10., 'B': 9.54}

    # Define numpy array of zeros to be filled
    pk2_seq = np.zeros([len(shortened_seqs), max_seq_length])

    for i, seq in enumerate(shortened_seqs):
        for j, amino_acid in enumerate(seq):
            pk2_seq[i, j] = amino_acid_pk2[amino_acid]

    return pk2_seq


def load_fasta(filename, label_index, identifier=False):
    # Load input strings
    records = list(SeqIO.parse(filename, 'fasta'))

    num_examples = len(records)

    seqs = [records[i].seq for i in range(num_examples)]
    seq_strings = [seqs[i]._data for i in range(num_examples)]

    if identifier:
        identifiers = [records[i].id for i in range(num_examples)]

    # Create one hot labels array
    if label_index is not None:
        labels = np.zeros([num_examples, 4])
        labels[:, label_index] = 1.
    else:
        labels = None

    # Create sequence length array
    seq_lengths = [len(seq) for seq in seq_strings]

    if identifier:
        return seq_strings, labels, seq_lengths, identifiers
    else:
        return seq_strings, labels, seq_lengths


def str2onehot(stringlist):
    # Define amino acid index dict
    amino_acid_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
                        'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20,
                        'U': 21, 'B': 22}

    # Create numpy array of zeros with correct size
    seq_lengths = [len(string) for string in stringlist]
    max_seq_length = max(seq_lengths)
    num_examples = len(stringlist)
    onehot = np.zeros([num_examples, max_seq_length, 23])  # [Number of examples, seq_length, embedding_size]

    for i in range(len(stringlist)):
        string = stringlist[i]
        for j in range(len(string)):
            onehot[i, j, amino_acid_index[string[j]]] = 1.
    return onehot


def str2index(stringlist):
    # Define amino acid index dict
    amino_acid_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, 'U': 21, 'B': 22}

    # Create numpy array of zeros with correct size
    seq_lengths = [len(string) for string in stringlist]
    max_seq_length = max(seq_lengths)
    num_examples = len(stringlist)
    indexes = np.zeros([num_examples, max_seq_length])  # [Number of examples, seq_length, embedding_size]

    for i in range(len(stringlist)):
        string = stringlist[i]
        for j in range(len(string)):
            indexes[i, j] = int(amino_acid_index[string[j]])
    return indexes


def combine_tensors(tensor_list, dim=0):
    n_tensors = len(tensor_list)

    # Get shapes of tensors
    n_dims = len(np.shape(tensor_list[0]))
    shape_list = [list(np.shape(tensor)) for tensor in tensor_list]
    shape_array = np.zeros([n_tensors, n_dims])
    for i in range(n_tensors):
        shape_array[i, :] = np.shape(tensor_list[i])

    # Get required dimensions and create padded tensors
    max_required_dims = np.amax(shape_array, 0)

    padded_tensor_list = []
    for i in range(n_tensors):
        required_dims = max_required_dims
        actual_dims = shape_list[i]
        required_dims[dim] = actual_dims[dim]

        pad_list = []
        for j in range(n_dims):
            pad_list.append([0, int(required_dims[j]-actual_dims[j])])

        padded_tensor_list.append(np.pad(tensor_list[0], pad_list, 'constant'))
        del tensor_list[0]

    # Concatenate tensors
    combined_tensor = np.concatenate(padded_tensor_list, dim)

    return combined_tensor


class DataSubContainer:
    def __init__(self, inputs, labels, seq_lengths, full_seq_lengths, molecular_weights, hydropathy_seqs, isoelectric_point_seqs, pk1_seqs, pk2_seqs):
        self.inputs = inputs
        self.labels = labels
        self.seq_lengths = seq_lengths
        self.full_seq_lengths = full_seq_lengths
        self.molecular_weights = molecular_weights
        self.hydropathy_seqs = hydropathy_seqs
        self.isoelectric_point_seqs = isoelectric_point_seqs
        self.pk1_seqs = pk1_seqs
        self.pk2_seqs = pk2_seqs

    def next_batch(self, batch_size):
        inputs = self.inputs[:batch_size]
        labels = self.labels[:batch_size]
        seq_lengths = self.seq_lengths[:batch_size]
        full_seq_lengths = self.full_seq_lengths[:batch_size]
        molecular_weights = self.molecular_weights[:batch_size]
        hydropathy_seqs = self.hydropathy_seqs[:batch_size]
        isoelectric_point_seqs = self.isoelectric_point_seqs[:batch_size]
        pk1_seqs = self.pk1_seqs[:batch_size]
        pk2_seqs = self.pk2_seqs[:batch_size]

        # Roll arrays for next time
        self.inputs = np.roll(self.inputs, -batch_size, 0)
        self.labels = np.roll(self.labels, -batch_size, 0)
        self.seq_lengths = np.roll(self.seq_lengths, -batch_size, 0)
        self.full_seq_lengths = np.roll(self.full_seq_lengths, -batch_size, 0)
        self.molecular_weights = np.roll(self.molecular_weights, -batch_size, 0)
        self.hydropathy_seqs = np.roll(self.hydropathy_seqs, -batch_size, 0)
        self.isoelectric_point_seqs = np.roll(self.isoelectric_point_seqs, -batch_size, 0)
        self.pk1_seqs = np.roll(self.pk1_seqs, -batch_size, 0)
        self.pk2_seqs = np.roll(self.pk2_seqs, -batch_size, 0)

        return inputs, labels, seq_lengths, full_seq_lengths, molecular_weights, hydropathy_seqs, isoelectric_point_seqs, pk1_seqs, pk2_seqs


class DataContainer:
    def __init__(self, train, test):
        test_frac = 0.1

        train_inputs = train[0]
        train_labels = train[1]
        train_seq_lengths = train[2]
        train_full_seq_lengths = train[3]
        train_molecular_weights = train[4]
        train_hydropathy_seqs = train[5]
        train_isoelectric_point_seqs = train[6]
        train_pk1_seqs = train[7]
        train_pk2_seqs = train[8]

        test_inputs = test[0]
        test_seq_lengths = test[1]
        test_full_seq_lengths = test[2]
        test_molecular_weights = test[3]
        test_hydropathy_seqs = test[4]
        test_isoelectric_point_seqs = test[5]
        test_pk1_seqs = test[6]
        test_pk2_seqs = test[7]
        test_identifiers = test[8]

        num_inputs = len(train_inputs)

        self.num_test_examples = int(test_frac*num_inputs)
        self.num_train_examples = num_inputs - self.num_test_examples
        self.max_seq_length = max([max(train_seq_lengths), max(test_seq_lengths)])
        self.mean_seq_length = np.mean([np.mean(train_seq_lengths), np.mean(test_seq_lengths)])

        self.blind_test_inputs = test_inputs
        self.blind_test_seq_lengths = test_seq_lengths
        self.blind_test_full_seq_lengths = test_full_seq_lengths
        self.blind_test_molecular_weights = test_molecular_weights
        self.blind_test_hydropathy_seqs = test_hydropathy_seqs
        self.blind_test_isoelectric_point_seqs = test_isoelectric_point_seqs
        self.blind_test_pk1_seqs = test_pk1_seqs
        self.blind_test_pk2_seqs = test_pk2_seqs
        self.blind_test_identifiers = test_identifiers

        self.train = DataSubContainer(train_inputs[:self.num_train_examples],
                                      train_labels[:self.num_train_examples],
                                      train_seq_lengths[:self.num_train_examples],
                                      train_full_seq_lengths[:self.num_train_examples],
                                      train_molecular_weights[:self.num_train_examples],
                                      train_hydropathy_seqs[:self.num_train_examples],
                                      train_isoelectric_point_seqs[:self.num_train_examples],
                                      train_pk1_seqs[:self.num_train_examples],
                                      train_pk2_seqs[:self.num_train_examples])

        self.test = DataSubContainer(train_inputs[self.num_train_examples + 1:],
                                     train_labels[self.num_train_examples + 1:],
                                     train_seq_lengths[self.num_train_examples + 1:],
                                     train_full_seq_lengths[self.num_train_examples + 1:],
                                     train_molecular_weights[self.num_train_examples + 1:],
                                     train_hydropathy_seqs[self.num_train_examples + 1:],
                                     train_isoelectric_point_seqs[self.num_train_examples + 1:],
                                     train_pk1_seqs[self.num_train_examples + 1:],
                                     train_pk2_seqs[self.num_train_examples + 1:])
