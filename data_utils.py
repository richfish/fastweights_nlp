from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

def prepare_datasets(snli_client):
    if snli_client.test_only:
        print("Preparing OOS test data.")
        _, test = snli_client.prepare_only_val_test()
        snli_client.processed["test"].append(test[0])
        snli_client.processed["test"].append(test[1])
        snli_client.processed["test"].append(test[2])
    else:
        print("Preparing datasets incl. training, this may take a while")
        if snli_client.use_small_file:
            print("Using small train file for testing purposes")
        training, validation, _ = snli_client.prepare_datasets()
        snli_client.processed["train"].append(training[0])
        snli_client.processed["train"].append(training[1])
        snli_client.processed["train"].append(training[2])
        snli_client.processed["val"].append(validation[0])
        snli_client.processed["val"].append(validation[1])
        snli_client.processed["val"].append(validation[2])
        print("Done preparing data for training")

def get_batch(snli_client, batch_size, mode="train"):
    x_set1 = snli_client.processed[mode][0]
    x_set2 = snli_client.processed[mode][1]
    y_set = snli_client.processed[mode][2]
    return _stochastically_build_batch_non_concat(x_set1, x_set2, y_set, batch_size)

def get_amount_data(snli_client, mode="train"):
    return len(snli_client.processed[mode][0])

def _stochastically_build_batch_non_concat(x_set1, x_set2, y_set, batch_size):
    """
    Default
    """
    batchx1 = []
    batchx2 = []
    batchy =[]
    for _ in range(batch_size):
        idx = random.randint(0,len(x_set1)-1)
        batchx1.append(x_set1[idx])
        batchx2.append(x_set2[idx])
        batchy.append(y_set[idx])
    return batchx1, batchx2, batchy

def _stochastically_build_batch_concat(x_set, y_set, batch_size):
    """
    For running version with pre-concatenation
    """
    batchx = []
    batchy =[]
    for _ in range(batch_size):
        idx = random.randint(0,len(x_set)-1)
        batchx.append(x_set[idx])
        batchy.append(y_set[idx])
    return batchx, batchy

def batch_with_embed(batch_list):
    """
    Optionally convert to GloVe manually rather than assignment
    """
    new_batch = []
    for seq in batch_list:
        new_seq = []
        for i in seq:
            new_seq.append(embedding_matrix[i])
        new_batch.append(new_seq)
    return new_batch
