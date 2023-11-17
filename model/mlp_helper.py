import re

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import random


import matplotlib.pyplot as plt
from matplotlib import cm

def to1hot(data):
    """Converts a dataframe into its 1hot encodings."""
    one_hot_encoded_data = pd.get_dummies(data, columns = ['sentiment', 'body']) 
    return one_hot_encoded_data

def data_iter(batch_size, features, labels):
    """Reads the dataset"""
    num_examples = len(features)

    # The examples are read at random, in no particular order
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i:i + batch_size]
        yield features[j], labels[j]


def sgd(params, grads, lr):
    """Minibatch stochastic gradient descent."""
    for p, g in zip(params, grads):
        p.data -= g.data * lr