import re

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import random


import matplotlib.pyplot as plt
from matplotlib import cm

def to1hot(dataframe):
    """Converts a dataframe into its 1hot encodings."""
    one_hot_encoded_data = pd.get_dummies(dataframe, columns = ['sentiment', 'title', 'body']) 
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


def sgd(model, lr, m):
    """Minibatch stochastic gradient descent w/ momentum."""
    for p in model.parameters():
        try:
          p_mom = p.data - p.prev_data
        except:
          p_mom = 0
        p.prev_p = p.data
        p.data = p.data - lr * p.grad - m * p_mom
        p.grad = None

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean()