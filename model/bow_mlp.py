# Credit to: https://www.kaggle.com/code/eliotbarr/text-mining-with-sklearn-keras-mlp-lstm-cnn
import re

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import random

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from matplotlib import cm
import mlp_helper
import csv
plt.style.use('ggplot')


def converter(train_path: Path, test_path: Path, num_epochs):
    """Converts from a dataframe to tensor"""
    project_root = Path(__file__).parent.parent
    train_dataframe = pd.read_csv(train_path)
    test_dataframe = pd.read_csv(test_path)
    
    split = train_dataframe.shape[0] / num_epochs
    split_test = test_dataframe.shape[0] / num_epochs
    split = int(split)
    split_test = int(split_test)

    for epoch in range(num_epochs):
        train_1hot = mlp_helper.to1hot(train_dataframe[:split])
        if epoch == 0:
            train_data = torch.Tensor(train_1hot.values)
        else:
            train_data = torch.cat((train_data, torch.tensor(train_1hot.values)), 1)
        train_dataframe = train_dataframe[split:]

        test_1hot = mlp_helper.to1hot(test_dataframe[:split_test])
        if epoch == 0:
            test_data = torch.Tensor(test_1hot.values)
        else:
            test_data = torch.cat((test_data, torch.tensor(test_1hot.values)), 1)
        test_dataframe = test_dataframe[split_test:]
    
    return train_data, test_data

def mlp(train_data, test_data, lr, num_epochs, batch_size):
    """Trains the data through an mlp"""

    model = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=lr)
    

    for epoch in range(num_epochs):
        # Train for one epoch
        losses = []
        for X_batch, y_batch in mlp_helper.data_iter(batch_size=batch_size, db=train_data):
                # Reformat data
            # Use model to compute predictions
            yhat = model(X_batch)
            l = mlp_helper.cross_entropy(yhat, y_batch)  # Minibatch loss in `X_batch` and `y_batch`

            # Compute gradients by back propagation
            l.backward()

            # Update parameters using their gradient
            mlp_helper.sgd(model, lr, 0.9)

            losses.append(l.detach().item())

        # Measure accuracy on the test set
        acc = []
        for X_batch, y_batch in mlp_helper.data_iter(batch_size=16, db=test_data):
            yhat = model(X_batch)
            acc.append(mlp_helper.accuracy(yhat, y_batch))

        print(f"Epoch {epoch+1}: Train Loss {np.mean(losses):.3f} Test Accuracy {np.mean(acc):.3f}", flush=True)



    print(train_data)
    #test_data=torch.tensor(test_data)

    return 0



def main():
    # parent directory
    project_root = Path(__file__).parent.parent
    raw_train_path = project_root.joinpath("dataset", "1formatted_train.csv")
    raw_test_path = project_root.joinpath("dataset", "1formatted_val.csv")

    # Hyperparameters
    num_epochs = 50
    lr = 0.01
    batch_size = 16

    train_data, test_data = converter(raw_train_path, raw_test_path,num_epochs)

    print("Training Data Shape: ", train_data.shape)

    #mlp(train_data, test_data, lr, num_epochs, batch_size)



if __name__ == "__main__":
    main()