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
plt.style.use('ggplot')


def converter(train_path: Path, test_path: Path, num_epochs):
    """Converts from a dataframe to tensor"""
    train_dataframe = pd.read_csv(train_path)
    test_dataframe = pd.read_csv(test_path)
    
    split = train_dataframe.shape[0] / num_epochs
    split_test = test_dataframe.shape[0] / num_epochs
    split = int(split)
    split_test = int(split_test)

    for epoch in range(num_epochs):
        train_1hot = mlp_helper.to1hot(train_dataframe[:split])
        print(train_1hot)
        #train_data = torch.tensor(train_1hot['sentiment'])
        train_dataframe = train_dataframe[split:]

        #test_1hot = mlp_helper.to1hot(test_dataframe[:split_test])
        #test_data = torch.tensor(test_1hot)
        #test_dataframe = test_dataframe[split_test:]
    
    return 0

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

    mlp_data = converter(raw_train_path, raw_test_path,num_epochs)


    # Write to csv
    #mlp_train.to_csv(project_root.joinpath("dataset", "mlp_train.csv"), index=False)

    # Check sizes
    print("--> Running mlp_data")
    rows, columns = mlp_data.shape
    print("\nDataframe shape for train\nrows: ", rows)



if __name__ == "__main__":
    main()