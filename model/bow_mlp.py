# Credit to: https://www.kaggle.com/code/eliotbarr/text-mining-with-sklearn-keras-mlp-lstm-cnn
import re

import pandas as pd
import torch
from pathlib import Path

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from matplotlib import cm
import mlp_helper
plt.style.use('ggplot')


def mlp_trainer(train_path: Path, test_path: Path, num_epochs=50, lr=0.01, batch_size=16):
    """1hot encodes the pandas data and then runs MLPClassifier on the 1hot encoded data"""
    project_root = Path(__file__).parent.parent
    train_dataframe = pd.read_csv(train_path)
    test_dataframe = pd.read_csv(test_path)
    
    split = train_dataframe.shape[0] / batch_size
    split_test = test_dataframe.shape[0] / batch_size
    split = int(split)
    split_test = int(split_test)

    for batch in range(batch_size):
        #Converting from pandas to tensor for training and testing
        train_1hot_labels = mlp_helper.to1hot(train_dataframe['sentiment'][:split])
        train_1hot_features = mlp_helper.to1hot(train_dataframe['body'][:split])
        train_dataframe = train_dataframe[split:]

        test_1hot_labels = mlp_helper.to1hot(test_dataframe['sentiment'][:split_test])
        test_1hot_features = mlp_helper.to1hot(test_dataframe['body'][:split_test])
        test_dataframe = test_dataframe[split_test:]

    
        clf = MLPClassifier(random_state=1, max_iter=300).fit(train_1hot_features, train_1hot_labels)
        clf.predict_proba(test_1hot_features[:1])
        clf.predict(test_1hot_features[:5, :])
        clf.score(test_1hot_features, test_1hot_labels)
    



def main():
    # parent directory
    project_root = Path(__file__).parent.parent
    raw_train_path = project_root.joinpath("dataset", "1formatted_train.csv")
    raw_test_path = project_root.joinpath("dataset", "1formatted_val.csv")

    # Hyperparameters
    num_epochs = 50
    batch_size = 16
    lr = 0.01

    mlp_trainer(raw_train_path, raw_test_path, num_epochs, lr, batch_size)


if __name__ == "__main__":
    main()