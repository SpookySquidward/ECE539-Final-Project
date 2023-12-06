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
    raw_train_path = project_root.joinpath("dataset", "formatted_train.csv")
    raw_test_path = project_root.joinpath("dataset", "formatted_test.csv")
    raw_val_path = project_root.joinpath("dataset", "formatted_val.csv")
    train_dataframe = pd.read_csv(raw_train_path)
    test_dataframe = pd.read_csv(raw_test_path)
    val_dataframe = pd.read_csv(raw_val_path)

    og_train_path = project_root.joinpath("dataset", "train.csv")
    og_test_path = project_root.joinpath("dataset", "test.csv")
    og_train_dataframe = pd.read_csv(og_train_path)
    og_test_dataframe = pd.read_csv(og_test_path)

    # Finding dataset information
    print("Original Training data shape: ", og_train_dataframe.shape)
    print("Original Testing data shape: ", og_test_dataframe.shape)
    print("Processed Training data shape: ", train_dataframe.shape)
    print("Processed Testing data shape: ", test_dataframe.shape)
    print("Processed Validation data shape: ", val_dataframe.shape)

    val_percent = test_dataframe.shape[0] / (val_dataframe.shape[0] + test_dataframe.shape[0] + train_dataframe.shape[0])
    print("Percentage: ", val_percent)

    # Splitting up data for demo purposes
    demo_train_dataframe = train_dataframe[:1000]
    demo_test_dataframe = test_dataframe[:200] #20% of the train size.
    print("Processed Training data shape: ", demo_train_dataframe.shape)
    print("Processed Testing data shape: ", demo_test_dataframe.shape)

    demo_train_dataframe.to_csv(project_root.joinpath("dataset", "demo_train.csv"), index=False)
    demo_test_dataframe.to_csv(project_root.joinpath("dataset", "demo_test.csv"), index=False)


    # Hyperparameters
    #num_epochs = 50
    #batch_size = 16
    #lr = 0.01

    #mlp_trainer(raw_train_path, raw_test_path, num_epochs, lr, batch_size)


if __name__ == "__main__":
    main()