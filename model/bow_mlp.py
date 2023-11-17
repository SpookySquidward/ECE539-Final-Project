# Credit to: https://www.kaggle.com/code/eliotbarr/text-mining-with-sklearn-keras-mlp-lstm-cnn
import re

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import random


import matplotlib.pyplot as plt
from matplotlib import cm
import mlp_helper
plt.style.use('ggplot')


def mlp_data(train_path: Path, test_path: Path):
    """Trains the data through an mlp"""
    train_dataframe = pd.read_csv(train_path)
    test_dataframe = pd.read_csv(test_path)

    train_1hot = mlp_helper.to1hot(train_dataframe)
    train_data = torch.tensor(train_1hot)


    print(train_data)
    #test_data=torch.tensor(test_data)

    return 0



def main():
    # parent directory
    project_root = Path(__file__).parent.parent
    raw_train_path = project_root.joinpath("dataset", "1formatted_train.csv")
    raw_test_path = project_root.joinpath("dataset", "1formatted_val.csv")

    # Reformat data
    mlp_train = mlp_data(raw_train_path, raw_test_path)

    # Write to csv
    #mlp_train.to_csv(project_root.joinpath("dataset", "mlp_train.csv"), index=False)

    # Check sizes
    print("--> Running mlp_data")
    rows, columns = mlp_train.shape
    print("\nDataframe shape for train\nrows: ", rows)



if __name__ == "__main__":
    main()