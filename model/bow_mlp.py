# Credit to: https://www.kaggle.com/code/eliotbarr/text-mining-with-sklearn-keras-mlp-lstm-cnn
import re

import pandas as pd
import numpy as np
from pathlib import Path

from bs4 import BeautifulSoup

#from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('ggplot')

def mlp_data(train_path: Path):
    train_data = pd.read_csv(filepath_or_buffer=train_path, names=["rating", "title", "body"])
    
    labels, texts = [], []
    for i, line in enumerate(train_data.split("\n")):
        content = line.split()
        labels.append(content[0])
        texts.append(" ".join(content[1:]))


def main():
    # parent directory
    project_root = Path(__file__).parent.parent
    raw_train_path = project_root.joinpath("dataset", "formatted_train.csv")
    raw_test_path = project_root.joinpath("dataset", "formatted_test.csv")

    # Reformat data
    mlp_train = mlp_data(raw_train_path)

    # Write to csv
    #mlp_train.to_csv(project_root.joinpath("dataset", "mlp_train.csv"), index=False)

    # Check sizes
    print("--> Running mlp_data")
    rows, columns = mlp_train.shape
    print("\nDataframe shape for train\nExpected rows: 2400000", "\nActual rows: ", rows)



if __name__ == "__main__":
    main()