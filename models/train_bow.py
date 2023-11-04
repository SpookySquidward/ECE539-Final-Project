from pathlib import Path  # https://realpython.com/python-pathlib/
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def train_bow(train_path: Path) -> LogisticRegression:
    """
    Train a logistic regression on data from input path.
    :param train_path: path to a CSV file of reviews
    :return: a logistic regression model
    """
    train_df = pd.read_csv(train_path)

    # Get data
    text_train = train_df[["title", "body"]]
    y_train = train_df["sentiment"]

    # The following code is primarily from https://www.kaggle.com/code/kashnitsky/topic-4-linear-models-part-4-pros-cons
    # which was used as a reference in our proposal.
    cv = CountVectorizer()
    text_train = text_train["body"]
    cv.fit(text_train)  # Figure out later how we handle titles
    x_train = cv.transform(text_train)

    # Train a logistic regression on the transformed data
    log_reg = LogisticRegression(solver="lbfgs", n_jobs=-1, random_state=7)
    log_reg.fit(x_train, y_train)

    return log_reg
