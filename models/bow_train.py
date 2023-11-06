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

    x_train, y_train = format_text_to_bow(train_path)

    # Train a logistic regression on the transformed data
    log_reg = LogisticRegression(solver="lbfgs", n_jobs=-1, random_state=7)
    log_reg.fit(x_train, y_train)

    return log_reg


def format_text_to_bow(path: Path) -> (pd.Series, pd.Series):
    """
    Function to format text to a bag of words (BoW).
    The following code is primarily from https://www.kaggle.com/code/kashnitsky/topic-4-linear-models-part-4-pros-cons
    :param path: path to CSV file to format
    :return: tuple of x (words/feature) and y (label)
    """
    df = pd.read_csv(path)
    text = df["body"]  # Figure out later how we handle titles
    cv = CountVectorizer()
    cv.fit(text)
    x = cv.transform(text)
    y = df["sentiment"]

    return x, y


def main():
    # First parent is ...\ECE539-Final-Project\preprocessing, second is project root
    project_root = Path(__file__).parent.parent
    train_path = project_root.joinpath("dataset", "formatted_train.csv")
    bow_model = train_bow(train_path)

    # Get accuracy
    test_path = project_root.joinpath("dataset", "formatted_test.csv")
    x_train, y_train = format_text_to_bow(train_path)
    x_test, y_test = format_text_to_bow(test_path)
    print("Train accuracy: ", round(bow_model.score(x_train, y_train), 3))
    print("Test Accuracy: ", round(bow_model.score(x_test, y_test), 3))


if __name__ == "__main__":
    main()
