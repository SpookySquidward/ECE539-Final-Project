from pathlib import Path  # https://realpython.com/python-pathlib/
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from data_visualization.metrics import get_metrics


def create_count_vector(df: pd.DataFrame) -> CountVectorizer:
    """
    Create a count vectorizer from review data frame. Necessary to do seperately to avoid this error:
    https://stackoverflow.com/questions/62371380/logistic-regression-x-has-667-features-per-sample-expecting-74869
    :param df: data frame of reviews
    :return: count vectorizer fitted to input dataframe
    """
    text = df["title"].astype(str) + " " + df["body"].astype(str)
    cv = CountVectorizer()
    cv.fit(text)
    return cv


def train_bow_lr(train_df: pd.DataFrame, cv: None | CountVectorizer = None) -> LogisticRegression:
    """
    Train a logistic regression on data from input path.
    :param cv: input CountVectorizer (if any)
    :param train_df: dataframe of training set
    :return: a logistic regression model
    """
    if cv is None:
        cv = create_count_vector(train_df)

    x_train, y_train = format_df_to_bow(cv, train_df)

    # Train a logistic regression on the transformed data
    log_reg = LogisticRegression(solver="lbfgs", n_jobs=-1, random_state=7, max_iter=4000).fit(x_train, y_train)
    # If we get a warning wrt. max_iter, look here:
    # https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter

    return log_reg


def train_bow_mlp(train_df: pd.DataFrame, cv: None | CountVectorizer = None) -> MLPClassifier:
    """
    Train a logistic regression on data from input path.
    :param cv: input CountVectorizer (if any)
    :param train_df: dataframe of training set
    :return: an MLP model
    """
    if cv is None:
        cv = create_count_vector(train_df)

    x_train, y_train = format_df_to_bow(cv, train_df)

    # Train an MLP on the transformed data
    mlp = MLPClassifier(random_state=7, max_iter=4000).fit(x_train, y_train)

    return mlp


def format_df_to_bow(cv: CountVectorizer, df: pd.DataFrame) -> (pd.Series, pd.Series):
    """
    Function to format text to a bag of words (BoW).
    :param cv: count vectorizer fitted to training data
    :param df: dataframe with reviews
    :return: tuple of x (words/feature) and y (label)
    """
    text = df["title"].astype(str) + " " + df["body"].astype(str)
    x = cv.transform(text)
    y = df["sentiment"]

    return x, y


def main():
    # First parent is ...\ECE539-Final-Project\models, second is project root
    project_root = Path(__file__).parent.parent
    train_path = project_root.joinpath("dataset", "formatted_train.csv")
    test_path = project_root.joinpath("dataset", "formatted_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Smaller dataset
    train_df = train_df.head(25000)
    test_df = test_df.head(25000)

    # Train model
    cv = create_count_vector(train_df)
    bow_mlp_model = train_bow_mlp(train_df, cv)
    bow_lr_model = train_bow_lr(train_df, cv)

    # Get test metrics
    x_test, y_test = format_df_to_bow(cv, test_df)
    mlp_metrics_dir = project_root.joinpath("metrics", "bow_mlp_test")
    Path(mlp_metrics_dir).mkdir(parents=True, exist_ok=True)
    get_metrics(bow_mlp_model, x_test, y_test, mlp_metrics_dir)

    lr_metrics_dir = project_root.joinpath("metrics", "bow_lr_test")
    Path(lr_metrics_dir).mkdir(parents=True, exist_ok=True)
    get_metrics(bow_lr_model, x_test, y_test, lr_metrics_dir)


if __name__ == "__main__":
    main()
