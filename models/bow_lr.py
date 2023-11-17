from pathlib import Path  # https://realpython.com/python-pathlib/
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import datetime


def create_count_vector(df: pd.DataFrame) -> CountVectorizer:
    """
    Create a count vectorizer from review data frame. Necessary to do seperately to avoid this error:
    https://stackoverflow.com/questions/62371380/logistic-regression-x-has-667-features-per-sample-expecting-74869
    :param df: data frame of reviews
    :return: count vectorizer fitted to input dataframe
    """
    text = df["body"]  # Figure out later how we handle titles
    cv = CountVectorizer()
    cv.fit(text)
    return cv


def train_bow(train_df: pd.DataFrame, cv: None | CountVectorizer = None) -> LogisticRegression:
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
    log_reg = LogisticRegression(solver="lbfgs", n_jobs=-1, random_state=7, max_iter=400).fit(x_train, y_train)
    # If we get a warning wrt. max_iter, look here:
    # https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter

    return log_reg


def format_df_to_bow(cv: CountVectorizer, df: pd.DataFrame) -> (pd.Series, pd.Series):
    """
    Function to format text to a bag of words (BoW).
    :param cv: count vectorizer fitted to training data
    :param df: dataframe with reviews
    :return: tuple of x (words/feature) and y (label)
    """
    text = df["body"]  # Figure out later how we handle titles
    x = cv.transform(text)
    y = df["sentiment"]

    return x, y


def main():
    # First parent is ...\ECE539-Final-Project\models, second is project root
    project_root = Path(__file__).parent.parent
    train_path = project_root.joinpath("dataset", "formatted_train.csv")
    val_path = project_root.joinpath("dataset", "formatted_val.csv")
    test_path = project_root.joinpath("dataset", "formatted_test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Train model
    cv = create_count_vector(train_df)
    bow_model = train_bow(train_df, cv)

    # Get accuracy
    x_train, y_train = format_df_to_bow(cv, train_df)
    x_val, y_val = format_df_to_bow(cv, val_df)
    x_test, y_test = format_df_to_bow(cv, test_df)
    print("Train accuracy: ", round(bow_model.score(x_train, y_train), 3))
    print("Validation Accuracy: ", round(bow_model.score(x_val, y_val), 3))
    print("Test Accuracy: ", round(bow_model.score(x_test, y_test), 3))

    # Writing to file
    current_time = datetime.datetime.now()

    with open("BoW_lr_res.txt", "a") as res_file:
        # Writing data to a file
        res_file.write(f'{current_time}')
        res_file.write(f'\nTrain accuracy: {round(bow_model.score(x_train, y_train), 3)}')
        res_file.write(f'\nValidation Accuracy: {round(bow_model.score(x_val, y_val), 3)}')
        res_file.write(f'\nTest Accuracy: {round(bow_model.score(x_test, y_test), 3)}\n')


if __name__ == "__main__":
    main()
