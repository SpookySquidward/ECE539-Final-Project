import pandas as pd
import os


def snap_reviews(rating: int):
    """
    Function to snap a rating (1-5) to either negative, neutral or positive.
    :param rating: star rating
    :return: sentiment as a string
    """
    if rating < 3:
        return "negative"
    elif rating > 3:
        return "positive"
    return "neutral"


def format_data(path: str):
    """
    Format data from csv file and return dataframe.
    :param path: path to review csv file
    :return: formatted data as dataframe
    """
    df = pd.read_csv(filepath_or_buffer=path, names=["star_rating", "headline", "body"])

    # From stars to sentiment
    df['star_rating'] = df['star_rating'].apply(snap_reviews)

    # Discard neutral reviews
    df = df[df['star_rating'] != "neutral"]

    # Change header
    df.set_axis(['sentiment', 'headline', 'body'], axis='columns')
    return df


def main():
    # parent directory
    parent = os.path.join(os.getcwd(), os.pardir)
    parent = os.path.abspath(parent)

    # Define paths of test and training
    test_path = os.path.join(parent, "dataset", "test.csv")
    train_path = os.path.join(parent, "dataset", "train.csv")

    # Paths for new files
    formatted_test_path = os.path.join(parent, "dataset", "formatted_test.csv")
    formatted_train_path = os.path.join(parent, "dataset", "formatted_train.csv")

    # Reformat data
    format_data(test_path).to_csv(path_or_buf=formatted_test_path, index=False)
    format_data(train_path).to_csv(path_or_buf=formatted_train_path, index=False)


if __name__ == "__main__":
    main()
