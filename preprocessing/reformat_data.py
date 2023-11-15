import pandas as pd
from pathlib import Path
import os
from csv import QUOTE_ALL


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


def format_data(path: Path):
    """
    Format data from csv file and return dataframe.
    :param path: path to review csv file
    :return: formatted data as dataframe
    """
    df = pd.read_csv(filepath_or_buffer=path, names=["star_rating", "title", "body"])

    # From stars to sentiment
    df['star_rating'] = df['star_rating'].apply(snap_reviews)

    # Discard neutral reviews
    df = df[df['star_rating'] != "neutral"]

    # Encapsulate in quotation marks
    # df = df.map(lambda x: f'"{x}"')

    # Change header
    df = df.set_axis(['sentiment', 'title', 'body'], axis='columns')
    return df


def main():
    # parent directory
    project_root = Path(__file__).parent.parent
    raw_train_path = project_root.joinpath("dataset", "train.csv")
    raw_test_path = project_root.joinpath("dataset", "test.csv")

    # Reformat data
    formatted_train = format_data(raw_train_path)
    formatted_test = format_data(raw_test_path)

    # Write to csv
    formatted_test.to_csv(project_root.joinpath("dataset", "formatted_test.csv"), index=False, quoting=QUOTE_ALL)
    formatted_train.to_csv(project_root.joinpath("dataset", "formatted_train.csv"), index=False, quoting=QUOTE_ALL)

    # Check sizes
    print("--> Running format_data")
    rows, columns = formatted_train.shape
    print("\nDataframe shape for train\nExpected rows: 2400000", "\nActual rows: ", rows)
    rows, columns = formatted_test.shape
    print("\nDataframe shape for test\nExpected rows: 520000", "\nActual rows: ", rows)



if __name__ == "__main__":
    main()
