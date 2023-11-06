import pandas as pd
from pathlib import Path  # https://realpython.com/python-pathlib/


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
    # First parent is ...\ECE539-Final-Project\preprocessing, second is project root
    project_root = Path(__file__).parent.parent
    train_path = project_root.joinpath("dataset", "train.csv")
    test_path = project_root.joinpath("dataset", "test.csv")

    # Reformat data
    formatted_train = format_data(train_path)
    formatted_test = format_data(test_path)

    # Write to csv
    formatted_train.to_csv(path_or_buf=project_root.joinpath("dataset", "formatted_train.csv"), index=False)
    formatted_test.to_csv(path_or_buf=project_root.joinpath("dataset", "formatted_test.csv"), index=False)

    # Check sizes
    rows, columns = formatted_train.shape
    print("\nDataframe shape for train\nExpected rows: 2400000", "\nActual rows: ", rows)
    rows, columns = formatted_test.shape
    print("Dataframe shape for test\nExpected rows: 520000", "\nActual rows: ", rows)


if __name__ == "__main__":
    main()