from pathlib import Path
import pandas as pd
from preprocessing.reformat_data import format_data
from preprocessing.train_val_split import split_in_train_and_validation


def run_preprocessing_pipeline(train_path: Path, test_path: Path, intermediate_csvs = False) -> None:
    """
    Run all preprocessing - this functions uses dataframes as intermediate representation.
    :param train_path: path to training data
    :param test_path: path to test data
    :return:
    """
    # Reformat data
    formatted_train = format_data(train_path)
    formatted_test = format_data(test_path)

    if intermediate_csvs:
        # Write to csv
        project_root = Path(__file__).parent.parent
        formatted_train_path = project_root.joinpath("dataset", "formatted_train.csv")
        formatted_train.to_csv(formatted_train_path, index=False)
        formatted_test.to_csv(project_root.joinpath("dataset", "formatted_test.csv"), index=False)

        split_in_train_and_validation(formatted_train_path, 0.2)
    else:
        # Use dataframes directly
        split_in_train_and_validation(formatted_train, 0.2)
        split_in_train_and_validation(formatted_test, 0.2)


def main():
    # First parent is ...\ECE539-Final-Project\preprocessing, second is project root
    project_root = Path(__file__).parent.parent
    raw_train_path = project_root.joinpath("dataset", "train.csv")
    raw_test_path = project_root.joinpath("dataset", "test.csv")

    run_preprocessing_pipeline(raw_train_path, raw_test_path)

if __name__ == "__main__":
    main()