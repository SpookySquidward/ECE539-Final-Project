from pathlib import Path
from reformat_data import format_data
from train_val_split import split_in_train_and_validation

def run_preprocessing_pipeline(project_root: Path, train_path: Path, test_path: Path) -> None:
    """
    Run all preprocessing - this functions uses dataframes as intermediate representation.
    :param train_path: path to training data
    :param test_path: path to test data
    :return:
    """
    # Reformat data
    print("\n--> Running format_data")
    formatted_train = format_data(train_path)
    formatted_test = format_data(test_path)

    # Check sizes
    rows, columns = formatted_test.shape
    print("Dataframe shape for test\nExpected rows: 520000", "\nActual rows: ", rows)
    rows, columns = formatted_train.shape
    print("\nDataframe shape for train\nExpected rows: 2400000", "\nActual rows: ", rows)

    # Write to csv
    formatted_test.to_csv(project_root.joinpath("dataset", "formatted_test.csv"), index=False)

    # Split train in train and validation
    split_in_train_and_validation(formatted_train, 0.2)


def main():
    # First parent is ...\ECE539-Final-Project\preprocessing, second is project root
    project_root = Path(__file__).parent.parent
    raw_train_path = project_root.joinpath("dataset", "train.csv")
    raw_test_path = project_root.joinpath("dataset", "test.csv")

    run_preprocessing_pipeline(project_root, raw_train_path, raw_test_path)

if __name__ == "__main__":
    main()