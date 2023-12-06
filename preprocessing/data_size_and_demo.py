import pandas as pd
from pathlib import Path
import datetime


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    train_path = project_root.joinpath("dataset", "formatted_train.csv")
    test_path = project_root.joinpath("dataset", "formatted_test.csv")
    val_path = project_root.joinpath("dataset", "formatted_val.csv")
    og_train_path = project_root.joinpath("dataset", "train.csv")
    og_test_path = project_root.joinpath("dataset", "test.csv")

    # Get dataframes
    train_dataframe = pd.read_csv(train_path)
    test_dataframe = pd.read_csv(test_path)
    val_dataframe = pd.read_csv(val_path)
    og_train_dataframe = pd.read_csv(og_train_path)
    og_test_dataframe = pd.read_csv(og_test_path)

    # Finding dataset information
    file_path = project_root.joinpath("metrics", "dataset_sizes.txt")
    current_time = datetime.datetime.now()

    with open(file_path, "a") as size_file:
        # Writing data to a file
        size_file.write(f'{current_time}')
        size_file.write(f'\nOriginal Training data shape: {og_train_dataframe.shape}')
        size_file.write(f'\nOriginal Testing data shape: {og_test_dataframe.shape}')
        size_file.write(f'\nProcessed Training data shape: {train_dataframe.shape}')
        size_file.write(f'\nProcessed Testing data shape: {test_dataframe.shape}')
        size_file.write(f'\nProcessed Validation data shape: {val_dataframe.shape}')

    # Splitting up formatted data for demo purposes
    demo_train_dataframe = train_dataframe[:1000]
    demo_test_dataframe = test_dataframe[:1000]

    demo_train_dataframe.to_csv(project_root.joinpath("dataset", "demo_train.csv"), index=False)
    demo_test_dataframe.to_csv(project_root.joinpath("dataset", "demo_test.csv"), index=False)


if __name__ == "__main__":
    main()