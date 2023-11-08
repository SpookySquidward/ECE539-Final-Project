from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_in_train_and_validation(train_data: Path | pd.DataFrame, validation_size: float) -> None:
    """
    Split a CSV into a training and validation set. Two new CSV files will be created.
    :param train_data: either path to training CSV OR dataframe with training data
    :param validation_size: size of validation set
    """

    if isinstance(train_data, Path):
        train_data = pd.read_csv(train_data)

    og_rows = train_data.shape[0]
    train_df, val_df = train_test_split(train_data, train_size=1 - validation_size, random_state=11)

    # Check sizes
    print("\n--> Running split_in_train_and_validation")
    rows, columns = train_df.shape
    print(f'Dataframe shape for train\nExpected rows: {og_rows * (1-validation_size)}', '\nActual rows: ', rows)
    rows, columns = val_df.shape
    print(f'\nDataframe shape for validation\nExpected rows: {og_rows * validation_size}', '\nActual rows: ', rows)

    # Save as CSV files - could be made to overwrite formatted train
    project_root = Path(__file__).parent.parent
    train_df.to_csv(project_root.joinpath("dataset", "formatted_train.csv"), index=False)
    val_df.to_csv(project_root.joinpath("dataset", "formatted_val.csv"), index=False)


def main():
    # First parent is ...\ECE539-Final-Project\preprocessing, second is project root
    project_root = Path(__file__).parent.parent
    train_path = project_root.joinpath("dataset", "formatted_train.csv")

    split_in_train_and_validation(train_path, 0.2)


if __name__ == "__main__":
    main()
