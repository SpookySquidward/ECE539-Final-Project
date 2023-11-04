from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_in_train_and_validation(path: Path, validation_size: float) -> None:
    """
    Split a CSV into a training and validation set. Two new CSV files will be created.
    :param path: path to training CSV
    :param validation_size: size of validation set
    """
    train_df = pd.read_csv(path)
    train_df, val_df = train_test_split(train_df, train_size=1 - validation_size, random_state=11)

    # Save as CSV files - could be made to overwrite formatted train
    project_root = Path(__file__).parent.parent
    train_df.to_csv(path_or_buf=project_root.joinpath("dataset", "new_formatted_train.csv"), index=False)
    val_df.to_csv(path_or_buf=project_root.joinpath("dataset", "new_formatted_val.csv"), index=False)


def main():
    # First parent is ...\ECE539-Final-Project\preprocessing, second is project root
    project_root = Path(__file__).parent.parent
    train_path = project_root.joinpath("dataset", "formatted_train.csv")

    split_in_train_and_validation(train_path, 0.2)


if __name__ == "__main__":
    main()
