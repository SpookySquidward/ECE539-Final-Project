import pandas as pd
from pathlib import Path


def main():
    # parent directory
    project_root = Path(__file__).parent
    raw_train_path = project_root.joinpath("dataset", "formatted_train.csv")
    raw_test_path = project_root.joinpath("dataset", "formatted_test.csv")
    raw_val_path = project_root.joinpath("dataset", "formatted_val.csv")
    train_dataframe = pd.read_csv(raw_train_path)
    test_dataframe = pd.read_csv(raw_test_path)
    val_dataframe = pd.read_csv(raw_val_path)

    og_train_path = project_root.joinpath("dataset", "train.csv")
    og_test_path = project_root.joinpath("dataset", "test.csv")
    og_train_dataframe = pd.read_csv(og_train_path)
    og_test_dataframe = pd.read_csv(og_test_path)

    # Finding dataset information
    print("Original Training data shape: ", og_train_dataframe.shape)
    print("Original Testing data shape: ", og_test_dataframe.shape)
    print("Processed Training data shape: ", train_dataframe.shape)
    print("Processed Testing data shape: ", test_dataframe.shape)
    print("Processed Validation data shape: ", val_dataframe.shape)

    # Splitting up data for demo purposes
    demo_train_dataframe = train_dataframe[:1000] # 60%
    demo_test_dataframe = test_dataframe[:1000] #20%
    val_percent = demo_test_dataframe.shape[0] / (demo_test_dataframe.shape[0] + demo_train_dataframe.shape[0])
    print("Processed Training data shape: ", demo_train_dataframe.shape)
    print("Processed Testing data shape: ", demo_test_dataframe.shape)

    demo_train_dataframe.to_csv(project_root.joinpath("dataset", "demo_train.csv"), index=False)
    demo_test_dataframe.to_csv(project_root.joinpath("dataset", "demo_test.csv"), index=False)


if __name__ == "__main__":
    main()