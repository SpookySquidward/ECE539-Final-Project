# Reads our testing and training data from a csv file in the format specified for Xiang Zhang's dataset

import numpy as np
import pandas as pd
import os


# This is the actual reader object
class data_reader:
    # CSV file format parameters
    csv_delimiter = ','
    csv_quotechar = '"'
    csv_doublequote = True
    csv_columns = ["stars", "title", "body"]
    csv_dtypes = {
        "stars": np.double,
        "title": str,
        "body": str
    }
    
    
    def __init__(self) -> None:
        """A data reader which can extract features and labels from a CSV dataset
        """
        
        # A pandas dataframe will store the CSV temporarily in RAM
        self._data = None
        self._data_length = None
        # Store the index of the next-to-read line of the dataframe
        self._read_location = 0
    
    
    def open_csv(self, csv_path: str) -> None:
        """Reads a csv file's contents

        Args:
            csv_path (str): The path to the csv file to read

        Raises:
            ValueError: if an invalid csv_path was specified
        """
        
        # Make sure we were actually passed a CSV file
        if os.path.splitext(csv_path)[1].lower() != '.csv':
            raise ValueError(f"Specified path {csv_path} is not a .csv file!")
        
        # Read the file
        self._data = pd.read_csv(csv_path, sep=self.csv_delimiter, quotechar=self.csv_quotechar, doublequote=self.csv_doublequote, header=None, names=self.csv_columns, dtype=self.csv_dtypes)
        self._data_length = self._data.shape[0]
        # Reset read index
        self._read_location = 0
    
    
    def shuffle(self, random_state: int = None) -> None:
        """Once data has been loaded, this method shuffles its order so read() calls are pseudo-random

        Args:
            random_state (int, optional): If specified, defines the seed for pseudo-random shuffling. Defaults to None.
        """
        
        self._data = self._data.sample(frac=1, random_state=random_state)
    
    
    def read(self, num_lines = 1) -> pd.DataFrame:
        """Read one or mode lines of data

        Args:
            num_lines (int, optional): The number of lines to read. Defaults to 1.

        Returns:
            pd.DataFrame: a list of up to num_lines data entries, with column labels "stars", "title", and "body". If fewer than num_lines data entries remain unread since the last shuffle call, returns the remaining unread lines; if all lines have been read, returns None.
        """
        
        if self._read_location <= self._data_length:
            return self._data.iloc[self._read_location : self._read_location + num_lines]
        else:
            return None