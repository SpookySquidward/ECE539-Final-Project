# Reads our testing and training data from a csv file in the format specified for Xiang Zhang's dataset

import os
import csv
import random
import typing
import math


# This is the actual reader object
class data_reader:
    # CSV file format parameters
    csv_delimiter = ','
    csv_quotechar = '"'
    csv_doublequote = True
    csv_encoding = "utf-8"
    
    
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
        with open(csv_path, encoding=self.csv_encoding) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=self.csv_delimiter, quotechar=self.csv_quotechar, doublequote=self.csv_doublequote)
            self._data = list(csv_reader)
        
        # Reset read index
        self._read_location = 0
    
    
    def shuffle(self, random_state: int = None) -> None:
        """Once data has been loaded, this method shuffles its order so read() calls are pseudo-random

        Args:
            random_state (int, optional): If specified, defines the seed for pseudo-random shuffling. Defaults to None.
        """
        
        # Shuffle the data array in-place
        random.seed(random_state)
        random.shuffle(self._data)
        
        # Reset read index
        self._read_location = 0
    
    
    def read(self, num_lines = 1) -> list[list[str]]:
        """Read one or mode lines of data

        Args:
            num_lines (int, optional): The number of lines to read. Defaults to 1.

        Returns:
            pd.DataFrame: a list of up to num_lines data entries, with column labels "stars", "title", and "body". If fewer than num_lines data entries remain unread since the last shuffle call, returns the remaining unread lines; if all lines have been read, returns None.
        """
        
        if self._read_location < len(self._data):
            starting_index = self._read_location
            self._read_location += num_lines
            self._read_location = min(self._read_location, len(self._data))
            return self._data[starting_index : self._read_location]
        else:
            return None
    
    
    def read_epoch(self, batch_size=10) -> typing.Generator[list[list[str]], None, None]:
        for i_batch in range(math.ceil((len(self._data) - self._read_location) / batch_size)):
            yield self.read(batch_size)
        
        # # The last batch might be smaller than the rest
        # if self._read_location < len(self._data):
        #     yield self.read(len(self._data) - self._read_location)