"""Module for handling Amazon reviews, including reading/writing to CSV files in the same format
specified for Xiang Zhang's dataset
"""

import os
import csv
import random
import typing
import math
import io


# CSV file format parameters
csv_delimiter = ','
csv_quotechar = '"'
csv_doublequote = True
csv_encoding = "utf-8"
csv_quoting = csv.QUOTE_ALL


class review:
    """A sample from one of our review datasets, including the title, body, and label of the review
    """

    def __init__(self, title: str = None, body: str = None, label: str = None) -> None:
        """Initializes a review with the specified data fields

        Args:
            title (str, optional): The title of the review. Defaults to None.
            body (str, optional): The body of the review. Defaults to None.
            label (str, optional): The review label. Defaults to None.
        """
    
        self.title = title
        self.body = body
        self.label = label
    

    def __str__(self) -> str:
        out_str = "Label: "
        if self.label:
            out_str += f'"{self.label}"'
        else:
            out_str += 'None'
        
        out_str += '; Title: '
        if self.title:
            out_str += f'"{self.title}"'
        else:
            out_str += 'None'
        
        out_str += '; Body: '
        if self.body:
            # Limit to 100 characters for easier parsing
            char_limit = 100
            remaining_chars = char_limit - len(out_str) - 5
            if remaining_chars <= 0:
                out_str += '"..."'
            else:
                out_str += f'"{self.body[:remaining_chars]}..."'
        else:
            out_str += 'None'
        
        return out_str
    

    def __repr__(self):
        return str(self)


class csv_reader:
    """A data reader which can extract features and labels from a CSV dataset
    """
    
    def __init__(self) -> None:
        """Creates a new data_reader object
        """
        # A list will store the CSV temporarily in RAM
        self._data = []
        # Store the index of the next-to-read line of the dataframe
        self._read_location = 0
    
    
    def open_csv(self, csv_path: str) -> None:
        """Reads a csv file's contents

        Args:
            csv_path (str): The path to the csv file to read

        Raises:
            ValueError: if an invalid csv_path was specified
        """

        # Re-initialize data list and read location
        self.__init__()
        
        # Make sure we were actually passed a CSV file
        if os.path.splitext(csv_path)[1].lower() != '.csv':
            raise ValueError(f"Specified path {csv_path} is not a .csv file!")
        
        # Read the file
        with open(csv_path, encoding=csv_encoding) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=csv_delimiter, quotechar=csv_quotechar, doublequote=csv_doublequote)
            
            for line_i, line_items in enumerate(csv_reader):
                # Ignore blank lines
                if len(line_items) == 0 or (len(line_items) == 1 and line_items[0].isspace()):
                    continue

                # Make sure we have a valid line ('"<label>","<title>","<body>"')
                if not len(line_items) == 3:
                    raise ValueError(f'Error on line {line_i + 1} of {csv_path}: got illegal csv line items {line_items}, expected three items in the format \'"<label>","<title>","<body>"\'.')

                # Create a new review object for each sample
                review_object = review(label=line_items[0], title=line_items[1], body=line_items[2])
                self._data.append(review_object)
    
    
    def shuffle(self, random_state: int = None) -> None:
        """Once data has been loaded, this method shuffles its order so read() calls yield data in
        a pseudo-random order

        Args:
            random_state (int, optional): If specified, defines the seed for pseudo-random
            shuffling. Defaults to None.
        """
        
        # Shuffle the data array in-place
        random.seed(random_state)
        random.shuffle(self._data)
        
        # Reset read index
        self._read_location = 0
    
    
    def read(self, num_samples = 1) -> list[review]:
        """Read one or mode lines of data

        Args:
            num_samples (int, optional): The number of samples to return at once. Defaults to 1.

        Returns:
            list[review]: a list of num_samples reviews which have not yet been read by a previous
            read() call since the last shuffle() call. If fewer than num_samples data entries
            remain unread since the last shuffle call, returns the remaining unread lines; if all
            lines have been read, returns None.
        """
        
        if self._read_location < len(self._data):
            starting_index = self._read_location
            self._read_location += num_samples
            self._read_location = min(self._read_location, len(self._data))
            return self._data[starting_index : self._read_location]
        else:
            return None
    
    
    def read_epoch(self, batch_size=10) -> typing.Iterable[list[review]]:
        """Returns as many read() call outputs as are required to read all the data which had not
        been read since initialization or the lase shuffle() call, using the requested batch size
        as the number of samples per read() request

        Args:
            batch_size (int, optional): The number of samples to return at each iteration. Defaults
            to 10.

        Yields:
            list[review]: a list of batch_size reviews. The last list of samples returned will
            contain fewer than batch_size samples if the number of unread samples is not evenly
            divisible by batch_size.
        """

        for i_batch in range(math.ceil((len(self._data) - self._read_location) / batch_size)):
            yield self.read(batch_size)


class csv_writer():
    """An object which can write reviews to a csv file
    """


    def __init__(self) -> None:
        """Creates a new csv_writer object
        """
        
        self._writer = None

    
    def use_csv(self, csv_file: io.TextIOWrapper) -> None:
        """Assigns a target csv file for writing

        Args:
            csv_file (io.TextIOWrapper): the file to write to; use `with open(<csv_path>, mode="w",
            encoding=dataset.csv_encoding) as csv_file:` to get this file
        """

        self._writer = csv.writer(csv_file, delimiter=csv_delimiter, quotechar=csv_quotechar, doublequote=csv_doublequote, quoting=csv_quoting)
    

    def _review_to_csv_list(review: review) -> list[str]:
        """Converts a review to a list of strings for writing to a CSV file

        Args:
            out_review (review): Review to be converted

        Returns:
            list[str]: a list of review fields in the form of `[<label>, <title>, <body>]`
        """

        return [review.label, review.title, review.body]


    def write_review(self, review: review) -> None:
        """Writes one review to the output csv file

        Args:
            out_review (review): The review to write
        """
        
        self._writer.writerow(csv_writer._review_to_csv_list(review))
    
    def write_reviews(self, reviews: list[review]) -> None:
        self._writer.writerows(csv_writer._review_to_csv_list(review) for review in reviews)