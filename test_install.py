# Make sure all required libraries can be imported
import numpy, pandas, torch, matplotlib, sklearn
import data_reader
import os

# Open test.csv
reader = data_reader.data_reader()
csv_path = os.path.join(os.path.curdir, "dataset", "test.csv")
reader.open_csv(csv_path)

# Read a few lines
print(reader.read(5))

# Shuffle the data and read a few more lines
reader.shuffle()
print(reader.read(5))