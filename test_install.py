# Make sure all required libraries can be imported
import numpy, torch, matplotlib, sklearn
import data_reader
import os
import csv


# Open test.csv
reader = data_reader.data_reader()
csv_path = os.path.join(os.path.curdir, "dataset", "test.csv")
reader.open_csv(csv_path)


# Read a few lines of data
print(reader.read(num_samples=5))


# Shuffle the data and read a few more lines
reader.shuffle()
print()
print(reader.read(num_samples=5))


# Read a full epoch of training data
# First, re-shuffle the data (this makes sure we can see it all; read() hides data that we have
# already read since the lase shuffle() call)
reader.shuffle()

# We want to count the number of data points with each target label
star_counts = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "total": 0
}

# We can read all the data from our CSV with read_epoch()
for batch in reader.read_epoch(batch_size=100):
    for review in batch:
        # Get the label
        stars = review[0]
        # Store the label and total counts
        star_counts[stars] += 1
        star_counts["total"] += 1

print(f"\nTest dataset sample counts by star rating (label):\n{star_counts}")



# Label remapping example
reader.shuffle()

# The csv module can give us a writer in just a few lines, so use that rather than implementing our
# own custom writer
remapped_path = os.path.join(os.path.curdir, "dataset", "remapped_test.csv")
with open(remapped_path, "w") as out_file:
    writer = csv.writer(out_file, delimiter=data_reader.data_reader.csv_delimiter, quotechar=data_reader.data_reader.csv_quotechar, doublequote=data_reader.data_reader.csv_doublequote, quoting=csv.QUOTE_ALL)
    
    # Read in all our data and get rid of all 3-star reviews; remap other reviews to either "pos"
    # or "neg"
    for batch in reader.read_epoch(batch_size=100):
        for review in batch:
            stars = review[0]
            if stars in ["1", "2"]:
                writer.writerow(["neg"] + review[1:])
            elif stars == "3":
                continue
            elif stars in ["4", "5"]:
                writer.writerow(["pos"] + review[1:])
    
    # Note: the csv module also has a 

# Now we can read in the new data to make sure it was processed correctly
reader_remapped = data_reader.data_reader()
reader_remapped.open_csv(remapped_path)
print()
print(reader_remapped.read(num_samples=10))