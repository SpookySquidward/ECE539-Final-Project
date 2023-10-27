# Make sure all required libraries can be imported
import numpy, torch, matplotlib, sklearn
import dataset
import os
import csv


# Open test.csv
reader = dataset.csv_reader()
csv_path = os.path.join(os.path.curdir, "dataset", "test.csv")
reader.open_csv(csv_path)


# Read a few lines of data
print("\nReading data from a csv:")
print(*reader.read(num_samples=5), sep='\n')


# Shuffle the data and read a few more lines
reader.shuffle()
print("\nReading csv data after shuffling:")
print(*reader.read(num_samples=5), sep='\n')


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
        stars = review.label
        # Store the label and total counts
        star_counts[stars] += 1
        star_counts["total"] += 1

print(f"\nTest dataset sample counts by star rating (label):\n{star_counts}")


# Label remapping example
reader.shuffle()

# Initialize a csv writer to save our remapped labels
writer = dataset.csv_writer()
remapped_path = os.path.join(os.path.curdir, "dataset", "remapped_test.csv")
with open(remapped_path, mode="w", encoding=dataset.csv_encoding) as csv_file:
    writer.use_csv(csv_file)

    # Read all our input data and remap the labels
    for batch in reader.read_epoch(batch_size=50):
        for review in batch:
            stars = review.label
            if stars in ["1", "2"]:
                review.label = "negative"
            elif stars == "3":
                continue
            elif stars in ["4", "5"]:
                review.label = "positive"
            writer.write_review(review)
    
        # Note: we could also use the writer.write_reviews() method to write more than one review
        # at a time:
        # writer.write_reviews(batch)

# Now we can read in the new data to make sure it was processed correctly
reader_remapped = dataset.csv_reader()
reader_remapped.open_csv(remapped_path)
print("\nFirst 10 samples of remapped dataset:")
print(*reader_remapped.read(num_samples=10), sep='\n')