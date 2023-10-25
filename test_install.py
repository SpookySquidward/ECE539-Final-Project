# Make sure all required libraries can be imported
import numpy, torch, matplotlib, sklearn
import data_reader
import os

# Open test.csv
reader = data_reader.data_reader()
csv_path = os.path.join(os.path.curdir, "dataset", "test.csv")
reader.open_csv(csv_path)

# Read a few lines
print(reader.read(num_lines=5))

# Shuffle the data and read a few more lines
reader.shuffle()
print()
print(reader.read(num_lines=5))

# Read a full epoch of training data
reader.shuffle()
star_counts = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "total": 0
}
for batch in reader.read_epoch(batch_size=100):
    for review in batch:
        stars = review[0]
        star_counts[stars] += 1
        star_counts["total"] += 1

print(f"\nTest dataset sample counts by star rating (label):\n{star_counts}")