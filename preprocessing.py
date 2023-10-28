#This is for preprocessing: First will be relabeling everything and then will be splitting the data

# Freya will do relabeling here:


# Data splitting
import torch, sklearn
import data_reader
import os
import numpy as np

# Open train.csv
reader = data_reader.data_reader()
csv_path = os.path.join(os.path.curdir, "dataset", "train.csv")
reader.open_csv(csv_path)

#Separate labels and features
i = 1
data = []
data.append(reader.read(3000000))
data = np.array(data)
data.reshape((3000000,3))

print(data)


X = data[:, :-1]
y = data[:, -1]
print('num_samples, num_features', X.shape)
print('labels', np.unique(y))
print('label size', y.shape)

#Split data
from sklearn.model_selection import train_test_split
