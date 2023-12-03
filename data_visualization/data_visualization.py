#data_visualization

# Accuracy visualization
import matplotlib.pyplot as plt

# Referenced models in paper compared to testing model
model_names = ['EDA/LSTM, @shahules [13]' ,
               'EDA/LSTM, @arbazkhan971 [1]',
               'GloVe/LSTM, @reshmikad [12]' ,
               'Bag-of-Words, @madz2000 [7]',
               'Bag-of-Words, @kashnitsky [5]',
               'Bag-of-Words, @Lakshmi25npathi [6]',
               'Project Model']
accuracies = [0.75, 0.878, 0.8909, 0.913, 0.926, 0.955, 0.9]

# Plotting bar chart
plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, color='blue')
plt.xlabel('Accuracy')
plt.title('Model Comparison - Accuracy')
plt.xlim(0, 1)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()


#f1 score visualization
import matplotlib.pyplot as plt

model_names = ['EDA/LSTM, @shahules [13]' ,
               'EDA/LSTM, @arbazkhan971 [1]',
               'GloVe/LSTM, @reshmikad [12]' ,
               'Bag-of-Words, @madz2000 [7]',
               'Bag-of-Words, @kashnitsky [5]',
               'Bag-of-Words, @Lakshmi25npathi [6]',
               'Project Model']
f1_scores = [1, 1, 1, 1, 1, 1, 1]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.barh(model_names, f1_scores)
plt.xlabel('Accuracy')
plt.title('Model Comparison - Accuracy')
plt.xlim(0, 1) 
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()


# Confusion Matrix Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

true_labels = np.array([1, 0, 1, 1, 0, 1, 0, 1])
predicted_labels = np.array([1, 1, 1, 1, 0, 0, 1, 1])

conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Confusion Matrix Pie Chart Visualization
import matplotlib.pyplot as plt

# Extract values from the confusion matrix
tp, fp, fn, tn = conf_matrix[1][1], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[0][0]

# Data for the pie chart
labels = ['True Positives', 'False Positives', 'False Negatives', 'True Negatives']
sizes = [tp, fp, fn, tn]
colors = ['lightcoral', 'gold', 'lightblue', 'lightgreen']
explode = (0.1, 0, 0, 0)  # explode the 1st slice (True Positives)

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Confusion Matrix Breakdown')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Classification Report Visualization
class_report = classification_report(true_labels, predicted_labels)
print("Classification Report:\n", class_report)

# Accuracy and F1 Score
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
