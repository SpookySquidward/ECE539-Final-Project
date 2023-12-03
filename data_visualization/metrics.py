import pandas as pd
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from pathlib import Path


def graphical_confusion_matrix(conf_matrix: confusion_matrix, save_dir: Path) -> None:
    """
    Create a graphical confusion metric and save it in save_dir.
    :param conf_matrix: confusion matrix
    :param save_dir: directory to save file in
    """
    fig = plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    fig.savefig(save_dir.joinpath("graphical_confusion_matrix.png"))


def pie_chart_confusion_matrix(conf_matrix: confusion_matrix, save_dir: Path) -> None:
    """
    Create a pie chart of the confusion matrix and save it in save_dir.
    :param conf_matrix: confusion matrix
    :param save_dir: directory to save file in
    """
    # Extract values from the confusion matrix
    tp, fp, fn, tn = conf_matrix[1][1], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[0][0]

    # Data for the pie chart
    labels = ['True Positives', 'False Positives', 'False Negatives', 'True Negatives']
    sizes = [tp, fp, fn, tn]
    colors = ['lightcoral', 'gold', 'lightblue', 'lightgreen']
    explode = (0.1, 0, 0, 0)  # explode the 1st slice (True Positives)

    # Plotting the pie chart
    fig = plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Confusion Matrix Breakdown')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig.savefig(save_dir.joinpath("pie_chart_confusion_matrix.png"))


def get_metrics(model, x_df, y_df, save_dir: Path) -> None:
    """
    Get all metrics for model - model must have .predict() method!
    :param model: trained model
    :param x_df: input for model
    :param y_df: target for model
    :param save_dir: directory to save figures etc. in
    """
    predicted_labels = model.predict(x_df)
    accuracy = accuracy_score(y_df, predicted_labels)
    f1_pos = f1_score(y_df, predicted_labels, average='binary', pos_label="positive")
    f1_neg = f1_score(y_df, predicted_labels, average='binary', pos_label="negative")

    current_time = datetime.datetime.now()
    file_path = save_dir.joinpath("metrics.txt")
    with open(file_path, "a") as res_file:
        # Writing data to a file
        res_file.write(f'{current_time}')
        res_file.write(f'\nAccuracy: {round(accuracy, 3)}')
        res_file.write(f'\nF1 positive label: {round(f1_pos, 3)}')
        res_file.write(f'\nF1 negative label: {round(f1_neg, 3)}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_df, predicted_labels)

    # Graphics
    graphical_confusion_matrix(conf_matrix, save_dir)
    pie_chart_confusion_matrix(conf_matrix, save_dir)
