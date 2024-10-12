'''
label_comparator.py

The LabelComparator class compares predicted labels with actual labels and computes
classification report and confusion matrix for the labels.
'''

import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class LabelComparator:
    def __init__(self, predicted_labels_dir, actual_labels_dir):
        self.predicted_labels_dir = predicted_labels_dir
        self.actual_labels_dir = actual_labels_dir

    def compare_labels(self):
        # Iterate over predicted label files
        all_y_true = []
        all_y_pred = []

        for label_file_name in os.listdir(self.predicted_labels_dir):
            if label_file_name.endswith('.npy'):
                predicted_label_file = os.path.join(
                    self.predicted_labels_dir, label_file_name)
                actual_label_file = os.path.join(
                    self.actual_labels_dir, label_file_name)

                if not os.path.exists(actual_label_file):
                    print(
                        f"Actual label file {actual_label_file} does not exist. Skipping.")
                    continue

                # Load predicted and actual labels
                # Shape: (timesteps, num_classes)
                y_pred = np.load(predicted_label_file)
                y_true = np.load(actual_label_file)

                # Flatten the arrays for metrics computation
                y_pred_flat = y_pred.flatten()
                y_true_flat = y_true.flatten()

                all_y_pred.extend(y_pred_flat)
                all_y_true.extend(y_true_flat)

        # Compute classification report
        report = classification_report(all_y_true, all_y_pred, digits=4)
        print("Classification Report:")
        print(report)

        # Compute confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        print("Confusion Matrix:")
        print(cm)
