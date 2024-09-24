"""
data_utils.py

Utility functions for loading and processing data.

Functions:
- load_data_from_csv: Load features and labels from a CSV file.
- get_max_sequence_length: Calculate the maximum sequence length from features and labels.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List


def load_data_from_csv(csv_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load features and labels from a CSV file.

    Args:
    - csv_file (str): Path to the CSV file containing 'feature_file' and 'label_file' columns.

    Returns:
    - features (List[np.ndarray]): List of feature arrays.
    - labels (List[np.ndarray]): List of label arrays.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    data = pd.read_csv(csv_file)
    features = []
    labels = []

    for _, row in data.iterrows():
        feature_path = row['feature_file']
        label_path = row['label_file']

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        feature = np.load(feature_path)
        label = np.load(label_path)

        features.append(feature)
        labels.append(label)

    return features, labels


def get_max_sequence_length(features: List[np.ndarray], labels: List[np.ndarray]) -> int:
    """
    Calculate the maximum sequence length from features and labels.

    Args:
    - features (List[np.ndarray]): List of feature arrays.
    - labels (List[np.ndarray]): List of label arrays.

    Returns:
    - maxlen (int): The maximum sequence length.
    """
    maxlen = max(
        max(feature.shape[0] for feature in features),
        max(label.shape[0] for label in labels)
    )
    return maxlen
