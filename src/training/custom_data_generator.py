"""
custom_data_generator.py

Custom data generator for training models with optional event split.
"""


from typing import List, Tuple
import numpy as np
import tensorflow as tf


class CustomDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for training models with optional event split.

    Attributes:
    - features (List[np.ndarray]): The features.
    - labels (List[np.ndarray]): The labels.
    - batch_size (int): The batch size.
    - maxlen (int): The maximum sequence length.
    - enforce_event_split (bool): Whether to enforce a custom event split (e.g., 50% of each batch contains events).

    Methods:
    - __init__: Initialize the data generator.
    - __len__: Calculate the number of batches in the generator.
    - __getitem__: Generate one batch of data. Optionally ensure that a certain percentage of each batch contains samples with events.
    - on_epoch_end: Shuffle the data at the end of each epoch.
    - get_input_shape: Get the input shape of the data.
    - get_num_classes: Get the number of classes.
    - pad_sequences: Pad sequences to a fixed length.
    """

    def __init__(self, features: List[np.ndarray], labels: List[np.ndarray], batch_size: int, maxlen: int, enforce_event_split: bool = False, event_ratio: float = 0.5, **kwargs):
        """
        Custom data generator that optionally ensures a certain percentage of each batch contains samples with events.

        Args:
        - features (List[np.ndarray]): The features.
        - labels (List[np.ndarray]): The labels.
        - batch_size (int): The batch size.
        - maxlen (int): The maximum sequence length.
        - enforce_event_split (bool): Whether to enforce a custom event split (e.g., 50% of each batch contains events).
        """
        super().__init__(**kwargs)
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.enforce_event_split = enforce_event_split
        self.event_ratio = event_ratio

        # Split data into samples with events and without events
        self.features_with_events = [f for f, l in zip(
            features, labels) if np.any(l == 1)]
        self.labels_with_events = [l for l in labels if np.any(l == 1)]
        self.features_without_events = [f for f, l in zip(
            features, labels) if not np.any(l == 1)]
        self.labels_without_events = [l for l in labels if not np.any(l == 1)]

        self.on_epoch_end()

    def __len__(self):
        """
        Calculate the number of batches in the generator.
        """
        return int(np.ceil(len(self.features) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data. Optionally ensure that a certain percentage of each batch contains samples with events.

        Args:
        - index (int): The index of the batch.
        """
        batch_features = []
        batch_labels = []

        if self.enforce_event_split:
            # Ensure at least 50% of samples contain events
            num_with_events = max(1, int(self.batch_size * self.event_ratio))
            if len(self.features_with_events) >= num_with_events:
                batch_features.extend(
                    self.features_with_events[:num_with_events])
                batch_labels.extend(self.labels_with_events[:num_with_events])

                self.features_with_events = self.features_with_events[num_with_events:]
                self.labels_with_events = self.labels_with_events[num_with_events:]
            else:
                batch_features.extend(self.features_with_events)
                batch_labels.extend(self.labels_with_events)

                self.features_with_events = []
                self.labels_with_events = []

            remaining_batch_size = self.batch_size - len(batch_features)
            if len(self.features_without_events) >= remaining_batch_size:
                batch_features.extend(
                    self.features_without_events[:remaining_batch_size])
                batch_labels.extend(
                    self.labels_without_events[:remaining_batch_size])

                # Rotate the data pool for samples without events
                self.features_without_events = self.features_without_events[
                    remaining_batch_size:] + batch_features[num_with_events:]
                self.labels_without_events = self.labels_without_events[
                    remaining_batch_size:] + batch_labels[num_with_events:]
            else:
                batch_features.extend(self.features_without_events)
                batch_labels.extend(self.labels_without_events)
        else:
            # If not enforcing event split, just shuffle and batch the data
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, len(self.features))
            batch_features = self.features[start_idx:end_idx]
            batch_labels = self.labels[start_idx:end_idx]

        # Shuffle the current batch
        batch = list(zip(batch_features, batch_labels))
        np.random.shuffle(batch)
        batch_features, batch_labels = zip(*batch)

        # Prepare the batch
        batch_features = self.pad_sequences(batch_features, self.maxlen)
        batch_labels_frame = self.pad_sequences(batch_labels, self.maxlen)
        batch_labels_utt = np.any(
            batch_labels_frame == 1, axis=1).astype(np.float32)

        return np.array(batch_features), (np.array(batch_labels_frame), np.array(batch_labels_utt))

    def on_epoch_end(self):
        """
        Shuffle the data at the end of each epoch.
        """
        combined_with_events = list(
            zip(self.features_with_events, self.labels_with_events))
        combined_without_events = list(
            zip(self.features_without_events, self.labels_without_events))

        np.random.shuffle(combined_with_events)
        np.random.shuffle(combined_without_events)

        if combined_with_events:
            self.features_with_events, self.labels_with_events = zip(
                *combined_with_events)
            self.features_with_events = list(self.features_with_events)
            self.labels_with_events = list(self.labels_with_events)

        if combined_without_events:
            self.features_without_events, self.labels_without_events = zip(
                *combined_without_events)
            self.features_without_events = list(self.features_without_events)
            self.labels_without_events = list(self.labels_without_events)

        if self.enforce_event_split:
            # Replenish the data pools after shuffling
            while len(self.features_with_events) < self.__len__() * (self.batch_size * self.event_ratio):
                self.features_with_events += self.features_with_events
                self.labels_with_events += self.labels_with_events

            while len(self.features_without_events) < self.__len__() * (self.batch_size * (1 - self.event_ratio)):
                self.features_without_events += self.features_without_events
                self.labels_without_events += self.labels_without_events

    def get_input_shape(self) -> Tuple[int, int]:
        """
        Get the input shape of the data.
        """
        return (self.maxlen, self.features[0].shape[1])

    def get_num_classes(self) -> int:
        """
        Get the number of classes.
        """
        return self.labels[0].shape[-1]

    def pad_sequences(self, sequences: List[np.ndarray], maxlen: int) -> np.ndarray:
        """
        Pad sequences to a fixed length.

        Args:
        - sequences (List[np.ndarray]): The sequences to pad.
        - maxlen (int): The maximum sequence length.

        Returns:
        - padded_sequences (np.ndarray): The padded sequences.
        """
        sequences = [seq if seq.ndim == 2 else np.expand_dims(
            seq, axis=-1) for seq in sequences]
        feature_dim = sequences[0].shape[1]
        padded_sequences = np.zeros((len(sequences), maxlen, feature_dim))

        for i, seq in enumerate(sequences):
            padded_sequences[i, :seq.shape[0], :] = seq

        return padded_sequences


if __name__ == "__main__":
    """
    Example usage of the CustomDataGenerator class. This is used for testing purposes only.
    """
    import pandas as pd
    from typing import Tuple

    def load_data_from_csv(csv_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load .npy files specified in the CSV file.

        Args:
        - csv_file (str): The path to the CSV file containing file names and labels.

        Returns:
        - features (List[np.ndarray]): The loaded features.
        - labels (List[np.ndarray]): The loaded labels.
        """
        data = pd.read_csv(csv_file)
        features = []
        labels = []

        for index, row in data.iterrows():
            feature_path = row['feature_file']
            label_path = row['label_file']

            feature = np.load(feature_path)
            label = np.load(label_path)

            features.append(feature)
            labels.append(label)

        return features, labels

    test_csv = "data/metadata/test.csv"

    # Load data once
    print(f"Loading test data from {test_csv}...")
    features, labels = load_data_from_csv(test_csv)
    maxlen = max(max([len(f) for f in features]),
                 max([len(l) for l in labels]))

    print("without event split")
    generator = CustomDataGenerator(
        features, labels, batch_size=32, maxlen=maxlen)

    for i in range(len(generator)):
        batch_features, (batch_labels_frame, batch_labels_utt) = generator[i]
        print(f"Batch {i} - Features shape: {batch_features.shape}")
        print(f"Batch {i} - Utterance labels: {batch_labels_utt.flatten()}")

    print()
    print("with event split")
    generator = CustomDataGenerator(
        features, labels, batch_size=32, maxlen=maxlen, enforce_event_split=True)

    for i in range(len(generator)):
        batch_features, (batch_labels_frame, batch_labels_utt) = generator[i]
        print(f"Batch {i} - Features shape: {batch_features.shape}")
        print(f"Batch {i} - Utterance labels: {batch_labels_utt.flatten()}")
