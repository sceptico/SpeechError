from typing import List
import numpy as np
import tensorflow as tf

from util import pad_sequences


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, features: List[np.ndarray], labels: List[np.ndarray], batch_size: int, maxlen: int, **kwargs):
        """
        Custom data generator that ensures that every batch has at least one sample with events.

        Args:
        - features (List[np.ndarray]): The features.
        - labels (List[np.ndarray]): The labels.
        - batch_size (int): The batch size.
        - maxlen (int): The maximum sequence length.
        """
        super().__init__(**kwargs)
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.maxlen = maxlen

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
        return int(np.floor(len(self.features) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data. Ensure that every batch has at least one sample with events.
        If there are not enough samples with events, repeat them to fill the batches.

        Args:
        - index (int): The index of the batch.
        """
        # Calculate how many samples with events to include in this batch
        samples_per_batch_with_events = max(
            1, len(self.features_with_events) // self.__len__())

        start_idx = index * samples_per_batch_with_events
        end_idx = (index + 1) * samples_per_batch_with_events

        if end_idx > len(self.features_with_events):
            end_idx = len(self.features_with_events)
            start_idx = end_idx - samples_per_batch_with_events

        batch_features_with_events = self.features_with_events[start_idx:end_idx]
        batch_labels_with_events = self.labels_with_events[start_idx:end_idx]

        # Fill the rest of the batch with samples without events
        remaining_batch_size = self.batch_size - \
            len(batch_features_with_events)

        if len(self.features_without_events) >= remaining_batch_size:
            batch_features_without_events = self.features_without_events[:remaining_batch_size]
            batch_labels_without_events = self.labels_without_events[:remaining_batch_size]
            self.features_without_events = self.features_without_events[remaining_batch_size:]
            self.labels_without_events = self.labels_without_events[remaining_batch_size:]
        else:
            batch_features_without_events = self.features_without_events
            batch_labels_without_events = self.labels_without_events
            self.features_without_events = []
            self.labels_without_events = []

        batch_features = batch_features_with_events + batch_features_without_events
        batch_labels = batch_labels_with_events + batch_labels_without_events

        batch_features = pad_sequences(batch_features, self.maxlen)
        batch_labels_frame = pad_sequences(batch_labels, self.maxlen)
        batch_labels_utt = np.any(
            batch_labels_frame == 1, axis=1).astype(np.float32)

        return np.array(batch_features), (np.array(batch_labels_frame), np.array(batch_labels_utt))

    def on_epoch_end(self):
        """
        Shuffle the data at the end of each epoch. If not enough samples with events, repeat them to fill the batches.
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
        else:
            self.features_with_events = []
            self.labels_with_events = []

        if combined_without_events:
            self.features_without_events, self.labels_without_events = zip(
                *combined_without_events)
            self.features_without_events = list(self.features_without_events)
            self.labels_without_events = list(self.labels_without_events)
        else:
            self.features_without_events = []
            self.labels_without_events = []

        # If not enough samples with events, repeat them to fill the batches
        while len(self.features_with_events) < self.__len__() * (self.batch_size // 2):
            self.features_with_events += self.features_with_events
            self.labels_with_events += self.labels_with_events

    def get_input_shape(self) -> tuple:
        """
        Get the input shape of the data.
        """
        return (self.maxlen, self.features[0].shape[1])

    def get_num_classes(self) -> int:
        """
        Get the number of classes.
        """
        return self.labels[0].shape[-1]
