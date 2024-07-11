import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List, Generator

# Define paths
features_dir = "data/features/"
labels_dir = "data/labels/"

# Data Generator


def data_generator(features_dir: str, labels_dir: str, batch_size: int, label_padding_value: int = -1) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields batches of features and labels.

    Args:
    - features_dir (str): The directory containing the feature files.
    - labels_dir (str): The directory containing the label files.
    - batch_size (int): The size of the batches to generate.
    - label_padding_value (int): The padding value for labels.

    Yields:
    - Tuple[np.ndarray, np.ndarray]: Batch of features and corresponding labels.
    """
    features_files = sorted(os.listdir(features_dir))
    labels_files = sorted(os.listdir(labels_dir))

    while True:
        for start in range(0, len(features_files), batch_size):
            end = min(start + batch_size, len(features_files))
            batch_features = []
            batch_labels = []

            for feature_file, label_file in zip(features_files[start:end], labels_files[start:end]):
                feature_path = os.path.join(features_dir, feature_file)
                label_path = os.path.join(labels_dir, label_file)

                feature = np.load(feature_path)
                label = np.load(label_path)

                batch_features.append(feature)
                batch_labels.append(label)

            # Pad the sequences
            padded_features = tf.keras.preprocessing.sequence.pad_sequences(
                batch_features, padding='post', dtype='float32'
            )
            padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
                batch_labels, padding='post', dtype='float32', value=label_padding_value
            )

            yield padded_features, padded_labels

# Create TensorFlow Dataset


def create_tf_dataset(features_dir: str, labels_dir: str, batch_size: int, buffer_size: int = 100, label_padding_value: int = -1) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset from the features and labels.

    Args:
    - features_dir (str): The directory containing the feature files.
    - labels_dir (str): The directory containing the label files.
    - batch_size (int): The size of the batches to generate.
    - buffer_size (int): The buffer size for shuffling.
    - label_padding_value (int): The padding value for labels.

    Returns:
    - dataset (tf.data.Dataset): The TensorFlow Dataset.
    """
    output_signature = (
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        # Adjust based on number of classes
        tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        args=[features_dir, labels_dir, batch_size, label_padding_value],
        output_signature=output_signature
    )

    return dataset.shuffle(buffer_size=buffer_size).prefetch(tf.data.experimental.AUTOTUNE)


# Set batch size and buffer size
batch_size = 16
buffer_size = 100  # Adjusted buffer size to reduce memory usage

# Create the dataset
train_dataset = create_tf_dataset(
    features_dir, labels_dir, batch_size, buffer_size)

# Define the model


def create_model(input_dim: int, num_classes: int, label_padding_value: int = -1) -> keras.Model:
    """
    Create a Keras model for speech error detection.

    Args:
    - input_dim (int): The dimensionality of the input features.
    - num_classes (int): The number of classes in the labels.
    - label_padding_value (int): The padding value for labels.

    Returns:
    - model (keras.Model): The compiled Keras model.
    """
    # Variable-length sequences
    inputs = keras.Input(shape=(None, input_dim), name='input_features')
    masked_inputs = layers.Masking(mask_value=0.0, name='masking_layer')(
        inputs)  # Mask the padded feature values
    lstm_layer_1 = layers.LSTM(64, return_sequences=True, name='lstm_layer_1')(
        masked_inputs)  # Return sequences for all LSTM layers
    lstm_layer_2 = layers.LSTM(64, return_sequences=True, name='lstm_layer_2')(
        lstm_layer_1)  # Ensure the last LSTM layer also returns sequences
    dense_layer = layers.Dense(
        64, activation='relu', name='dense_layer')(lstm_layer_2)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output_layer')(
        dense_layer)  # Ensure output shape matches target shape

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name='speech_error_detection_model')

    # Custom loss to ignore padded labels
    def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Custom loss function to ignore padded labels.

        Args:
        - y_true (tf.Tensor): The true labels.
        - y_pred (tf.Tensor): The predicted labels.

        Returns:
        - loss (tf.Tensor): The computed loss value.
        """
        mask = tf.cast(tf.not_equal(y_true, label_padding_value), tf.float32)
        loss = mask * tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    return model


# Create the model
# Define input_dim and num_classes based on your data
input_dim = 128  # Adjust based on your feature shape
num_classes = 6  # Number of classes
model = create_model(input_dim, num_classes)

# Train the model
model.fit(train_dataset, epochs=10)

# Save the model
model.save("speech_error_detection_model.h5")
