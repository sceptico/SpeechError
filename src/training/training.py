import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List

from Attention import Attention
from custom_loss import custom_loss


def load_data(features_dir: str, labels_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load .npy files from the specified directories.

    Args:
    - features_dir (str): The directory containing the feature files.
    - labels_dir (str): The directory containing the label files.

    Returns:
    - features (List[np.ndarray]): The loaded features.
    - labels (List[np.ndarray]): The loaded labels.
    """
    features_files = sorted(os.listdir(features_dir))
    labels_files = sorted(os.listdir(labels_dir))

    features = []
    labels = []

    for feature_file, label_file in zip(features_files, labels_files):
        feature_path = os.path.join(features_dir, feature_file)
        label_path = os.path.join(labels_dir, label_file)

        feature = np.load(feature_path)
        label = np.load(label_path)

        features.append(feature)
        labels.append(label)

    return features, labels


def pad_sequences(sequences: List[np.ndarray], maxlen: int) -> np.ndarray:
    """
    Pad sequences to the same length.

    Args:
    - sequences (List[np.ndarray]): List of sequences to pad.
    - maxlen (int): The length to pad the sequences to.

    Returns:
    - padded_sequences (np.ndarray): The padded sequences.
    """
    padded_sequences = np.zeros(
        (len(sequences), maxlen, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.shape[0], :] = seq
    return padded_sequences


def create_tf_dataset(features: List[np.ndarray], labels: List[np.ndarray], maxlen: int) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset from the features and labels.

    Args:
    - features (List[np.ndarray]): The features.
    - labels (List[np.ndarray]): The labels.
    - maxlen (int): The length to pad the sequences to.

    Returns:
    - dataset (tf.data.Dataset): The TensorFlow Dataset.
    """
    features = pad_sequences(features, maxlen)
    labels = pad_sequences(labels, maxlen)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=1000).batch(32)
    return dataset


def create_model(input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(
        shape=(input_shape[0], input_shape[1]), name="input_layer")

    # Masking layer to handle padding in sequences
    mask = layers.Masking(mask_value=0.0, name="masking_layer")(inputs)
    # Input: (batch_size, timesteps, features)
    # Output: (batch_size, timesteps, features)

    # First LSTM layer
    lstm_layer_1 = layers.LSTM(
        64, return_sequences=True, name="lstm_layer_1")(mask)
    # Input: (batch_size, timesteps, features)
    # Output: (batch_size, timesteps, 64)

    # Second LSTM layer
    lstm_layer_2 = layers.LSTM(
        64, return_sequences=True, name="lstm_layer_2")(lstm_layer_1)
    # Input: (batch_size, timesteps, 64)
    # Output: (batch_size, timesteps, 64)

    # Frame-level prediction dense layer
    dense_layer_frame = layers.Dense(
        64, activation='relu', name="dense_layer_frame")(lstm_layer_2)
    # Input: (batch_size, timesteps, 64)
    # Output: (batch_size, timesteps, 64)

    # Frame-level prediction output layer
    output_frame = layers.Dense(
        num_classes, activation='sigmoid', name="output_layer_frame")(dense_layer_frame)
    # Input: (batch_size, timesteps, 64)
    # Output: (batch_size, timesteps, num_classes)

    # Attention mechanism for utterance-level prediction
    attention_layer = Attention(name="attention_layer")(output_frame)
    # Input: (batch_size, timesteps, num_classes)
    # Output: (batch_size, num_classes)

    # Utterance-level dense layer
    dense_layer_utt = layers.Dense(
        64, activation='relu', name="dense_layer_utt")(attention_layer)
    # Input: (batch_size, num_classes)
    # Output: (batch_size, 64)

    # Utterance-level output layer
    output_utt = layers.Dense(
        num_classes, activation='sigmoid', name="output_layer_utt")(dense_layer_utt)
    # Input: (batch_size, 64)
    # Output: (batch_size, num_classes)

    model = keras.Model(inputs=inputs, outputs=[output_frame, output_utt])

    # Define custom loss function
    losses = {
        "output_layer_frame": custom_loss,
        "output_layer_utt": custom_loss,
    }

    lossWeights = {"output_layer_frame": 1.0, "output_layer_utt": 1.0}

    # Define the metrics for each output
    metrics = {
        "output_layer_frame": ["accuracy"],
        "output_layer_utt": ["accuracy"],
    }

    model.compile(optimizer='adam', loss=losses,
                  loss_weights=lossWeights, metrics=metrics)

    return model


if __name__ == "__main__":
    # Define paths
    features_dir = "data/features/"
    labels_dir = "data/labels/"

    # Ensure Eager Execution
    tf.config.run_functions_eagerly(True)

    # Load data
    features, labels = load_data(features_dir, labels_dir)

    # Determine the maximum sequence length
    maxlen = max(max(feature.shape[0] for feature in features), max(
        label.shape[0] for label in labels))

    # Create TensorFlow Dataset
    train_dataset = create_tf_dataset(features, labels, maxlen)

    # Create the model
    input_shape = (maxlen, features[0].shape[1])
    num_classes = labels[0].shape[1]
    model = create_model(input_shape, num_classes)

    # Show model summary
    model.summary()

    # Test model output types
    test_input = np.random.rand(1, maxlen, input_shape[1]).astype(np.float32)
    test_output = model(test_input)

    print("Test model output types:")
    print("Type of test_output:", type(test_output))
    print("Length of test_output:", len(test_output))
    print("Shapes of test_output elements:", [o.shape for o in test_output])

    # Train the model
    model.fit(train_dataset, epochs=10)

    # Save the model
    model.save("speech_error_detection_model.keras")
