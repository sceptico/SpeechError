
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List

from Attention import Attention
from custom_loss import custom_loss
from CustomErrorRateMetric import CustomErrorRateMetric, ErrorRateLoggingCallback


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
    # Ensure each sequence is 2D
    sequences = [seq if seq.ndim == 2 else np.expand_dims(
        seq, axis=-1) for seq in sequences]

    feature_dim = sequences[0].shape[1]
    padded_sequences = np.zeros((len(sequences), maxlen, feature_dim))

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
    labels_frame = pad_sequences(labels, maxlen)
    labels_utt = np.any(labels_frame == 1, axis=1).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(
        (features, (labels_frame, labels_utt)))
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
        32, return_sequences=True, name="lstm_layer_1")(mask)
    # Input: (batch_size, timesteps, features)
    # Output: (batch_size, timesteps, 32)

    # Second LSTM layer
    lstm_layer_2 = layers.LSTM(
        32, return_sequences=True, name="lstm_layer_2")(lstm_layer_1)
    # Input: (batch_size, timesteps, 32)
    # Output: (batch_size, timesteps, 32)

    # Frame-level prediction dense layer
    dense_layer_frame = layers.Dense(
        32, activation='relu', name="dense_layer_frame")(lstm_layer_2)
    # Input: (batch_size, timesteps, 32)
    # Output: (batch_size, timesteps, 32)

    # Frame-level prediction output layer
    output_frame = layers.Dense(
        num_classes, activation='sigmoid', name="output_layer_frame")(dense_layer_frame)
    # Input: (batch_size, timesteps, 32)
    # Output: (batch_size, timesteps, num_classes)

    # Attention mechanism for utterance-level prediction
    attention_layer = Attention(name="attention_layer")(output_frame)
    # Input: (batch_size, timesteps, num_classes)
    # Output: (batch_size, num_classes)

    # Utterance-level dense layer
    dense_layer_utt = layers.Dense(
        32, activation='relu', name="dense_layer_utt")(attention_layer)
    # Input: (batch_size, num_classes)
    # Output: (batch_size, 32)

    # Utterance-level output layer
    output_utt = layers.Dense(
        num_classes, activation='sigmoid', name="output_layer_utt")(dense_layer_utt)
    # Input: (batch_size, 32)
    # Output: (batch_size, num_classes)

    model = keras.Model(inputs=inputs, outputs=[output_frame, output_utt])

    # Define custom loss function
    losses = {
        "output_layer_frame": custom_loss,
        "output_layer_utt": custom_loss,
    }

    lossWeights = {"output_layer_frame": 1.0, "output_layer_utt": 1.0}

    # metrics = {
    #     "output_layer_frame": [CustomErrorRateMetric(), "FalsePositives", "FalseNegatives", "TruePositives", "TrueNegatives", "accuracy"],
    #     "output_layer_utt": ["FalsePositives", "FalseNegatives", "TruePositives", "TrueNegatives", "accuracy", "F1Score"],
    # }

    metrics = {
        "output_layer_frame": [CustomErrorRateMetric()],
        "output_layer_utt": ["accuracy"],
    }

    model.compile(optimizer='adam', loss=losses,
                  loss_weights=lossWeights, metrics=metrics)

    return model


def get_layer_outputs(model: keras.Model, inputs: np.ndarray) -> List[np.ndarray]:
    """
    Get the outputs of all layers in the model for the given inputs.

    Args:
    - model (keras.Model): The original model.
    - inputs (np.ndarray): The input data.

    Returns:
    - List[np.ndarray]: The outputs of each layer.
    """
    layer_outputs = [layer.output for layer in model.layers]
    intermediate_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    intermediate_outputs = intermediate_model.predict(inputs)
    return intermediate_outputs


def print_layer_outputs(layer_outputs: List[np.ndarray]) -> None:
    """
    Print the outputs of each layer for debugging.

    Args:
    - layer_outputs (List[np.ndarray]): The outputs of each layer.
    """
    for i, output in enumerate(layer_outputs):
        print(f"Layer {i}: {model.layers[i].name}")
        print(f"Output of layer {i} ({output.shape}):\n{output}")
        print("-" * 50)


if __name__ == "__main__":
    # Define paths
    features_dir = "data/features/"
    labels_dir = "data/labels/"

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

    # '''Test debugging model with one batch of data'''
    # for batch_features, (batch_labels_frame, batch_labels_utt) in train_dataset.take(1):
    #     intermediate_outputs = get_layer_outputs(model, batch_features)
    #     print_layer_outputs(intermediate_outputs)

    #     # Calculate the loss for this batch
    #     output_frame, output_utt = model(batch_features, training=False)
    #     loss_frame = custom_loss(batch_labels_frame, output_frame)
    #     loss_utt = custom_loss(batch_labels_utt, output_utt)
    #     total_loss = 1.0 * loss_frame + 1.0 * loss_utt

    #     print(f"Frame-level loss: {loss_frame.numpy()}")
    #     print(f"Utterance-level loss: {loss_utt.numpy()}")
    #     print(f"Total loss: {total_loss.numpy()}")

    # Train the model
    model.fit(train_dataset, epochs=20)
    # model.fit(train_dataset, epochs=1, callbacks=[ErrorRateLoggingCallback()])

    # Save the model
    model.save("speech_error_detection_model.keras")
