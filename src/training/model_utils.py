"""
model_utils.py

Utility functions for creating and compiling models.

Functions:
- create_model: Create and compile the model.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple

from attention import Attention
from custom_frame_level_loss import CustomFrameLevelLoss
from custom_f1_score import CustomF1Score


def create_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    frame_loss_function: str,
    utt_loss_function: str,
    frame_loss_weight: float,
    utt_loss_weight: float,
    optimizer_name: str,
    learning_rate: float
) -> keras.Model:
    """
    Create and compile the model.

    Args:
    - input_shape (Tuple[int, int]): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - frame_loss_function (str): Loss function for frame-level outputs.
    - utt_loss_function (str): Loss function for utterance-level outputs.
    - frame_loss_weight (float): Weight for the frame-level loss.
    - utt_loss_weight (float): Weight for the utterance-level loss.
    - optimizer_name (str): Name of the optimizer to use.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - model (keras.Model): Compiled Keras model.
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input_layer")

    # Masking layer
    mask = layers.Masking(mask_value=0.0, name="masking_layer")(inputs)

    # LSTM layers
    lstm_layer_1 = layers.LSTM(
        64, return_sequences=True, name="lstm_layer_1")(mask)
    lstm_layer_2 = layers.LSTM(
        64, return_sequences=True, name="lstm_layer_2")(lstm_layer_1)

    # Frame-level prediction layers
    dense_layer_frame = layers.Dense(
        64, activation='relu', name="dense_layer_frame")(lstm_layer_2)
    output_frame = layers.Dense(
        num_classes, activation='sigmoid', name="frame")(dense_layer_frame)

    # Attention mechanism for utterance-level prediction
    attention_layer = Attention(name="attention_layer")(output_frame)
    dense_layer_utt = layers.Dense(
        64, activation='relu', name="dense_layer_utt")(attention_layer)
    output_utt = layers.Dense(
        num_classes, activation='sigmoid', name="utt")(dense_layer_utt)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=[output_frame, output_utt])

    # Configure loss functions
    frame_loss = CustomFrameLevelLoss(
    ) if frame_loss_function == "custom_frame_level_loss" else frame_loss_function
    utt_loss = CustomFrameLevelLoss(
    ) if utt_loss_function == "custom_frame_level_loss" else utt_loss_function

    losses = {
        "frame": frame_loss,
        "utt": utt_loss,
    }

    loss_weights = {
        "frame": frame_loss_weight,
        "utt": utt_loss_weight,
    }

    # Configure metrics
    metrics = {
        "frame": [
            tf.keras.metrics.Precision(name='precision'),
            CustomF1Score(name='f1_score'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
        ],
        "utt": [
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.F1Score(name='f1_score'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
        ],
    }

    # Configure optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    print("Model compiled successfully.")
    model.summary()

    return model
