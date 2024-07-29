
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List, Dict
import argparse
import configparser

from Attention import Attention
from custom_frame_level_loss import custom_frame_level_loss
from CustomDataGenerator import CustomDataGenerator
from CustomF1Score import CustomF1Score


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


def create_model(input_shape: Tuple[int, int], num_classes: int, frame_loss_function: str, utt_loss_function: str, frame_to_utt_loss_ratio: float, optimizer: str, learning_rate: float) -> keras.Model:
    """
    Create the model.

    The model consists of two LSTM layers followed by a frame-level prediction dense layer and output layer.
    The output of the frame-level prediction layer is passed through an attention mechanism to obtain the utterance-level prediction.

    Args:
    - input_shape (Tuple[int, int]): The input shape of the model.
    - num_classes (int): The number of classes.
    - frame_loss_function (str): The frame-level loss function.
    - utt_loss_function (str): The utterance-level loss function.
    - frame_to_utt_loss_ratio (float): The ratio of frame-level loss to utterance-level loss.
    - optimizer (str): The optimizer.
    - learning_rate (float): The learning rate.

    Returns:
    - model (keras.Model): The compiled model.
    """
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
        num_classes, activation='sigmoid', name="frame")(dense_layer_frame)
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
        num_classes, activation='sigmoid', name="utt")(dense_layer_utt)
    # Input: (batch_size, 64)
    # Output: (batch_size, num_classes)

    model = keras.Model(inputs=inputs, outputs=[output_frame, output_utt])

    frame_loss_function = custom_frame_level_loss if frame_loss_function == "custom_frame_level_loss" else "binary_crossentropy"
    utt_loss_function = custom_frame_level_loss if utt_loss_function == "custom_frame_level_loss" else "binary_crossentropy"

    losses = {
        "frame": frame_loss_function,
        "utt": utt_loss_function,
    }

    lossWeights = {
        "frame": 1.0,
        "utt": frame_to_utt_loss_ratio,
    }

    metrics = {
        "frame": [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.Precision(name='precision'),
            CustomF1Score(name='f1_score'),
            tf.keras.metrics.AUC(name='auc'),],
        "utt": [
            tf.keras.metrics.F1Score(name='f1_score'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
        ],
    }

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=lossWeights,
        metrics=metrics
    )

    return model


def training(**kwargs) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Train the model on the training set and evaluate on the evaluation set.

    Args:
    - train_csv_path (str): The path to the training CSV file.
    - eval_csv_path (str): The path to the evaluation CSV file.
    - test_csv_path (str): The path to the test CSV file.
    - frame_loss_function (str): The frame-level loss function.
    - utt_loss_function (str): The utterance-level loss function.
    - frame_to_utt_loss_ratio (float): The ratio of frame-level loss to utterance-level loss.
    - optimizer = (str): The optimizer.
    - learning_rate (float): The learning rate.
    - epochs (int): The number of epochs to train the model.
    - batch_size (int): The batch size.

    Returns:
    - results (Dict[str, float]): The evaluation results on the test set.
    - history (Dict[str, List[float]]): The training history.

    Raises:
    - FileNotFoundError: If the CSV file is not found.
    """
    # Unpack the keyword arguments
    train_csv_path = kwargs["train_csv_path"] if "train_csv_path" in kwargs else None
    eval_csv_path = kwargs["eval_csv_path"] if "eval_csv_path" in kwargs else None
    test_csv_path = kwargs["test_csv_path"] if "test_csv_path" in kwargs else None
    frame_loss_function = kwargs["frame_loss_function"] if "frame_loss_function" in kwargs else "binary_crossentropy"
    utt_loss_function = kwargs["utt_loss_function"] if "utt_loss_function" in kwargs else "binary_crossentropy"
    frame_to_utt_loss_ratio = float(
        kwargs["frame_to_utt_loss_ratio"]) if "frame_to_utt_loss_ratio" in kwargs else 1.0
    optimizer = kwargs["optimizer"] if "optimizer" in kwargs else "adam"
    learning_rate = float(kwargs["learning_rate"]
                          ) if "learning_rate" in kwargs else 0.001
    epochs = int(kwargs["epochs"]) if "epochs" in kwargs else 10
    batch_size = int(kwargs["batch_size"]) if "batch_size" in kwargs else 32

    # Load data
    for path in [train_csv_path, eval_csv_path, test_csv_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    train_features, train_labels = load_data_from_csv(train_csv_path)
    eval_features, eval_labels = load_data_from_csv(eval_csv_path)
    test_features, test_labels = load_data_from_csv(test_csv_path)

    # Determine the maximum sequence length
    maxlen = max(
        max(feature.shape[0] for feature in train_features +
            eval_features + test_features),
        max(label.shape[0]
            for label in train_labels + eval_labels + test_labels)
    )

    # Create TensorFlow Dataset
    train_generator = CustomDataGenerator(
        train_features, train_labels, batch_size, maxlen)
    eval_generator = CustomDataGenerator(
        eval_features, eval_labels, batch_size, maxlen)
    test_generator = CustomDataGenerator(
        test_features, test_labels, batch_size, maxlen)

    # Create and compile the model
    input_shape = (maxlen, train_features[0].shape[1])
    num_classes = train_labels[0].shape[1]
    model = create_model(input_shape, num_classes, frame_loss_function,
                         utt_loss_function, frame_to_utt_loss_ratio, optimizer, learning_rate)

    model.summary()

    # Train the model
    history = model.fit(train_generator, epochs=epochs,
                        validation_data=eval_generator)

    # Evaluate the model on the test set
    results = model.evaluate(test_generator, return_dict=True)

    # Sample Prediction
    sample = test_generator[0]
    frame_pred, utt_pred = model.predict(sample[0])
    print()
    print("Frame-level prediction:")
    print("-----------------------")
    print(frame_pred.flatten())
    print()
    print("Utterance-level prediction:")
    print("---------------------------")
    print(utt_pred.flatten())
    print()

    frame_classification = np.where(frame_pred > 0.5, 1, 0)
    utt_classification = np.where(utt_pred > 0.5, 1, 0)

    print("Frame-level classification:")
    print("---------------------------")
    frame_1_count = np.count_nonzero(frame_classification == 1)
    frame_0_count = np.count_nonzero(frame_classification == 0)
    print(f"1: {frame_1_count}")
    print(f"0: {frame_0_count}")
    print()
    print("Utterance-level classification:")
    print("-------------------------------")
    print(utt_classification.flatten())
    print()

    return results, history.history


def parse_config(config_path: str) -> Dict[str, str]:
    """
    Parse the configuration file.

    Args:
    - config_path (str): The path to the configuration file.

    Returns:
    - config (Dict[str, str]): The configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    dict_config = {}
    for section in config.sections():
        dict_config[section] = dict(config.items(section))

    return dict_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("config_path", type=str, default="training_default.cfg",
                        help="The path to the configuration file.")
    config_path = parser.parse_args().config_path

    config = parse_config(config_path)

    training_config = config["training"]
    log_config = config["log"]

    print("Training configuration:")
    print("-----------------------")
    for key, value in training_config.items():
        print(f"{key}: {value}")
    print()

    print("Log configuration:")
    print("------------------")
    for key, value in log_config.items():
        print(f"{key}: {value}")
    print()

    results, history = training(**training_config)

    model_name = log_config["model_name"]
    log_dir = log_config["log_dir"]

    print("Results on the test set:")
    print("------------------------")
    for key, value in results.items():
        print(f"{key}: {value}")
    print()

    # Save the training log and results to CSV files
    log = pd.DataFrame(history)
    log['epoch'] = range(1, len(log['loss']) + 1)
    cols = log.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    log = log[cols]
    log_path = os.path.join(log_dir, f"{model_name}_log.csv")
    log.to_csv(log_path, index=False)

    results = pd.DataFrame(results, index=[0])
    results_path = os.path.join(log_dir, f"{model_name}_results.csv")
    results.to_csv(results_path, index=False)
