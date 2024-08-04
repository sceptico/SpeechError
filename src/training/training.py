import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List, Dict
import argparse
import configparser
import math

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


def create_model(input_shape: Tuple[int, int], num_classes: int, frame_loss_function: str, utt_loss_function: str, utt_to_frame_loss_ratio: float, optimizer: str, learning_rate: float) -> keras.Model:
    """
    Create the model.

    The model consists of two LSTM layers followed by a frame-level prediction dense layer and output layer.
    The output of the frame-level prediction layer is passed through an attention mechanism to obtain the utterance-level prediction.

    Args:
    - input_shape (Tuple[int, int]): The input shape of the model.
    - num_classes (int): The number of classes.
    - frame_loss_function (str): The frame-level loss function.
    - utt_loss_function (str): The utterance-level loss function.
    - utt_to_frame_loss_ratio_loss_ratio (float): The ratio of utterance-level loss to frame-level loss.
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
        "utt": utt_to_frame_loss_ratio,
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

    print("Model compiled successfully.")

    model.summary()

    return model


def print_experiment_results(results: Dict[str, float], model: keras.Model, test_generator: CustomDataGenerator):
    """
    Print the results of the experiment.

    Args:
    - results (Dict[str, float]): Evaluation results from model.evaluate
    - model (keras.Model): The trained model
    - test_generator (CustomDataGenerator): The test data generator
    """
    # Print evaluation results
    print("Evaluation Results:")
    print("-------------------")
    for key, value in results.items():
        print(f"{key}: {value}")
    print()

    # Sample Prediction
    sample = test_generator[0]
    frame_pred, utt_pred = model.predict(sample[0])
    print("Sample Predictions:")
    print("-------------------")
    print("Frame-level prediction:")
    print(frame_pred.flatten())
    print()
    print("Utterance-level prediction:")
    print(utt_pred.flatten())
    print()

    frame_classification = np.where(frame_pred > 0.5, 1, 0)
    utt_classification = np.where(utt_pred > 0.5, 1, 0)

    print("Frame-level classification counts:")
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


def run_experiment(train_generator, eval_generator, test_generator, config_path: str):
    """
    Run a single experiment using the given configuration file.

    Args:
    - train_generator: Pre-initialized CustomDataGenerator for training data
    - eval_generator: Pre-initialized CustomDataGenerator for evaluation data
    - test_generator: Pre-initialized CustomDataGenerator for test data
    - config_path (str): Path to the configuration file
    """
    # Parse the configuration file
    config = parse_config(config_path)
    training_config = config["training"]
    log_config = config["log"]
    model_name = log_config["model_name"]
    log_dir = log_config["log_dir"]
    model_dir = log_config["model_dir"]
    checkpoint_dir = log_config["checkpoint_dir"]

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unpack the keyword arguments
    frame_loss_function = training_config.get(
        "frame_loss_function", "binary_crossentropy")
    utt_loss_function = training_config.get(
        "utt_loss_function", "binary_crossentropy")
    utt_to_frame_loss_ratio = float(
        training_config.get("utt_to_frame_loss_ratio", 1.0))
    optimizer = training_config.get("optimizer", "adam")
    learning_rate = float(training_config.get("learning_rate", 0.001))
    epochs = int(training_config.get("epochs", 10))

    # Convert loss function from string to function
    if frame_loss_function == "custom_frame_level_loss":
        frame_loss_function = custom_frame_level_loss

    if utt_loss_function == "custom_frame_level_loss":
        utt_loss_function = custom_frame_level_loss

    # Determine the input shape and number of classes
    input_shape = train_generator.get_input_shape()
    num_classes = train_generator.get_num_classes()

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = f"{checkpoint_dir}/{model_name}_checkpoint_{{epoch:03d}}.keras"

    # Create a callback that saves the model's weights every 5 epochs
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )

    # Create and compile the model
    model = create_model(input_shape, num_classes, frame_loss_function,
                         utt_loss_function, utt_to_frame_loss_ratio, optimizer, learning_rate)

    # Load the latest checkpoint if available
    completed_epochs = 0
    latest_checkpoint = None
    model_name_log = model_name
    for epoch in range(epochs, 0, -1):
        checkpoint = checkpoint_path.format(epoch=epoch)
        if os.path.exists(checkpoint):
            latest_checkpoint = checkpoint
            completed_epochs = epoch
            model_name_log += f"_from_epoch_{epoch}"
            break

    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        print(f"Loaded weights from {latest_checkpoint}")
        print(f"Resuming training from epoch {completed_epochs + 1}.")
    else:
        print("No checkpoint found. Training from scratch.")

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        initial_epoch=completed_epochs,
        validation_data=eval_generator,
        callbacks=[checkpoint_callback]
    )

    model_name_log += f"_to_epoch_{epochs}"

    # Evaluate the model on the test set
    results = model.evaluate(test_generator, return_dict=True)

    # Print experiment results
    print(f"Results for configuration: {config_path}")
    print_experiment_results(results, model, test_generator)

    # Save the training log and results to CSV files
    # Only if training happened (i.e. completed_epochs < epochs)
    if completed_epochs >= epochs:
        print("No training happened. Not saving logs and results.")
        return

    log = pd.DataFrame(history.history)
    # Offset the epoch numbers by the number of completed epochs
    log['epoch'] = [
        epoch + completed_epochs for epoch in range(1, len(log['loss']) + 1)]
    cols = log.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    log = log[cols]
    log_path = os.path.join(log_dir, f"{model_name_log}_log.csv")
    log.to_csv(log_path, index=False)

    results_df = pd.DataFrame(results, index=[0])
    results_path = os.path.join(log_dir, f"{model_name_log}_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Log and results saved to: {log_dir}")

    model_path = os.path.join(model_dir, f"{model_name}_epoch_{epochs}.keras")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    print(f"Experiment with configuration \"{config_path}\" completed.")


def parse_config(config_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse the configuration file.

    Args:
    - config_path (str): The path to the configuration file.

    Returns:
    - config (Dict[str, Dict[str, str]]): The configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    dict_config = {}
    for section in config.sections():
        dict_config[section] = dict(config.items(section))

    return dict_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("config_paths", nargs='+', type=str,
                        help="List of configuration files for different experiments.")

    args = parser.parse_args()

    # Check if all configuration files exist
    for config_path in args.config_paths:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

    # Load the first configuration to extract dataset paths and batch size
    first_config = parse_config(args.config_paths[0])
    train_csv = first_config['data']['train_csv']
    eval_csv = first_config['data']['eval_csv']
    test_csv = first_config['data']['test_csv']
    batch_size = int(first_config['training'].get('batch_size', 32))
    config_paths = args.config_paths

    # Check if the CSV files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV file not found: {train_csv}")

    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"Evaluation CSV file not found: {eval_csv}")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV file not found: {test_csv}")

    # Load data once
    print(f"Loading training data from {train_csv}...")
    train_features, train_labels = load_data_from_csv(train_csv)

    print(f"Loading evaluation data from {eval_csv}...")
    eval_features, eval_labels = load_data_from_csv(eval_csv)

    print(f"Loading test data from {test_csv}...")
    test_features, test_labels = load_data_from_csv(test_csv)

    # Determine the maximum sequence length
    maxlen = max(
        max(feature.shape[0] for feature in train_features +
            eval_features + test_features),
        max(label.shape[0]
            for label in train_labels + eval_labels + test_labels)
    )

    # Create reusable CustomDataGenerator objects
    train_generator = CustomDataGenerator(
        train_features, train_labels, batch_size=batch_size, maxlen=maxlen)
    eval_generator = CustomDataGenerator(
        eval_features, eval_labels, batch_size=batch_size, maxlen=maxlen)
    test_generator = CustomDataGenerator(
        test_features, test_labels, batch_size=batch_size, maxlen=maxlen)

    # Run experiments for each configuration file
    for config_path in config_paths:
        print(f"Running experiment with configuration: {config_path}")
        run_experiment(train_generator, eval_generator,
                       test_generator, config_path)
