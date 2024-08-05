import tensorflow as tf
from tensorflow.keras import backend as K


def custom_f1_score(y_true, y_pred):
    """
    Custom F1 score that reshapes 3D inputs to 2D.

    Args:
    - y_true (tf.Tensor): The true labels.
    - y_pred (tf.Tensor): The predicted labels.

    Returns:
    - f1_score (tf.Tensor): The computed F1 score.
    """
    # Ensure both tensors have the same data type
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Reshape the inputs to 2D
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    # Calculate precision and recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1_score
