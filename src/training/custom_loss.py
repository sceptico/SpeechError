
import tensorflow as tf
from tensorflow import keras


def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function for predictions at any level.

    Args:
    - y_true (tf.Tensor): The true labels.
    - y_pred (tf.Tensor): The predicted labels.

    Returns:
    - loss (tf.Tensor): The computed loss.
    """
    # Create a mask for the true labels
    # Only consider the frames and utterances with speech when calculating the loss
    mask = tf.equal(y_true, 1)

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    if tf.rank(y_true) == 3:  # Frame-level
        # Convert y_true to float32 and apply the mask
        y_true_frame = tf.cast(y_true, tf.float32)
        y_true_masked = tf.boolean_mask(y_true_frame, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        loss = keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)
    elif tf.rank(y_true) == 2:  # Utterance-level
        # Convert y_true to float32 and apply the mask
        y_true_utt = tf.cast(y_true, tf.float32)
        y_true_masked = tf.boolean_mask(y_true_utt, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        loss = keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)
    else:
        raise ValueError(
            f"Unsupported tensor rank: expected rank 2 or 3, got rank {tf.rank(y_true)}")

    return tf.reduce_mean(loss)
