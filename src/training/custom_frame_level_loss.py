import tensorflow as tf
from tensorflow import keras


@tf.function
def custom_frame_level_loss(y_true: tf.Tensor, y_pred_frame: tf.Tensor) -> tf.Tensor:
    """
    Custom frame-level loss function that only considers frames within event-containing utterances.

    Args:
    - y_true (tf.Tensor): The true labels.
    - y_pred_frame (tf.Tensor): The predicted labels at the frame level.

    Returns:
    - loss (tf.Tensor): The computed frame-level loss.
    """
    # Create a mask for the true utterance labels
    # Only consider the utterances with speech events when calculating the loss
    SMALL_LOSS = tf.constant(1e-7, dtype=tf.float32)

    # Compute y_true_utt to check if an utterance contains an event or not
    y_true_utt = tf.cast(tf.reduce_any(tf.equal(y_true, 1), axis=-1), tf.int32)

    # Create a mask for the utterances where y_true_utt is 1 (indicating event presence)
    mask = tf.equal(y_true_utt, 1)
    mask_expanded = tf.expand_dims(mask, axis=-1)

    # Mask the y_true and y_pred_frame to exclude non-event utterances
    y_true_masked = tf.boolean_mask(y_true, mask_expanded)
    y_pred_frame_masked = tf.boolean_mask(y_pred_frame, mask_expanded)

    # Compute the loss
    loss = tf.cond(tf.equal(tf.size(y_true_masked), 0),
                   lambda: SMALL_LOSS,
                   lambda: tf.reduce_mean(keras.losses.binary_crossentropy(y_true_masked, y_pred_frame_masked)))

    return loss
