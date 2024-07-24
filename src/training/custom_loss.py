import tensorflow as tf
from tensorflow import keras


@tf.function
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
    SMALL_LOSS = tf.constant(1e-7, dtype=tf.float32)
    mask = tf.equal(y_true, 1)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    def frame_level_loss():
        y_true_frame = tf.cast(y_true, tf.float32)
        y_true_masked = tf.boolean_mask(y_true_frame, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        return tf.cond(tf.equal(tf.size(y_true_masked), 0),
                       lambda: SMALL_LOSS,
                       lambda: tf.reduce_mean(keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)))

    def utterance_level_loss():
        y_true_utt = tf.cast(y_true, tf.float32)
        y_true_masked = tf.boolean_mask(y_true_utt, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        return tf.cond(tf.equal(tf.size(y_true_masked), 0),
                       lambda: SMALL_LOSS,
                       lambda: tf.reduce_mean(keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)))

    def unsupported_rank():
        tf.print(
            "Unsupported tensor rank: expected rank 2 or 3, got rank", tf.rank(y_true))
        return tf.constant(float('nan'))

    rank = tf.rank(y_true)

    return tf.cond(tf.equal(rank, 3),
                   frame_level_loss,
                   lambda: tf.cond(tf.equal(rank, 2),
                                   utterance_level_loss,
                                   unsupported_rank))
