"""
attention.py

Custom Keras layer to compute the context vector using the attention mechanism.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Attention(Layer):
    """
    Custom Keras layer to compute the context vector using the attention mechanism.
    """

    def __init__(self, **kwargs):
        """
        Initializes the layer.
        """
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, mask=None) -> tf.Tensor:
        """
        Defines the computation that will be performed on the input data.

        Args:
        - inputs (tf.Tensor): Input tensor.

        Returns:
        - context_vector (tf.Tensor): Context vector computed using the attention mechanism.
        """
        # inputs.shape = (batch_size, timesteps, num_classes)
        p_t = inputs  # Assuming inputs are the frame-level predictions

        if mask is not None:
            # mask.shape = (batch_size, timesteps)
            # Convert mask to same dtype as inputs
            mask = tf.cast(mask, dtype=inputs.dtype)
            # Shape: (batch_size, timesteps, 1)
            mask_expanded = tf.expand_dims(mask, axis=-1)
            # Apply mask to p_t
            p_t_masked = p_t * mask_expanded
        else:
            p_t_masked = p_t

        # Compute attention weights
        # Sum over time dimension
        sum_p_t = tf.reduce_sum(p_t_masked, axis=1, keepdims=True)
        # Avoid division by zero
        sum_p_t += tf.keras.backend.epsilon()
        a_t = p_t_masked / sum_p_t  # Normalize over the time dimension

        # Compute context vector as a weighted sum of the frame-level features
        context_vector = tf.reduce_sum(a_t * inputs, axis=1)
        return context_vector

    def compute_mask(self, inputs: tf.Tensor, mask=None) -> tf.Tensor:
        """
        Computes the output mask tensor. This is required when the layer's call method returns a tensor with a different
        time dimension than the input tensor.

        Args:
        - inputs: Input tensor.
        - mask: Input mask tensor.
        """
        return None
