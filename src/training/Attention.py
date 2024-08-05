import tensorflow as tf
from tensorflow.keras.layers import Layer


class Attention(Layer):
    """
    Custom Keras layer to compute the context vector using the attention mechanism.
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        The call method is used to define the computation that will be performed on the input data.

        Args:
        - inputs (tf.Tensor): The input tensor to the layer.

        Returns:
        - context_vector (tf.Tensor): The context vector computed using the attention mechanism.
        """
        # inputs.shape = (batch_size, timesteps, num_classes)
        p_t = inputs  # assuming inputs are the frame-level predictions

        # Compute attention weights
        sum_p_t = tf.reduce_sum(p_t, axis=1, keepdims=True)
        a_t = p_t / sum_p_t  # normalize over the time dimension

        # Compute context vector as a weighted sum of the frame-level features
        context_vector = tf.reduce_sum(a_t * inputs, axis=1)
        return context_vector
