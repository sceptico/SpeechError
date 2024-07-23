from tensorflow.keras import backend as K


def error_rate(y_true, y_pred):
    """
    Custom metric to calculate the error rate (ER).

    Args:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.

    Returns:
    - Error rate (ER).
    """
    # Ensure both tensors are of the same type
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # Convert predictions to binary
    y_pred = K.round(y_pred)

    # Calculate true positives, false positives, false negatives
    FP = K.sum(K.cast(y_pred, 'int32') - K.cast(y_true * y_pred, 'int32'))
    FN = K.sum(K.cast(y_true, 'int32') - K.cast(y_true * y_pred, 'int32'))

    # Calculate substitutions (S), deletions (D), and insertions (I)
    S = K.cast(K.minimum(FN, FP), 'float32')
    D = K.cast(K.maximum(0, FN - FP), 'float32')
    I = K.cast(K.maximum(0, FP - FN), 'float32')

    # Calculate total number of active sound events (N)
    N = K.sum(y_true)

    # Error rate (ER) calculation
    # Add epsilon to avoid division by zero
    # Divide by the total number of active sound events (N)
    # Divide by the total number of frames (y_true.shape[0])
    ER = (S + D + I) / (N + K.epsilon())

    return ER
