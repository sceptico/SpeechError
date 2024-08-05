from typing import List
import numpy as np


def pad_sequences(sequences: List[np.ndarray], maxlen: int) -> np.ndarray:
    sequences = [seq if seq.ndim == 2 else np.expand_dims(
        seq, axis=-1) for seq in sequences]
    feature_dim = sequences[0].shape[1]
    padded_sequences = np.zeros((len(sequences), maxlen, feature_dim))

    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.shape[0], :] = seq

    return padded_sequences
