"""
Representation of a synthetic dataset of signals from different function types.
"""
import os
from pathlib import Path

import numpy as np
from scipy import signal
from tensorflow.keras.utils import to_categorical

from prosenet.datasets import BaseDataset


class SyntheticSignalsDataset(BaseDataset):
    """
    Upon `load_data`, generates synthetic saw, square, and sine wave signals.

    `n_range` specifies minimum and maximum number of oscillations per sequence.
    """
    def __init__(self, examples_per_class=5000, sequence_length=200, n_range=(1, 10), test_size=0.2):
        self.examples_per_class = examples_per_class
        self.sequence_length = sequence_length

        self.min_n, self.max_n = n_range
        assert self.max_n > self.min_n, 'Must be a positive range'

        assert test_size < 1.0, 'Must be a fraction'
        self.test_size = test_size


    def data_dir(self):
        return None


    def load_data(self):
        """Create the data"""
        t = np.linspace(0, 1, self.sequence_length)

        fs = np.random.random(size=self.examples_per_class)
        fs *= (self.max_n - self.min_n)
        fs += self.min_n

        saws = [signal.sawtooth(2 * np.pi * f * t) for f in fs]
        squares = [signal.square(2 * np.pi * f * t) for f in fs]
        sines = [np.sin(2 * np.pi * f * t) for f in fs]

        X = np.concatenate([saws, squares, sines])
        y = to_categorical([0]*fs.size + [1]*fs.size + [2]*fs.size)

        # shuffle and add features axis
        p = np.random.permutation(len(X))
        X, y = X[p, :, np.newaxis], y[p]

        # split
        split_idx = np.int((1 - self.test_size) * X.shape[0])
        self.X_train, self.y_train = X[:split_idx], y[:split_idx]
        self.X_test, self.y_test = X[split_idx:], y[split_idx:]
