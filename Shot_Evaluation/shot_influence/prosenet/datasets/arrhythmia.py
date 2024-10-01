"""
Representation of the MIT-BIH ECG Arrhythmia Dataset.
"""
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical

from prosenet.datasets import BaseDataset


class ArrhythmiaDataset(BaseDataset):
    """
    Parameters
    ----------
    data_dir : str
        Path to directory containing 'mitbih_{train/test}.csv' files
    load_data : bool, optional
        Whether to load data on __init__, or delay until `load_data` call.
    normalize : bool, optional
        Whether to shift the data from range [0, 1] -> range[-1, 1]
    """
    def __init__(self, data_dir, load_data=True, normalize=True):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / 'mitbih_train.csv'
        assert self.train_path.exists() and self.train_path.is_file(), 'File must exist'

        self.test_path = self.data_dir / 'mitbih_test.csv'
        assert self.test_path.exists() and self.test_path.is_file(), 'File must exist'

        self.sequence_length = 187
        self.input_shape = (self.sequence_length,1)

        self.num_classes = 5
        self.output_shape = (self.num_classes,)

        self.normalize=normalize
        if load_data:
            self.load_data(normalize=self.normalize)


    def data_dirname(self):
        return self.data_dir


    def load_data(self, normalize=None):
        """
        Define X/y train/test.
        """
        normalize = normalize if normalize is not None else self.normalize

        train_data = np.loadtxt(self.train_path, delimiter=',')
        test_data = np.loadtxt(self.test_path, delimiter=',')

        self.X_train = train_data[:, :-1, np.newaxis]
        if normalize:
            self.X_train = self.X_train * 2.0 - 1.0
        self.y_train = to_categorical(train_data[:, -1].astype(np.int))

        self.X_test = test_data[:, :-1, np.newaxis]
        if normalize:
            self.X_test = self.X_test * 2.0 - 1.0
        self.y_test = to_categorical(test_data[:, -1].astype(np.int))


    def __repr__(self):
        return (
            'MIT-BIH Arrhythmia Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Input shape: {self.input_shape}\n'
            f'Train, Test counts: {self.X_train.shape[0]}, {self.X_test.shape[0]}\n'
        )
