"""
Just here to define mandatory Dataset subclass methods.
"""

class BaseDataset:
    def data_dirname(self):
        raise NotImplementedError()

    def load_data(self):
        raise NotImplementedError()
