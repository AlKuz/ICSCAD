from abc import abstractmethod
import numpy as np


class DataGenerator(object):

    def __init__(self, path: str):
        self._path = path
        self._counter = 0

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise Exception("Method doesn't implemented")


class HDF5Generator(DataGenerator):
    pass


class CSVGenerator(DataGenerator):

    def __init__(self, path: str):
        assert path[-3] == "csv", "Wrong file extension"
        super(CSVGenerator, self).__init__(path=path)
        self._data = np.loadtxt(path, delimiter=',')

    def __next__(self):
        if self._counter < self._data.shape[0]:
            data = self._data[self._counter, ...]
            self._counter += 1
            return data
        else:
            self._counter = 0
            return self._data[self._counter, ...]


class SynthericGenerator(DataGenerator):
    pass