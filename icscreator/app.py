"""
Main file for launch application
"""
import numpy as np
import matplotlib.pyplot as plt
from icscreator.kernel.models import multilayer_srnn

NAME = 'engine'
MODEL_FOLDER = './static/models/'
DATA = "/home/alexander/Projects/ICSCreator/static/data/Data_JC.csv"


class DataGenerator(object):

    def __init__(self, input_data: np.ndarray, target_data: np.ndarray):
        assert len(input_data) == len(target_data), "Data should have equal length"
        self._input_data = input_data
        self._target_data = target_data
        self._data_length = len(input_data)

    def __len__(self):
        return self._data_length

    def run(self, batch=1) -> (np.ndarray, np.ndarray):
        while True:
            for i in range(self._data_length // batch):
                inp = self._input_data[i * batch: (i + 1) * batch]
                tar = self._target_data[i * batch: (i + 1) * batch]
                yield [inp * 0.0 + 0.2], [tar * 0.0 + 0.7]


data = np.genfromtxt(DATA, delimiter=',', skip_header=True)
fuel = data[::100, 1] / 4.0
freq = data[::100, 2] / 200000.0
temp = data[::100, 3] / 1000.0

traing_generator = DataGenerator(fuel, freq)

model = multilayer_srnn(1, 10, 1, name=NAME)
print(model.input.shape)
print(model.output.shape)
model.compile(optimizer='adam', loss='mse')
model.fit_generator(
    generator=traing_generator.run(batch=1),
    steps_per_epoch=10000,
    epochs=100,
)

results = []
for i in fuel:
    r = int(model.predict([i]))
    results.append(r)

plt.plot(results)
plt.show()
